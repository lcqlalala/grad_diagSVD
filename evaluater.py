import torch
import numpy as np
from tqdm import tqdm
import time
import itertools
from utils.data_utils import get_test_data
import os
import sys
import gc

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


def _dense_from_lowrank_pair(v_proj, u_proj):
    """Materialize u_proj(v_proj(x)) as one dense Linear for faster eval."""
    if isinstance(u_proj, torch.nn.Identity):
        return v_proj
    if not isinstance(v_proj, torch.nn.Linear) or not isinstance(u_proj, torch.nn.Linear):
        return None

    device = v_proj.weight.device
    dtype = v_proj.weight.dtype
    has_bias = u_proj.bias is not None
    dense = torch.nn.Linear(
        v_proj.in_features,
        u_proj.out_features,
        bias=has_bias,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        dense.weight.copy_(u_proj.weight.float().matmul(v_proj.weight.float()).to(dtype))
        if has_bias:
            dense.bias.copy_(u_proj.bias.to(dtype))
    return dense


def _materialize_lowrank_module(module):
    if not hasattr(module, "v_proj") or not hasattr(module, "u_proj"):
        return None
    return _dense_from_lowrank_pair(module.v_proj, module.u_proj)


def materialize_svd_attention_to_dense(model):
    """
    Convert only attention low-rank projections to dense Linear for evaluation.

    This keeps MLP projections low-rank, but removes the two-GEMM overhead from
    q/k/v/o attention projections. It is mathematically equivalent up to normal
    floating-point associativity differences.
    """
    converted = 0
    for module in model.modules():
        # LLaMA/Mistral SVD attention modules call q_u_proj(q_v_proj(x))
        # directly, so replace the V projection with a dense projection and
        # make the U projection an identity.
        for prefix in ("q", "k", "v", "o"):
            v_name = f"{prefix}_v_proj"
            u_name = f"{prefix}_u_proj"
            if not hasattr(module, v_name) or not hasattr(module, u_name):
                continue
            dense = _dense_from_lowrank_pair(getattr(module, v_name), getattr(module, u_name))
            if dense is None:
                continue
            setattr(module, v_name, dense)
            setattr(module, u_name, torch.nn.Identity())
            converted += 1

        # OPT SVD attention uses HuggingFace's forward and stores low-rank
        # projections as q_proj/k_proj/v_proj/out_proj modules.
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            child = getattr(module, name, None)
            dense = _materialize_lowrank_module(child)
            if dense is None:
                continue
            setattr(module, name, dense)
            converted += 1
    if converted > 0:
        print(f"[ppl_eval] materialized {converted} attention low-rank projections to dense Linear.")
    return converted


def _get_actual_device(model, fallback="cuda"):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def _cuda_synchronize_if_needed(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def ppl_eval(
    model,
    tokenizer,
    datasets=['wikitext2', 'ptb', 'c4'],
    model_seq_len=2048,
    batch_size=32,
    device="cuda",
    materialize_attn_dense=True,
):
    model.to(device)
    model.eval()

    actual_device = _get_actual_device(model, fallback=device)
    if actual_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(actual_device)
        pure_lowrank_weight_memory_mib = torch.cuda.memory_allocated(actual_device) / (1024 ** 2)
    else:
        pure_lowrank_weight_memory_mib = 0.0

    materialized_count = 0
    if materialize_attn_dense:
        materialized_count = materialize_svd_attention_to_dense(model)
        gc.collect()
        if actual_device.type == "cuda":
            torch.cuda.empty_cache()

    if actual_device.type == "cuda":
        materialized_weight_memory_mib = torch.cuda.memory_allocated(actual_device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(actual_device)
    else:
        materialized_weight_memory_mib = pure_lowrank_weight_memory_mib

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_id)
    ppls = {}
    total_effective_tokens = 0
    total_inference_time = 0.0
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size = batch_size)
        total_nll = 0.0
        total_tokens = 0
        skipped_batches = 0
        correct_tokens = 0
        first_stats = None
        for batch in tqdm(test_loader, desc=f"Evaluating {dataset}"):
            batch = batch.to(actual_device)
            if tokenizer.pad_token_id is None:
                attention_mask = torch.ones_like(batch, device=batch.device)
            else:
                attention_mask = batch.ne(tokenizer.pad_token_id).long()
            total_effective_tokens += int(attention_mask.sum().item())

            _cuda_synchronize_if_needed(actual_device)
            start_time = time.time()
            output = model(
                input_ids=batch,
                attention_mask=attention_mask,
                use_cache=False,
            )
            _cuda_synchronize_if_needed(actual_device)
            total_inference_time += time.time() - start_time

            lm_logits = output.logits
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = batch[:, 1:].contiguous().long()
                valid_mask = shift_labels.ne(pad_id).reshape(-1)
                
                loss = loss_fct(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                )
                valid_loss = loss[valid_mask]
                ntok = int(valid_mask.sum().item())
                batch_nll = float(valid_loss.sum().item())
                total_tokens += ntok
                total_nll += batch_nll
                with torch.no_grad():
                    pred = shift_logits.argmax(dim=-1)
                    correct_tokens += int(((pred == shift_labels) & shift_labels.ne(pad_id)).sum().item())
                if first_stats is None and ntok > 0:
                    first_stats = {
                        "batch_shape": tuple(batch.shape),
                        "logits_shape": tuple(lm_logits.shape),
                        "label_min": int(shift_labels.min().item()),
                        "label_max": int(shift_labels.max().item()),
                        "manual_loss": float(valid_loss.float().mean().item()) if valid_loss.numel() > 0 else None,
                        "logits_min": float(lm_logits.float().min().item()),
                        "logits_max": float(lm_logits.float().max().item()),
                    }
            else:
                skipped_batches += 1
        if total_tokens <= 0:
            raise RuntimeError(f"No finite logits were evaluated for {dataset}; skipped_batches={skipped_batches}.")
        mean_nll = total_nll / max(1, total_tokens)
        ppl = float(np.exp(mean_nll))
        ppls[dataset] = ppl
        if ppl < 1.01:
            acc = correct_tokens / max(1, total_tokens)
            print(
                f"[ppl_eval] WARNING: suspicious near-perfect PPL on {dataset}: "
                f"mean_nll={mean_nll:.8f}, next_token_acc={acc:.6f}, "
                f"total_tokens={total_tokens}, skipped_batches={skipped_batches}, stats={first_stats}"
            )
        elif skipped_batches > 0:
            print(f"[ppl_eval] WARNING: skipped {skipped_batches} non-finite batches on {dataset}.")
    effective_tokens_per_second = (
        total_effective_tokens / total_inference_time
        if total_inference_time > 0
        else 0.0
    )
    if actual_device.type == "cuda":
        peak_memory_mib = torch.cuda.max_memory_allocated(actual_device) / (1024 ** 2)
    else:
        peak_memory_mib = 0.0

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print(f"PPL: {ppls}")
    eval_graph = "attention dense materialized" if materialize_attn_dense else "pure low-rank"
    print(f"Effective Throughput ({eval_graph}): {effective_tokens_per_second:.2f} tokens/s (Excluding Padding)")
    print(f"Materialized Attention Projections: {materialized_count}")
    print(f"Model Weight Memory (Pure Low-Rank FP16): {pure_lowrank_weight_memory_mib:.2f} MiB")
    print(f"Model Weight Memory (After Attention Dense Materialization FP16): {materialized_weight_memory_mib:.2f} MiB")
    print(f"Peak VRAM Usage (Total): {peak_memory_mib:.2f} MiB")
    print("=" * 50 + "\n")
    return ppls

# only call this function when for 65b or more model    
@torch.no_grad()
def ppl_eval_large(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], seq_len=2048, batch_size=32, device="cuda"):
    import  torch.nn as nn
    class LlamaRMSNorm(nn.Module):
        def __init__(self, hidden_size=model.config.hidden_size, eps=model.config.rms_norm_eps):
            """
            LlamaRMSNorm is equivalent to T5LayerNorm
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
    norm = LlamaRMSNorm().half().cuda()
    lm_head = model.lm_head.cuda()
    model.eval()
    ppls = {}
    layers = model.model.layers
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            model.model.embed_tokens = model.model.embed_tokens.cuda()
            model.model.norm = model.model.norm.cuda()
            layers[0] = layers[0].cuda()

            dtype = next(iter(model.parameters())).dtype
            inps = torch.zeros(
                (batch.shape[0], model.seqlen, model.config.hidden_size), dtype=dtype, device="cuda"
            )
            cache = {'i': 0, 'attention_mask': None, "position_ids": None}
            class Catcher(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, **kwargs):
                    inps[cache['i']] = inp
                    cache['i'] += 1
                    if cache['attention_mask'] is None:
                        cache['attention_mask'] = kwargs['attention_mask']
                        cache['position_ids'] = kwargs['position_ids']
                    else:
                        cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                        cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
                    raise ValueError
            layers[0] = Catcher(layers[0])
            for j in range(batch.shape[0]):
                try:
                    model(batch[j].unsqueeze(0).cuda())
                except ValueError:
                    pass
            layers[0] = layers[0].module
            layers[0] = layers[0].cpu()
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            model.model.norm = model.model.norm.cpu()
            torch.cuda.empty_cache()
            attention_masks = cache['attention_mask']
            position_ids = cache['position_ids']
            for i in range(len(layers)):
                layer = layers[i].cuda()
                outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
                layers[i] = layer.cpu()
                inps = outs
                torch.cuda.empty_cache()
            hidden_states = norm(outs)
            lm_logits = lm_head(hidden_states)
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous().cuda()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
            else:
                print("warning: nan or inf in lm_logits")
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
    print("PPL after pruning: {}".format(ppls))
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda"):
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    weight_memory = torch.cuda.memory_allocated()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
            throughput += end_time - start_time
            print("time: {}".format(end_time - start_time))
    print("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
    print("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
    print("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
    print("Throughput: {} tokens/sec".format(token_num / throughput))
