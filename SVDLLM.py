#coding:utf8
import os
import sys
import argparse
import heapq
import math
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import * 

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)



@torch.no_grad()
def profle_svdllm(model_name, model, calib_loader, dev):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    handles = []
    for mod_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            handles.append(module.register_forward_hook(hook))
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for h in handles:
        h.remove()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat
        

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix
            if hasattr(subset[name], 'scaling_diag_matrix'):
                del subset[name].scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat


def _safe_cholesky(mat, eps=1e-6):
    try:
        return torch.linalg.cholesky(mat)
    except Exception:
        eig = torch.linalg.eigvalsh(mat)
        mat = mat + (-eig[0] + eps) * torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
        return torch.linalg.cholesky(mat)


def _min_eig_value(grad_inv_max, grad_eps):
    if grad_inv_max is None:
        return grad_eps
    return max(grad_eps, 1.0 / (grad_inv_max ** 2))


def _shift_psd_min_eig(mat, min_eig):
    eig = torch.linalg.eigvalsh(mat)
    if eig[0] < min_eig:
        mat = mat + (min_eig - eig[0]) * torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    return mat


def _get_block_size(name, model, attn_block_size, mlp_block_size):
    hidden_size = model.config.hidden_size
    num_heads = getattr(model.config, "num_attention_heads", None)
    head_dim = hidden_size // num_heads if num_heads else None
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj")):
        if attn_block_size and attn_block_size > 0:
            return attn_block_size
        return head_dim if head_dim else 0
    if any(k in name for k in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2")):
        if mlp_block_size and mlp_block_size > 0:
            return mlp_block_size
        return hidden_size
    return 0


def profile_grad_diag(model_name, model, calib_loader, dev, max_batches=None, grad_eps=1e-6, block_diag=False, attn_block_size=0, mlp_block_size=0, use_checkpointing=False, grad_inv_max=None, g_center=False, g_seq_level=False, g_cross_layer_norm=0.0):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    if g_seq_level and block_diag:
        print("[grad_diag] g_seq_level: block accumulators (grad_seq_blocks_sum) allocated on CPU "
              "to save ~870 MB GPU memory for LLaMA-7B. Hook transfers [B, block_size] slices to "
              "CPU before outer-product accumulation — negligible speed cost.")
    prev_training = model.training
    prev_use_cache = getattr(model.config, "use_cache", None)
    prev_dropout_cfg = {}
    if prev_use_cache is not None:
        model.config.use_cache = False
    prev_grad_ckpt = getattr(model, "is_gradient_checkpointing", False) or getattr(model, "gradient_checkpointing", False)
    if use_checkpointing:
        for key in ("attention_dropout", "hidden_dropout", "dropout", "activation_dropout", "classifier_dropout"):
            if hasattr(model.config, key):
                prev_dropout_cfg[key] = getattr(model.config, key)
                setattr(model.config, key, 0.0)
    if use_checkpointing and hasattr(model, "gradient_checkpointing_enable") and not prev_grad_ckpt:
        model.gradient_checkpointing_enable()
    if use_checkpointing:
        model.train()
    else:
        model.eval()
    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(True)

    handles = []
    def hook(module, grad_input, grad_output):
        if grad_output is None or grad_output[0] is None:
            return
        gout = grad_output[0].detach().float()
        if gout.dim() == 2:  # for opt
            gout = gout.unsqueeze(0)
        # gout: [B, T, D]
        B, T, D = gout.shape
        if getattr(module, "grad_block_ranges", None) is not None:
            gout2d = gout.reshape(-1, D)  # [B*T, D]
            n = gout2d.shape[0]
            module.grad_blocks_count += n
            module.grad_seq_count += B
            for bi, (s, e) in enumerate(module.grad_block_ranges):
                block = gout2d[:, s:e]               # [B*T, block_size]
                module.grad_blocks_sum[bi] += block.t().matmul(block)
                if g_center:
                    module.grad_blocks_mean_sum[bi] += block.sum(dim=0)
                if g_seq_level:
                    # Sum over T first → [B, block_size] on GPU, then move to CPU
                    # before the outer product.  The accumulator lives on CPU to avoid
                    # allocating ~870 MB of extra GPU block matrices for LLaMA-7B.
                    seq_block = gout[:, :, s:e].sum(dim=1).cpu()   # [B, block_size], CPU
                    module.grad_seq_blocks_sum[bi] += seq_block.t().matmul(seq_block)
                    if g_center:
                        module.grad_seq_blocks_mean_sum[bi] += seq_block.sum(dim=0)
        else:
            sumsq = gout.pow(2).sum(dim=(0, 1))
            module.grad_squares_sum += sumsq
            module.grad_squares_count += B * T
            if g_center:
                module.grad_squares_mean_sum += gout.sum(dim=(0, 1))
            if g_seq_level:
                # Sum over sequence dimension first: [B, D]
                seq_gout = gout.sum(dim=1)
                module.grad_seq_squares_sum += seq_gout.pow(2).sum(dim=0)
                module.grad_seq_squares_count += B
                if g_center:
                    module.grad_seq_squares_mean_sum += seq_gout.sum(dim=0)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if block_diag:
                block_size = _get_block_size(name, model, attn_block_size, mlp_block_size)
                out_dim = module.weight.shape[0]
                if block_size and block_size > 0:
                    ranges = []
                    for s in range(0, out_dim, block_size):
                        e = min(s + block_size, out_dim)
                        ranges.append((s, e))
                    module.grad_block_ranges = ranges
                    module.grad_blocks_sum = [
                        torch.zeros((e - s, e - s), device=dev) for (s, e) in ranges
                    ]
                    module.grad_blocks_count = 0
                    module.grad_seq_count = 0
                    if g_center:
                        module.grad_blocks_mean_sum = [
                            torch.zeros(e - s, device=dev) for (s, e) in ranges
                        ]
                    if g_seq_level:
                        # Allocated on CPU to save GPU memory: each block is (block_size×block_size)
                        # which totals ~870 MB for LLaMA-7B — enough to cause OOM during forward.
                        # The seq_block tensor that feeds into these accumulators is tiny ([B, block_size]),
                        # so the .cpu() transfer in the hook is essentially free.
                        module.grad_seq_blocks_sum = [
                            torch.zeros((e - s, e - s), device="cpu") for (s, e) in ranges
                        ]
                        if g_center:
                            module.grad_seq_blocks_mean_sum = [
                                torch.zeros(e - s, device="cpu") for (s, e) in ranges
                            ]
                else:
                    module.grad_block_ranges = None
                    module.grad_squares_sum = torch.zeros(out_dim, device=dev)
                    module.grad_squares_count = 0
                    if g_center:
                        module.grad_squares_mean_sum = torch.zeros(out_dim, device=dev)
                    if g_seq_level:
                        module.grad_seq_squares_sum = torch.zeros(out_dim, device=dev)
                        module.grad_seq_squares_count = 0
                        if g_center:
                            module.grad_seq_squares_mean_sum = torch.zeros(out_dim, device=dev)
            else:
                module.grad_block_ranges = None
                module.grad_squares_sum = torch.zeros(module.weight.shape[0], device=dev)
                module.grad_squares_count = 0
                if g_center:
                    module.grad_squares_mean_sum = torch.zeros(module.weight.shape[0], device=dev)
                if g_seq_level:
                    module.grad_seq_squares_sum = torch.zeros(module.weight.shape[0], device=dev)
                    module.grad_seq_squares_count = 0
                    if g_center:
                        module.grad_seq_squares_mean_sum = torch.zeros(module.weight.shape[0], device=dev)
            handles.append(module.register_full_backward_hook(hook))

    for i, batch in enumerate(tqdm(calib_loader)):
        if max_batches is not None and i >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        batch = {k: v.to(dev) for k, v in batch.items()}
        labels = batch["input_ids"]
        outputs = model(**batch, labels=labels, use_cache=False, output_attentions=False, output_hidden_states=False)
        loss = outputs.loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    grad_diag = {}
    max_raw_eig = 0.0  # track max eigenvalue across all blocks for diagnostics
    total_drift_ratio = 0.0   # fraction of G explained by mean gradient (centering diagnostic)
    total_drift_blocks = 0
    for i in range(len(layers)):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            module = subset[name]
            if getattr(module, "grad_block_ranges", None) is not None and module.grad_block_ranges:
                count     = max(module.grad_blocks_count, 1)
                seq_count = max(getattr(module, "grad_seq_count", 1), 1)
                blocks = []
                for bi, (s, e) in enumerate(module.grad_block_ranges):
                    if g_seq_level:
                        # G^seq normalized by NT (= count), not N (= seq_count).
                        # Formula: (1/NT) Σ_n (Σ_t δy_{n,t})(Σ_t δy_{n,t})^T
                        #
                        # Why NT not N:
                        #   G^seq/N  = (1/N)  Σ_n g_n g_n^T  ≈ T×G^tok×(1+(T-1)ρ)
                        #   G^seq/NT = (1/NT) Σ_n g_n g_n^T  ≈ G^tok×(1+(T-1)ρ)
                        #
                        # Dividing by N amplifies every layer by T and early layers
                        # (high temporal correlation ρ) by up to T² = 2048² = 4M×.
                        # This creates a 500× cross-layer G imbalance, causing Lagrange
                        # to push all free budget into early layer qkv (rank→2048) while
                        # starving mlp/down → PPL collapses (observed: 107 vs 68).
                        #
                        # Dividing by NT normalises to the same scale as G^tok:
                        # uncorrelated layers give G^seq/NT ≈ G^tok (same scale),
                        # while temporally-coherent layers give G^seq/NT > G^tok
                        # proportionally to their true correlation ρ.  Cross-layer
                        # imbalance is reduced from 500× back to the typical 10-50×.
                        mat = module.grad_seq_blocks_sum[bi] / count   # ÷ NT
                        if g_center:
                            mean_vec = module.grad_seq_blocks_mean_sum[bi] / count
                            mat = mat - mean_vec.unsqueeze(1).matmul(mean_vec.unsqueeze(0))
                    else:
                        # G^tok: (1/NT) Σ_{n,t} δy_{n,t} δy_{n,t}^T
                        mat = module.grad_blocks_sum[bi] / count
                        if g_center:
                            # G^c = G - μμ^T  where μ = (1/NT) Σ δy
                            mean_vec = module.grad_blocks_mean_sum[bi] / count
                            drift = mean_vec.unsqueeze(1).matmul(mean_vec.unsqueeze(0))
                            # Diagnostic: fraction of trace(G) explained by the mean direction
                            tr_g = float(mat.trace().item())
                            tr_d = float(drift.trace().item())
                            if tr_g > 0:
                                total_drift_ratio += tr_d / tr_g
                                total_drift_blocks += 1
                            mat = mat - drift
                    # Symmetrize to repair any floating-point asymmetry after centering
                    mat = (mat + mat.t()) * 0.5
                    # Track max raw eigenvalue before adding regularisation.
                    # If max_raw_eig stays near 0 for ALL blocks, the model is
                    # likely still in float16 during backprop and BUG 3 (model.float()
                    # before gradient profiling) has not taken effect.
                    raw_eig = float(torch.linalg.eigvalsh(mat)[-1].item())
                    if raw_eig > max_raw_eig:
                        max_raw_eig = raw_eig
                    if grad_eps:
                        mat = mat + grad_eps * torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
                    blocks.append({"start": s, "end": e, "mat": mat.cpu()})
                layer_profile[name] = {"type": "block", "blocks": blocks}
                del module.grad_blocks_sum
                del module.grad_blocks_count
                if hasattr(module, "grad_seq_count"):
                    del module.grad_seq_count
                if g_center and hasattr(module, "grad_blocks_mean_sum"):
                    del module.grad_blocks_mean_sum
                if g_seq_level and hasattr(module, "grad_seq_blocks_sum"):
                    del module.grad_seq_blocks_sum
                if g_seq_level and g_center and hasattr(module, "grad_seq_blocks_mean_sum"):
                    del module.grad_seq_blocks_mean_sum
            else:
                if g_seq_level:
                    # G^seq diagonal normalized by NT (= grad_squares_count), not N.
                    # Consistent with the block case fix: (1/NT) Σ_n (Σ_t δy_{n,t})^2
                    # For uncorrelated tokens this equals G^tok; for correlated early
                    # layers it is (1+(T-1)ρ) × G^tok, capturing temporal coherence
                    # without the T-fold scale inflation that causes cross-layer imbalance.
                    count = max(module.grad_squares_count, 1)   # NT, not N
                    diag = module.grad_seq_squares_sum / count
                    if g_center:
                        mean_vec = module.grad_seq_squares_mean_sum / count
                        diag = diag - mean_vec.pow(2)
                    diag = diag.clamp_min(grad_eps)
                else:
                    count = max(module.grad_squares_count, 1)
                    diag = module.grad_squares_sum / count
                    if g_center:
                        mean_vec = module.grad_squares_mean_sum / count
                        diag = diag - mean_vec.pow(2)
                    diag = diag.clamp_min(grad_eps)
                raw_max = float(diag.max().item())
                if raw_max > max_raw_eig:
                    max_raw_eig = raw_max
                layer_profile[name] = diag.cpu()
                del module.grad_squares_sum
                del module.grad_squares_count
                if g_center and hasattr(module, "grad_squares_mean_sum"):
                    del module.grad_squares_mean_sum
                if g_seq_level and hasattr(module, "grad_seq_squares_sum"):
                    del module.grad_seq_squares_sum
                    del module.grad_seq_squares_count
                if g_seq_level and g_center and hasattr(module, "grad_seq_squares_mean_sum"):
                    del module.grad_seq_squares_mean_sum
        grad_diag[i] = layer_profile

    # -----------------------------------------------------------------------
    # Cross-layer G normalization (--g_cross_layer_norm BETA, 0.0 = disabled).
    #
    # Problem: --g_seq_level (and to a lesser extent plain token-level G) can
    # create extreme cross-layer G dynamic range (e.g. 453×).  The Lagrange
    # water-filling then dumps all free budget into the single highest-G layer
    # (typically layer 0 qkv), starving mlp/down and collapsing PPL.
    #
    # Fix: after computing all G values, scale each layer's G matrix toward the
    # global geometric mean by a factor of (geom_mean / layer_max_G)^BETA:
    #
    #   G'_layer = G_layer * (geom_mean / max_eig(G_layer))^BETA
    #
    # Effect on the cross-layer ratio R = max_G / min_G:
    #   After normalization:  R' = R^(1 - BETA)
    #   BETA=0.0: R'=R  (disabled, no change)
    #   BETA=0.5: R' = sqrt(453) ≈ 21×   (moderate compression)
    #   BETA=0.75: R' = 453^0.25 ≈ 4.6×  (strong compression)
    #   BETA=1.0: R' = 1×                 (full normalization, all layers equal)
    #
    # Within-layer G structure (relative eigenvalue ratios within each block)
    # is preserved — only the cross-layer SCALE is adjusted.  The Lagrange
    # search still uses G signal to distinguish which modules within a layer
    # are more important; it only redistributes the budget MORE EVENLY across
    # layers.
    # -----------------------------------------------------------------------
    if g_cross_layer_norm > 0.0:
        eps_cn = grad_eps if grad_eps and grad_eps > 0 else 1e-9
        # Step 1: compute max raw eigenvalue per layer
        layer_max_eig = {}
        for lid, layer_profile in grad_diag.items():
            lmax = eps_cn
            for entry in layer_profile.values():
                if isinstance(entry, dict) and entry.get("type") == "block":
                    for b in entry["blocks"]:
                        eig = float(torch.linalg.eigvalsh(b["mat"])[-1].item())
                        lmax = max(lmax, eig)
                else:
                    lmax = max(lmax, float(entry.max().item()))
            layer_max_eig[lid] = max(lmax, eps_cn)
        # Step 2: geometric mean across layers
        log_vals = [math.log(v) for v in layer_max_eig.values()]
        geom_mean = math.exp(sum(log_vals) / len(log_vals))
        orig_ratio = max(layer_max_eig.values()) / min(layer_max_eig.values())
        # Step 3: apply normalization scale to each layer's G entries
        for lid, layer_profile in grad_diag.items():
            lmax = layer_max_eig[lid]
            scale = (geom_mean / lmax) ** g_cross_layer_norm
            if abs(scale - 1.0) < 1e-9:
                continue
            for name, entry in layer_profile.items():
                if isinstance(entry, dict) and entry.get("type") == "block":
                    for b in entry["blocks"]:
                        b["mat"] = b["mat"] * scale
                else:
                    layer_profile[name] = entry * scale
        compressed_ratio = orig_ratio ** (1.0 - g_cross_layer_norm)
        print(f"[grad_diag] g_cross_layer_norm={g_cross_layer_norm:.2f}: "
              f"cross-layer G ratio compressed {orig_ratio:.0f}× → {compressed_ratio:.1f}× "
              f"(geom_mean={geom_mean:.2e}, BETA={g_cross_layer_norm:.2f})")

    # Report centering diagnostic: how much of G was explained by the mean gradient
    if g_center and total_drift_blocks > 0:
        mean_drift_pct = 100.0 * total_drift_ratio / total_drift_blocks
        print(f"[grad_diag] g_center: mean drift ratio = {mean_drift_pct:.1f}%  "
              f"(fraction of trace(G) explained by mean gradient μμᵀ; "
              f"high % → systematic gradient bias removed by centering)")
    if g_seq_level:
        label = "sequence-level G^seq/NT" if not g_center else "centered sequence-level G^seq/NT"
        print(f"[grad_diag] g_seq_level: using {label} — "
              f"G_j = (1/NT) Σ_n (Σ_t δy_{{n,t}})^[j] (Σ_t δy_{{n,t}})^[j]ᵀ  "
              f"(normalized by NT so G^seq ≈ G^tok for uncorrelated tokens)")
    # Warn when cross-layer G dynamic range is extreme (>100×) — this causes
    # Lagrange to dump all free budget into the highest-G layer, starving
    # other modules and collapsing PPL.  Common trigger: --g_seq_level with
    # old /N normalization (now fixed to /NT), or very aggressive cascade_alpha.
    if len(grad_diag) >= 2:
        eps_diag = grad_eps if grad_eps and grad_eps > 0 else 1e-6
        layer_maxes = []
        for lid in sorted(grad_diag.keys()):
            layer_max = 0.0
            for entry in grad_diag[lid].values():
                if isinstance(entry, dict) and entry.get("type") == "block":
                    for b in entry["blocks"]:
                        v = float(torch.linalg.eigvalsh(b["mat"])[-1].item()) - eps_diag
                        layer_max = max(layer_max, max(v, 0.0))
                else:
                    layer_max = max(layer_max, max(0.0, float(entry.max().item()) - eps_diag))
            layer_maxes.append(layer_max)
        valid = [v for v in layer_maxes if v > 0]
        if len(valid) >= 2:
            cross_layer_ratio = max(valid) / min(valid)
            if cross_layer_ratio > 100:
                beta50 = cross_layer_ratio ** 0.5
                beta75 = cross_layer_ratio ** 0.25
                print(
                    f"\n[grad_diag] WARNING: cross-layer G dynamic range = {cross_layer_ratio:.0f}×  "
                    f"(max layer G / min layer G).\n"
                    f"  Lagrange water-filling dumps all free budget into the single highest-G layer,\n"
                    f"  allocating rank >> uniform there while starving mlp/down → PPL >> uniform.\n"
                    f"  Causes & recommended fixes:\n"
                    f"    1. Use --g_cross_layer_norm BETA to compress the cross-layer ratio:\n"
                    f"         BETA=0.50 → {beta50:.0f}×  (moderate, recommended first try)\n"
                    f"         BETA=0.75 → {beta75:.1f}×  (strong compression)\n"
                    f"         BETA=1.00 → 1×      (full equalization, disables cross-layer G signal)\n"
                    f"       This compresses the scale of each layer's G toward the geometric mean\n"
                    f"       while PRESERVING within-layer/within-block G structure.\n"
                    f"    2. --g_seq_level adds temporal correlation amplification to early layers.\n"
                    f"       Even with /NT normalization, high-ρ early layers can be 100–500× above\n"
                    f"       low-ρ late layers.  Try removing --g_seq_level if ratio stays >100×.\n"
                    f"    3. Remove --layer_ratio_floor (or set to 0.0): if it consumes >90%% of\n"
                    f"       the budget the imbalance is moot (Lagrange has no free budget anyway).\n"
                    f"    4. Set --module_rank_max 1280 to cap extreme individual module ranks.\n"
                    f"    5. Raise --module_rank_min_mlp / --module_rank_min_down to ~uniform rank\n"
                    f"       ({cross_layer_ratio:.0f}× imbalance will always starve late-layer mlp/down).\n"
                    f"    6. --grad_cascade_alpha too large → try 0.0.\n"
                )
    # Diagnostic: if max_raw_eig is near 0, gradient profiling captured no signal.
    # Two different root causes produce the same G≈I symptom; the diagnostic
    # distinguishes them so the user can apply the right fix.
    eps_ref = grad_eps if grad_eps and grad_eps > 0 else 1e-6

    # Threshold to call the signal "real": at least 10x above grad_eps so the
    # regularisation does not drown it out.
    has_real_signal = max_raw_eig > 10 * eps_ref

    if has_real_signal:
        signal_label = "REAL SIGNAL"
    elif max_raw_eig < 1e-20:
        signal_label = "G≈I — gradient is effectively ZERO (float16 underflow?)"
    else:
        signal_label = (f"G≈I — signal exists but is masked by --grad_eps "
                        f"({max_raw_eig:.1e} < 10×grad_eps={10*eps_ref:.1e})")

    print(f"[grad_diag] max raw block eigenvalue = {max_raw_eig:.3e}  "
          f"(grad_eps={eps_ref:.1e}; ratio={max_raw_eig/eps_ref:.1f}x; {signal_label})")

    # -----------------------------------------------------------------------
    # BUG 6 CHECK: --grad_eps is too large — it masks real gradient signal.
    #
    # profile_grad_diag adds  grad_eps * I  to every gradient block.  If
    # grad_eps ≥ max_raw_eig the regularisation dominates every block's
    # eigenvalue, making G ≈ grad_eps · I uniformly.  The G-weighted rank
    # allocation then degenerates to pure WH-energy (G is a no-op), just as
    # BUG 5 does via over-clamping.
    #
    # This is DIFFERENT from BUG 3 (float16 underflow where max_raw_eig ≈ 0).
    # Here the model IS in float32 and gradients ARE real, but grad_eps is just
    # too large for this model.  Typical LLaMA-7B gradient block eigenvalues
    # are ~ 1e-8 to 1e-7 — far below the default grad_eps = 1e-6.
    # -----------------------------------------------------------------------
    if (not has_real_signal) and max_raw_eig > 1e-30:
        # Gradient signal IS present (max_raw_eig > 0) but grad_eps is too big.
        safe_eps = max_raw_eig * 1e-2          # 100× below the signal
        safe_eps_str = f"{safe_eps:.0e}"
        safe_inv_max = max(1, int(math.ceil(1.0 / math.sqrt(max_raw_eig / 10)))) + 1
        print(
            f"\n[grad_diag] *** BUG 6 WARNING: --grad_eps IS TOO LARGE ***\n"
            f"  max raw block eigenvalue = {max_raw_eig:.2e}\n"
            f"  grad_eps                 = {eps_ref:.2e}  "
            f"({eps_ref/max_raw_eig:.0f}× LARGER than the signal)\n"
            f"\n"
            f"  profile_grad_diag adds grad_eps·I to every block, so every\n"
            f"  block's eigenvalue becomes  ≈ grad_eps = {eps_ref:.2e}.\n"
            f"  This makes G ≈ {math.sqrt(eps_ref):.2e}·I uniformly — a NO-OP.\n"
            f"\n"
            f"  Root cause:  --grad_eps {eps_ref:.0e}  >>  actual gradient\n"
            f"               eigenvalues ({max_raw_eig:.2e}) for this model.\n"
            f"  This is NOT a float16 issue (BUG 3). The model IS in float32;\n"
            f"  gradients are real but tiny for LLaMA-scale models.\n"
            f"\n"
            f"  FIX — use both of these flags:\n"
            f"    --grad_eps {safe_eps_str}          "
            f"(100× below max signal = {max_raw_eig:.1e})\n"
            f"    --grad_inv_max {safe_inv_max}      "
            f"(min_eig = 1/{safe_inv_max}² ≈ {1.0/safe_inv_max**2:.1e} "
            f"< {max_raw_eig:.1e})\n"
            f"\n"
            f"  After this fix, G-weighting will correctly allocate more rank\n"
            f"  to later (high-gradient) layers and less to early layers,\n"
            f"  matching the KFAC theory and improving PPL vs uniform allocation.\n"
        )

    # -----------------------------------------------------------------------
    # BUG 5 CHECK: --grad_inv_max clamps every gradient block (previous fix).
    # Only warn here if the gradient signal is real (has_real_signal).
    # -----------------------------------------------------------------------
    if grad_inv_max is not None and grad_inv_max > 0:
        min_eig_effective = max(eps_ref, 1.0 / (grad_inv_max ** 2))
        if has_real_signal and max_raw_eig < min_eig_effective:
            safe_inv_max = int(math.ceil(1.0 / math.sqrt(max_raw_eig))) + 1
            print(
                f"\n[grad_diag] *** CRITICAL WARNING: G-weighting DISABLED by --grad_inv_max ***\n"
                f"  min_eig = 1/{grad_inv_max}² = {min_eig_effective:.2e}  >  "
                f"max_raw_eig = {max_raw_eig:.2e}\n"
                f"  Every gradient block is clamped to the same constant ({min_eig_effective:.2e}),\n"
                f"  making G = {math.sqrt(min_eig_effective):.3f}·I uniformly.  "
                f"G-based rank redistribution is a NO-OP.\n"
                f"  With --layer_ratio_floor below the target, the Lagrange search will\n"
                f"  reallocate ranks using ONLY WH-energy, which under-allocates rank to\n"
                f"  PPL-critical attention modules (q/k/v/o_proj), often raising PPL far\n"
                f"  above the uniform-allocation baseline.\n"
                f"  FIX: lower --grad_inv_max to at most {safe_inv_max} so that\n"
                f"  min_eig = 1/{safe_inv_max}² ≈ {1.0/safe_inv_max**2:.1e} < {max_raw_eig:.1e}.\n"
                f"  Alternatively add --module_rank_min_qkv <floor_rank> to protect\n"
                f"  attention projections regardless of the G estimate.\n"
            )

    # ------------------------------------------------------------------
    # Per-layer gradient diagnostic: show how gradient magnitude varies
    # across layers.  For LLaMA-7B the ratio between early and late
    # layers is typically ~10,000x, which causes extreme rank
    # redistribution even when the global G signal is real.
    # ------------------------------------------------------------------
    min_eig_clamp = (
        max(eps_ref, 1.0 / (grad_inv_max ** 2))
        if (grad_inv_max is not None and grad_inv_max > 0)
        else eps_ref
    )
    print(f"[grad_diag] Per-layer max raw G eigenvalue "
          f"(min_eig_clamp={min_eig_clamp:.1e}, eps_ref={eps_ref:.1e}):")
    # Three activation tiers:
    #   clamped    : layer_max_raw < min_eig_clamp (G = constant)
    #   weak       : 1× ≤ ratio < 10×  (G barely above floor; still ~= constant)
    #   ACTIVE     : ratio ≥ 10×        (meaningful differential G-weighting)
    # "weak" layers appear ACTIVE in the binary sense but their G eigenvalues
    # are so close to min_eig that G ≈ constant·I for the whole block.  They
    # behave similarly to clamped layers for rank allocation purposes.
    STRONG_RATIO = 10.0
    n_clamped_layers = 0
    n_weak_layers = 0
    for layer_id in sorted(grad_diag.keys()):
        layer_max_raw = 0.0
        mod_type_max   = {}   # "qkv" / "o" / "mlp" / "other" → max raw eig (over all blocks)
        mod_type_sum   = {}   # sum of per-block raw eig  (for mean computation)
        mod_type_count = {}   # number of blocks per module type
        def _mtype(nm):
            if any(t in nm for t in ("q_proj", "k_proj", "v_proj")):
                return "qkv"
            if any(t in nm for t in ("o_proj", "out_proj")):
                return "o"
            if any(t in nm for t in ("gate_proj", "up_proj", "down_proj")):
                return "mlp"
            return "other"
        for mname, entry in grad_diag[layer_id].items():
            mtype = _mtype(mname)
            if isinstance(entry, dict) and entry.get("type") == "block":
                for b in entry["blocks"]:
                    # mat = raw_mat + grad_eps*I; subtracting eps_ref recovers raw eig
                    eig_val = float(torch.linalg.eigvalsh(b["mat"])[-1].item())
                    raw = max(0.0, eig_val - eps_ref)
                    layer_max_raw = max(layer_max_raw, raw)
                    mod_type_max[mtype]    = max(mod_type_max.get(mtype, 0.0), raw)
                    mod_type_sum[mtype]    = mod_type_sum.get(mtype, 0.0) + raw
                    mod_type_count[mtype]  = mod_type_count.get(mtype, 0) + 1
            else:
                # diagonal stored after clamp_min(grad_eps); subtract eps_ref for raw
                raw = max(0.0, float(entry.max().item()) - eps_ref)
                layer_max_raw = max(layer_max_raw, raw)
                mod_type_max[mtype]    = max(mod_type_max.get(mtype, 0.0), raw)
                mod_type_sum[mtype]    = mod_type_sum.get(mtype, 0.0) + raw
                mod_type_count[mtype]  = mod_type_count.get(mtype, 0) + 1
        ratio_to_floor = (layer_max_raw / min_eig_clamp
                          if min_eig_clamp > 0 else float("inf"))
        if layer_max_raw < min_eig_clamp:
            label = "clamped              "
            n_clamped_layers += 1
        elif ratio_to_floor < STRONG_RATIO:
            label = f"weak   ×{ratio_to_floor:5.1f} floor"
            n_weak_layers += 1
        else:
            label = f"ACTIVE ×{ratio_to_floor:7.0f} floor"
        # Per-module-type breakdown: shows max and mean G per block, plus
        # concentration ratio (max/mean).  A high concentration (e.g. ×8c)
        # means only a few head-blocks drive the max G; most blocks sit at
        # min_eig, so the effective natural rank is still low despite the
        # high peak — explaining why a module can be at its rank floor even
        # when its max_raw_G looks large.
        parts = []
        for mt in ("qkv", "o", "mlp"):
            v     = mod_type_max.get(mt, 0.0)
            cnt   = mod_type_count.get(mt, 1)
            mean_v = mod_type_sum.get(mt, 0.0) / max(cnt, 1)
            conc  = v / max(mean_v, eps_ref)   # concentration ratio: max/mean
            ratio_mt = v / min_eig_clamp if min_eig_clamp > 0 else 0.0
            if ratio_mt >= STRONG_RATIO:
                tier = "A"   # strongly ACTIVE
            elif v >= min_eig_clamp:
                tier = "w"   # weak
            else:
                tier = "-"   # clamped
            parts.append(f"{mt}={v:.1e}/{mean_v:.1e}[{tier}]×{conc:.1f}c")
        module_detail = "  " + "  ".join(parts)
        print(f"  layer {layer_id:02d}: max_raw_G={layer_max_raw:.2e}  [{label}]{module_detail}")
    n_layers = len(grad_diag)
    if n_layers > 0:
        n_strong = n_layers - n_clamped_layers - n_weak_layers
        pct_clamped = 100.0 * n_clamped_layers / n_layers
        pct_weak = 100.0 * n_weak_layers / n_layers
        print(f"  [grad_diag] G activation: {n_strong} ACTIVE (≥10× floor)  "
              f"{n_weak_layers} weak (1–10×)  {n_clamped_layers} clamped (<1×)  "
              f"out of {n_layers} layers")
        # Warn when clamped OR weak layers dominate — both lead to G ≈ constant·I
        # for those layers, which concentrates rank in the few strongly ACTIVE layers.
        n_not_strong = n_clamped_layers + n_weak_layers
        pct_not_strong = 100.0 * n_not_strong / n_layers
        if pct_not_strong > 50:
            print(
                f"  [grad_diag] WARNING: only {n_strong}/{n_layers} layers have "
                f"strong G signal (≥10× floor).  {n_not_strong} layers "
                f"({'clamped+weak' if n_clamped_layers and n_weak_layers else 'clamped' if n_clamped_layers else 'weak'}) "
                f"act as G≈I and receive a disproportionately small rank share.\n"
                f"  Early-layer compression errors cascade and often raise PPL above "
                f"the uniform-allocation baseline.\n"
                f"  Options to reduce rank concentration:\n"
                f"    1. Raise --layer_ratio_floor (e.g. 0.38–0.40)\n"
                f"    2. Add --module_rank_min_qkv / --module_rank_min_mlp\n"
                f"    3. Adjust --grad_inv_max to change the floor "
                f"(current min_eig={min_eig_clamp:.1e})\n"
            )

    model = model.cpu()
    if use_checkpointing and hasattr(model, "gradient_checkpointing_disable") and not prev_grad_ckpt:
        model.gradient_checkpointing_disable()
    for key, value in prev_dropout_cfg.items():
        setattr(model.config, key, value)
    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache
    model.train(prev_training)
    torch.set_grad_enabled(prev_grad)
    return grad_diag


def compute_layer_ratios(model_name, model, grad_diag, base_ratio, min_ratio=0.01, max_ratio=0.99, eps=1e-6):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        return None
    layer_sizes = []
    layer_scores = []
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        size_i = 0
        score_i = 0.0
        for name in subset:
            W = subset[name].weight
            size_i += W.shape[0] * W.shape[1]
            if grad_diag is not None and i in grad_diag and name in grad_diag[i]:
                entry = grad_diag[i][name]
                if isinstance(entry, dict) and entry.get("type") == "block":
                    for b in entry["blocks"]:
                        mat = b["mat"]
                        score_i += float(torch.trace(mat).item())
                else:
                    score_i += float(entry.sum().item())
        if score_i <= 0:
            score_i = eps
        layer_sizes.append(size_i)
        layer_scores.append(score_i)
    total_size = sum(layer_sizes)
    if total_size == 0:
        return None
    weighted_mean = sum(s * w for s, w in zip(layer_sizes, layer_scores)) / total_size
    ratios = [base_ratio * (w / weighted_mean) for w in layer_scores]
    target = base_ratio * total_size

    # Iteratively clamp and rescale to hit the global budget.
    for _ in range(5):
        fixed = []
        free = []
        for i, r in enumerate(ratios):
            if r < min_ratio:
                ratios[i] = min_ratio
                fixed.append(i)
            elif r > max_ratio:
                ratios[i] = max_ratio
                fixed.append(i)
            else:
                free.append(i)
        fixed_size = sum(layer_sizes[i] for i in fixed)
        free_size = sum(layer_sizes[i] for i in free)
        if free_size <= 0:
            break
        fixed_budget = sum(layer_sizes[i] * ratios[i] for i in fixed)
        remain = target - fixed_budget
        if remain <= 0:
            break
        scale = remain / sum(layer_sizes[i] * ratios[i] for i in free)
        updated = False
        for i in free:
            new_r = ratios[i] * scale
            if new_r < min_ratio:
                ratios[i] = min_ratio
                updated = True
            elif new_r > max_ratio:
                ratios[i] = max_ratio
                updated = True
            else:
                ratios[i] = new_r
        if not updated:
            break
    return ratios


@torch.no_grad()
def profile_module_spectrum(model_name, model, profiling_mat, dev, grad_diag=None, grad_eps=1e-6, grad_inv_max=None, cascade_alpha=0.0, g_blend_alpha=1.0, g_sigma_space=False):
    """Compute the G-H-W spectrum for each module.

    cascade_alpha: depth-based correction factor that compensates for the
      first-order KFAC approximation ignoring sequential forward error
      cascading.  Layer i's G matrix is scaled by (1 + cascade_alpha * i/(L-1))
      before the spectrum is computed.  Later layers get a larger correction
      because their compression errors are compounded by all preceding layers
      (the KFAC first-order approximation does not model these interaction terms
      and systematically underestimates the importance of later layers when
      multiple layers are compressed simultaneously).

      cascade_alpha = 0.0  →  no correction (default, original KFAC)
      cascade_alpha = 1.0  →  last layer G is 2× the raw value
      cascade_alpha = 2.0  →  last layer G is 3× the raw value

      Try 0.5–2.0 when the per-layer G diagnostic shows that ACTIVE layers are
      concentrated in early layers and late layers are all at the rank floor.

    g_blend_alpha: within-block G concentration reduction factor (1.0 = pure
      KFAC block Cholesky, 0.0 = block-mean scalar only).

      The block-diagonal G for attention/MLP groups can be highly concentrated:
      a few head-blocks have max eigenvalue 10–20× larger than the block mean
      (shown as ×Nc in the per-module-type diagnostic).  With g_blend_alpha=1.0
      the full block Cholesky G^{1/2} weights input rows differently within each
      block, focusing the SVD budget on the block's top principal directions.
      This causes the KFAC natural rank to be very low (all energy in ~2/32 head
      blocks), leading to floor-binding for most modules.

      With g_blend_alpha < 1.0 we replace the block Cholesky with a blend:
          G_eff^{1/2}_block = alpha * Chol(G_block) + (1-alpha) * sqrt(mean_eig) * I
      The second term gives each input dim of the block the SAME weight = sqrt
      of the block's mean eigenvalue, preserving cross-block variation (hot blocks
      still outscale cold blocks) while eliminating within-block concentration.

      g_blend_alpha = 1.0  →  full KFAC (default, original behaviour)
      g_blend_alpha = 0.5  →  50% KFAC, 50% block-mean (typical concentration
                                ×19.8c → ×2.5c after blend)
      g_blend_alpha = 0.0  →  block-mean only (pure cross-block variation,
                                concentration ≈ ×1.0c within each block)

      For diagonal G, blends g_sqrt with mean(g_sqrt) over the whole module,
      achieving the same concentration-reduction effect.

      Try 0.3–0.7 when all modules are floor-bound and the grad_diag shows
      high within-block concentration (×Nc >> 1).

    g_sigma_space: project G to the left-singular-vector basis of W before
      applying the Cholesky factor (only affects block-diagonal G entries).

      Standard KFAC minimises ½‖G^{1/2}(W−W̃)H‖²_F in row-space.  The optimal
      rank-r approximation to the G-preconditioned weight is:
          W̃* = U_r U_r^T W  (project to top-r left singular vectors of G^{1/2}W)
      When G is block-diagonal but NOT aligned with the SVD basis of W, the
      Cholesky rotation G^{1/2} mixes the SVD components in a way that the
      subsequent truncated SVD cannot fully undo — some budget is wasted on
      off-axis directions.

      g_sigma_space re-expresses G in the singular-vector basis of W_block:
          G^σ_j = U_j^T  G_j  U_j          (projection, h×h)
          G_eff^{1/2} = Chol(G^σ_j) @ diag(Σ_j) @ Vh_j
      so W_scaled_block = Chol(G^σ_j) @ diag(Σ_j) @ Vh_j.
      The singular values are now scored in the basis where G is expressed in
      σ-space: the off-diagonal elements of G^σ couple different singular modes,
      so the final svdvals(W_scaled @ H^{1/2}) reflect the true KFAC cost of
      retaining each σ-direction including its cross-mode interactions.

      g_sigma_space = False  →  standard KFAC (default)
      g_sigma_space = True   →  σ-space projected G with off-diagonal coupling

      Only affects modules with block-diagonal G (requires --g_block_diag).
      For diagonal G, g_sigma_space is a no-op.
    """
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        return None
    spectra = {}
    model.eval()
    n_layers = len(layers)
    if cascade_alpha and cascade_alpha != 0.0:
        print(f"[spectrum] cascade correction enabled (alpha={cascade_alpha:.2f}): "
              f"layer 0 G×1.00,  layer {n_layers//2} G×{1+cascade_alpha*0.5:.2f},  "
              f"layer {n_layers-1} G×{1+cascade_alpha:.2f}")
    if g_blend_alpha < 1.0:
        if g_blend_alpha <= 0.0:
            print(f"[spectrum] G blend: block-mean scalar only (alpha=0.0) — "
                  f"within-block concentration fully removed, cross-block variation preserved.")
        else:
            print(f"[spectrum] G blend: alpha={g_blend_alpha:.2f} — "
                  f"G_eff^{{1/2}}_block = {g_blend_alpha:.2f}×Chol(G) + {1-g_blend_alpha:.2f}×sqrt(mean_eig)×I. "
                  f"Reduces within-block concentration (×Nc) toward ×1.0c.")
    if g_sigma_space:
        print(f"[spectrum] g_sigma_space=True: G projected to W σ-space with off-diagonal coupling. "
              f"G^σ_j = U_j^T G_j U_j; W_scaled = Chol(G^σ_j) @ diag(Σ_j) @ Vh_j. "
              f"Requires --g_block_diag; no-op for diagonal G modules.")
    for i in tqdm(range(n_layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_spec = {}
        # Cascade correction: later layers get proportionally higher G-weighting
        # to compensate for KFAC underestimating their importance under
        # simultaneous multi-layer compression.
        cascade_scale = 1.0 + cascade_alpha * i / max(n_layers - 1, 1)
        for name in subset:
            # clone to avoid any accidental in-place modification of model weights
            W = subset[name].weight.data.float().to(dev).clone()
            if grad_diag is not None and i in grad_diag and name in grad_diag[i]:
                entry = grad_diag[i][name]
                if isinstance(entry, dict) and entry.get("type") == "block":
                    W_scaled = W.clone()
                    for b in entry["blocks"]:
                        s, e = b["start"], b["end"]
                        mat = b["mat"].to(dev)
                        if cascade_scale != 1.0:
                            mat = mat * cascade_scale
                        min_eig = _min_eig_value(grad_inv_max, grad_eps)
                        mat = _shift_psd_min_eig(mat, min_eig)
                        chol = _safe_cholesky(mat, grad_eps)
                        if g_sigma_space:
                            # σ-space projected G with off-diagonal coupling.
                            # Project G into the left-singular-vector basis of W_block so
                            # that the Cholesky factor scores each singular mode by its
                            # true G-weighted cost including cross-mode interactions:
                            #   G^σ_j = U_j^T G_j U_j         (G in σ-basis, h×h)
                            #   W_scaled = Chol(G^σ_j) @ diag(Σ_j) @ Vh_j
                            W_block = W[s:e, :]                                # [h, d_in]
                            U_j, S_j, Vh_j = torch.linalg.svd(W_block, full_matrices=False)
                            mat_sigma = U_j.t().matmul(mat).matmul(U_j)        # [h, h]
                            chol_sigma = _safe_cholesky(mat_sigma, grad_eps)
                            W_scaled[s:e, :] = chol_sigma.matmul(
                                torch.diag(S_j).matmul(Vh_j)
                            )
                        elif g_blend_alpha >= 1.0:
                            # Pure KFAC: full block Cholesky
                            W_scaled[s:e, :] = chol.matmul(W[s:e, :])
                        else:
                            # Blended: alpha*Chol(G) + (1-alpha)*sqrt(mean_eig)*I
                            # Preserves cross-block variation (hot vs cold blocks)
                            # while reducing within-block concentration (max/mean).
                            mean_eig_val = float(torch.linalg.eigvalsh(mat).mean().item())
                            uniform_scale = math.sqrt(max(mean_eig_val, grad_eps))
                            if g_blend_alpha <= 0.0:
                                # Pure block-mean: each dim in block gets same scale
                                W_scaled[s:e, :] = uniform_scale * W[s:e, :]
                            else:
                                blk_sz = e - s
                                eye = torch.eye(blk_sz, device=dev, dtype=chol.dtype)
                                chol_blend = g_blend_alpha * chol + (1.0 - g_blend_alpha) * uniform_scale * eye
                                W_scaled[s:e, :] = chol_blend.matmul(W[s:e, :])
                    W = W_scaled
                else:
                    g_diag = entry.to(dev)
                    if cascade_scale != 1.0:
                        g_diag = g_diag * cascade_scale
                    min_eig = _min_eig_value(grad_inv_max, grad_eps)
                    g_diag = torch.clamp(g_diag, min=min_eig)
                    g_sqrt = torch.sqrt(torch.clamp(g_diag, min=grad_eps))
                    if g_blend_alpha < 1.0:
                        # For diagonal G, blend g_sqrt with the module-mean
                        # to reduce the within-module concentration effect.
                        mean_scale = float(g_sqrt.mean().item())
                        g_sqrt = g_blend_alpha * g_sqrt + (1.0 - g_blend_alpha) * mean_scale
                    W = W * g_sqrt.unsqueeze(1)
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev).float()
                W_scale = torch.matmul(W, scaling_diag_matrix)
            else:
                W_scale = W
            try:
                svals = torch.linalg.svdvals(W_scale)
            except Exception:
                _, svals, _ = torch.linalg.svd(W_scale, full_matrices=False)
            layer_spec[name] = {
                "s2": (svals * svals).cpu(),
                "shape": tuple(W.shape),
            }
            W = W_scale = svals = None
            del W, W_scale, svals
        spectra[i] = layer_spec
        torch.cuda.empty_cache()
    return spectra


def allocate_module_ranks(spectra, target_ratio, min_rank=1, max_rank=None, min_rank_overrides=None, layer_floor_ratio=0.0, qkv_multiple=None, early_layers=None, early_layers_min_qkv=None, eps=1e-12):
    if spectra is None:
        return None, 0.0
    entries = []
    total_full = 0
    min_total = 0
    max_score = 0.0
    def _is_qkv(name):
        return "q_proj" in name or "k_proj" in name or "v_proj" in name
    def _ceil_multiple(val, step):
        return int((val + step - 1) // step) * step
    def _align_rank(k, step, k_max, cap=None):
        if step is None or step <= 1:
            if cap is not None:
                k = min(k, cap)
            return min(k, k_max)
        if cap is not None:
            k = min(k, cap)
        k = min(k, k_max)
        k = _ceil_multiple(k, step)
        if k > k_max:
            k = (k_max // step) * step
        if cap is not None and k > cap:
            k = (cap // step) * step
        if k <= 0:
            k = min(step, k_max)
        return k
    def _next_rank(k, step, k_max):
        if step is None or step <= 1:
            return min(k + 1, k_max)
        return min(k + step, k_max)
    def _block_score(s2, k, next_k, cost):
        if s2.numel() == 0 or next_k <= k:
            return 0.0
        delta = next_k - k
        block_sum = float(s2[k:next_k].sum().item())
        return block_sum / (delta * cost)
    def _pick_min_rank(name, base):
        if not min_rank_overrides:
            return base
        if "o_proj" in name or "out_proj" in name:
            return min_rank_overrides.get("o_proj", base)
        if "down_proj" in name:
            return min_rank_overrides.get("down_proj", base)
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            return min_rank_overrides.get("qkv", min_rank_overrides.get("attn", base))
        if "gate_proj" in name or "up_proj" in name:
            return min_rank_overrides.get("mlp", base)
        return base
    for layer_id in spectra:
        for name, info in spectra[layer_id].items():
            m, n = info["shape"]
            s2 = info["s2"]
            k_max = int(min(m, n, s2.shape[0]))
            if k_max <= 0:
                continue
            cost = m + n
            total_full += m * n
            base_min = min_rank if min_rank is not None else 1
            r_min = _pick_min_rank(name, base_min)
            if early_layers_min_qkv is not None and early_layers is not None:
                if layer_id < early_layers and _is_qkv(name):
                    r_min = max(r_min, early_layers_min_qkv)
            step = qkv_multiple if qkv_multiple and _is_qkv(name) else 1
            r_min = max(1, min(r_min, k_max))
            r_min = _align_rank(r_min, step, k_max, cap=max_rank)
            min_total += r_min * cost
            if s2.numel() > 0:
                max_score = max(max_score, float((s2[0] / cost).item()))
            entries.append({
                "layer": layer_id,
                "name": name,
                "s2": s2,
                "cost": cost,
                "k_max": k_max,
                "r_min": r_min,
                "m": m,
                "n": n,
                "step": step,
            })
    if total_full <= 0:
        return None, 0.0

    # Enforce per-module floor: each module retains at least layer_floor_ratio
    # of its own parameters.  The old greedy approach used gradient-weighted
    # _block_score which biased budget toward q/k/v/gate/up and starved
    # o_proj/down_proj.  A proportional per-module floor treats all modules
    # equally and lets the Lagrange search optimise above the floor.
    if layer_floor_ratio and layer_floor_ratio > 0:
        for entry in entries:
            m, n = entry["m"], entry["n"]
            cost = entry["cost"]  # m + n
            # rank k such that k*(m+n) = floor_ratio * m*n
            per_module_floor = int(math.ceil(layer_floor_ratio * m * n / cost))
            per_module_floor = max(1, min(per_module_floor, entry["k_max"]))
            per_module_floor = _align_rank(per_module_floor, entry["step"],
                                           entry["k_max"], cap=max_rank)
            if per_module_floor < entry["r_min"]:
                per_module_floor = entry["r_min"]
            if per_module_floor > entry["r_min"]:
                min_total += (per_module_floor - entry["r_min"]) * cost
                entry["r_min"] = per_module_floor

    target_params = target_ratio * total_full
    if min_total > target_params + eps:
        overshoot = min_total - target_params
        print(
            f"[allocate_module_ranks] CRITICAL: floor budget ({min_total:,.0f}) "
            f"exceeds target ({target_params:,.0f}) by {overshoot:,.0f} params "
            f"({100*overshoot/max(target_params,eps):.1f}% over-budget).\n"
            f"  Lagrange search BYPASSED — all modules assigned their floor rank.\n"
            f"  KFAC G-weighting has ZERO effect.  Per-module-type floor vs uniform:\n"
        )
        # Per-module-type breakdown to identify which floors are over-protected.
        def _mtype_alloc(nm):
            if any(t in nm for t in ("q_proj", "k_proj", "v_proj")): return "qkv"
            if any(t in nm for t in ("o_proj", "out_proj")):          return "o"
            if "down_proj" in nm:                                      return "down"
            if any(t in nm for t in ("gate_proj", "up_proj")):        return "mlp"
            return "other"
        type_floor = {}   # mtype → (floor_rank, uniform_rank, cost, count)
        for entry in entries:
            mt = _mtype_alloc(entry["name"])
            m, n, cost = entry["m"], entry["n"], entry["cost"]
            ur = int(math.ceil(target_ratio * m * n / cost))
            fk = entry["r_min"]
            if mt not in type_floor:
                type_floor[mt] = [fk, ur, cost, 1]
            else:
                type_floor[mt][3] += 1
        for mt, (fk, ur, cost, cnt) in sorted(type_floor.items()):
            delta = fk - ur
            tag = f"↑ +{delta}" if delta > 0 else (f"↓ {delta}" if delta < 0 else "= uniform")
            extra_params = delta * cost * cnt
            print(f"    {mt:5s}: floor={fk}  uniform~={ur}  {tag}  "
                  f"({extra_params:+,.0f} params vs uniform × {cnt} modules)")
        print(
            f"  Action: lower floors of ↑ (over-protected) modules until total floor < target.\n"
            f"  Modules with ↓ (deficit) cannot be raised until ↑ modules are reduced first."
        )
        ranks = {}
        for entry in entries:
            ranks.setdefault(entry["layer"], {})[entry["name"]] = entry["r_min"]
        effective = min_total / total_full
        return ranks, effective

    # Warn if the floor consumes nearly all of the target budget, which means the
    # Lagrange water-filling has almost no free budget left to redistribute across
    # modules based on gradient importance.  In practice this disables the G-weighted
    # optimisation: every module ends up at its floor rank regardless of gradient
    # signal.  Typical symptom: PPL is much worse than expected even though
    # --use_grad_g and --use_module_rank_allocation are set.
    #
    # Rule of thumb: floor should leave at least 20% of the target budget free for
    # redistribution, i.e.  layer_floor_ratio  ≤  0.80 * target_ratio.
    lagrange_free = target_params - min_total
    lagrange_free_frac = lagrange_free / max(target_params, eps)
    if lagrange_free_frac < 0.05:
        floor_ratio_eff = min_total / total_full
        print(
            f"[allocate_module_ranks] WARNING: layer_floor_ratio is too close to "
            f"the target ratio — the floor consumes {(1-lagrange_free_frac)*100:.1f}% "
            f"of the target budget (floor_eff={floor_ratio_eff:.4f}, "
            f"target={target_ratio:.4f}, free={lagrange_free_frac*100:.2f}%).\n"
            f"  Gradient-weighted redistribution is effectively DISABLED: every "
            f"module will be allocated its floor rank with at most ~1 extra rank "
            f"across all {len(entries)} modules combined.\n"
            f"  FIX: lower --layer_ratio_floor to ≤ {0.8*target_ratio:.3f} so that "
            f"at least 20%% of the budget is free for gradient-based reallocation."
        )
    low = 0.0
    high = max_score if max_score > 0 else 1.0
    best_ranks = None
    best_total = None
    for _ in range(40):
        mid = (low + high) / 2
        total = 0
        ranks = {}
        for entry in entries:
            s2 = entry["s2"]
            cost = entry["cost"]
            k_max = entry["k_max"]
            r_min = entry["r_min"]
            if s2.numel() == 0:
                k = r_min
            else:
                k = int((s2 / cost >= mid).sum().item())
                k = max(k, r_min)
                k = _align_rank(k, entry["step"], k_max, cap=max_rank)
                if k < r_min:
                    k = r_min
            total += k * cost
            ranks.setdefault(entry["layer"], {})[entry["name"]] = k
        if total > target_params:
            low = mid
        else:
            high = mid
            best_ranks = ranks
            best_total = total
    if best_ranks is None:
        return None, 0.0

    # Greedy fill to use remaining budget with highest marginal gains.
    total = best_total
    heap = []
    entry_map = {}
    for entry in entries:
        layer = entry["layer"]
        name = entry["name"]
        s2 = entry["s2"]
        cost = entry["cost"]
        k = best_ranks[layer][name]
        entry_map[(layer, name)] = entry
        if k < entry["k_max"]:
            next_k = _next_rank(k, entry["step"], entry["k_max"])
            score = _block_score(s2, k, next_k, cost)
            heapq.heappush(heap, (-score, layer, name))
    while heap and total + eps < target_params:
        neg_score, layer, name = heapq.heappop(heap)
        entry = entry_map[(layer, name)]
        cost = entry["cost"]
        k = best_ranks[layer][name]
        next_k = _next_rank(k, entry["step"], entry["k_max"])
        delta = next_k - k
        if delta <= 0:
            continue
        if total + delta * cost > target_params + eps:
            # This module's next step would overshoot the budget. Skip it (do
            # not put it back in the heap) and try the next-best candidate.
            # Using `break` here would permanently stop the fill even when
            # cheaper modules (e.g. step=1 o_proj vs step=128 q_proj) still
            # fit within the remaining budget.
            continue
        best_ranks[layer][name] = next_k
        total += delta * cost
        if next_k < entry["k_max"]:
            next_score = _block_score(entry["s2"], next_k, _next_rank(next_k, entry["step"], entry["k_max"]), cost)
            heapq.heappush(heap, (-next_score, layer, name))
    effective = total / total_full
    return best_ranks, effective


def summarize_module_ranks(spectra, module_ranks):
    if spectra is None or module_ranks is None:
        return 0.0, {}
    total_full = 0
    total_low = 0
    layer_stats = {}
    for layer_id in spectra:
        layer_full = 0
        layer_low = 0
        for name, info in spectra[layer_id].items():
            m, n = info["shape"]
            k = module_ranks.get(layer_id, {}).get(name, 0)
            layer_full += m * n
            layer_low += k * (m + n)
        total_full += layer_full
        total_low += layer_low
        if layer_full > 0:
            layer_stats[layer_id] = layer_low / layer_full
    effective = (total_low / total_full) if total_full > 0 else 0.0
    return effective, layer_stats


def _split_module_ranks(ranks_layer):
    attn_ranks = {}
    mlp_ranks = {}
    if not ranks_layer:
        return attn_ranks, mlp_ranks
    for name, r in ranks_layer.items():
        if "q_proj" in name:
            attn_ranks["q_proj"] = r
        elif "k_proj" in name:
            attn_ranks["k_proj"] = r
        elif "v_proj" in name:
            attn_ranks["v_proj"] = r
        elif "o_proj" in name:
            attn_ranks["o_proj"] = r
        elif "out_proj" in name:
            attn_ranks["out_proj"] = r
        elif "gate_proj" in name:
            mlp_ranks["gate_proj"] = r
        elif "up_proj" in name:
            mlp_ranks["up_proj"] = r
        elif "down_proj" in name:
            mlp_ranks["down_proj"] = r
        elif "fc1" in name:
            mlp_ranks["fc1"] = r
        elif "fc2" in name:
            mlp_ranks["fc2"] = r
    return attn_ranks, mlp_ranks


def _layer_param_sizes(model_name, model):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        return None
    sizes = []
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        size_i = 0
        for name in subset:
            W = subset[name].weight
            size_i += W.shape[0] * W.shape[1]
        sizes.append(size_i)
    return sizes


def _get_head_dim(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)
    hidden = getattr(cfg, "hidden_size", None)
    heads = getattr(cfg, "num_attention_heads", None)
    if hidden is None or heads is None:
        return None
    if heads == 0:
        return None
    return int(hidden // heads)


def _normalize_layer_ratios(ratios, layer_sizes, target_ratio, eps=1e-12):
    total = sum(layer_sizes)
    if total <= 0:
        return ratios, 0.0
    effective = sum(s * r for s, r in zip(layer_sizes, ratios)) / total
    if abs(effective - target_ratio) <= eps:
        return ratios, effective
    scale = target_ratio / max(effective, eps)
    ratios = [r * scale for r in ratios]
    effective = sum(s * r for s, r in zip(layer_sizes, ratios)) / total
    return ratios, effective


def _default_spectrum_path(save_path, model_name, dataset, nsamples, seed):
    safe_model = model_name.replace("/", "_").replace("-", "_")
    filename = f"{safe_model}_spectrum_{dataset}_{nsamples}_{seed}.pt"
    return os.path.join(save_path, filename)


def _module_rank_overrides(args):
    overrides = {}
    if args.module_rank_min_attn is not None:
        overrides["attn"] = args.module_rank_min_attn
    if args.module_rank_min_qkv is not None:
        overrides["qkv"] = args.module_rank_min_qkv
    if args.module_rank_min_o is not None:
        overrides["o_proj"] = args.module_rank_min_o
    if args.module_rank_min_mlp is not None:
        overrides["mlp"] = args.module_rank_min_mlp
    if args.module_rank_min_down is not None:
        overrides["down_proj"] = args.module_rank_min_down
    return overrides if overrides else None


def _save_model_fp16(model, tokenizer, path):
    prev_dtype = next(iter(model.parameters())).dtype
    model = model.to(torch.float16)
    torch.save({'model': model, 'tokenizer': tokenizer}, path)
    model = model.to(prev_dtype)


def _grad_diag_fingerprint(grad_diag, grad_eps):
    """Return an integer fingerprint of gradient quality to detect stale spectrum caches.

    The blocks stored in grad_diag already have grad_eps·I added, so the
    maximum eigenvalue seen here is  max_raw_eig + grad_eps.

    Fingerprint semantics:
      0            → no gradient data or G≈I (max block eigenvalue ≤ 2×grad_eps)
      positive k   → round(log10((max_eig - grad_eps) / grad_eps)), i.e.
                      gradient signal is 10^k × grad_eps.  Changes whenever
                      the gradient magnitude or grad_eps changes.

    The fingerprint is stored in the spectrum cache meta together with grad_eps
    so that changing --grad_eps correctly invalidates old caches.
    """
    if grad_diag is None:
        return 0
    eps = grad_eps if grad_eps and grad_eps > 0 else 1e-6
    max_eig = eps  # lower bound (blocks contain at least grad_eps·I)
    for layer_profile in grad_diag.values():
        for entry in layer_profile.values():
            if isinstance(entry, dict) and entry.get("type") == "block":
                for b in entry["blocks"]:
                    eig = torch.linalg.eigvalsh(b["mat"])
                    max_eig = max(max_eig, float(eig[-1].item()))
            else:
                max_eig = max(max_eig, float(entry.max().item()))
    # Subtract the regularisation contribution to get the raw signal level.
    raw_signal = max_eig - eps
    if raw_signal <= eps:
        return 0
    return max(1, int(round(math.log10(raw_signal / eps))))


def _spectrum_meta(args, grad_nsamples, grad_seq_len, grad_fingerprint=0):
    return {
        "model": args.model,
        "dataset": args.dataset,
        "whitening_nsamples": args.whitening_nsamples,
        "grad_nsamples": grad_nsamples,
        "model_seq_len": args.model_seq_len,
        "grad_seq_len": grad_seq_len,
        "seed": args.seed,
        "use_grad_g": bool(args.use_grad_g),
        "g_block_diag": bool(args.g_block_diag),
        "attn_block_size": int(args.attn_block_size),
        "mlp_block_size": int(args.mlp_block_size),
        "grad_inv_max": args.grad_inv_max,
        # grad_eps is included so that changing --grad_eps invalidates the cache.
        # A different grad_eps changes the eigenvalue structure of every G block
        # (profile_grad_diag adds grad_eps·I to each block), so the stored
        # spectrum would be wrong for the new grad_eps setting.
        "grad_eps": float(args.grad_eps) if args.grad_eps is not None else None,
        # Fingerprint detects G≈I caches (BUG 3/6 active) vs caches with real
        # gradient signal that exceeds the regularisation floor.
        # 0 = no gradient data / signal ≤ grad_eps.
        # Positive k = round(log10(raw_signal / grad_eps)).
        "grad_fingerprint": int(grad_fingerprint),
        # Cascade correction alpha: changing this changes every layer's G-weighted
        # spectrum, so the cached spectrum is invalid for a different alpha.
        "grad_cascade_alpha": float(args.grad_cascade_alpha) if hasattr(args, "grad_cascade_alpha") and args.grad_cascade_alpha is not None else 0.0,
        # Blend alpha changes the G-weighted spectrum for every module.
        "g_blend_alpha": float(args.g_blend_alpha) if hasattr(args, "g_blend_alpha") and args.g_blend_alpha is not None else 1.0,
        # Centering and sequence-level G change grad_diag values, so the spectrum is invalid.
        "g_center": bool(getattr(args, "g_center", False)),
        "g_seq_level": bool(getattr(args, "g_seq_level", False)),
        # Cross-layer normalization rescales every G matrix in grad_diag.
        "g_cross_layer_norm": float(getattr(args, "g_cross_layer_norm", 0.0)),
        # σ-space projection changes the G-weighted spectrum for every block-diag module.
        "g_sigma_space": bool(getattr(args, "g_sigma_space", False)),
    }


def _load_spectrum(path, expected_meta=None):
    data = torch.load(path)
    if isinstance(data, dict) and "spectra" in data:
        meta = data.get("meta")
        spectra = data.get("spectra")
        if expected_meta is not None and meta != expected_meta:
            return None, False
        return spectra, True
    # Backward-compatible: raw spectra dict
    return data, True
     
 
@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev, grad_diag=None, grad_eps=1e-6, layer_ratios=None, module_ranks=None, grad_inv_max=None, debug_svd=False, g_sigma_space=False, g_blend_alpha=1.0):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        ratio_i = layer_ratios[i] if layer_ratios is not None else ratio
        ranks_layer = module_ranks[i] if module_ranks is not None and i in module_ranks else None
        attn_ranks, mlp_ranks = _split_module_ranks(ranks_layer)
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio, ranks=attn_ranks if attn_ranks else None)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio, ranks=mlp_ranks if mlp_ranks else None)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio, ranks=attn_ranks if attn_ranks else None)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio, ranks=mlp_ranks if mlp_ranks else None)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio, ranks=ranks_layer if ranks_layer else None)
        #### Replace Attn, MLP ####
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            g_inv_sqrt = None
            g_blocks = None
            debug_g_info = ""
            if grad_diag is not None:
                entry = grad_diag[i][name]
                if isinstance(entry, dict) and entry.get("type") == "block":
                    g_blocks = []
                    debug_block_min = None
                    min_eig = _min_eig_value(grad_inv_max, grad_eps)
                    for b in entry["blocks"]:
                        s, e = b["start"], b["end"]
                        mat = b["mat"].to(dev)
                        if debug_svd:
                            eig = torch.linalg.eigvalsh(mat)
                            min_raw = float(eig[0].item())
                            debug_block_min = min_raw if debug_block_min is None else min(debug_block_min, min_raw)
                        mat = _shift_psd_min_eig(mat, min_eig)
                        if g_sigma_space:
                            # σ-space: project G to the left-singular-vector basis of W_block.
                            # MUST be consistent with profile_module_spectrum g_sigma_space path:
                            #   G^σ_j = U_j^T G_j U_j     (G in σ-basis)
                            #   Effective whitening:  G_eff^{1/2} = Chol(G^σ_j) @ U_j^T
                            #   Effective inverse:    G_eff^{-1/2} = U_j @ inv(Chol(G^σ_j))
                            # W_block = U_j Σ_j Vh_j  →  U_j^T W_block = diag(Σ_j) Vh_j
                            W_block = W[s:e, :].clone()
                            U_j, S_j, Vh_j = torch.linalg.svd(W_block, full_matrices=False)
                            mat_sigma = U_j.t().matmul(mat).matmul(U_j)
                            chol_sigma = _safe_cholesky(mat_sigma, grad_eps)
                            inv_chol_sigma = torch.linalg.inv(chol_sigma)
                            # Apply σ-space whitening (same as profile_module_spectrum)
                            W[s:e, :] = chol_sigma.matmul(torch.diag(S_j).matmul(Vh_j))
                            g_inv_sqrt_block = U_j.matmul(inv_chol_sigma)
                        elif g_blend_alpha < 1.0:
                            # Blended Chol: alpha*Chol(G) + (1-alpha)*sqrt(mean_eig)*I
                            # MUST be consistent with profile_module_spectrum g_blend_alpha path.
                            chol = _safe_cholesky(mat, grad_eps)
                            mean_eig_val = float(torch.linalg.eigvalsh(mat).mean().item())
                            uniform_scale = math.sqrt(max(mean_eig_val, grad_eps))
                            blk_sz = e - s
                            if g_blend_alpha <= 0.0:
                                chol_blend = uniform_scale * torch.eye(blk_sz, device=dev, dtype=chol.dtype)
                            else:
                                eye = torch.eye(blk_sz, device=dev, dtype=chol.dtype)
                                chol_blend = g_blend_alpha * chol + (1.0 - g_blend_alpha) * uniform_scale * eye
                            inv_chol_blend = torch.linalg.inv(chol_blend)
                            W[s:e, :] = chol_blend.matmul(W[s:e, :])
                            g_inv_sqrt_block = inv_chol_blend
                        else:
                            # Standard KFAC block Cholesky
                            chol = _safe_cholesky(mat, grad_eps)
                            inv_chol = torch.linalg.inv(chol)
                            W[s:e, :] = chol.matmul(W[s:e, :])
                            g_inv_sqrt_block = inv_chol
                        g_blocks.append({"start": s, "end": e, "g_inv_sqrt": g_inv_sqrt_block})
                    if debug_svd and debug_block_min is not None:
                        debug_g_info = f" g_block_min={debug_block_min:.3e}"
                else:
                    g_diag = entry.to(dev)
                    min_eig = _min_eig_value(grad_inv_max, grad_eps)
                    g_diag = torch.clamp(g_diag, min=min_eig)
                    g_sqrt = torch.sqrt(torch.clamp(g_diag, min=grad_eps))
                    g_inv_sqrt = 1.0 / g_sqrt
                    W = W * g_sqrt.unsqueeze(1)
                    if debug_svd:
                        g_min = float(g_diag.min().item())
                        g_max = float(g_diag.max().item())
                        debug_g_info = f" g_min={g_min:.3e} g_max={g_max:.3e}"
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            if ranks_layer is not None and name in ranks_layer:
                num_s_after_trunc = int(ranks_layer[name])
            else:
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio_i / (W.shape[0] + W.shape[1]))
            num_s_after_trunc = max(1, min(num_s_after_trunc, min(W.shape[0], W.shape[1])))
            if debug_svd:
                s2 = (S.float() ** 2)
                denom = float(s2.sum().item())
                if denom > 0:
                    tail = float(s2[num_s_after_trunc:].sum().item())
                    rel_err = math.sqrt(tail / denom)
                else:
                    rel_err = 0.0
                print(f"[debug] layer {i:02d} {name} k={num_s_after_trunc} rel_err={rel_err:.6f}{debug_g_info}")
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            if g_blocks is not None:
                for b in g_blocks:
                    s, e = b["start"], b["end"]
                    truc_u[s:e, :] = b["g_inv_sqrt"].matmul(truc_u[s:e, :])
            elif g_inv_sqrt is not None:
                truc_u = g_inv_sqrt.unsqueeze(1) * truc_u
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        # Replace modules after all weights have been assigned
        if 'opt' in model_name:
            svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
            svd_decoder.final_layer_norm = layer.final_layer_norm
            layers[i] = svd_decoder
        else:
            layer.self_attn = svd_attn
            layer.mlp = svd_mlp
        del layer
        torch.cuda.empty_cache()


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False, grad_diag=None, grad_eps=1e-6, layer_ratios=None, module_ranks=None, grad_inv_max=None, debug_svd=False):
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        ratio_i = layer_ratios[i] if layer_ratios is not None else ratio
        ranks_layer = module_ranks[i] if module_ranks is not None and i in module_ranks else None
        attn_ranks, mlp_ranks = _split_module_ranks(ranks_layer)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio, ranks=attn_ranks if attn_ranks else None)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio, ranks=mlp_ranks if mlp_ranks else None)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio, ranks=attn_ranks if attn_ranks else None)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio, ranks=mlp_ranks if mlp_ranks else None)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio, ranks=ranks_layer if ranks_layer else None)
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            if grad_diag is not None:
                g_entry = grad_diag[i][name]
                if isinstance(g_entry, dict) and g_entry.get("type") == "block":
                    g_diag = g_entry  # block-diag dict, local_update handles device transfer
                else:
                    g_diag = g_entry.to(dev)
            else:
                g_diag = None
            rank_override = ranks_layer[name] if ranks_layer is not None and name in ranks_layer else None
            debug_name = f"layer {i:02d} {name}" if debug_svd else name
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=ratio_i, name=debug_name, direct_update=direct_update, g_diag=g_diag, grad_eps=grad_eps, rank=rank_override, grad_inv_max=grad_inv_max, debug_svd=debug_svd)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
        # Replace modules after all weights have been assigned
        if 'opt' in model_name:
            svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
            svd_decoder.final_layer_norm = layer.final_layer_norm
            layers[i] = svd_decoder
        else:
            layer.self_attn = svd_attn
            layer.mlp = svd_mlp
        layer = layer.to(dev)
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False, g_diag=None, grad_eps=1e-6, rank=None, grad_inv_max=None, debug_svd=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        g_inv_sqrt = None
        self.g_sqrt = None
        self.g_inv_sqrt = None
        self.g_blocks = None
        if g_diag is not None:
            if isinstance(g_diag, dict) and g_diag.get("type") == "block":
                self.g_blocks = []
                min_eig = _min_eig_value(grad_inv_max, grad_eps)
                for b in g_diag["blocks"]:
                    s, e = b["start"], b["end"]
                    mat = b["mat"].to(self.dev)
                    mat = _shift_psd_min_eig(mat, min_eig)
                    chol = _safe_cholesky(mat, grad_eps)
                    inv_chol = torch.linalg.inv(chol)
                    self.g_blocks.append({"start": s, "end": e, "g_sqrt": chol, "g_inv_sqrt": inv_chol})
                    W[s:e, :] = chol.matmul(W[s:e, :])
            else:
                g_diag = g_diag.to(self.dev)
                min_eig = _min_eig_value(grad_inv_max, grad_eps)
                g_diag = torch.clamp(g_diag, min=min_eig)
                g_sqrt = torch.sqrt(torch.clamp(g_diag, min=grad_eps))
                g_inv_sqrt = 1.0 / g_sqrt
                self.g_sqrt = g_sqrt
                self.g_inv_sqrt = g_inv_sqrt
                W = W * g_sqrt.unsqueeze(1)
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # truncation SVD
        if rank is not None:
            num_s_after_trunc = int(rank)
        else:
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        num_s_after_trunc = max(1, min(num_s_after_trunc, min(W.shape[0], W.shape[1])))
        if debug_svd:
            s2 = (self.S.float() ** 2)
            denom = float(s2.sum().item())
            if denom > 0:
                tail = float(s2[num_s_after_trunc:].sum().item())
                rel_err = math.sqrt(tail / denom)
            else:
                rel_err = 0.0
            print(f"[debug] {self.name} k={num_s_after_trunc} rel_err={rel_err:.6f}")
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if self.g_blocks is not None:
            for b in self.g_blocks:
                s, e = b["start"], b["end"]
                self.truc_u[s:e, :] = b["g_inv_sqrt"].matmul(self.truc_u[s:e, :])
        elif g_inv_sqrt is not None:
            self.truc_u = g_inv_sqrt.unsqueeze(1) * self.truc_u
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        if self.g_blocks is not None:
            outs_w = outs.clone()
            new_output_w = new_output.clone()
            for b in self.g_blocks:
                s, e = b["start"], b["end"]
                outs_w[:, s:e] = outs_w[:, s:e].matmul(b["g_sqrt"].t())
                new_output_w[:, s:e] = new_output_w[:, s:e].matmul(b["g_sqrt"].t())
            self.error = torch.sqrt(torch.sum((outs_w - new_output_w) ** 2)).item() / torch.norm(outs_w, p='fro').item()
            x = torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
            updated_uT_w = torch.linalg.lstsq(x, outs_w).solution
            updated_uT = updated_uT_w.clone()
            for b in self.g_blocks:
                s, e = b["start"], b["end"]
                updated_uT[:, s:e] = updated_uT[:, s:e].matmul(b["g_inv_sqrt"].t())
            self.updated_uT = updated_uT
            updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
            updated_output_w = updated_output.clone()
            for b in self.g_blocks:
                s, e = b["start"], b["end"]
                updated_output_w[:, s:e] = updated_output_w[:, s:e].matmul(b["g_sqrt"].t())
            self.updated_error = torch.sqrt(torch.sum((outs_w - updated_output_w) ** 2)).item() / torch.norm(outs_w, p='fro').item()
        elif self.g_sqrt is not None:
            outs_w = outs * self.g_sqrt
            new_output_w = new_output * self.g_sqrt
            self.error = torch.sqrt(torch.sum((outs_w - new_output_w) ** 2)).item() / torch.norm(outs_w, p='fro').item()
            # print(f"truncted error: {self.error}")
            x = torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
            updated_uT_w = torch.linalg.lstsq(x, outs_w).solution
            self.updated_uT = updated_uT_w * self.g_inv_sqrt.unsqueeze(0)
            updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
            updated_output_w = updated_output * self.g_sqrt
            self.updated_error = torch.sqrt(torch.sum((outs_w - updated_output_w) ** 2)).item() / torch.norm(outs_w, p='fro').item()
        else:
            self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
            # print(f"truncted error: {self.error}")
            x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
            self.updated_uT = torch.linalg.lstsq(x,outs).solution
            updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
            self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"updated error: {self.updated_error}")
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
        # print(f"Finish {self.name}"
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


def _obtain_grad_diag(args, model, tokenizer, cali_white_data=None):
    if args.grad_path is not None:
        return torch.load(args.grad_path)
    grad_nsamples = args.grad_nsamples if args.grad_nsamples is not None else args.whitening_nsamples
    grad_seq_len = args.grad_seq_len if args.grad_seq_len is not None else args.model_seq_len
    grad_batch_size = args.grad_batch_size if args.grad_batch_size is not None else 1
    if cali_white_data is not None and grad_seq_len == args.model_seq_len and grad_batch_size == 1:
        cali_grad_data = cali_white_data
    else:
        cali_grad_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=grad_seq_len, batch_size=grad_batch_size)
    grad_diag = profile_grad_diag(
        args.model, model, cali_grad_data, args.DEV, max_batches=grad_nsamples, grad_eps=args.grad_eps,
        block_diag=args.g_block_diag, attn_block_size=args.attn_block_size, mlp_block_size=args.mlp_block_size,
        use_checkpointing=args.grad_checkpointing, grad_inv_max=args.grad_inv_max,
        g_center=getattr(args, "g_center", False),
        g_seq_level=getattr(args, "g_seq_level", False),
        g_cross_layer_norm=getattr(args, "g_cross_layer_norm", 0.0),
    )
    if args.save_path is not None:
        torch.save(grad_diag, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_grad_diag_' + args.dataset + '_' + str(grad_nsamples) + '_' + str(args.seed) + '.pt')
    return grad_diag


def _obtain_layer_ratios(args, model, grad_diag):
    layer_ratios = compute_layer_ratios(args.model, model, grad_diag, args.ratio, min_ratio=args.layer_ratio_min, max_ratio=args.layer_ratio_max, eps=args.grad_eps)
    layer_sizes = _layer_param_sizes(args.model, model)
    if layer_ratios is not None and layer_sizes is not None:
        if args.layer_ratio_strict:
            layer_ratios, effective_ratio = _normalize_layer_ratios(layer_ratios, layer_sizes, args.ratio)
        else:
            _, effective_ratio = _normalize_layer_ratios(layer_ratios, layer_sizes, args.ratio)
        if args.print_layer_ratios:
            print(f"Layerwise keep ratios (target={args.ratio:.6f}, effective={effective_ratio:.6f})")
            for i, r in enumerate(layer_ratios):
                print(f"layer {i:02d} ratio={r:.6f} size={layer_sizes[i]}")
    return layer_ratios


def _obtain_module_ranks(args, model, profiling_mat, grad_diag):
    grad_nsamples = args.grad_nsamples if args.grad_nsamples is not None else args.whitening_nsamples
    grad_seq_len = args.grad_seq_len if args.grad_seq_len is not None else args.model_seq_len
    # Compute gradient quality fingerprint once so it is consistent between the
    # cache-load check and the save.  A fingerprint of 0 means G≈I (no real
    # gradient signal); a positive fingerprint means real gradient signal was
    # captured and would be wasted if a G≈I spectrum were loaded from cache.
    grad_fingerprint = _grad_diag_fingerprint(
        grad_diag if args.use_grad_g else None, args.grad_eps
    )
    print(f"[spectrum] grad_diag fingerprint={grad_fingerprint} "
          f"(0=G≈I/no-grad, >0=real gradient signal, units=log10 scale)")
    spectrum_path = args.spectrum_path
    if spectrum_path is None and args.save_path is not None:
        spectrum_path = _default_spectrum_path(args.save_path, args.model, args.dataset, grad_nsamples, args.seed)
    spectrum = None
    if spectrum_path is not None and os.path.exists(spectrum_path):
        expected_meta = _spectrum_meta(args, grad_nsamples, grad_seq_len, grad_fingerprint)
        spectrum, ok = _load_spectrum(spectrum_path, expected_meta)
        if not ok:
            print("Warning: spectrum cache meta mismatch (possible stale G≈I cache after float32 fix), recomputing...")
            spectrum = None
    if spectrum is None:
        spectrum = profile_module_spectrum(
            args.model, model, profiling_mat, args.DEV,
            grad_diag=grad_diag if args.use_grad_g else None, grad_eps=args.grad_eps, grad_inv_max=args.grad_inv_max,
            cascade_alpha=args.grad_cascade_alpha if hasattr(args, "grad_cascade_alpha") else 0.0,
            g_blend_alpha=args.g_blend_alpha if hasattr(args, "g_blend_alpha") and args.g_blend_alpha is not None else 1.0,
            g_sigma_space=bool(getattr(args, "g_sigma_space", False)),
        )
        if spectrum_path is not None:
            meta = _spectrum_meta(args, grad_nsamples, grad_seq_len, grad_fingerprint)
            torch.save({"meta": meta, "spectra": spectrum}, spectrum_path)
    qkv_multiple = None
    if args.module_rank_min_qkv is not None or args.early_layers_min_qkv is not None:
        qkv_multiple = _get_head_dim(model)
    early_layers = args.early_layers if args.early_layers is not None and args.early_layers > 0 else None
    module_ranks, effective_ratio = allocate_module_ranks(
        spectrum,
        args.ratio,
        min_rank=args.module_rank_min,
        max_rank=args.module_rank_max,
        min_rank_overrides=_module_rank_overrides(args),
        layer_floor_ratio=args.layer_ratio_floor,
        qkv_multiple=qkv_multiple,
        early_layers=early_layers,
        early_layers_min_qkv=args.early_layers_min_qkv,
    )
    # Over-clamping / rank-concentration diagnostic.
    # When G-weighting is partially disabled by gradient vanishing (early-layer
    # blocks clamped to min_eig) the rank allocation can concentrate heavily
    # in a few late layers.  Warn unconditionally so the user sees the issue
    # even without --print_module_ranks.
    if module_ranks is not None and args.use_grad_g:
        _eff, _lstats = summarize_module_ranks(spectrum, module_ranks)
        if _lstats:
            _ratios = list(_lstats.values())
            _min_r, _max_r = min(_ratios), max(_ratios)
            if _max_r > 0 and _min_r >= 0:
                _spread = _max_r / max(_min_r, 1e-9)
                if _spread > 2.5 and _max_r > 1.5 * args.ratio:
                    _n_at_floor = sum(1 for r in _ratios if r < args.ratio * 0.85)
                    _n_above = sum(1 for r in _ratios if r > args.ratio * 1.5)
                    print(
                        f"[ranks] WARNING: rank distribution is highly concentrated.\n"
                        f"  Layer ratios: min={_min_r:.3f}  target={args.ratio:.3f}  "
                        f"max={_max_r:.3f}  (spread={_spread:.1f}x)\n"
                        f"  {_n_at_floor} layer(s) near floor (<85% of target ratio), "
                        f"{_n_above} layer(s) well above target (>150%).\n"
                        f"  This pattern typically means gradient vanishing has left most "
                        f"early layers clamped to min_eig in G, while 1-2 late layers drive "
                        f"all the rank redistribution.  Early-layer compression errors then "
                        f"cascade and raise PPL far above the uniform-allocation baseline.\n"
                        f"  Recommended mitigations (choose one or combine):\n"
                        f"    --layer_ratio_floor 0.38   # tighten floor to prevent over-squeeze\n"
                        f"    --module_rank_min_qkv <N>  # protect attention projections\n"
                        f"    --module_rank_min_mlp <N>  # protect MLP projections\n"
                    )
    if args.print_module_ranks and module_ranks is not None:
        effective_ratio, layer_stats = summarize_module_ranks(spectrum, module_ranks)
        print(f"Module ranks (target={args.ratio:.6f}, effective={effective_ratio:.6f})")
        for layer_id in sorted(layer_stats.keys()):
            print(f"layer {layer_id:02d} ratio={layer_stats[layer_id]:.6f}")
        ranks_flat = []
        for layer_id in sorted(module_ranks.keys()):
            for name, k in module_ranks[layer_id].items():
                shape = spectrum[layer_id][name]["shape"]
                print(f"layer {layer_id:02d} {name} rank={k} shape={shape}")
                ranks_flat.append(k)
        if ranks_flat:
            ranks_flat = sorted(ranks_flat)
            print(f"rank stats: min={ranks_flat[0]} p50={ranks_flat[len(ranks_flat)//2]} max={ranks_flat[-1]}")
            tiny = sum(1 for r in ranks_flat if r <= 1)
            if tiny > 0:
                print(f"Warning: {tiny} modules have rank<=1. Consider increasing --module_rank_min.")
        # Per-module-type rank summary: reveals whether attention is uniformly at
        # its floor while MLP has high variance (the pattern that causes G-weighted
        # PPL to be worse than uniform allocation).
        def _rank_mtype(nm):
            if any(t in nm for t in ("q_proj", "k_proj", "v_proj")):
                return "qkv"
            if any(t in nm for t in ("o_proj", "out_proj")):
                return "o"
            if "down_proj" in nm:
                return "down"
            if any(t in nm for t in ("gate_proj", "up_proj")):
                return "mlp"
            return "other"
        mtype_ranks = {}
        for layer_id in sorted(module_ranks.keys()):
            for name, k in module_ranks[layer_id].items():
                mt = _rank_mtype(name)
                mtype_ranks.setdefault(mt, []).append(k)
        # Compute uniform-allocation rank for each module type, for comparison.
        # uniform_rank = target_ratio * m*n / (m+n) — the rank each module would
        # get if Lagrange water-filling were run WITHOUT any G weighting (flat signal).
        # This is what plain SVD-LLM computes.  When KFAC-weighted allocation gives
        # a floor below the uniform rank, KFAC is under-serving that module type —
        # the G signal is pushing budget toward other modules, but the PPL impact of
        # the reduced rank here likely dominates.
        def _uniform_rank_for_type(mt):
            for lid in spectrum:
                for nm, info in spectrum[lid].items():
                    if _rank_mtype(nm) != mt:
                        continue
                    m, n = info["shape"]
                    cost = m + n
                    k_max = int(min(m, n))
                    ur = args.ratio * m * n / cost
                    # align qkv to head_dim
                    if mt == "qkv":
                        hd = _get_head_dim(model) or 128
                        ur = int(math.ceil(ur / hd)) * hd
                    else:
                        ur = int(math.ceil(ur))
                    return min(ur, k_max)
            return None
        _cli_flag = {"qkv": "--module_rank_min_qkv", "o": "--module_rank_min_o",
                     "mlp": "--module_rank_min_mlp", "down": "--module_rank_min_down"}
        # Detect global over-budget: effective_ratio > target means the sum of all
        # module floors exceeded the target budget, so allocate_module_ranks bypassed
        # the Lagrange search entirely and returned r_min for every module.  In this
        # state KFAC G-weighting has ZERO effect — all modules are pinned to their
        # floor regardless of gradient signal.  The fix is to lower the floors of
        # modules that are ABOVE their uniform-allocation rank (over-protected).
        floor_over_budget = effective_ratio > args.ratio + 1e-4
        if floor_over_budget and getattr(args, "use_grad_g", False):
            print(
                f"[ranks] CRITICAL: total floor budget EXCEEDS target ratio "
                f"(effective={effective_ratio:.4f} > target={args.ratio:.4f}).\n"
                f"[ranks]   Lagrange search BYPASSED — all modules stuck at r_min.\n"
                f"[ranks]   KFAC G-weighting has ZERO effect on rank allocation.\n"
                f"[ranks]   Find and lower over-protected module floors (marked ↑ below)."
            )
        print("[ranks] Per-module-type rank summary (min / median / max):")
        for mt in ("qkv", "o", "mlp", "down", "other"):
            ks = sorted(mtype_ranks.get(mt, []))
            if not ks:
                continue
            kmin, kmed, kmax = ks[0], ks[len(ks) // 2], ks[-1]
            n_at_min = sum(1 for k in ks if k == kmin)
            uniform_flag = " ← ALL AT SAME RANK (floor binding?)" if kmin == kmax else ""
            print(f"  {mt:5s}: min={kmin}  p50={kmed}  max={kmax}  "
                  f"(n={len(ks)}, {n_at_min} at min){uniform_flag}")
            # Compare current floor to the uniform-allocation rank and print:
            #   ↑ OVER-PROTECTED  when floor > uniform (consuming excess budget)
            #   ↓ DEFICIT         when floor < uniform (under-served by KFAC)
            # When the global floor is over-budget, suppress "raise floor" advice
            # for deficit modules — adding more floor would only worsen things.
            if kmin == kmax and getattr(args, "use_grad_g", False) and mt in _cli_flag:
                ur = _uniform_rank_for_type(mt)
                if ur is not None:
                    if kmin > ur:
                        # Over-protected: floor above uniform → burns extra budget
                        excess = kmin - ur
                        pct = 100.0 * excess / max(ur, 1)
                        # Estimate extra params using the first module's shape.
                        sample_cost = next(
                            (sum(info["shape"]) for lid in spectrum
                             for nm, info in spectrum[lid].items()
                             if _rank_mtype(nm) == mt),
                            8192
                        )
                        extra_params = excess * sample_cost * len(ks)
                        print(
                            f"  [ranks] ↑ {mt}: floor={kmin} EXCEEDS uniform ~{ur} "
                            f"(+{excess} = +{pct:.0f}% over-protected).\n"
                            f"  [ranks]   Wastes {extra_params:,} extra params vs uniform "
                            f"that KFAC cannot redistribute.\n"
                            f"  [ranks]   Lower: {_cli_flag[mt]} {ur}  "
                            f"(back to uniform level, frees budget for Lagrange)"
                        )
                    elif kmin < ur:
                        deficit = ur - kmin
                        pct = 100.0 * deficit / ur
                        if floor_over_budget:
                            # Deficit, but total floor is already over-budget — don't
                            # suggest raising this floor (would make the over-budget worse).
                            print(
                                f"  [ranks] ↓ {mt}: KFAC under-allocates vs uniform "
                                f"({kmin} vs ~{ur}, deficit={deficit} = {pct:.0f}%).\n"
                                f"  [ranks]   Cannot raise floor — total floor already exceeds budget.\n"
                                f"  [ranks]   First lower over-protected (↑) module floors to free room."
                            )
                        else:
                            print(
                                f"  [ranks] ↓ {mt}: KFAC under-allocates vs uniform "
                                f"({kmin} vs ~{ur}, deficit={deficit} = {pct:.0f}%).\n"
                                f"  [ranks]   KFAC natural rank < floor for ALL {len(ks)} {mt} "
                                f"layers — G signal pushes budget elsewhere.\n"
                                f"  [ranks]   Fix: {_cli_flag[mt]} {ur}  "
                                f"(raises floor to match uniform allocation)\n"
                                f"  [ranks]   Or:  {_cli_flag[mt]} {int(kmin + deficit*0.5)}  "
                                f"(raises floor halfway toward uniform)"
                            )
    return module_ranks


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--use_grad_g', action='store_true', help='whether to use output-gradient diag weighting')
    parser.add_argument('--grad_nsamples', type=int, default=None, help='Number of calibration batches for grad diag estimation')
    parser.add_argument('--grad_path', type=str, default=None, help='Local path to load grad diag (G) matrices')
    parser.add_argument('--grad_eps', type=float, default=1e-6, help='Epsilon to avoid zero in grad diag')
    parser.add_argument('--grad_seq_len', type=int, default=None, help='Sequence length for grad diag profiling (defaults to model_seq_len)')
    parser.add_argument('--grad_batch_size', type=int, default=None, help='Batch size for grad diag profiling (defaults to 1)')
    parser.add_argument('--grad_checkpointing', action='store_true', help='Enable gradient checkpointing for grad diag profiling to reduce memory')
    parser.add_argument('--g_block_diag', action='store_true', help='use block-diagonal G (head blocks + MLP groups)')
    parser.add_argument('--attn_block_size', type=int, default=0, help='attention block size (0 uses head_dim)')
    parser.add_argument('--mlp_block_size', type=int, default=256, help='MLP block size')
    parser.add_argument('--use_layerwise_ratio', action='store_true', help='allocate per-layer ratios based on G importance')
    parser.add_argument('--layer_ratio_min', type=float, default=0.01, help='Minimum per-layer keep ratio')
    parser.add_argument('--layer_ratio_max', type=float, default=0.99, help='Maximum per-layer keep ratio')
    parser.add_argument('--layer_ratio_strict', action='store_true', help='Rescale layer ratios to exactly match the global keep ratio')
    parser.add_argument('--print_layer_ratios', action='store_true', help='Print per-layer keep ratios and effective global ratio')
    parser.add_argument('--use_module_rank_allocation', action='store_true', help='Allocate per-module ranks via spectrum + Lagrange')
    parser.add_argument('--spectrum_path', type=str, default=None, help='Path to load/save module spectrum cache')
    parser.add_argument('--module_rank_min', type=int, default=1, help='Minimum rank per module')
    parser.add_argument('--module_rank_min_attn', type=int, default=None, help='Minimum rank for attention proj modules')
    parser.add_argument('--module_rank_min_qkv', type=int, default=None, help='Minimum rank for q/k/v proj modules')
    parser.add_argument('--early_layers', type=int, default=0, help='Apply early-layer qkv min to first N layers')
    parser.add_argument('--early_layers_min_qkv', type=int, default=None, help='Minimum rank for q/k/v in early layers')
    parser.add_argument('--module_rank_min_o', type=int, default=None, help='Minimum rank for o/out proj modules')
    parser.add_argument('--module_rank_min_mlp', type=int, default=None, help='Minimum rank for gate/up proj modules')
    parser.add_argument('--module_rank_min_down', type=int, default=None, help='Minimum rank for down proj modules')
    parser.add_argument('--layer_ratio_floor', type=float, default=0.0, help='Minimum per-layer keep ratio when allocating module ranks')
    parser.add_argument('--module_rank_max', type=int, default=None, help='Maximum rank per module (default: min(out,in))')
    parser.add_argument('--print_module_ranks', action='store_true', help='Print per-module ranks and per-layer effective ratios')
    parser.add_argument('--grad_inv_max', type=float, default=None, help='Clamp max value of g_inv_sqrt to avoid huge scaling')
    parser.add_argument('--grad_cascade_alpha', type=float, default=0.0,
        help='Cascade correction for first-order KFAC approximation. '
             'Scales layer i G by (1 + alpha * i / (L-1)). Later layers get '
             'higher G-weighting to compensate for KFAC ignoring sequential '
             'forward error cascading (simultaneous compression of many layers). '
             'Try 0.5-2.0 when G-weighted PPL is worse than uniform allocation. '
             'Default 0.0 (no correction, pure KFAC).')
    parser.add_argument('--g_blend_alpha', type=float, default=1.0,
        help='Within-block G concentration reduction factor for spectrum computation. '
             '1.0 = pure KFAC full block Cholesky (original behaviour). '
             '0.0 = block-mean scalar only (replaces each block Cholesky with '
             'sqrt(mean_eig(G_block))*I, eliminating within-block concentration). '
             'Intermediate values blend the two: '
             '  G_eff^{1/2}_block = alpha * Chol(G_block) + (1-alpha) * sqrt(mean_eig) * I. '
             'For attention heads with G concentration ×Nc >> 1 (shown in grad_diag output), '
             'the full Cholesky focuses the SVD budget on 1-2 hot principal directions per '
             'block, leaving the remaining directions at min_eig.  This causes KFAC natural '
             'rank to be much lower than the uniform allocation rank, making all modules '
             'floor-bound and G-weighting ineffective. '
             'Try 0.3-0.7 to reduce concentration while keeping cross-block variation. '
             'Invalidates spectrum cache (recomputes spectrum for new alpha).  Default 1.0.')
    parser.add_argument('--g_cross_layer_norm', type=float, default=0.0,
        help='Cross-layer G normalization strength BETA (0.0=disabled, 1.0=full). '
             'Compresses the cross-layer G dynamic range by scaling each layer\'s G '
             'toward the global geometric mean: G\'_layer = G_layer × (geom_mean/max_G_layer)^BETA. '
             'Effect on cross-layer ratio R: R\' = R^(1-BETA). '
             'Example: R=453×, BETA=0.50 → R\'=21×, BETA=0.75 → R\'=4.6×, BETA=1.0 → R\'=1×. '
             'Only the cross-layer SCALE is adjusted — within-layer/within-block G '
             'structure (relative eigenvalue ratios) is fully preserved. '
             'Use when the cross-layer G dynamic range warning shows >100× despite '
             'correct configuration. Try BETA=0.5 first, then 0.75. '
             'Invalidates grad_diag cache (rescales all G matrices). '
             'CRITICAL: also remove --layer_ratio_floor (or set to 0.0) so Lagrange '
             'has enough free budget to actually use the G signal.')
    parser.add_argument('--g_center', action='store_true',
        help='Centered second-moment G: subtract mean gradient direction μμᵀ from each G block. '
             'Isolates input-dependent gradient variance from systematic batch drift. '
             'G^c_j = (1/NT)Σ δy δyᵀ - μ_j μ_j^T  where μ_j = (1/NT)Σ δy^[j]. '
             'Reports drift ratio (fraction of trace(G) removed) during profiling. '
             'Effective when mean gradient is non-zero (e.g., language model loss has '
             'a dominant softmax direction), filtering that bias from the covariance. '
             'Compatible with --g_block_diag and --g_seq_level. Invalidates grad_diag cache.')
    parser.add_argument('--g_seq_level', action='store_true',
        help='Sequence-level gradient covariance G: aggregate gradients over the token '
             'dimension before computing the outer product. '
             'G^seq_j = (1/N) Σ_n (Σ_t δy_{n,t}^[j])(Σ_t δy_{n,t}^[j])^T. '
             'Captures temporal gradient coherence: if the model consistently pushes in '
             'the same direction across tokens for a given input, that direction gets '
             'higher G weight than token-level averaging would give it. '
             'This is distinct from all three reference methods (Ref 1: full Kronecker G⊗A, '
             'Ref 2/3: Fisher in σ-space), which all use independent token-level outer products. '
             'Especially relevant for autoregressive LLMs where adjacent token gradients are '
             'correlated through the causal mask. Compatible with --g_block_diag and --g_center. '
             'Invalidates grad_diag cache.')
    parser.add_argument('--g_sigma_space', action='store_true',
        help='Project block-diagonal G to the left-singular-vector basis of W before '
             'applying the Cholesky factor (σ-space projection with off-diagonal coupling). '
             'For each weight block W_j = U_j Σ_j Vh_j, the σ-space G is: '
             'G^σ_j = U_j^T G_j U_j  (G expressed in the basis of W\'s left singular vectors). '
             'The effective G^{1/2} applied to W is then: '
             'W_scaled = Chol(G^σ_j) @ diag(Σ_j) @ Vh_j. '
             'Off-diagonal elements of G^σ couple different singular modes so the final '
             'svdvals(W_scaled @ H^{1/2}) reflect the true KFAC cost including cross-mode '
             'interactions, rather than the Chol(G) rotation that may not align with σ-space. '
             'Requires --g_block_diag; no-op for diagonal G modules. '
             'Invalidates spectrum cache. '
             'Incompatible with --g_blend_alpha < 1.0 (σ-space takes priority over blend).')
    parser.add_argument('--debug_svd', action='store_true', help='Print per-module truncation error and G stats for debugging')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        grad_diag = None
        layer_ratios = None
        module_ranks = None
        cali_white_data = None
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        if args.use_grad_g or args.use_layerwise_ratio or args.use_module_rank_allocation:
            # Convert to float32 before gradient profiling so that backward-pass
            # gradients do not underflow to zero in float16.  Step 2 already does
            # this (model = model.float() near the top of the step-2 branch).
            # Without this, all gradient blocks end up at the regularisation floor
            # (g_block_min == grad_eps) because float16 backprop loses signal for
            # small gradient values, making G ≈ epsilon * I and the gradient
            # weighting completely ineffective.
            model = model.float()
            grad_diag = _obtain_grad_diag(args, model, tokenizer, cali_white_data)
        if args.use_layerwise_ratio:
            layer_ratios = _obtain_layer_ratios(args, model, grad_diag)
        if args.use_module_rank_allocation:
            module_ranks = _obtain_module_ranks(args, model, profiling_mat, grad_diag)
            layer_ratios = None
        whitening(args.model, model, profiling_mat, args.ratio, args.DEV, grad_diag=grad_diag if args.use_grad_g else None, grad_eps=args.grad_eps, layer_ratios=layer_ratios, module_ranks=module_ranks, grad_inv_max=args.grad_inv_max, debug_svd=args.debug_svd,
                  g_sigma_space=bool(getattr(args, "g_sigma_space", False)),
                  g_blend_alpha=float(args.g_blend_alpha) if hasattr(args, "g_blend_alpha") and args.g_blend_alpha is not None else 1.0)
        if args.save_path is not None:
            _save_model_fp16(model, tokenizer, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + '.pt')
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()  # need to set to float
        grad_diag = None
        layer_ratios = None
        module_ranks = None
        cali_white_data = None
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        if args.use_grad_g or args.use_layerwise_ratio or args.use_module_rank_allocation:
            grad_diag = _obtain_grad_diag(args, model, tokenizer, cali_white_data)
        if args.use_layerwise_ratio:
            layer_ratios = _obtain_layer_ratios(args, model, grad_diag)
        if args.use_module_rank_allocation:
            module_ranks = _obtain_module_ranks(args, model, profiling_mat, grad_diag)
            layer_ratios = None
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV, grad_diag=grad_diag if args.use_grad_g else None, grad_eps=args.grad_eps, layer_ratios=layer_ratios, module_ranks=module_ranks, grad_inv_max=args.grad_inv_max, debug_svd=args.debug_svd)
        if args.save_path is not None:
            _save_model_fp16(model, tokenizer, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        grad_diag = None
        layer_ratios = None
        module_ranks = None
        if args.use_grad_g or args.use_layerwise_ratio or args.use_module_rank_allocation:
            grad_diag = _obtain_grad_diag(args, model, tokenizer)
        if args.use_layerwise_ratio:
            layer_ratios = _obtain_layer_ratios(args, model, grad_diag)
        if args.use_module_rank_allocation:
            module_ranks = _obtain_module_ranks(args, model, None, grad_diag)
            layer_ratios = None
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True, grad_diag=grad_diag if args.use_grad_g else None, grad_eps=args.grad_eps, layer_ratios=layer_ratios, module_ranks=module_ranks, grad_inv_max=args.grad_inv_max, debug_svd=args.debug_svd)
        if args.save_path is not None:
            _save_model_fp16(model, tokenizer, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
