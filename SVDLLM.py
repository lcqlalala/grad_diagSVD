#coding:utf8
import os
import sys
import argparse
import heapq
import math
from contextlib import nullcontext
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
def profile_module_spectrum(model_name, model, profiling_mat, dev):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        return None
    spectra = {}
    model.eval()
    n_layers = len(layers)
    for i in tqdm(range(n_layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_spec = {}
        for name in subset:
            # clone to avoid any accidental in-place modification of model weights
            W = subset[name].weight.data.float().to(dev).clone()
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
    # signal.  Typical symptom: PPL is much worse than expected.
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


def _print_layer_compression_stats(prefix, layer_id, full_params, low_params, cost_sum, rank_values):
    if full_params <= 0 or cost_sum <= 0:
        return
    keep_ratio = low_params / full_params
    compress_ratio = 1.0 - keep_ratio
    effective_rank = low_params / cost_sum
    rmin = min(rank_values) if rank_values else 0
    rmax = max(rank_values) if rank_values else 0
    rmean = (sum(rank_values) / len(rank_values)) if rank_values else 0.0
    print(
        f"[{prefix}] layer {layer_id:02d}: "
        f"keep={keep_ratio:.4f} compress={compress_ratio:.4f} "
        f"effective_rank={effective_rank:.2f} "
        f"rank[min/mean/max]={rmin}/{rmean:.1f}/{rmax}"
    )


def _module_rank_type(name):
    if any(t in name for t in ("q_proj", "k_proj", "v_proj")):
        return "qkv"
    if any(t in name for t in ("o_proj", "out_proj")):
        return "o"
    if "down_proj" in name:
        return "down"
    if any(t in name for t in ("gate_proj", "up_proj")):
        return "mlp"
    return "other"


def _print_layer_rank_detail(layer_id, rank_map):
    groups = {"qkv": [], "o": [], "mlp": [], "down": [], "other": []}
    for name, k in rank_map.items():
        groups[_module_rank_type(name)].append(int(k))
    parts = []
    for key in ("qkv", "o", "mlp", "down", "other"):
        vals = groups[key]
        if not vals:
            continue
        vmin, vmax = min(vals), max(vals)
        vmean = sum(vals) / len(vals)
        parts.append(f"{key}={vmin}/{vmean:.1f}/{vmax}")
    if parts:
        print(f"[layer-ranks] layer {layer_id:02d}: " + "  ".join(parts))


def _spectrum_meta(args):
    return {
        "model": args.model,
        "dataset": args.dataset,
        "whitening_nsamples": args.whitening_nsamples,
        "model_seq_len": args.model_seq_len,
        "seed": args.seed,
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
def whitening(
    model_name,
    model,
    profiling_mat,
    ratio,
    dev,
    layer_ratios=None,
    module_ranks=None,
    debug_svd=False,
    print_layer_rank_detail=False,
):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    global_full_params = 0
    global_low_params = 0
    global_cost_sum = 0
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        ratio_i = layer_ratios[i] if layer_ratios is not None else ratio
        ranks_layer = module_ranks[i] if module_ranks is not None and i in module_ranks else None
        attn_ranks, mlp_ranks = _split_module_ranks(ranks_layer)
        subset = find_layers(layer)
        layer_full_params = 0
        layer_low_params = 0
        layer_cost_sum = 0
        layer_rank_values = []
        layer_rank_map = {}
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
            debug_g_info = ""
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
            m, n = int(W.shape[0]), int(W.shape[1])
            layer_full_params += m * n
            layer_low_params += num_s_after_trunc * (m + n)
            layer_cost_sum += (m + n)
            layer_rank_values.append(int(num_s_after_trunc))
            layer_rank_map[name] = int(num_s_after_trunc)
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
        _print_layer_compression_stats(
            "layer-stats",
            i,
            layer_full_params,
            layer_low_params,
            layer_cost_sum,
            layer_rank_values,
        )
        if print_layer_rank_detail:
            _print_layer_rank_detail(i, layer_rank_map)
        global_full_params += layer_full_params
        global_low_params += layer_low_params
        global_cost_sum += layer_cost_sum
        del layer
        torch.cuda.empty_cache()
    if global_full_params > 0 and global_cost_sum > 0:
        keep_ratio = global_low_params / global_full_params
        compress_ratio = 1.0 - keep_ratio
        effective_rank = global_low_params / global_cost_sum
        print(
            f"[layer-stats] overall: keep={keep_ratio:.4f} compress={compress_ratio:.4f} "
            f"effective_rank={effective_rank:.2f}"
        )


@torch.no_grad()
def whitening_local_update(
    model_name,
    model,
    dataloader,
    profiling_mat,
    ratio,
    dev,
    direct_update=False,
    layer_ratios=None,
    module_ranks=None,
    debug_svd=False,
    use_bi_closed_form=True,
    use_weighted_update=True,
    bi_weight_mode="residual",
    bi_weight_alpha=0.5,
    bi_weight_clip=10.0,
    bi_u_ridge=1e-5,
    bi_v_ridge=1e-5,
    bi_sigma_eps=1e-6,
    print_layer_rank_detail=False,
    update_layer_batch_size=1,
):
    print("Start SVD decomposition then update "
          f"(bi_closed_form={use_bi_closed_form}, weighted={use_weighted_update}, "
          f"weight_mode={bi_weight_mode}, layer_batch_size={update_layer_batch_size})...")
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
    update_layer_batch_size = max(1, int(update_layer_batch_size))

    def _layer_forward_chunked(layer_mod, hs, attn, pos_ids=None):
        n = hs.shape[0]
        out_all = torch.empty_like(hs)
        for st in range(0, n, update_layer_batch_size):
            ed = min(n, st + update_layer_batch_size)
            if "opt" not in model_name:
                out_all[st:ed] = layer_mod(
                    hs[st:ed],
                    attention_mask=attn[st:ed],
                    position_ids=pos_ids[st:ed],
                )[0]
            else:
                out_all[st:ed] = layer_mod(
                    hs[st:ed],
                    attention_mask=attn[st:ed],
                )[0]
        return out_all

    global_full_params = 0
    global_low_params = 0
    global_cost_sum = 0
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        ratio_i = layer_ratios[i] if layer_ratios is not None else ratio
        ranks_layer = module_ranks[i] if module_ranks is not None and i in module_ranks else None
        attn_ranks, mlp_ranks = _split_module_ranks(ranks_layer)
        subset = find_layers(layer)
        gpts = {}
        layer_full_params = 0
        layer_low_params = 0
        layer_cost_sum = 0
        layer_rank_values = []
        layer_rank_map = {}
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
            rank_override = ranks_layer[name] if ranks_layer is not None and name in ranks_layer else None
            debug_name = f"layer {i:02d} {name}" if debug_svd else name
            gpts[name] = local_update(
                subset[name],
                scaling_diag_matrix=scaling_diag_matrix,
                ratio=ratio_i,
                name=debug_name,
                direct_update=direct_update,
                rank=rank_override,
                debug_svd=debug_svd,
                use_bi_closed_form=use_bi_closed_form,
                use_weighted_update=use_weighted_update,
                bi_weight_mode=bi_weight_mode,
                bi_weight_alpha=bi_weight_alpha,
                bi_weight_clip=bi_weight_clip,
                bi_u_ridge=bi_u_ridge,
                bi_v_ridge=bi_v_ridge,
                bi_sigma_eps=bi_sigma_eps,
            )
            m, n = int(subset[name].weight.shape[0]), int(subset[name].weight.shape[1])
            rk = int(gpts[name].rank)
            layer_full_params += m * n
            layer_low_params += rk * (m + n)
            layer_cost_sum += (m + n)
            layer_rank_values.append(rk)
            layer_rank_map[name] = rk
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        if "opt" not in model_name:
            outs = _layer_forward_chunked(layer, inps, attention_masks, position_ids)
        else:
            outs = _layer_forward_chunked(layer, inps, attention_masks, None)
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
        _print_layer_compression_stats(
            "layer-stats",
            i,
            layer_full_params,
            layer_low_params,
            layer_cost_sum,
            layer_rank_values,
        )
        if print_layer_rank_detail:
            _print_layer_rank_detail(i, layer_rank_map)
        global_full_params += layer_full_params
        global_low_params += layer_low_params
        global_cost_sum += layer_cost_sum
        layer = layer.to(dev)
        if "opt" not in model_name:
            outs = _layer_forward_chunked(layer, inps, attention_masks, position_ids)
        else:
            outs = _layer_forward_chunked(layer, inps, attention_masks, None)
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache
    if global_full_params > 0 and global_cost_sum > 0:
        keep_ratio = global_low_params / global_full_params
        compress_ratio = 1.0 - keep_ratio
        effective_rank = global_low_params / global_cost_sum
        print(
            f"[layer-stats] overall: keep={keep_ratio:.4f} compress={compress_ratio:.4f} "
            f"effective_rank={effective_rank:.2f}"
        )


class local_update:
    def __init__(
        self,
        layer,
        scaling_diag_matrix,
        ratio,
        name,
        direct_update=False,
        rank=None,
        debug_svd=False,
        use_bi_closed_form=True,
        use_weighted_update=True,
        bi_weight_mode="residual",
        bi_weight_alpha=0.5,
        bi_weight_clip=10.0,
        bi_u_ridge=1e-5,
        bi_v_ridge=1e-5,
        bi_sigma_eps=1e-6,
    ):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        self.debug_svd = debug_svd
        self.use_bi_closed_form = bool(use_bi_closed_form)
        self.use_weighted_update = bool(use_weighted_update)
        self.bi_weight_mode = str(bi_weight_mode)
        self.bi_weight_alpha = float(bi_weight_alpha)
        self.bi_weight_clip = float(bi_weight_clip)
        self.bi_u_ridge = float(bi_u_ridge)
        self.bi_v_ridge = float(bi_v_ridge)
        self.bi_sigma_eps = float(bi_sigma_eps)
        # Keep model weights in original dtype (e.g. fp16), but run closed-form
        # solves in fp32 for numerical stability and dtype consistency.
        W = layer.weight.data.detach().to(self.dev, dtype=torch.float32).clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            scaling_diag_matrix = scaling_diag_matrix.to(self.dev, dtype=torch.float32)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(
                    scaling_diag_matrix.shape[0],
                    device=self.dev,
                    dtype=scaling_diag_matrix.dtype,
                )
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # truncation SVD
        if rank is not None:
            num_s_after_trunc = int(rank)
        else:
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        num_s_after_trunc = max(1, min(num_s_after_trunc, min(W.shape[0], W.shape[1])))
        self.rank = num_s_after_trunc
        if debug_svd:
            s2 = (self.S.float() ** 2)
            denom = float(s2.sum().item())
            if denom > 0:
                tail = float(s2[num_s_after_trunc:].sum().item())
                rel_err = math.sqrt(tail / denom)
            else:
                rel_err = 0.0
            print(f"[debug] {self.name} k={num_s_after_trunc} rel_err={rel_err:.6f}")
        self.truc_s = self.S[:num_s_after_trunc].to(self.dev, dtype=torch.float32)
        self.truc_u = self.U[:, :num_s_after_trunc].to(self.dev, dtype=torch.float32)
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].to(self.dev, dtype=torch.float32)
        else:
            self.truc_v = torch.matmul(
                self.VT[:num_s_after_trunc, :].to(self.dev, dtype=torch.float32),
                scaling_matrix_inv,
            )
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0
        # One-shot initialization: if hooks are skipped, fallback to truncated factors.
        self.updated_uT = self.truc_u.t().clone()
        self.updated_truc_v = self.truc_v.clone()

    def _solve_weighted_ridge(self, X, Y, weights=None, ridge=0.0, prior=None):
        if weights is not None:
            sw = torch.sqrt(weights).unsqueeze(1)
            Xw = X * sw
            Yw = Y * sw
        else:
            Xw = X
            Yw = Y
        lhs = Xw.t().matmul(Xw)
        rhs = Xw.t().matmul(Yw)
        if ridge > 0:
            eye = torch.eye(lhs.shape[0], device=lhs.device, dtype=lhs.dtype)
            lhs = lhs + ridge * eye
            if prior is not None:
                rhs = rhs + ridge * prior
        try:
            sol = torch.linalg.solve(lhs, rhs)
        except Exception:
            sol = torch.linalg.lstsq(lhs, rhs).solution
        return sol

    def _compute_sample_weights(self, outs, base_output, eps=1e-12):
        if (not self.use_weighted_update) or self.bi_weight_mode == "uniform":
            return None
        mode = self.bi_weight_mode
        if mode == "output_norm":
            base = outs.pow(2).mean(dim=1)
        else:
            # Default residual-based weighting: prioritize hard tokens.
            base = (outs - base_output).pow(2).mean(dim=1)
        alpha = max(0.0, self.bi_weight_alpha)
        weights = (base + eps).pow(alpha)
        mean_w = float(weights.mean().item())
        if mean_w > 0:
            weights = weights / mean_w
        clip = max(1.0, self.bi_weight_clip)
        weights = torch.clamp(weights, min=1.0 / clip, max=clip)
        return weights

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2]).float()
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2]).float()
        # Rank-space feature used by both U-step and V-step.
        x_latent = torch.matmul(torch.matmul(inps, self.truc_v.t()), self.truc_sigma)  # [N, r]
        base_output = torch.matmul(x_latent, self.truc_u.t())  # [N, d_out]
        out_norm = torch.norm(outs, p="fro").clamp_min(1e-12)
        self.error = torch.sqrt(torch.sum((outs - base_output) ** 2)).item() / out_norm.item()

        weights = self._compute_sample_weights(outs, base_output)

        # Step 1: fixed V, solve U in closed form (weighted ridge LS).
        u_prior = self.truc_u.t()
        self.updated_uT = self._solve_weighted_ridge(
            x_latent,
            outs,
            weights=weights,
            ridge=self.bi_u_ridge,
            prior=u_prior,
        )

        # Step 2: fixed U, solve right factor one-shot in rank space.
        if self.use_bi_closed_form:
            u_mat = self.updated_uT.t()                     # [d_out, r]
            target_latent = torch.matmul(outs, u_mat)      # [N, r]
            r_prior = torch.eye(self.rank, device=self.dev, dtype=x_latent.dtype)
            r_corr = self._solve_weighted_ridge(
                x_latent,
                target_latent,
                weights=weights,
                ridge=self.bi_v_ridge,
                prior=r_prior,
            )  # [r, r]
            sigma = self.truc_sigma
            sigma_inv = torch.diag(1.0 / torch.clamp(self.truc_s, min=self.bi_sigma_eps))
            # Enforce x_new = x_latent @ r_corr via V update:
            # V_new^T = V^T Σ R Σ^{-1}  =>  V_new = Σ^{-1} R^T Σ V.
            self.updated_truc_v = sigma_inv.matmul(r_corr.t().matmul(sigma.matmul(self.truc_v)))
        else:
            self.updated_truc_v = self.truc_v

        updated_latent = torch.matmul(torch.matmul(inps, self.updated_truc_v.t()), self.truc_sigma)
        updated_output = torch.matmul(updated_latent, self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output) ** 2)).item() / out_norm.item()
        if self.debug_svd:
            if weights is None:
                print(f"[debug] {self.name} one-shot: rel_err {self.error:.6f} -> {self.updated_error:.6f}")
            else:
                wmin = float(weights.min().item())
                wmax = float(weights.max().item())
                print(f"[debug] {self.name} one-shot: rel_err {self.error:.6f} -> {self.updated_error:.6f}  "
                      f"weights[{wmin:.3f},{wmax:.3f}]")

        inps = outs = x_latent = base_output = updated_latent = updated_output = None
        del inps, outs, x_latent, base_output, updated_latent, updated_output
        torch.cuda.empty_cache()
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.updated_truc_v)
        return self.appendU, self.appendV


def _obtain_layer_ratios(args, model, grad_diag):
    layer_ratios = compute_layer_ratios(args.model, model, grad_diag, args.ratio, min_ratio=args.layer_ratio_min, max_ratio=args.layer_ratio_max, eps=1e-6)
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


def _get_transformer_layers(model_name, model):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        return model.model.layers
    if "opt" in model_name:
        return model.model.decoder.layers
    return None


@torch.no_grad()
def _eval_calib_loss(
    model,
    calib_data,
    dev,
    max_batches=None,
    use_autocast_bf16=False,
    early_stop_loss=None,
    early_stop_min_batches=0,
):
    prev_use_cache = getattr(model.config, "use_cache", None)
    if prev_use_cache is not None:
        model.config.use_cache = False
    model.eval()
    dev_obj = dev if isinstance(dev, torch.device) else torch.device(dev)
    device_type = dev_obj.type
    enable_autocast = bool(use_autocast_bf16 and device_type == "cuda" and torch.cuda.is_available())
    early_stop_min_batches = max(0, int(early_stop_min_batches))
    total_loss = 0.0
    total_tokens = 0
    for bi, batch in enumerate(calib_data):
        if max_batches is not None and bi >= max_batches:
            break
        # If calibration batches are preloaded on device, avoid repeated copies.
        first_tensor = next(iter(batch.values()))
        if first_tensor.device == dev_obj:
            batch = batch
        else:
            batch = {k: v.to(dev_obj, non_blocking=True) for k, v in batch.items()}
        labels = batch["input_ids"]
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enable_autocast else nullcontext()
        with ctx:
            outputs = model(
                **batch,
                labels=labels,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        loss_val = float(outputs.loss.item())
        ntok = int(labels.numel())
        total_loss += loss_val * ntok
        total_tokens += ntok
        if (
            early_stop_loss is not None
            and total_tokens > 0
            and (bi + 1) >= early_stop_min_batches
        ):
            running_loss = total_loss / total_tokens
            if running_loss > float(early_stop_loss):
                break
    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache
    if total_tokens <= 0:
        return float("inf")
    return total_loss / total_tokens


def _rank_from_ratio(m, n, ratio, max_rank=None):
    k = int(m * n * ratio / (m + n))
    k = max(1, min(k, min(m, n)))
    if max_rank is not None:
        k = min(k, int(max_rank))
    return k


@torch.no_grad()
def _low_rank_weight_from_ratio(module, ratio, dev, scaling_diag_matrix=None, max_rank=None):
    W = module.weight.data.float().to(dev).clone()

    scaling_inv = None
    if scaling_diag_matrix is not None:
        scaling_diag_matrix = scaling_diag_matrix.to(dev).float()
        try:
            scaling_inv = torch.linalg.inv(scaling_diag_matrix)
        except Exception:
            scaling_diag_matrix = scaling_diag_matrix + 1e-6 * torch.eye(
                scaling_diag_matrix.shape[0], device=dev, dtype=scaling_diag_matrix.dtype
            )
            scaling_inv = torch.linalg.inv(scaling_diag_matrix)
        W_scale = torch.matmul(W, scaling_diag_matrix)
    else:
        W_scale = W

    U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
    m, n = W.shape
    k = _rank_from_ratio(m, n, ratio, max_rank=max_rank)
    truc_s = S[:k]
    truc_u = U[:, :k]
    if scaling_inv is not None:
        truc_v = torch.matmul(VT[:k, :], scaling_inv)
    else:
        truc_v = VT[:k, :]
    W_hat = torch.matmul(truc_u, torch.diag(truc_s).matmul(truc_v))
    return W_hat, k


def _loss_aware_candidate_ratios(args):
    lo = max(1e-4, float(args.layer_ratio_min))
    hi = min(0.9999, float(args.layer_ratio_max))
    if args.loss_aware_ratio_candidates is not None:
        vals = []
        for token in str(args.loss_aware_ratio_candidates).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                vals.append(min(hi, max(lo, float(token))))
            except Exception:
                pass
        if vals:
            if args.loss_aware_include_bounds:
                vals.extend([lo, hi])
            vals.append(min(hi, max(lo, float(args.ratio))))
            return sorted({round(v, 8) for v in vals})
        print("[loss-aware] WARNING: --loss_aware_ratio_candidates parse failed, falling back to generated grid.")
    n = max(3, int(args.loss_aware_num_candidates))
    span = max(0.0, float(args.loss_aware_ratio_span))
    center = min(hi, max(lo, float(args.ratio)))
    if n == 1 or span <= 0:
        return [center]
    start = center - span
    end = center + span
    vals = []
    for i in range(n):
        t = i / max(n - 1, 1)
        vals.append(min(hi, max(lo, start + (end - start) * t)))
    vals.append(center)
    return sorted({round(v, 8) for v in vals})


def _solve_layerwise_mckp(layer_tables, budget_params, total_full, dp_bins=2000):
    n_layers = len(layer_tables)
    if n_layers == 0:
        return [], 0.0
    dp_bins = max(200, int(dp_bins))
    bin_size = max(1, int(math.ceil(budget_params / dp_bins)))
    budget_bin = max(1, int(budget_params // bin_size))

    layer_bins = []
    min_bin_sum = 0
    max_bin_sum = 0
    for table in layer_tables:
        items = []
        for ci, entry in enumerate(table):
            cb = max(1, int(math.ceil(entry["cost"] / bin_size)))
            items.append((ci, cb, float(entry["delta"])))
        min_bin_sum += min(x[1] for x in items)
        max_bin_sum += max(x[1] for x in items)
        layer_bins.append(items)

    if min_bin_sum > budget_bin:
        chosen = [min(range(len(t)), key=lambda i: t[i]["cost"]) for t in layer_tables]
        cost = sum(layer_tables[i][chosen[i]]["cost"] for i in range(n_layers))
        return chosen, cost / max(total_full, 1)
    if max_bin_sum < budget_bin:
        chosen = [max(range(len(t)), key=lambda i: t[i]["cost"]) for t in layer_tables]
        cost = sum(layer_tables[i][chosen[i]]["cost"] for i in range(n_layers))
        return chosen, cost / max(total_full, 1)

    inf = float("inf")
    prev = [inf] * (budget_bin + 1)
    prev[0] = 0.0
    trace = []
    for li in range(n_layers):
        curr = [inf] * (budget_bin + 1)
        back = [(-1, -1)] * (budget_bin + 1)
        for bprev in range(budget_bin + 1):
            if prev[bprev] == inf:
                continue
            base = prev[bprev]
            for ci, cb, delta in layer_bins[li]:
                b = bprev + cb
                if b > budget_bin:
                    continue
                cand = base + delta
                if cand < curr[b]:
                    curr[b] = cand
                    back[b] = (bprev, ci)
        prev = curr
        trace.append(back)

    feasible = [b for b, v in enumerate(prev) if v < inf]
    if not feasible:
        chosen = [min(range(len(t)), key=lambda i: t[i]["cost"]) for t in layer_tables]
        cost = sum(layer_tables[i][chosen[i]]["cost"] for i in range(n_layers))
        return chosen, cost / max(total_full, 1)
    best_b = min(feasible, key=lambda b: (prev[b], -b))

    chosen = [0] * n_layers
    b = best_b
    for li in range(n_layers - 1, -1, -1):
        bprev, ci = trace[li][b]
        if bprev < 0 or ci < 0:
            chosen[li] = min(range(len(layer_tables[li])), key=lambda i: layer_tables[li][i]["cost"])
            b = max(b, 0)
        else:
            chosen[li] = ci
            b = bprev
    cost = sum(layer_tables[i][chosen[i]]["cost"] for i in range(n_layers))
    return chosen, cost / max(total_full, 1)


@torch.no_grad()
def _obtain_loss_aware_layer_ratios(args, model, tokenizer, profiling_mat, cali_white_data=None):
    layers = _get_transformer_layers(args.model, model)
    if layers is None:
        return None

    eval_nsamples = max(1, int(args.loss_aware_nsamples))
    eval_seq_len = int(args.loss_aware_seq_len) if args.loss_aware_seq_len is not None else min(args.model_seq_len, 512)
    eval_batch = max(1, int(args.loss_aware_batch_size))
    eval_windows = eval_nsamples * eval_batch
    can_reuse_white = (
        cali_white_data is not None
        and eval_seq_len == args.model_seq_len
        and eval_batch == 1
        and len(cali_white_data) >= eval_nsamples
    )
    if can_reuse_white:
        eval_data = cali_white_data[:eval_nsamples]
    else:
        eval_data = get_calib_train_data(
            args.dataset,
            tokenizer,
            eval_windows,
            seqlen=eval_seq_len,
            batch_size=eval_batch,
        )
    if len(eval_data) > eval_nsamples:
        eval_data = eval_data[:eval_nsamples]
    if args.loss_aware_cache_on_device:
        try:
            eval_data = [{k: v.to(args.DEV, non_blocking=True) for k, v in batch.items()} for batch in eval_data]
            print(f"[loss-aware] calibration batches cached on device={args.DEV}")
        except RuntimeError as e:
            print(f"[loss-aware] WARNING: cache_on_device failed ({e}), fallback to host batches.")
            if "cuda" in str(args.DEV):
                torch.cuda.empty_cache()

    prev_device = next(iter(model.parameters())).device
    model = model.to(args.DEV)
    full_max_batches = int(args.loss_aware_eval_max_batches) if int(args.loss_aware_eval_max_batches) > 0 else None
    stage1_batches = max(1, int(args.loss_aware_stage1_batches))
    stage1_topk = max(1, int(args.loss_aware_stage1_topk))
    use_autocast_bf16 = bool(args.loss_aware_autocast_bf16)
    use_early_stop = bool(args.loss_aware_early_stop)
    early_stop_margin = float(args.loss_aware_early_stop_margin)
    early_stop_min_batches = max(1, int(args.loss_aware_early_stop_min_batches))
    eval_batches = min(len(eval_data), full_max_batches) if full_max_batches is not None else len(eval_data)
    base_loss = _eval_calib_loss(
        model,
        eval_data,
        args.DEV,
        max_batches=full_max_batches,
        use_autocast_bf16=use_autocast_bf16,
    )
    print(f"[loss-aware] baseline calib loss={base_loss:.6f} using {eval_batches} batches (seq_len={eval_seq_len}, batch_size={eval_batch})")

    candidates = _loss_aware_candidate_ratios(args)
    print(f"[loss-aware] ratio candidates: {', '.join(f'{r:.4f}' for r in candidates)}")
    use_two_stage = bool(args.loss_aware_two_stage and len(candidates) > stage1_topk)
    if use_two_stage:
        print(
            f"[loss-aware] two-stage enabled: stage1_batches={stage1_batches}, "
            f"stage1_topk={stage1_topk}, full_batches={eval_batches}"
        )
    if use_autocast_bf16:
        print("[loss-aware] eval autocast: bfloat16")
    if use_early_stop:
        print(
            f"[loss-aware] early-stop enabled: margin={early_stop_margin:.6f}, "
            f"min_batches={early_stop_min_batches}"
        )

    layer_tables = []
    layer_full_sizes = []
    for li in tqdm(range(len(layers)), desc="loss-aware layers"):
        layer = layers[li]
        subset = find_layers(layer)
        profiling_layer = profiling_mat[li] if (profiling_mat is not None and li in profiling_mat) else None
        # Build per-module low-rank caches once for this layer. Candidate sweeps then
        # only change the truncation rank k, avoiding repeated full SVD/inversion.
        layer_cache = {}
        layer_full = 0
        for name, mod in subset.items():
            m, n = mod.weight.shape
            layer_full += int(m * n)
            k_cap = max(_rank_from_ratio(int(m), int(n), r, max_rank=args.module_rank_max) for r in candidates)
            k_cap = max(1, min(k_cap, min(int(m), int(n))))

            scale = profiling_layer[name] if (profiling_layer is not None and name in profiling_layer) else None
            W = mod.weight.data.detach().float().to(args.DEV)
            if scale is not None:
                scale = scale.to(args.DEV).float()
                try:
                    scale_inv = torch.linalg.inv(scale)
                except Exception:
                    scale = scale + 1e-6 * torch.eye(scale.shape[0], device=args.DEV, dtype=scale.dtype)
                    scale_inv = torch.linalg.inv(scale)
                W_scale = torch.matmul(W, scale)
            else:
                scale_inv = None
                W_scale = W

            U_full, S_full, VT_full = torch.linalg.svd(W_scale, full_matrices=False)
            U = U_full[:, :k_cap].contiguous()
            S = S_full[:k_cap].contiguous()
            if scale_inv is not None:
                V = torch.matmul(VT_full[:k_cap, :], scale_inv).contiguous()
            else:
                V = VT_full[:k_cap, :].contiguous()
            layer_cache[name] = {
                "backup": mod.weight.data.detach().clone(),
                "m": int(m),
                "n": int(n),
                "k_cap": int(k_cap),
                "U": U,
                "S": S,
                "V": V,
            }
            W = W_scale = scale = scale_inv = U_full = S_full = VT_full = None
            del W, W_scale, scale, scale_inv, U_full, S_full, VT_full
        layer_full_sizes.append(layer_full)

        table = []

        def _apply_ratio(ratio_i):
            cost_i = 0
            for name, mod in subset.items():
                info = layer_cache[name]
                rank_k = _rank_from_ratio(info["m"], info["n"], ratio_i, max_rank=args.module_rank_max)
                rank_k = max(1, min(rank_k, info["k_cap"]))
                U = info["U"][:, :rank_k]
                S = info["S"][:rank_k]
                V = info["V"][:rank_k, :]
                w_hat = torch.matmul(U * S.unsqueeze(0), V)
                mod.weight.data.copy_(w_hat.to(mod.weight.device, dtype=mod.weight.dtype))
                cost_i += int(rank_k * (info["m"] + info["n"]))
            return cost_i

        if use_two_stage:
            coarse_table = []
            for ratio_i in candidates:
                cost_i = _apply_ratio(ratio_i)
                loss_i = _eval_calib_loss(
                    model,
                    eval_data,
                    args.DEV,
                    max_batches=stage1_batches,
                    use_autocast_bf16=use_autocast_bf16,
                )
                coarse_table.append({
                    "ratio": float(ratio_i),
                    "cost": int(cost_i),
                    "loss": float(loss_i),
                    "delta": float(loss_i - base_loss),
                })
            coarse_sorted = sorted(coarse_table, key=lambda x: x["delta"])
            selected_ratios = {x["ratio"] for x in coarse_sorted[:stage1_topk]}
            selected_ratios.add(min(candidates))
            selected_ratios.add(max(candidates))
            selected_ratios.add(min(candidates, key=lambda r: abs(float(r) - float(args.ratio))))
            selected_candidates = sorted(selected_ratios)
            coarse_delta_map = {x["ratio"]: x["delta"] for x in coarse_sorted}
            eval_order = sorted(selected_candidates, key=lambda r: coarse_delta_map.get(float(r), float("inf")))
            if args.print_layer_ratios:
                kept = ",".join([f"{x:.4f}" for x in selected_candidates])
                print(f"[loss-aware] layer {li:02d} stage1 keep candidates: {kept}")
        else:
            selected_candidates = candidates
            eval_order = list(selected_candidates)

        layer_best_loss = float("inf")
        for ratio_i in eval_order:
            cost_i = _apply_ratio(ratio_i)
            stop_loss = None
            if use_early_stop and layer_best_loss < float("inf"):
                stop_loss = layer_best_loss + early_stop_margin
            loss_i = _eval_calib_loss(
                model,
                eval_data,
                args.DEV,
                max_batches=full_max_batches,
                use_autocast_bf16=use_autocast_bf16,
                early_stop_loss=stop_loss,
                early_stop_min_batches=early_stop_min_batches,
            )
            layer_best_loss = min(layer_best_loss, loss_i)
            delta_i = loss_i - base_loss
            table.append({
                "ratio": float(ratio_i),
                "cost": int(cost_i),
                "loss": float(loss_i),
                "delta": float(delta_i),
            })
        table.sort(key=lambda x: x["ratio"])
        for name, mod in subset.items():
            mod.weight.data.copy_(layer_cache[name]["backup"].to(mod.weight.device, dtype=mod.weight.dtype))
        layer_cache.clear()
        layer_tables.append(table)
        if args.print_layer_ratios:
            msg = "  ".join([f"r={x['ratio']:.4f}:Δ={x['delta']:+.5f}" for x in table])
            print(f"[loss-aware] layer {li:02d} {msg}")

    total_full = sum(layer_full_sizes)
    target_params = args.ratio * total_full
    chosen_idx, _ = _solve_layerwise_mckp(
        layer_tables,
        target_params,
        total_full,
        dp_bins=args.loss_aware_dp_bins,
    )
    chosen_ratios = [layer_tables[i][chosen_idx[i]]["ratio"] for i in range(len(layer_tables))]
    chosen_cost = sum(layer_tables[i][chosen_idx[i]]["cost"] for i in range(len(layer_tables)))
    chosen_delta = sum(layer_tables[i][chosen_idx[i]]["delta"] for i in range(len(layer_tables)))
    print(f"[loss-aware] selected effective_ratio={chosen_cost/max(total_full,1):.6f} "
          f"(target={args.ratio:.6f})  total_delta={chosen_delta:+.6f}")
    if args.print_layer_ratios:
        for li, ratio_i in enumerate(chosen_ratios):
            info = layer_tables[li][chosen_idx[li]]
            print(f"[loss-aware] layer {li:02d} ratio={ratio_i:.6f} "
                  f"cost={info['cost']} delta={info['delta']:+.6f}")

    if str(prev_device) == "cpu":
        model = model.cpu()
    return chosen_ratios


def _obtain_module_ranks(args, model, profiling_mat):
    spectrum_path = args.spectrum_path
    if spectrum_path is None and args.save_path is not None:
        spectrum_path = _default_spectrum_path(args.save_path, args.model, args.dataset, args.whitening_nsamples, args.seed)
    spectrum = None
    if spectrum_path is not None and os.path.exists(spectrum_path):
        expected_meta = _spectrum_meta(args)
        spectrum, ok = _load_spectrum(spectrum_path, expected_meta)
        if not ok:
            print("Warning: spectrum cache meta mismatch, recomputing...")
            spectrum = None
    if spectrum is None:
        spectrum = profile_module_spectrum(args.model, model, profiling_mat, args.DEV)
        if spectrum_path is not None:
            meta = _spectrum_meta(args)
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
        if floor_over_budget:
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
            if kmin == kmax and mt in _cli_flag:
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
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--use_layerwise_ratio', action='store_true', help='allocate per-layer ratios based on G importance')
    parser.add_argument('--layer_ratio_min', type=float, default=0.01, help='Minimum per-layer keep ratio')
    parser.add_argument('--layer_ratio_max', type=float, default=0.99, help='Maximum per-layer keep ratio')
    parser.add_argument('--layer_ratio_strict', action='store_true', help='Rescale layer ratios to exactly match the global keep ratio')
    parser.add_argument('--print_layer_ratios', action='store_true', help='Print per-layer keep ratios and effective global ratio')
    parser.add_argument('--use_loss_aware_layerwise', action='store_true',
        help='Allocate one uniform keep ratio per layer by real calibration loss (NLL) + dynamic programming under global budget.')
    parser.add_argument('--loss_aware_nsamples', type=int, default=8,
        help='Number of calibration batches for loss-aware layerwise allocation.')
    parser.add_argument('--loss_aware_seq_len', type=int, default=512,
        help='Sequence length for loss-aware calibration (shorter is faster).')
    parser.add_argument('--loss_aware_batch_size', type=int, default=1,
        help='Batch size for loss-aware calibration.')
    parser.add_argument('--loss_aware_num_candidates', type=int, default=5,
        help='Number of keep-ratio candidates per layer when auto-generating the candidate grid.')
    parser.add_argument('--loss_aware_ratio_span', type=float, default=0.08,
        help='Candidate span around target keep ratio (target±span).')
    parser.add_argument('--loss_aware_ratio_candidates', type=str, default=None,
        help='Optional comma-separated keep-ratio candidates (after internal ratio conversion), e.g. "0.32,0.36,0.40,0.44,0.48".')
    parser.add_argument('--loss_aware_include_bounds', action='store_true',
        help='When custom loss-aware candidates are provided, also include layer_ratio_min/max to keep allocator freedom.')
    parser.add_argument('--loss_aware_two_stage', action='store_true',
        help='Enable two-stage candidate evaluation: coarse screen all candidates, then fully evaluate top-k per layer.')
    parser.add_argument('--loss_aware_stage1_batches', type=int, default=4,
        help='Number of calibration batches used in stage-1 coarse screening (only when --loss_aware_two_stage).')
    parser.add_argument('--loss_aware_stage1_topk', type=int, default=3,
        help='Per-layer number of stage-1 best candidates kept for full evaluation (only when --loss_aware_two_stage).')
    parser.add_argument('--loss_aware_eval_max_batches', type=int, default=0,
        help='Cap the number of calibration batches per loss evaluation. 0 means using all available batches.')
    parser.add_argument('--loss_aware_autocast_bf16', action='store_true',
        help='Run loss-aware evaluation forwards under bfloat16 autocast on CUDA for speed.')
    parser.add_argument('--loss_aware_cache_on_device', action='store_true',
        help='Preload loss-aware calibration batches onto args.DEV to reduce host->device copy overhead.')
    parser.add_argument('--loss_aware_early_stop', action='store_true',
        help='Enable candidate loss early-stop when running loss is already worse than current layer-best by a margin.')
    parser.add_argument('--loss_aware_early_stop_margin', type=float, default=0.003,
        help='Loss margin for early-stop (stop when running_loss > layer_best + margin).')
    parser.add_argument('--loss_aware_early_stop_min_batches', type=int, default=4,
        help='Minimum number of batches evaluated before early-stop can trigger.')
    parser.add_argument('--loss_aware_dp_bins', type=int, default=2000,
        help='DP budget bins for loss-aware layerwise allocation (larger = finer budget match, slower DP).')
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
    parser.add_argument('--print_layer_rank_detail', action='store_true',
        help='Print per-layer rank detail by module type (qkv/o/mlp/down).')
    parser.add_argument('--disable_bi_closed_form', action='store_true',
        help='Disable one-shot bi-side correction and use U-only closed-form update.')
    parser.add_argument('--disable_weighted_update', action='store_true',
        help='Disable weighted update and use uniform token weights in local correction.')
    parser.add_argument('--bi_weight_mode', type=str, default='residual', choices=['residual', 'output_norm', 'uniform'],
        help='Sample weighting mode for one-shot correction.')
    parser.add_argument('--bi_weight_alpha', type=float, default=0.5,
        help='Power for sample weighting (w=(signal+eps)^alpha).')
    parser.add_argument('--bi_weight_clip', type=float, default=10.0,
        help='Clip range for normalized sample weights [1/clip, clip].')
    parser.add_argument('--bi_u_ridge', type=float, default=1e-5,
        help='Ridge coefficient for U-step closed-form solve.')
    parser.add_argument('--bi_v_ridge', type=float, default=1e-5,
        help='Ridge coefficient for V-step closed-form solve.')
    parser.add_argument('--bi_sigma_eps', type=float, default=1e-6,
        help='Numerical epsilon for inverting singular values in bi-side V-step.')
    parser.add_argument('--update_use_fp32', action='store_true',
        help='Cast model to fp32 before local update (step 2/3). Default keeps model dtype (e.g. fp16).')
    parser.add_argument('--update_layer_batch_size', type=int, default=1,
        help='Micro-batch size for per-layer forward in local update (step 2/3). Smaller value reduces VRAM peak.')
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
        if args.use_loss_aware_layerwise:
            if args.use_module_rank_allocation:
                print("[loss-aware] --use_loss_aware_layerwise takes priority; --use_module_rank_allocation is ignored.")
            layer_ratios = _obtain_loss_aware_layer_ratios(
                args,
                model,
                tokenizer,
                profiling_mat,
                cali_white_data=cali_white_data,
            )
            module_ranks = None
        elif args.use_layerwise_ratio:
            layer_ratios = _obtain_layer_ratios(args, model, None)
        if args.use_module_rank_allocation and not args.use_loss_aware_layerwise:
            module_ranks = _obtain_module_ranks(args, model, profiling_mat)
            layer_ratios = None
        whitening(
            args.model,
            model,
            profiling_mat,
            args.ratio,
            args.DEV,
            layer_ratios=layer_ratios,
            module_ranks=module_ranks,
            debug_svd=args.debug_svd,
            print_layer_rank_detail=args.print_layer_rank_detail,
        )
        if args.save_path is not None:
            _save_model_fp16(model, tokenizer, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + '.pt')
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        if args.update_use_fp32:
            model = model.float()
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
        if args.use_loss_aware_layerwise:
            if args.use_module_rank_allocation:
                print("[loss-aware] --use_loss_aware_layerwise takes priority; --use_module_rank_allocation is ignored.")
            layer_ratios = _obtain_loss_aware_layer_ratios(
                args,
                model,
                tokenizer,
                profiling_mat,
                cali_white_data=cali_white_data,
            )
            module_ranks = None
        elif args.use_layerwise_ratio:
            layer_ratios = _obtain_layer_ratios(args, model, None)
        if args.use_module_rank_allocation and not args.use_loss_aware_layerwise:
            module_ranks = _obtain_module_ranks(args, model, profiling_mat)
            layer_ratios = None
        if args.use_loss_aware_layerwise:
            print("[method] Loss-aware rank allocation + weighted one-shot bi-side closed-form correction")
        whitening_local_update(
            args.model,
            model,
            dataloader,
            profiling_mat,
            args.ratio,
            args.DEV,
            layer_ratios=layer_ratios,
            module_ranks=module_ranks,
            debug_svd=args.debug_svd,
            use_bi_closed_form=not args.disable_bi_closed_form,
            use_weighted_update=not args.disable_weighted_update,
            bi_weight_mode=args.bi_weight_mode,
            bi_weight_alpha=args.bi_weight_alpha,
            bi_weight_clip=args.bi_weight_clip,
            bi_u_ridge=args.bi_u_ridge,
            bi_v_ridge=args.bi_v_ridge,
            bi_sigma_eps=args.bi_sigma_eps,
            print_layer_rank_detail=args.print_layer_rank_detail,
            update_layer_batch_size=args.update_layer_batch_size,
        )
        if args.save_path is not None:
            _save_model_fp16(model, tokenizer, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        if args.update_use_fp32:
            model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        layer_ratios = None
        module_ranks = None
        if args.use_loss_aware_layerwise:
            if args.use_module_rank_allocation:
                print("[loss-aware] --use_loss_aware_layerwise takes priority; --use_module_rank_allocation is ignored.")
            layer_ratios = _obtain_loss_aware_layer_ratios(
                args,
                model,
                tokenizer,
                profiling_mat=None,
                cali_white_data=None,
            )
            module_ranks = None
        elif args.use_layerwise_ratio:
            layer_ratios = _obtain_layer_ratios(args, model, None)
        if args.use_module_rank_allocation and not args.use_loss_aware_layerwise:
            module_ranks = _obtain_module_ranks(args, model, None)
            layer_ratios = None
        if args.use_loss_aware_layerwise:
            print("[method] Loss-aware rank allocation + weighted one-shot bi-side closed-form correction")
        whitening_local_update(
            model_name=args.model,
            model=model,
            dataloader=dataloader,
            profiling_mat=None,
            ratio=args.ratio,
            dev=args.DEV,
            direct_update=True,
            layer_ratios=layer_ratios,
            module_ranks=module_ranks,
            debug_svd=args.debug_svd,
            use_bi_closed_form=not args.disable_bi_closed_form,
            use_weighted_update=not args.disable_weighted_update,
            bi_weight_mode=args.bi_weight_mode,
            bi_weight_alpha=args.bi_weight_alpha,
            bi_weight_clip=args.bi_weight_clip,
            bi_u_ridge=args.bi_u_ridge,
            bi_v_ridge=args.bi_v_ridge,
            bi_sigma_eps=args.bi_sigma_eps,
            print_layer_rank_detail=args.print_layer_rank_detail,
            update_layer_batch_size=args.update_layer_batch_size,
        )
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
