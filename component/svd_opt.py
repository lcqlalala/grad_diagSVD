# coding=utf-8
"""SVD parameterization for OPT layers.

The forward pass intentionally delegates to HuggingFace's OPT modules.  We only
replace selected Linear projections with low-rank Linear pairs, so causal masks,
cache handling, dropout placement, and layer-norm ordering stay identical to the
installed transformers implementation.
"""

from typing import Optional

from torch import nn

from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer


class SVDOPTLowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.v_proj = nn.Linear(in_features, rank, bias=False)
        self.u_proj = nn.Linear(rank, out_features, bias=bias)

    def forward(self, hidden_states):
        return self.u_proj(self.v_proj(hidden_states))


def _rank_from_ratio(m: int, n: int, ratio: float) -> int:
    return max(1, int(m * n * float(ratio) / max(1, m + n)))


def _rank_override(ranks, name: str, default: int, max_rank: int) -> Optional[int]:
    if ranks is None or name not in ranks:
        return None
    return max(1, min(int(ranks[name]), int(max_rank)))


def _maybe_low_rank_linear(in_features: int, out_features: int, rank: Optional[int], bias: bool):
    if rank is None:
        return nn.Linear(in_features, out_features, bias=bias)
    return SVDOPTLowRankLinear(in_features, out_features, rank, bias=bias)


def _u_proj(module):
    if isinstance(module, SVDOPTLowRankLinear):
        return module.u_proj
    raise AttributeError("projection is not low-rank")


def _v_proj(module):
    if isinstance(module, SVDOPTLowRankLinear):
        return module.v_proj
    raise AttributeError("projection is not low-rank")


class SVDOPTAttention(OPTAttention):
    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        ratio=1,
        ranks=None,
        **kwargs,
    ):
        try:
            super().__init__(config=config, is_decoder=is_decoder, **kwargs)
        except TypeError:
            # Older transformers releases used the explicit OPTAttention
            # constructor.  Keep this path so local pydeps versions remain usable.
            super().__init__(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=is_decoder,
                bias=config.enable_bias,
            )
        self.ratio = ratio

        embed_dim = int(getattr(self, "embed_dim", config.hidden_size))
        enable_bias = bool(getattr(self, "enable_bias", getattr(config, "enable_bias", True)))
        default_rank = _rank_from_ratio(embed_dim, embed_dim, ratio)

        def pick(name):
            r = _rank_override(ranks, name, default_rank, embed_dim)
            if r is None and float(ratio) != 1.0:
                r = default_rank
            return r

        q_rank = pick("q_proj")
        if q_rank is not None:
            self.q_proj = SVDOPTLowRankLinear(embed_dim, embed_dim, q_rank, bias=enable_bias)
        k_rank = pick("k_proj")
        if k_rank is not None:
            self.k_proj = SVDOPTLowRankLinear(embed_dim, embed_dim, k_rank, bias=enable_bias)
        v_rank = pick("v_proj")
        if v_rank is not None:
            self.v_proj = SVDOPTLowRankLinear(embed_dim, embed_dim, v_rank, bias=enable_bias)
        out_rank = pick("out_proj")
        if out_rank is not None:
            self.out_proj = SVDOPTLowRankLinear(embed_dim, embed_dim, out_rank, bias=enable_bias)

    @property
    def q_u_proj(self):
        return _u_proj(self.q_proj)

    @property
    def q_v_proj(self):
        return _v_proj(self.q_proj)

    @property
    def k_u_proj(self):
        return _u_proj(self.k_proj)

    @property
    def k_v_proj(self):
        return _v_proj(self.k_proj)

    @property
    def v_u_proj(self):
        return _u_proj(self.v_proj)

    @property
    def v_v_proj(self):
        return _v_proj(self.v_proj)

    @property
    def out_u_proj(self):
        return _u_proj(self.out_proj)

    @property
    def out_v_proj(self):
        return _v_proj(self.out_proj)


class SVDOPTDecoderLayer(OPTDecoderLayer):
    def __init__(self, config: OPTConfig, ratio=1, ranks=None):
        super().__init__(config)
        self.ratio = ratio

        attn_ranks = None
        mlp_ranks = None
        if ranks:
            attn_ranks = {k: v for k, v in ranks.items() if k in ("q_proj", "k_proj", "v_proj", "out_proj")}
            mlp_ranks = {k: v for k, v in ranks.items() if k in ("fc1", "fc2")}
        self.self_attn = SVDOPTAttention(config=config, ratio=ratio, is_decoder=True, ranks=attn_ranks)

        embed_dim = int(config.hidden_size)
        ffn_dim = int(config.ffn_dim)
        enable_bias = bool(getattr(config, "enable_bias", True))
        default_rank = _rank_from_ratio(ffn_dim, embed_dim, ratio)

        def pick(name):
            r = _rank_override(mlp_ranks, name, default_rank, min(embed_dim, ffn_dim))
            if r is None and float(ratio) != 1.0:
                r = default_rank
            return r

        fc1_rank = pick("fc1")
        if fc1_rank is not None:
            self.fc1 = SVDOPTLowRankLinear(embed_dim, ffn_dim, fc1_rank, bias=enable_bias)
        fc2_rank = pick("fc2")
        if fc2_rank is not None:
            self.fc2 = SVDOPTLowRankLinear(ffn_dim, embed_dim, fc2_rank, bias=enable_bias)

    @property
    def fc1_u_proj(self):
        return _u_proj(self.fc1)

    @property
    def fc1_v_proj(self):
        return _v_proj(self.fc1)

    @property
    def fc2_u_proj(self):
        return _u_proj(self.fc2)

    @property
    def fc2_v_proj(self):
        return _v_proj(self.fc2)
