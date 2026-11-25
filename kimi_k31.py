# Kimi K3.1 Pro for mlx-lm: the K3 model plus DSA (DeepSeek Sparse Attention).
# Copy this file to <mlx_lm>/models/kimi_k31.py.
#
# The MLA layers listed in dsa_config.dsa_layers get an indexer
# (self_attn.indexer.{wq_b, wk, k_norm, weights_proj}). For every query it
# scores all earlier keys and keeps the index_topk (2048) best; attention
# then runs over those keys only. Math follows the vLLM implementation
# (vllm/models/kimi_k3/nvidia/dsa.py):
#   q   = wq_b(q_a_layernorm(q_a_proj(x)))            [T, H, D]
#   k   = LayerNorm_fp32(wk(x)) -> model dtype        [T, D]
#   w   = weights_proj(x) * D**-0.5 * H**-0.5         [T, H]
#   score[t, s] = sum_h relu(q[t, h] . k[s]) * w[t, h],  s <= t
# Default: bf16 projections, fp32 scores (as NeMo AutoModel; reproducible across
# machines and tensor-parallel layouts). K31_INDEXER_FP8=1 quantizes q and k to
# FP8 e4m3 with power-of-two (ue8m0) scales first, as vLLM's serving kernels do.

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from . import kimi_k3
from .base import scaled_dot_product_attention
from .cache import CacheList, KVCache

FP8_MAX = 448.0
INDEXER_FP8 = os.environ.get("K31_INDEXER_FP8", "0") == "1"


@dataclass
class ModelArgs(kimi_k3.ModelArgs):
    model_type: str = "kimi_k31"
    dsa_config: Optional[Dict[str, Any]] = None
    # Written by scripts/presplit.py: this folder holds only per-rank parts.
    k31_presplit: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # TextArgs drops unknown keys, so keep dsa_config before parsing.
        if isinstance(self.text_config, dict) and self.dsa_config is None:
            self.dsa_config = self.text_config.get("dsa_config")
        super().__post_init__()


def _fp8_round_trip(x: mx.array, amax_floor: float) -> mx.array:
    # Per-row FP8 e4m3 quantize + dequantize with a ue8m0 (power of two) scale,
    # as vLLM's indexer_q_quant_fold_weights / indexer_k_norm_quant_insert.
    x = x.astype(mx.float32)
    amax = mx.maximum(mx.abs(x).max(axis=-1, keepdims=True), amax_floor)
    scale = mx.power(2.0, mx.ceil(mx.log2(amax / FP8_MAX)))
    y = mx.clip(x / scale, -FP8_MAX, FP8_MAX)
    return mx.from_fp8(mx.to_fp8(y), dtype=mx.float32) * scale


class DSACache(CacheList):
    """CacheList(MLA cache, indexer key cache) that also builds masks."""

    def __init__(self):
        super().__init__(KVCache(), KVCache())

    @property
    def offset(self):
        return self.caches[0].offset

    def make_mask(self, *args, **kwargs):
        return self.caches[0].make_mask(*args, **kwargs)

    def empty(self):
        return self.caches[0].empty()

    def extract(self, idx):
        c = DSACache()
        c.caches = tuple(x.extract(idx) for x in self.caches)
        return c


class KimiK31Indexer(nn.Module):
    def __init__(self, args, dsa: Dict[str, Any]):
        super().__init__()
        if dsa.get("index_rope_dim", 0) != 0:
            raise ValueError("kimi_k31: index_rope_dim > 0 is not supported")
        if dsa.get("index_k_norm_type", "layernorm") != "layernorm":
            raise ValueError("kimi_k31: only index_k_norm_type=layernorm is supported")
        if dsa.get("index_topk_freq", 1) != 1:
            raise ValueError("kimi_k31: index_topk_freq != 1 is not supported")
        self.n_heads = dsa["index_num_heads"]
        self.head_dim = dsa["index_head_dim"]
        self.index_topk = dsa["index_topk"]
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=args.rms_norm_eps)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.weight_scale = self.head_dim**-0.5 * self.n_heads**-0.5
        self.score_block = 4096  # keys scored per step; lower it to save memory

    def __call__(self, x, qr, mask, cache=None):
        """Top-k key indices [B, 1, L, K], or None when all keys are kept."""
        B, L, _ = x.shape
        q = self.wq_b(qr).reshape(B, L, self.n_heads, self.head_dim)
        k = self.wk(x)
        k = mx.fast.layer_norm(
            k.astype(mx.float32),
            self.k_norm.weight.astype(mx.float32),
            self.k_norm.bias.astype(mx.float32),
            self.k_norm.eps,
        ).astype(x.dtype)
        k = k.reshape(B, 1, L, self.head_dim)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, k)
            cache.values = cache.keys  # values unused; avoid a second copy
        N = k.shape[2]
        if N <= self.index_topk:
            return None

        if INDEXER_FP8:
            q = _fp8_round_trip(q, 1e-10)
            k = _fp8_round_trip(k, 1e-4)
        else:
            q, k = q.astype(mx.float32), k.astype(mx.float32)
        q = q.transpose(0, 2, 1, 3)  # [B, H, L, D]
        w = self.weights_proj(x).astype(mx.float32) * self.weight_scale
        w = w.transpose(0, 2, 1)[..., None]  # [B, H, L, 1]

        best_v = best_i = None
        K, block = self.index_topk, self.score_block
        for s0 in range(0, N, block):
            s1 = min(N, s0 + block)
            sc = (mx.maximum(q @ k[:, :, s0:s1].swapaxes(-1, -2), 0) * w).sum(
                axis=1, keepdims=True
            )  # [B, 1, L, blk]
            if mask is not None:
                sc = mx.where(mask[..., s0:s1], sc, -mx.inf)
            ids = mx.broadcast_to(mx.arange(s0, s1, dtype=mx.int32), sc.shape)
            if best_v is not None:
                sc = mx.concatenate([best_v, sc], axis=-1)
                ids = mx.concatenate([best_i, ids], axis=-1)
            if sc.shape[-1] > K:
                pos = mx.argpartition(sc, kth=-K, axis=-1)[..., -K:]
                sc = mx.take_along_axis(sc, pos, axis=-1)
                ids = mx.take_along_axis(ids, pos, axis=-1)
            best_v, best_i = sc, ids
        return mx.stop_gradient(best_i)


class KimiK31DSAAttention(kimi_k3.KimiK3MLAAttention):
    def __init__(self, args, dsa: Dict[str, Any]):
        super().__init__(args)
        self.indexer = KimiK31Indexer(args, dsa)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        mla_cache = idx_cache = None
        if cache is not None:
            mla_cache, idx_cache = cache[0], cache[1]

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = mx.expand_dims(self.kv_a_layernorm(compressed_kv), axis=1)
        if mla_cache is not None:
            kv_latent, k_pe = mla_cache.update_and_fetch(kv_latent, k_pe)

        topk = self.indexer(x, qr, mask, idx_cache)
        if topk is not None:
            N = kv_latent.shape[2]
            sel = mx.put_along_axis(
                mx.zeros((B, 1, L, N), dtype=mx.bool_), topk, mx.array(True), axis=-1
            )
            mask = sel if mask is None else mx.logical_and(mask, sel)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
            )
        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
        output = scaled_dot_product_attention(
            q_nope, k, v, cache=mla_cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_gate:
            output = output * mx.sigmoid(self.g_proj(x))
        return self.o_proj(output)


class Model(kimi_k3.Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.presplit = args.k31_presplit
        dsa = dict(args.dsa_config or {})
        self.dsa_layers = set(dsa.get("dsa_layers") or [])  # 1-based
        if self.dsa_layers:
            targs = self.language_model.args
            for i, layer in enumerate(self.language_model.model.layers):
                if (i + 1) in self.dsa_layers and not layer.is_linear:
                    layer.self_attn = KimiK31DSAAttention(targs, dsa)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        super().shard(group)
        if not self.presplit:
            return
        # Pre-split folder (scripts/presplit.py): the model was built without
        # weights; load this rank's part, already cut exactly as shard() cuts.
        import glob

        n, r = group.size(), group.rank()
        want = self.presplit.get("tp_size")
        if n != want:
            raise ValueError(f"model was pre-split for {want} Macs, launched on {n}")
        d = os.path.expanduser(os.environ.get("K31_PRESPLIT_DIR", ""))
        if not d:
            raise ValueError("pre-split model: set K31_PRESPLIT_DIR to its folder")
        files = sorted(glob.glob(os.path.join(d, f"tp{n}-r{r}-*.safetensors")))
        if not files:
            raise FileNotFoundError(
                f"rank {r}: no tp{n}-r{r}-*.safetensors in {d} "
                "(wrong part on this Mac, or hostfile order differs from the split)"
            )
        weights = {}
        for f in files:
            weights.update(mx.load(f))
        self.load_weights(list(weights.items()), strict=True)

    def make_cache(self):
        caches = self.language_model.make_cache()
        for c, layer in zip(range(len(caches)), self.layers):
            if isinstance(layer.self_attn, KimiK31DSAAttention):
                caches[c] = DSACache()
        return caches
