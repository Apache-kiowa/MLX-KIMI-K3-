"""Check the MLX DSA (kimi_k31.py) against the PyTorch reference.

1. MLX vs official code + dsa_torch_ref (full prompt), logits and top-1.
2. MLX full prompt vs MLX prefill + step-by-step decode with the cache.
--topk lowers index_topk on both sides so short prompts already go sparse.

Needs: torch transformers fla-core einops tiktoken; experts decoded first
with dequant_src.py (pass that folder as --src).
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_map

import dsa_torch_ref
import fla_torch_shim  # noqa: F401
from mlx_lm.models import kimi_k31
from mlx_lm.utils import load_model
from transformers import AutoModelForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="HF folder with decoded experts")
    p.add_argument("--mlx", required=True)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--decode-steps", type=int, default=8)
    p.add_argument("--fp8", action="store_true", help="FP8 indexer scoring (vLLM serving kernels)")
    p.add_argument("--last", type=int, default=64, help="compare logits of the last N positions")
    p.add_argument("--score-block", type=int, default=4096)
    a = p.parse_args()
    fp8 = a.fp8
    kimi_k31.INDEXER_FP8 = fp8

    hf = AutoModelForCausalLM.from_pretrained(
        a.src, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    n = dsa_torch_ref.install(hf, a.src, use_fp8=fp8, topk=a.topk)

    model, cfg = load_model(Path(a.mlx))
    model.update(tree_map(lambda v: v.astype(mx.float32) if v.dtype == mx.bfloat16 else v,
                          model.parameters()))
    m = 0
    for layer in model.layers:
        if isinstance(layer.self_attn, kimi_k31.KimiK31DSAAttention):
            if a.topk:
                layer.self_attn.indexer.index_topk = a.topk
            layer.self_attn.indexer.score_block = a.score_block
            m += 1
    topk = a.topk or cfg["text_config"]["dsa_config"]["index_topk"]
    print(f"DSA layers: torch {n}, mlx {m}; index_topk={topk}; fp8={fp8}; tokens={a.tokens}")

    vocab = cfg["text_config"]["vocab_size"]
    toks = np.random.default_rng(1).integers(0, vocab, size=(1, a.tokens))
    M = min(a.last, a.tokens)
    lm = hf.language_model
    with torch.no_grad():
        hs = lm.model(input_ids=torch.tensor(toks)).last_hidden_state[:, -M:]
        ref = lm.lm_head(hs).float().numpy()
    del hs
    mlm = model.language_model
    h = mlm.model(mx.array(toks))[:, -M:]
    out = np.array(mlm.lm_head(h).astype(mx.float32))
    diff = np.abs(out - ref).max()
    top = (out.argmax(-1) == ref.argmax(-1)).mean()
    print(f"[1] MLX vs PyTorch: max|diff| = {diff:.2e} (max|ref| = {np.abs(ref).max():.2e}), "
          f"same top-1 = {top:.0%}")

    S = a.decode_steps
    cache = model.make_cache()
    steps = [model(mx.array(toks[:, : a.tokens - S]), cache=cache)[:, -1]]
    for i in range(a.tokens - S, a.tokens - 1):
        steps.append(model(mx.array(toks[:, i : i + 1]), cache=cache)[:, -1])
    inc = np.array(mx.concatenate(steps).astype(mx.float32))
    assert S + 1 <= M
    full = out[0, M - S - 1 : M - 1]
    d2 = np.abs(inc - full).max()
    print(f"[2] MLX cache decode vs full prompt: max|diff| = {d2:.2e}, "
          f"same top-1 = {(inc.argmax(-1) == full.argmax(-1)).mean():.0%}")


if __name__ == "__main__":
    main()
