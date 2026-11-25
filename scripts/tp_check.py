"""Tensor-parallel check: run with mlx.launch on N ranks.

Rank 0 compares sharded logits with an unsharded run (--ref file, made with N=1).
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--fp32", action="store_true", help="cast bf16 weights to float32")
    p.add_argument("--topk", type=int, default=None, help="lower DSA index_topk to test sparse attention")
    a = p.parse_args()

    group = mx.distributed.init()
    rank, n = group.rank(), group.size()
    import json
    presplit = "k31_presplit" in json.loads((Path(a.model) / "config.json").read_text())
    model, config = load_model(Path(a.model), lazy=True, strict=not presplit)
    if a.topk:
        for layer in model.layers:
            if hasattr(layer.self_attn, "indexer"):
                layer.self_attn.indexer.index_topk = a.topk
    if n > 1 or presplit:
        model.shard(group)
    if a.fp32:
        from mlx.utils import tree_map
        model.update(tree_map(lambda v: v.astype(mx.float32) if v.dtype == mx.bfloat16 else v, model.parameters()))
    mx.eval(model.parameters())

    vocab = config["text_config"]["vocab_size"]
    toks = mx.array(np.random.default_rng(1).integers(0, vocab, size=(1, a.tokens)))

    # Full prompt, then 4 decode steps with a cache.
    cache = model.make_cache()
    logits = [model(toks, cache=cache)[:, -1]]
    nxt = mx.argmax(logits[-1], -1)[:, None]
    for _ in range(4):
        logits.append(model(nxt, cache=cache)[:, -1])
        nxt = mx.argmax(logits[-1], -1)[:, None]
    out = mx.concatenate(logits).astype(mx.float32)
    mx.eval(out)

    if n == 1:
        np.save(a.ref, np.array(out))
        print(f"saved reference logits {out.shape} to {a.ref}")
        return
    if rank == 0:
        ref = np.load(a.ref)
        diff = np.abs(np.array(out) - ref).max()
        top = (np.array(out).argmax(-1) == ref.argmax(-1)).mean()
        print(f"ranks={n}: max|diff| vs 1 rank = {diff:.2e}, same top-1 = {top:.0%}")
        print("[ok] TP logits match" if diff < (1e-3 if a.fp32 else 2e-1) and top == 1.0 else "[FAIL] TP logits differ")


if __name__ == "__main__":
    main()
