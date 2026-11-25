"""Check a converted FULL model without loading it into memory.

1. Builds the model lazily and loads the converted weights with strict=True:
   every tensor name and shape must match the K3.1 code (nothing missing, nothing extra).
2. DSA indexer: same number of tensors in the source and the converted model.
3. Expert bytes: for a few layers and experts, MLX weight/scales == source
   weight_packed/weight_scale (reads only those tensors).

Usage: python verify_full.py --src ~/models/Kimi-K3.1-Pro --mlx ~/models/Kimi-K3.1-Pro-mlx
"""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm.utils import load_model

PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def weight_map(folder: Path):
    idx = folder / "model.safetensors.index.json"
    if idx.exists():
        return json.load(open(idx))["weight_map"]
    out = {}
    for f in sorted(folder.glob("*.safetensors")):
        for k in mx.load(str(f)):
            out[k] = f.name
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--mlx", required=True)
    p.add_argument("--samples", type=int, default=3, help="layers and experts to sample")
    a = p.parse_args()
    src, dst = Path(a.src).expanduser(), Path(a.mlx).expanduser()

    model, cfg = load_model(dst, lazy=True, strict=True)
    params = dict(tree_flatten(model.parameters()))
    print(f"[ok] {len(params)} MLX tensors: names and shapes match the model code")

    smap = weight_map(src)
    n_src = sum(".indexer." in k for k in smap)
    n_mlx = sum(".indexer." in k for k in params)
    dsa = cfg["text_config"].get("dsa_config") or {}
    want = 5 * len(dsa.get("dsa_layers", []))
    assert n_src == n_mlx == want, (n_src, n_mlx, want)
    print(f"[ok] DSA indexer tensors: {n_src} in source, {n_mlx} in MLX ({len(dsa['dsa_layers'])} layers x 5)")

    pat = re.compile(r"^(.*)model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.weight_packed$")
    layers, experts, pre = set(), set(), ""
    for k in smap:
        m = pat.match(k)
        if m:
            pre = m[1]
            layers.add(int(m[2]))
            experts.add(int(m[3]))
    layers, experts = sorted(layers), sorted(experts)
    pick = lambda xs: sorted({xs[0], xs[len(xs) // 2], xs[-1]} | set(xs[:: max(1, len(xs) // a.samples)][: a.samples]))
    lsel, esel = pick(layers), pick(experts)
    print(f"MoE layers: {len(layers)}, experts per layer: {len(experts)}; checking layers {lsel}, experts {esel}")

    cache = {}

    def src_tensor(key):
        f = smap[key]
        if f not in cache:
            cache.clear()
            cache[f] = mx.load(str(src / f))
        return cache[f][key]

    n = 0
    for layer in lsel:
        for w, proj in PROJ.items():
            base = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}"
            qw, qs = params[f"{base}.weight"], params[f"{base}.scales"]
            for e in esel:
                k = f"{pre}model.layers.{layer}.block_sparse_moe.experts.{e}.{w}"
                packed = np.array(src_tensor(k + ".weight_packed"))
                scale = np.array(src_tensor(k + ".weight_scale"))
                assert np.array_equal(np.array(qw[e].view(mx.uint8)), packed), f"bytes differ: {k}"
                assert np.array_equal(np.array(qs[e]), scale), f"scales differ: {k}"
                n += 1
    print(f"[ok] expert bytes identical: {n} sampled matrices")


if __name__ == "__main__":
    main()
