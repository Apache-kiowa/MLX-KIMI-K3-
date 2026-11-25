"""Check an MLX K3.1 conversion against its source checkpoint.

1. Expert bytes: MLX weight/scales equal the source weight_packed/weight_scale.
2. Decode: mx.dequantize(mxfp4) equals an independent numpy MXFP4 decoder.
3. DSA indexer tensors are all kept.
4. Logits: the converted model equals a float32 reference model with experts
   decoded by numpy (no MLX quantized kernels in the reference).
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

sys.path.insert(0, str(Path(__file__).parent))
import mxfp4_ref  # noqa: E402
from mlx_lm.utils import _get_classes, load_model  # noqa: E402

PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def load_src(path):
    w = {}
    for f in sorted(glob.glob(str(Path(path) / "*.safetensors"))):
        w.update(mx.load(f))
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--mlx", required=True)
    p.add_argument("--tokens", type=int, default=32)
    a = p.parse_args()

    src = load_src(a.src)
    model, config = load_model(Path(a.mlx))
    params = dict(tree_flatten(model.parameters()))

    pat = re.compile(r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight_packed$")
    experts = {}
    for k in src:
        m = pat.search(k)
        if m:
            pre = k[: m.start()]
            experts.setdefault((int(m[1]), m[3]), []).append((int(m[2]), pre))
    print(f"source: {len(src)} tensors, {len(experts)} expert matrices groups")

    byte_ok = decode_ok = 0
    ref_w = {}
    for (layer, w), ids in sorted(experts.items()):
        base = f"language_model.model.layers.{layer}.mlp.switch_mlp.{PROJ[w]}"
        qw, qs = params[f"{base}.weight"], params[f"{base}.scales"]
        src_key = lambda e, pre, t: f"{pre}model.layers.{layer}.block_sparse_moe.experts.{e}.{w}.{t}"
        packed = np.stack([np.array(src[src_key(e, pre, "weight_packed")]) for e, pre in sorted(ids)])
        scale = np.stack([np.array(src[src_key(e, pre, "weight_scale")]) for e, pre in sorted(ids)])
        assert np.array_equal(np.array(qw.view(mx.uint8)), packed), f"bytes differ: {base}.weight"
        assert np.array_equal(np.array(qs), scale), f"bytes differ: {base}.scales"
        byte_ok += 1
        ref = mxfp4_ref.decode(packed, scale)
        mlx_deq = np.array(mx.dequantize(qw, qs, group_size=32, bits=4, mode="mxfp4").astype(mx.float32))
        assert np.array_equal(ref, mlx_deq), f"decode differs: {base}"
        decode_ok += 1
        ref_w[f"{base}.weight"] = mx.array(ref)
    print(f"[ok] expert bytes identical: {byte_ok} matrices")
    print(f"[ok] mx.dequantize == numpy MXFP4 decode: {decode_ok} matrices")

    n_src = sum(".indexer." in k for k in src)
    n_mlx = sum(".indexer." in k for k in params)
    assert n_src == n_mlx, f"indexer tensors: {n_src} in source, {n_mlx} in model"
    print(f"[ok] DSA indexer tensors kept: {n_src} in source, {n_mlx} in model")

    # Reference model: same class, no quantization, float32, numpy-decoded experts.
    cfg = json.load(open(Path(a.mlx) / "config.json"))
    cfg.pop("quantization", None)
    cfg.pop("quantization_config", None)
    model_cls, args_cls = _get_classes(cfg)
    ref_model = model_cls(args_cls.from_dict(cfg))
    weights = {k: v for k, v in params.items() if not k.endswith(".scales") and k not in ref_w}
    weights.update(ref_w)
    weights = {k: (v.astype(mx.float32) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in weights.items()}
    ref_model.load_weights(list(weights.items()), strict=True)
    model.update(tree_map(lambda v: v.astype(mx.float32) if mx.issubdtype(v.dtype, mx.floating) else v, model.parameters()))

    vocab = cfg["text_config"]["vocab_size"]
    toks = mx.array(np.random.default_rng(1).integers(0, vocab, size=(1, a.tokens)))
    out = model(toks, cache=None).astype(mx.float32)
    ref = ref_model(toks, cache=None).astype(mx.float32)
    diff = mx.abs(out - ref).max().item()
    scale = mx.abs(ref).max().item()
    same_top = (mx.argmax(out, -1) == mx.argmax(ref, -1)).astype(mx.float32).mean().item()
    print(f"logits: max|diff| = {diff:.2e} (max|ref| = {scale:.2e}), same top-1 = {same_top:.0%}")
    assert diff <= 1e-3 * max(scale, 1.0), "logits differ"
    print("[ok] logits match the float32 reference")


if __name__ == "__main__":
    main()
