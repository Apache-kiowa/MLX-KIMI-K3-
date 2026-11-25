import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from mlx_lm.utils import load_model

rng = np.random.default_rng(0)
L, N, K = 512, 2048, 64
idx = np.argsort(rng.random((1, 1, L, N)), axis=-1)[..., :K].astype(np.int32)
want = np.zeros((1, 1, L, N), bool)
np.put_along_axis(want, idx, True, axis=-1)
for dev in (mx.gpu, mx.cpu):
    with mx.stream(dev):
        for dt in (mx.bool_, mx.uint32):
            z = mx.zeros((1, 1, L, N), dtype=dt)
            got = mx.put_along_axis(z, mx.array(idx), mx.array(1, dt), axis=-1)
            got = np.array(got.astype(mx.bool_))
            print(f"[1] put_along_axis {dev} {dt}: wrong cells = {int((got != want).sum())}")
        sc = mx.array(rng.normal(size=(1, 1, L, N)).astype(np.float32))
        p = np.array(mx.argpartition(sc, kth=-K, axis=-1)[..., -K:])
        ref = np.argsort(-np.array(sc), axis=-1)[..., :K]
        bad = sum(set(a) != set(b) for a, b in zip(p.reshape(L, K), ref.reshape(L, K)))
        print(f"[2] argpartition {dev}: wrong rows = {bad}")

if len(sys.argv) > 1:
    model, cfg = load_model(Path(sys.argv[1]))
    model.update(tree_map(lambda v: v.astype(mx.float32) if v.dtype == mx.bfloat16 else v,
                          model.parameters()))
    for layer in model.layers:
        if hasattr(layer.self_attn, "indexer"):
            layer.self_attn.indexer.index_topk = 8
    toks = mx.array(np.random.default_rng(1).integers(0, cfg["text_config"]["vocab_size"], (1, 64)))
    outs = {}
    for name, dev in (("gpu1", mx.gpu), ("gpu2", mx.gpu), ("cpu", mx.cpu)):
        with mx.stream(dev):
            outs[name] = np.array(model(toks).astype(mx.float32))
    for a, b in (("gpu1", "gpu2"), ("gpu1", "cpu")):
        d = np.abs(outs[a] - outs[b]).max()
        t = (outs[a].argmax(-1) == outs[b].argmax(-1)).mean()
        print(f"[3] model {a} vs {b}: max|diff| = {d:.2e}, same top-1 = {t:.0%}")
