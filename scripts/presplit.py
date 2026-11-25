"""Cut a converted Kimi K3.1 MLX model into per-rank tensor-parallel parts.

Each Mac then stores only its own part (~1/4 of the experts) instead of the
whole 1.56 TB model. The cut is exactly what model.shard() does at load time,
so the loaded weights are bit-identical to loading the full model and sharding.

    python presplit.py --model FULL_DIR --out OUT_DIR --size 4 --rank 3

writes OUT_DIR with config/tokenizer files plus tp4-r3-*.safetensors.
Run once per rank (one at a time saves disk), copy OUT_DIR to that Mac, and
start the server with K31_PRESPLIT_DIR=<that dir on every Mac>.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model


class FakeGroup:
    """Stands in for mx.distributed.Group: shard() only asks rank() and size()."""

    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def quant_map(src: Path):
    """Which modules are quantized, and how. A model built without weights
    can't infer this (load_model looks for '<path>.scales' in the weights), so
    it is written into the config as per-layer quantization entries."""
    model, _ = load_model(src, lazy=True, strict=True)
    q = {}
    for path, m in model.named_modules():
        if hasattr(m, "bits") and hasattr(m, "group_size") and "scales" in m:
            q[path] = {"group_size": m.group_size, "bits": m.bits, "mode": getattr(m, "mode", "affine")}
    return q


def copy_aux(src: Path, out: Path, size: int):
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
    cfg = json.loads((src / "config.json").read_text())
    cfg["k31_presplit"] = {"tp_size": size}
    qmap = quant_map(src)
    if qmap:
        cfg.setdefault("quantization", {}).update(qmap)
        if "quantization_config" in cfg and isinstance(cfg["quantization_config"], dict):
            cfg["quantization_config"].update(qmap)
    print(f"config: {len(qmap)} quantized modules recorded")
    (out / "config.json").write_text(json.dumps(cfg, indent=4))


def split_rank(src: Path, out: Path, rank: int, size: int, chunk_gb: float):
    t0 = time.time()
    model, _ = load_model(src, lazy=True, strict=True)
    model.shard(FakeGroup(rank, size))
    flat = dict(tree_flatten(model.parameters()))
    del model
    total = sum(v.nbytes for v in flat.values())
    print(f"rank {rank}/{size}: {len(flat)} tensors, {total / 1e9:.1f} GB", flush=True)

    for old in out.glob(f"tp{size}-r{rank}-*.safetensors"):
        old.unlink()
    limit = int(chunk_gb * 1e9)
    keys = list(flat)
    part, cur, cur_bytes, done, names = 0, {}, 0, 0, {}

    def flush():
        nonlocal part, cur, cur_bytes
        if not cur:
            return
        part += 1
        name = f"tp{size}-r{rank}-{part:05d}.safetensors"
        mx.eval(cur)
        mx.save_safetensors(str(out / name), cur, metadata={"format": "mlx"})
        for k in cur:
            names[k] = name
        cur, cur_bytes = {}, 0
        mx.clear_cache()

    for k in keys:
        v = flat.pop(k)
        if cur and cur_bytes + v.nbytes > limit:
            flush()
            el = time.time() - t0
            print(f"  {done / 1e9:7.1f} / {total / 1e9:.1f} GB  {el / 60:5.1f} min", flush=True)
        cur[k] = v
        cur_bytes += v.nbytes
        done += v.nbytes
    flush()

    index = {"metadata": {"tp_size": size, "rank": rank, "total_size": total}, "weight_map": names}
    (out / f"tp{size}-r{rank}.index.json").write_text(json.dumps(index, indent=1))
    print(f"[ok] rank {rank}: {part} files, {total / 1e9:.1f} GB in {(time.time() - t0) / 60:.1f} min -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="converted full MLX model")
    p.add_argument("--out", required=True, help="output folder for this rank")
    p.add_argument("--size", type=int, default=4, help="number of Macs (tensor-parallel size)")
    p.add_argument("--rank", type=int, nargs="+", required=True, help="rank(s) to write")
    p.add_argument("--chunk-gb", type=float, default=5.0)
    a = p.parse_args()
    src, out = Path(a.model).expanduser(), Path(a.out).expanduser()
    copy_aux(src, out, a.size)
    for r in a.rank:
        assert 0 <= r < a.size, f"rank {r} out of range for size {a.size}"
        split_rank(src, out, r, a.size, a.chunk_gb)


if __name__ == "__main__":
    main()
