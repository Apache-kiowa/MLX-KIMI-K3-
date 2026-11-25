"""Show what a checkpoint holds, from the safetensors headers only (no weights read)."""

import argparse
import collections
import glob
import json
import re
import struct
from pathlib import Path


def headers(folder):
    for f in sorted(glob.glob(str(Path(folder) / "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        yield from h.items()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    a = p.parse_args()
    cfg = json.load(open(Path(a.folder) / "config.json"))
    tc = cfg.get("text_config", cfg)
    print("model_type:", cfg.get("model_type"), "/ text:", tc.get("model_type"))
    print("quantization_config:", cfg.get("quantization_config") or tc.get("quantization_config"))
    print("layers:", tc.get("num_hidden_layers"), "experts:", tc.get("num_experts"),
          "linear_attn_config:", tc.get("linear_attn_config"))
    idx = {k: v for k, v in tc.items() if "index" in k.lower() or "dsa" in k.lower()}
    print("indexer config keys:", idx)

    groups = collections.Counter()
    dtypes = collections.Counter()
    indexer, total = [], 0
    for name, info in headers(a.folder):
        total += 1
        shape_free = re.sub(r"\.\d+\.", ".N.", name)
        groups[shape_free] += 1
        dtypes[info["dtype"]] += 1
        if "index" in name:
            indexer.append((name, info["dtype"], info["shape"]))
    print(f"\n{total} tensors, dtypes: {dict(dtypes)}")
    print("\nname patterns (N = layer/expert number):")
    for k, c in sorted(groups.items()):
        print(f"  {c:6d}  {k}")
    print(f"\nindexer tensors: {len(indexer)}")
    for row in indexer[:12]:
        print("  ", *row)


if __name__ == "__main__":
    main()
