"""Copy a Moonshot checkpoint with MXFP4 experts decoded to plain bf16 weights.

The official PyTorch code can then load it without compressed-tensors.
Decoding uses the independent numpy MXFP4 decoder (mxfp4_ref.py).
Meant for tiny models only.
"""

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mxfp4_ref  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=False)

    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy(f, dst / f.name)

    weights = {}
    for f in sorted(glob.glob(str(src / "*.safetensors"))):
        weights.update(mx.load(f))
    out, n = {}, 0
    for k, v in weights.items():
        if k.endswith(".weight_scale"):
            continue
        if k.endswith(".weight_packed"):
            base = k[: -len(".weight_packed")]
            w = mxfp4_ref.decode(np.array(v), np.array(weights[base + ".weight_scale"]))
            out[base + ".weight"] = mx.array(w).astype(mx.bfloat16)
            n += 1
            continue
        out[k] = v
    mx.save_safetensors(str(dst / "model.safetensors"), out, metadata={"format": "pt"})

    cfg = json.load(open(src / "config.json"))
    cfg.pop("quantization_config", None)
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"].pop("quantization_config", None)
    json.dump(cfg, open(dst / "config.json", "w"), indent=2)
    print(f"decoded {n} expert matrices -> {dst}")


if __name__ == "__main__":
    main()
