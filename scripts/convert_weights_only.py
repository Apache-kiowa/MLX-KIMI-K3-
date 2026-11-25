"""Same steps as mlx_lm.convert without -q, but without a tokenizer.

The real tiny-random/kimi-k3.1-pro has a tokenizer: use mlx_lm.convert there.
"""

import argparse
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map_with_path
from mlx_lm.utils import load_model, save_config, save_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    a = p.parse_args()
    model, config = load_model(Path(a.src), lazy=True)
    cast = model.cast_predicate

    def to_bf16(k, v):
        if cast(k) and mx.issubdtype(v.dtype, mx.floating):
            return v.astype(mx.bfloat16)
        return v

    model.update(tree_map_with_path(to_bf16, model.parameters()))
    dst = Path(a.dst)
    save_model(dst, model)
    save_config(config, config_path=dst / "config.json")
    print("saved", dst, "quantization:", config.get("quantization"))


if __name__ == "__main__":
    main()
