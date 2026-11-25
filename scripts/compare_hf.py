"""Compare MLX logits with the official PyTorch code (trust_remote_code).

Needs: pip install torch transformers compressed-tensors fla-core einops tiktoken
First decode experts: dequant_src.py --src <hf> --dst <hf_deq>, then pass --src <hf_deq>.
On CPU the Triton kernels of fla-core are replaced by plain PyTorch
(fla_torch_shim.py next to this file).
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

import fla_torch_shim  # noqa: F401  (must come before the model is loaded)
from mlx_lm.utils import load_model
from transformers import AutoModelForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="original HF folder")
    p.add_argument("--mlx", required=True, help="converted MLX folder")
    p.add_argument("--tokens", type=int, default=32)
    a = p.parse_args()

    hf, info = AutoModelForCausalLM.from_pretrained(
        a.src, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True,
    )
    hf.eval()
    for k in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        keys = sorted(info.get(k) or [])
        print(f"{k}: {len(keys)}", *keys[:6], "..." if len(keys) > 6 else "", sep="\n  ")
    # Decompressed MXFP4 experts come back as bf16: cast everything to float32.
    for t in list(hf.parameters()) + list(hf.buffers()):
        if t.dtype == torch.bfloat16:
            t.data = t.data.float()
    vocab = hf.config.get_text_config().vocab_size
    toks = np.random.default_rng(1).integers(0, vocab, size=(1, a.tokens))
    with torch.no_grad():
        ref = hf(torch.tensor(toks)).logits.float().numpy()

    model, _ = load_model(Path(a.mlx))
    from mlx.utils import tree_map

    model.update(tree_map(lambda v: v.astype(mx.float32) if v.dtype == mx.bfloat16 else v, model.parameters()))
    out = np.array(model(mx.array(toks)).astype(mx.float32))

    diff = np.abs(out - ref).max()
    top = (out.argmax(-1) == ref.argmax(-1)).mean()
    print(f"max|diff| = {diff:.2e}, max|ref| = {np.abs(ref).max():.2e}, same top-1 = {top:.0%}")


if __name__ == "__main__":
    main()
