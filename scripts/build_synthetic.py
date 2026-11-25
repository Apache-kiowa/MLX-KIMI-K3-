"""Build a tiny random K3.1-style checkpoint in Moonshot's on-disk format.

Routed experts are stored per expert as MXFP4 (weight_packed + weight_scale),
other tensors as bf16, plus fake DSA indexer tensors on the MLA layers.
Used to test the convert/load/shard pipeline without network access.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent))
import mxfp4_ref  # noqa: E402
from mlx_lm.models import kimi_k3  # noqa: E402

TEXT_CONFIG = {
    "model_type": "kimi_k31",
    "vocab_size": 1000,
    "hidden_size": 128,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 256,
    "rms_norm_eps": 1e-5,
    "hidden_act": "situ",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "linear_attn_config": {
        "kda_layers": [1, 2, 4, 5],
        "full_attn_layers": [3, 6],
        "num_heads": 4,
        "head_dim": 32,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "use_full_rank_gate": True,
    },
    "num_experts": 16,
    "num_experts_per_token": 4,
    "moe_intermediate_size": 128,
    "routed_expert_hidden_size": 64,
    "latent_moe_use_norm": True,
    "num_shared_experts": 1,
    "first_k_dense_replace": 1,
    "q_lora_rank": 32,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 16,
    "v_head_dim": 16,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "attn_res_block_size": 2,
    "index_topk": 2048,
    "index_n_heads": 4,
    "index_head_dim": 32,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = {
        "model_type": "kimi_k31",
        "architectures": ["KimiK31ForConditionalGeneration"],
        "torch_dtype": "bfloat16",
        "text_config": TEXT_CONFIG,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
        },
    }
    model = kimi_k3.Model(kimi_k3.ModelArgs.from_dict(cfg))
    params = dict(tree_flatten(model.parameters()))

    tensors = {}
    pre = "language_model."
    for k, v in params.items():
        name = k[len(pre):] if k.startswith(pre) else k
        if ".switch_mlp." in name:
            # model.layers.L.mlp.switch_mlp.gate_proj.weight -> per expert MXFP4
            layer, _, rest = name.partition(".mlp.switch_mlp.")
            proj = rest.split(".")[0]
            src = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[proj]
            w = rng.normal(0, 0.05, size=v.shape).astype(np.float32)
            for e in range(v.shape[0]):
                packed, scale = mxfp4_ref.encode(w[e])
                base = f"{layer}.block_sparse_moe.experts.{e}.{src}"
                tensors[f"{base}.weight_packed"] = mx.array(packed)
                tensors[f"{base}.weight_scale"] = mx.array(scale)
            continue
        if ".mlp." in name and not name.endswith(("mlp.gate_proj.weight",
                                                   "mlp.up_proj.weight",
                                                   "mlp.down_proj.weight")):
            # MoE layer tensors live under block_sparse_moe in the source.
            name = name.replace(".mlp.", ".block_sparse_moe.", 1)
        if name.endswith("e_score_correction_bias"):
            tensors[name] = mx.zeros(v.shape, dtype=mx.float32)
            continue
        if name.endswith(("A_log", "dt_bias")):
            tensors[name] = (mx.random.normal(v.shape) * 0.1).astype(mx.float32)
            continue
        scale = 1.0 if name.endswith(("norm.weight", "layernorm.weight")) else 0.05
        base = mx.ones(v.shape) if scale == 1.0 else mx.random.normal(v.shape) * scale
        tensors[name] = base.astype(mx.bfloat16)

    # Fake DSA indexer tensors on MLA layers; the loader must drop them.
    for li in [i - 1 for i in TEXT_CONFIG["linear_attn_config"]["full_attn_layers"]]:
        ip = f"model.layers.{li}.self_attn.indexer"
        h, d, hid = (TEXT_CONFIG["index_n_heads"], TEXT_CONFIG["index_head_dim"],
                     TEXT_CONFIG["hidden_size"])
        tensors[f"{ip}.wq_b.weight"] = mx.zeros((h * d, TEXT_CONFIG["q_lora_rank"]),
                                                dtype=mx.bfloat16)
        tensors[f"{ip}.wk.weight"] = mx.zeros((d, hid), dtype=mx.bfloat16)
        tensors[f"{ip}.k_norm.weight"] = mx.ones((d,), dtype=mx.bfloat16)
        tensors[f"{ip}.weights_proj.weight"] = mx.zeros((h, hid), dtype=mx.bfloat16)

    mx.save_safetensors(str(out / "model.safetensors"), tensors, metadata={"format": "pt"})
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    n_exp = sum(1 for k in tensors if ".experts." in k)
    print(f"wrote {len(tensors)} tensors ({n_exp} expert tensors) to {out}")


if __name__ == "__main__":
    main()
