"""PyTorch reference of the Kimi K3.1 DSA indexer on top of the official code.

The official tiny-model code (modeling_kimi_linear.py) runs dense MLA and
ignores the indexer weights. install(model, src_dir) adds the indexer to the
DSA layers following vLLM (vllm/models/kimi_k3/nvidia/dsa.py): full score
matrix, causal mask, top-k, then attention only over the selected keys.
"""

import glob
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

FP8_MAX = 448.0


def fp8_round_trip(x, amax_floor):
    x = x.float()
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(amax_floor)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    y = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return y * scale


def topk_mask(ix, x, q_c, use_fp8):
    """Bool [B, 1, T, T]: key s is kept for query t; None if all are kept."""
    B, T, _ = x.shape
    if T <= ix["topk"]:
        return None
    H, D = ix["heads"], ix["dim"]
    q = (q_c @ ix["wq_b"].T.to(q_c.dtype)).view(B, T, H, D)
    k = x @ ix["wk"].T.to(x.dtype)
    k = F.layer_norm(k.float(), (D,), ix["k_w"].float(), ix["k_b"].float(), ix["eps"]).to(x.dtype)
    if use_fp8:
        q, k = fp8_round_trip(q, 1e-10), fp8_round_trip(k, 1e-4)
    else:
        q, k = q.float(), k.float()
    w = (x @ ix["wp"].T.to(x.dtype)).float() * (D**-0.5 * H**-0.5)  # [B, T, H]
    scores = 0
    for h in range(H):  # heads one at a time: no [B, T, H, S] tensor
        scores = scores + (q[:, :, h] @ k.transpose(-1, -2)).clamp_min(0) * w[:, :, h, None]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    idx = scores.topk(ix["topk"], dim=-1).indices
    keep = torch.zeros(B, T, T, dtype=torch.bool).scatter_(-1, idx, True) & causal
    return keep[:, None]


def install(model, src_dir, use_fp8=False, topk=None):
    src_dir = Path(src_dir)
    cfg = json.load(open(src_dir / "config.json"))
    tc = cfg.get("text_config", cfg)
    dsa = tc["dsa_config"]
    weights = {}
    for f in glob.glob(str(src_dir / "*.safetensors")):
        weights.update(load_file(f))
    lm = model.language_model if hasattr(model, "language_model") else model
    layers = lm.model.layers
    n = 0
    for li, layer in enumerate(layers):
        if (li + 1) not in dsa["dsa_layers"]:
            continue
        attn = layer.self_attn
        p = f"language_model.model.layers.{li}.self_attn.indexer."
        if p + "wq_b.weight" not in weights:
            p = p[len("language_model."):]
        attn._dsa = {
            "wq_b": weights[p + "wq_b.weight"].float(),
            "wk": weights[p + "wk.weight"].float(),
            "wp": weights[p + "weights_proj.weight"].float(),
            "k_w": weights[p + "k_norm.weight"].float(),
            "k_b": weights[p + "k_norm.bias"].float(),
            "heads": dsa["index_num_heads"],
            "dim": dsa["index_head_dim"],
            "topk": topk or dsa["index_topk"],
            "eps": tc["rms_norm_eps"],
        }
        attn._dsa_fp8 = use_fp8
        attn.forward = types.MethodType(_dsa_forward, attn)
        n += 1
    return n


def _dsa_forward(self, hidden_states, attention_mask=None, position_ids=None,
                 past_key_values=None, **kwargs):
    # Official KimiMLAAttention.forward (no cache), plus the DSA key mask.
    assert past_key_values is None or past_key_values.get_seq_length() == 0
    B, T = hidden_states.shape[:-1]
    q_c = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_c).view(B, T, -1, self.q_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    ckv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(ckv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(
        B, T, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
    k_pass, v = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(B, 1, T, self.qk_rope_head_dim).expand(*k_pass.shape[:-1], -1)
    qs = torch.cat((q_pass, q_rot), dim=-1)
    ks = torch.cat((k_pass, k_rot), dim=-1)

    keep = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
    sel = topk_mask(self._dsa, hidden_states, q_c, self._dsa_fp8)
    if sel is not None:
        keep = keep & sel
    scores = torch.einsum("bhqd,bhkd->bhqk", qs, ks) * self.scaling
    scores = scores.masked_fill(~keep, float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(qs.dtype)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, v).transpose(1, 2).reshape(B, T, -1)
    if self.use_output_gate:
        out = out * self.g_proj(hidden_states).sigmoid()
    return self.o_proj(out)
