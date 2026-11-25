"""Run the official Kimi K3/K3.1 PyTorch code without Triton (CPU, macOS).

Import this module BEFORE loading the model. It replaces the Triton kernels
of fla-core with plain PyTorch:
  ShortConvolution   -> causal depthwise conv1d + activation
  FusedRMSNormGated  -> RMSNorm(x) * weight * act(g)
  chunk_kda / fused_recurrent_kda -> fla.ops.kda.naive.naive_recurrent_kda
Slow, but exact enough for a reference on tiny models.
"""

import torch
import torch.nn.functional as F

import fla.modules as fm
import fla.ops.kda as fk
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda.naive import naive_recurrent_kda

_ACT = {
    None: lambda x: x,
    "silu": F.silu,
    "swish": F.silu,
    "sigmoid": torch.sigmoid,
}


def _conv_forward(self, x, residual=None, mask=None, cache=None,
                  output_final_state=False, cu_seqlens=None, **kwargs):
    if cu_seqlens is not None or mask is not None:
        raise NotImplementedError("fla_torch_shim: packed/masked input not supported")
    k = self.kernel_size[0]
    xt = x.transpose(1, 2)  # [B, C, T]
    if cache is not None:
        xt = torch.cat([cache.to(xt.dtype), xt], dim=-1)
    else:
        xt = F.pad(xt, (k - 1, 0))
    y = F.conv1d(xt, self.weight.to(xt.dtype), self.bias, groups=self.weight.shape[0])
    y = _ACT[self.activation](y).transpose(1, 2)
    new_cache = xt[..., -(k - 1):] if output_final_state else None
    if residual is not None:
        y = y + residual
    return y, new_cache


def _norm_forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
    if residual is not None:
        x = x + residual
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    if self.weight is not None:
        y = y * self.weight.float()
    y = y * _ACT[self.activation](g.float())
    y = y.to(x.dtype)
    return (y, x) if prenorm else y


def _kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
         use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
         use_beta_sigmoid_in_kernel=False, transpose_state_layout=False,
         cu_seqlens=None, **kwargs):
    if cu_seqlens is not None:
        raise NotImplementedError("fla_torch_shim: cu_seqlens not supported")
    if use_gate_in_kernel:
        raise NotImplementedError("fla_torch_shim: gate inside kernel not supported")
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        k = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta.float())
    if initial_state is not None and transpose_state_layout:
        initial_state = initial_state.transpose(-1, -2)
    o, s = naive_recurrent_kda(q, k, v, g, beta, scale=scale,
                               initial_state=initial_state,
                               output_final_state=output_final_state)
    if s is not None and transpose_state_layout:
        s = s.transpose(-1, -2)
    return o, s


ShortConvolution.forward = _conv_forward
FusedRMSNormGated.forward = _norm_forward
fk.chunk_kda = _kda
fk.fused_recurrent_kda = _kda
fm.ShortConvolution = ShortConvolution
fm.FusedRMSNormGated = FusedRMSNormGated
