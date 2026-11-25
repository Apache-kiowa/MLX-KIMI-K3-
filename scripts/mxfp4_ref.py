"""Independent MXFP4 (OCP MX v1.0) encoder/decoder in numpy.

Layout used by Moonshot checkpoints (compressed-tensors "mxfp4-pack-quantized"):
  weight_packed: uint8 [out, in/2], two E2M1 codes per byte, low nibble first
  weight_scale:  uint8 [out, in/32], E8M0 exponent, value = 2**(e - 127)
"""

import numpy as np

# E2M1 magnitudes for codes 0..7; bit 3 is the sign.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
GROUP = 32


def decode(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    lo = packed & 0x0F
    hi = packed >> 4
    codes = np.stack([lo, hi], axis=-1).reshape(*packed.shape[:-1], -1)
    mag = E2M1[codes & 0x7]
    val = np.where(codes & 0x8, -mag, mag)
    s = np.exp2(scale.astype(np.float32) - 127.0)
    s = np.repeat(s, GROUP, axis=-1)
    return (val * s).astype(np.float32)


def encode(w: np.ndarray):
    out, inp = w.shape
    assert inp % GROUP == 0
    g = w.reshape(out, inp // GROUP, GROUP).astype(np.float32)
    amax = np.abs(g).max(axis=-1, keepdims=True)
    # Shared exponent: floor(log2(amax)) - 2, so amax maps near 4..6 (OCP rule).
    e = np.floor(np.log2(np.maximum(amax, 2.0**-126))) - 2
    e = np.clip(e, -127, 127)
    scale_u8 = (e + 127).astype(np.uint8)
    x = g / np.exp2(e)
    # Round each value to the nearest E2M1 magnitude.
    idx = np.abs(np.abs(x)[..., None] - E2M1).argmin(axis=-1).astype(np.uint8)
    codes = idx | np.where(x < 0, 8, 0).astype(np.uint8)
    codes = codes.reshape(out, inp)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale_u8.reshape(out, inp // GROUP)
