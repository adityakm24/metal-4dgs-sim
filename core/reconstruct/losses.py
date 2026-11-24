"""Photometric / geometric losses in MLX (images are (H,W,C) float32)."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np


@lru_cache(maxsize=8)
def _gauss_window(size: int = 11, sigma: float = 1.5, channels: int = 3) -> mx.array:
    x = np.arange(size) - size // 2
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    w = np.outer(g, g).astype(np.float32)
    # depthwise conv weight: (C_out, kh, kw, C_in/groups=1)
    return mx.array(np.tile(w[None, :, :, None], (channels, 1, 1, 1)))


def _blur(x: mx.array, w: mx.array) -> mx.array:
    c = x.shape[-1]
    return mx.conv2d(x[None], w, padding=w.shape[1] // 2, groups=c)[0]


def ssim(a: mx.array, b: mx.array, size: int = 11) -> mx.array:
    w = _gauss_window(size, 1.5, a.shape[-1])
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = _blur(a, w), _blur(b, w)
    s_aa = _blur(a * a, w) - mu_a * mu_a
    s_bb = _blur(b * b, w) - mu_b * mu_b
    s_ab = _blur(a * b, w) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (s_aa + s_bb + c2))
    return mx.mean(m)


def l1(a: mx.array, b: mx.array, mask: mx.array | None = None) -> mx.array:
    d = mx.abs(a - b)
    if mask is None:
        return mx.mean(d)
    if d.ndim == 3 and mask.ndim == 2:
        mask = mask[..., None]
    return mx.sum(d * mask) / (mx.sum(mask) * (d.shape[-1] if d.ndim == 3 else 1) + 1e-6)


def photometric(pred: mx.array, gt: mx.array, lam: float = 0.2) -> mx.array:
    return (1 - lam) * l1(pred, gt) + lam * (1 - ssim(pred, gt))


def inverse_depth_l1(pred_depth: mx.array, gt_depth: mx.array, max_depth: float = 80.0) -> mx.array:
    mask = ((gt_depth > 0.1) & (gt_depth < max_depth)).astype(mx.float32)
    inv_p = 1.0 / mx.maximum(pred_depth, 0.1)
    inv_g = 1.0 / mx.maximum(gt_depth, 0.1)
    return l1(inv_p, inv_g, mask)


def psnr(a, b) -> float:
    mse = float(mx.mean(mx.square(mx.clip(a, 0, 1) - b)).item())
    return 10 * np.log10(1.0 / max(mse, 1e-10))
