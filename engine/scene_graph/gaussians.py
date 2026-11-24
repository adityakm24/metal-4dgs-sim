"""4D Gaussian parameter sets (μ, Σ via scale+rotation, α, SH, velocity v, temporal centre/scale)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from engine.metal_splat.project import rgb_to_sh0

PARAM_KEYS = ("means", "log_scales", "quats", "opacity_logits", "sh", "velocity", "t_center", "t_log_scale")
STATIC_T_LOG_SCALE = 10.0  # σ_t = e^10 s: effectively time-invariant


def num_sh(degree: int) -> int:
    return (degree + 1) ** 2


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_gaussians(
    means: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray | float,
    sh_degree: int = 1,
    opacity: float | np.ndarray = 0.1,
    quats: np.ndarray | None = None,
    t_center: float | np.ndarray = 0.0,
    t_scale: float | np.ndarray | None = None,
    velocity: np.ndarray | None = None,
) -> dict[str, mx.array]:
    n = len(means)
    scales = np.broadcast_to(np.asarray(scales, dtype=np.float64), (n, 3)) if np.ndim(scales) < 2 else scales
    sh = np.zeros((n, num_sh(sh_degree), 3), dtype=np.float32)
    sh[:, 0] = rgb_to_sh0(np.asarray(colors, dtype=np.float32))
    if quats is None:
        quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    t_log = np.full((n, 1), STATIC_T_LOG_SCALE) if t_scale is None else np.log(np.broadcast_to(t_scale, (n,))).reshape(n, 1)
    g = {
        "means": means,
        "log_scales": np.log(np.maximum(scales, 1e-7)),
        "quats": quats,
        "opacity_logits": logit(np.broadcast_to(opacity, (n,))).reshape(n, 1),
        "sh": sh,
        "velocity": np.zeros((n, 3)) if velocity is None else velocity,
        "t_center": np.broadcast_to(np.asarray(t_center, dtype=np.float64), (n,)).reshape(n, 1),
        "t_log_scale": t_log,
    }
    return {k: mx.array(np.asarray(v, dtype=np.float32)) for k, v in g.items()}


def count(g: dict) -> int:
    return int(g["means"].shape[0])


def concat(*gs: dict) -> dict:
    gs = [g for g in gs if g is not None and count(g) > 0]
    return {k: mx.concatenate([g[k] for g in gs], axis=0) for k in PARAM_KEYS}


def take(g: dict, idx) -> dict:
    idx = mx.array(np.asarray(idx)) if not isinstance(idx, mx.array) else idx
    return {k: v[idx] for k, v in g.items()}


def pad_sh(g: dict, degree: int) -> dict:
    k = num_sh(degree)
    sh = g["sh"]
    if sh.shape[1] == k:
        return g
    if sh.shape[1] > k:
        return {**g, "sh": sh[:, :k]}
    pad = mx.zeros((sh.shape[0], k - sh.shape[1], 3), dtype=sh.dtype)
    return {**g, "sh": mx.concatenate([sh, pad], axis=1)}


def estimate_scales_knn(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Isotropic initial scale = mean distance to the k nearest neighbours."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= k:
        return np.full(len(pts), 0.05)
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    return np.clip(d[:, 1:].mean(1), 1e-4, None)
