"""Geometric-fidelity metrics: metric depth consistency (AbsRel) and temporal warping error."""

from __future__ import annotations

import cv2
import numpy as np


def abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None, align_scale: bool = False) -> float:
    m = (gt > 0.1) & np.isfinite(pred) & (pred > 0)
    if mask is not None:
        m &= mask
    if not m.any():
        return float("nan")
    p, g = pred[m], gt[m]
    if align_scale:   # for scale-ambiguous monocular estimates
        p = p * np.median(g) / np.median(p)
    return float(np.mean(np.abs(p - g) / g))


def warp_backward(prev_img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Sample prev_img at (x + flow) for every pixel of the current frame (flow from rendered 4DGS)."""
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(prev_img.astype(np.float32), xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def warping_error(curr: np.ndarray, prev: np.ndarray, flow: np.ndarray, valid: np.ndarray | None = None) -> float:
    """E_warp: mean abs difference between frame t and frame t-1 warped by the geometric flow."""
    warped = warp_backward(prev, flow)
    m = np.all(np.isfinite(warped), axis=-1)
    if valid is not None:
        m &= valid
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(curr[m] - warped[m])))


def depth_reprojection_error(depth_t: np.ndarray, depth_ref: np.ndarray, flow: np.ndarray) -> float:
    """Relative difference between depth at t and the reference depth warped along the flow (consistency
    of the metric geometry the refiner is conditioned on)."""
    warped = warp_backward(depth_ref[..., None], flow)[..., 0]
    m = np.isfinite(warped) & (warped > 0.1) & (depth_t > 0.1)
    return float(np.median(np.abs(depth_t[m] - warped[m]) / depth_t[m])) if m.any() else float("nan")
