"""Dynamic-actor segmentation: split initial 3D points into background and per-actor sets using
tracked 3D boxes (from nuScenes/Waymo annotations, or boxes drawn in the MacBook app)."""

from __future__ import annotations

import numpy as np

from .tracks import Track


def points_in_box(points: np.ndarray, pose: np.ndarray, size, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mask, points in box-local frame) for an oriented box centred at pose."""
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    half = np.asarray(size, dtype=np.float64) / 2 + margin
    mask = np.all(np.abs(local) <= half, axis=1)
    return mask, local


def segment_points(points: np.ndarray, times: np.ndarray, actors: list[tuple[str, Track, list]],
                   margin: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Label each point with the index of the actor box containing it at its observation time (-1 =
    background) and return actor-local coordinates for labelled points (world coords otherwise)."""
    labels = np.full(len(points), -1, dtype=np.int64)
    local_pts = points.copy()
    uniq_t = np.unique(times)
    for t in uniq_t:
        sel = np.where(times == t)[0]
        for ai, (_, track, size) in enumerate(actors):
            if t < track.t_range[0] - 0.05 or t > track.t_range[1] + 0.05:
                continue
            free = sel[labels[sel] < 0]
            if len(free) == 0:
                break
            mask, local = points_in_box(points[free], track.pose(float(t)), size, margin)
            labels[free[mask]] = ai
            local_pts[free[mask]] = local[mask]
    return labels, local_pts
