"""Initial point clouds for 4DGS optimisation.

Priority: dataset points (LiDAR / ARKit feature points / synthetic) -> per-frame metric depth maps
(LiDAR sceneDepth) -> Depth Anything V2 monocular depth scale-aligned to ARKit VIO points -> random.
"""

from __future__ import annotations

import numpy as np

from capture.dataset import Dataset, Frame


def backproject(frame: Frame, depth: np.ndarray, rgb: np.ndarray, n: int, rng, max_depth: float = 80.0) -> np.ndarray:
    cam = frame.camera.scaled(depth.shape[1], depth.shape[0])
    v, u = np.nonzero((depth > 0.2) & (depth < max_depth))
    if len(v) == 0:
        return np.zeros((0, 7))
    k = rng.choice(len(v), size=min(n, len(v)), replace=False)
    v, u = v[k], u[k]
    z = depth[v, u]
    xc = np.stack([(u + 0.5 - cam.cx) / cam.fx * z, (v + 0.5 - cam.cy) / cam.fy * z, z], 1)
    xw = xc @ cam.c2w[:3, :3].T + cam.c2w[:3, 3]
    col = rgb[v, u] if rgb.shape[:2] == depth.shape else rgb[(v * rgb.shape[0] // depth.shape[0]), (u * rgb.shape[1] // depth.shape[1])]
    return np.concatenate([xw, col, np.full((len(z), 1), frame.t)], 1)


def scene_extent(ds: Dataset) -> tuple[np.ndarray, float]:
    centers = np.stack([f.camera.center for f in ds.frames])
    c = centers.mean(0)
    return c, float(max(np.linalg.norm(centers - c, axis=1).max() * 1.1, 1.0))


def initial_points(ds: Dataset, max_points: int = 300_000, mono_depth: bool = False, seed: int = 0,
                   image_size: tuple[int, int] | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if ds.points is not None and len(ds.points) >= 2000:
        pts = ds.points
        if np.isnan(pts[:, 3:6]).any():
            pts = pts.copy()
            pts[np.isnan(pts[:, 3:6]).any(1), 3:6] = 0.5
        if len(pts) > max_points:
            pts = pts[rng.choice(len(pts), max_points, replace=False)]
        print(f"[init] using {len(pts)} dataset points ({ds.source})")
        return pts
    per_frame = max(max_points // max(len(ds.frames), 1), 200)
    with_depth = [f for f in ds.frames if f.depth_path is not None]
    if with_depth:
        pts = [backproject(f, f.depth(), f.image(image_size), per_frame, rng) for f in with_depth]
        print(f"[init] back-projected metric depth from {len(with_depth)} frames")
        return np.concatenate(pts)
    if mono_depth:
        from .mono_depth import MonoDepth

        est = MonoDepth()
        sparse = ds.points
        pts = []
        for f in ds.frames[:: max(1, len(ds.frames) // 60)]:
            img = f.image(image_size)
            d = est.metric_depth(img, f.camera.scaled(img.shape[1], img.shape[0]), sparse)
            if d is not None:
                pts.append(backproject(f, d, img, per_frame * 2, rng))
        if pts:
            print(f"[init] Depth Anything V2 init from {len(pts)} frames")
            return np.concatenate(pts)
    c, r = scene_extent(ds)
    n = min(max_points, 100_000)
    xyz = c + rng.uniform(-1, 1, (n, 3)) * r * np.array([1.5, 1.5, 0.5])
    print("[init] WARNING: no geometry available, using random initialisation")
    return np.concatenate([xyz, np.full((n, 3), 0.5), np.full((n, 1), np.nan)], 1)
