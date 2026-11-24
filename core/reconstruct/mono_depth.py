"""Feed-forward monocular depth (Depth Anything V2) aligned to metric ARKit VIO feature points.

Optional dependency: `pip install transformers torch` (model: depth-anything/Depth-Anything-V2-Small-hf).
Depth Anything V2 predicts affine-invariant *inverse* depth; we fit scale/shift per frame against the
sparse metric points that ARKit's VIO triangulated (capture `points.ply`), which makes it metric.
"""

from __future__ import annotations

import numpy as np

from engine.metal_splat.camera import Camera


class MonoDepth:
    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        try:
            from transformers import pipeline
        except ImportError as e:  # pragma: no cover - optional
            raise RuntimeError("Depth Anything V2 init needs `pip install transformers torch`") from e
        self.pipe = pipeline("depth-estimation", model=model)

    def relative_inverse_depth(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        out = self.pipe(Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)))
        pred = np.asarray(out["predicted_depth"], dtype=np.float32).squeeze()
        import cv2

        return cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    def metric_depth(self, rgb: np.ndarray, cam: Camera, sparse_points: np.ndarray | None,
                     min_matches: int = 20) -> np.ndarray | None:
        if sparse_points is None or len(sparse_points) == 0:
            return None
        inv_rel = self.relative_inverse_depth(rgb)
        uv, z = cam.project(sparse_points[:, :3])
        H, W = inv_rel.shape
        ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if ok.sum() < min_matches:
            return None
        u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        x, y = inv_rel[v, u], 1.0 / z[ok]
        # robust affine fit y ≈ s x + b (iteratively reweighted least squares)
        w = np.ones_like(x)
        for _ in range(5):
            A = np.stack([x, np.ones_like(x)], 1) * w[:, None]
            s, b = np.linalg.lstsq(A, y * w, rcond=None)[0]
            r = np.abs(s * x + b - y)
            w = 1.0 / np.maximum(r / (np.median(r) + 1e-9), 1.0)
        inv_metric = s * inv_rel + b
        depth = np.where(inv_metric > 1e-3, 1.0 / np.maximum(inv_metric, 1e-3), 0.0)
        return depth.astype(np.float32)
