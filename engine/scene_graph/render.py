"""Render a SceneGraph from a camera at time t: RGB, metric depth, alpha and optical flow."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from engine.metal_splat.camera import Camera
from engine.metal_splat.project import project_gaussians, project_points
from engine.metal_splat.rasterize import rasterize

from .scene import SceneGraph


@dataclass
class RenderOutput:
    rgb: mx.array          # (H,W,3) in [0,1] (background composited)
    depth: mx.array        # (H,W) metric z-depth, alpha-normalised (0 where empty)
    alpha: mx.array        # (H,W)
    flow: mx.array | None  # (H,W,2) backward flow: pixel offset to its location in the reference frame
    means2d: mx.array      # (N,2) projected centres (exposed for densification statistics)
    radii: mx.array        # (N,)
    num_rendered: int      # tile/Gaussian pairs composited

    def numpy(self) -> dict[str, np.ndarray]:
        d = {"rgb": np.array(self.rgb), "depth": np.array(self.depth), "alpha": np.array(self.alpha)}
        if self.flow is not None:
            d["flow"] = np.array(self.flow)
        return d


def render(
    scene: SceneGraph,
    cam: Camera,
    t: float,
    *,
    params: dict[str, dict] | None = None,
    flow_ref: tuple[Camera, float] | None = None,
    background: tuple[float, float, float] | mx.array | None = None,
    means2d_offset: mx.array | None = None,
    realtime: bool = False,
) -> RenderOutput:
    """realtime=True encodes the frame without a CPU<->GPU sync (binning capacity from the previous frame)."""
    world = scene.evaluate(t, cam.center, params=params)
    w2c = mx.array(cam.w2c.astype(np.float32))
    means2d, conics, depths, radii, valid = project_gaussians(
        world["means"], world["quats"], world["log_scales"], w2c,
        cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height,
    )
    if means2d_offset is not None:
        means2d = means2d + means2d_offset
    opac = world["opacities"] * valid.astype(mx.float32)
    feats = [world["rgb"], depths[:, None]]
    if flow_ref is not None:
        ref_cam, ref_t = flow_ref
        ref_pts = scene.positions(ref_t, params=params, visible_at=t)
        uv_ref, _ = project_points(ref_pts, mx.array(ref_cam.w2c.astype(np.float32)),
                                   ref_cam.fx, ref_cam.fy, ref_cam.cx, ref_cam.cy)
        feats.append(mx.stop_gradient(uv_ref - means2d))
    feats = mx.concatenate(feats, axis=1)
    out, alpha, binning = rasterize(means2d, conics, opac, feats, radii, depths, cam.width, cam.height,
                                     exact=not realtime)
    bg = scene.background_color if background is None else background
    bg = bg if isinstance(bg, mx.array) else mx.array(np.asarray(bg, dtype=np.float32))
    rgb = out[..., :3] + (1.0 - alpha)[..., None] * bg
    depth = mx.where(alpha > 1e-3, out[..., 3] / mx.maximum(alpha, 1e-3), 0.0)
    flow = None
    if flow_ref is not None:
        flow = mx.where((alpha > 1e-3)[..., None], out[..., 4:6] / mx.maximum(alpha, 1e-3)[..., None], 0.0)
    return RenderOutput(rgb, depth, alpha, flow, means2d, radii, binning.num_entries)
