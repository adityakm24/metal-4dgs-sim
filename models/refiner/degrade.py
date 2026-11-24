"""Paired (degraded splat render, ground truth) data for refiner training.

* `synthetic` mode — a reconstructed scene and the ground-truth world it was reconstructed from are both
  available, so we render *both* from strongly extrapolated viewpoints (lateral shifts up to ±2.5 m, yaw and
  height jitter): perfectly aligned pairs that contain exactly the artefacts seen at sim time.
* `holdout` mode — real captures: render the reconstruction at every logged camera (train + held-out) and
  pair it with the real photo (Difix3D-style); optionally from an early, under-fitted checkpoint for harder
  degradations.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from engine.metal_splat.camera import Camera
from engine.scene_graph import SceneGraph, render
from engine.scene_graph.tracks import pose_matrix, yaw_quat


def shifted_ego_camera(scene: SceneGraph, t: float, lateral: float = 0.0, yaw: float = 0.0, dz: float = 0.0,
                       forward: float = 0.0) -> Camera:
    ego = scene.ego.track.pose(t)
    delta = pose_matrix([forward, lateral, dz], yaw_quat(yaw))
    return scene.ego.camera_at(ego @ delta)


def _buffers(out):
    flow = out.flow if out.flow is not None else mx.zeros(out.depth.shape + (2,))
    return np.concatenate([np.array(out.rgb), np.array(out.depth)[..., None], np.array(out.alpha)[..., None],
                           np.array(flow)], axis=-1).astype(np.float16)


def synthetic_pairs(recon: SceneGraph, gt: SceneGraph, n: int, seed: int = 0, max_lateral: float = 2.5,
                    max_yaw: float = 0.12, max_dz: float = 0.3, dt_flow: float = 0.1):
    """Returns (cond_buffers (n,H,W,7: rgb,depth,alpha,flow), targets (n,H,W,3)) as float16 numpy."""
    rng = np.random.default_rng(seed)
    t0, t1 = recon.time_range
    conds, tgts = [], []
    for _ in range(n):
        t = float(rng.uniform(t0 + dt_flow, t1))
        lat = float(rng.uniform(-max_lateral, max_lateral)) if rng.random() > 0.15 else 0.0
        yaw = float(rng.uniform(-max_yaw, max_yaw))
        dz = float(rng.uniform(-max_dz, max_dz))
        cam = shifted_ego_camera(recon, t, lat, yaw, dz)
        prev = shifted_ego_camera(recon, t - dt_flow, lat, yaw, dz)
        out = render(recon, cam, t, flow_ref=(prev, t - dt_flow))
        ref = render(gt, cam, t)
        conds.append(_buffers(out))
        tgts.append(np.clip(np.array(ref.rgb), 0, 1).astype(np.float16))
    return np.stack(conds), np.stack(tgts)


def holdout_pairs(recon: SceneGraph, frames, size: tuple[int, int]):
    conds, tgts = [], []
    for k, f in enumerate(frames):
        cam = f.camera.scaled(*size)
        ref = frames[k - 1] if k > 0 else f
        out = render(recon, cam, f.t, flow_ref=(ref.camera.scaled(*size), ref.t))
        conds.append(_buffers(out))
        tgts.append(f.image(size).astype(np.float16))
    return np.stack(conds), np.stack(tgts)
