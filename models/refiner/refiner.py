"""Geometry-grounded flow-matching refiner.

Rectified flow *from the raw splat render to the real image*: x_0 = render (+ small noise during
training), x_1 = ground truth, x_t = (1-t) x_0 + t x_1, target velocity v = x_1 - x_0, conditioned on the
render's metric depth, alpha and optical flow. Because the source distribution is already close to the
target, 2 Euler steps suffice at inference. The refiner only changes appearance; the metric depth/flow
buffers handed to downstream consumers come from the 4DGS scene graph untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .unet import RefinerUNet

COND_CH = 7


def make_condition(rgb: mx.array, depth: mx.array, alpha: mx.array, flow: mx.array | None) -> mx.array:
    """Stack (B,H,W,·) or (H,W,·) render buffers into the 7-channel conditioning tensor."""
    inv_d = mx.where(depth > 0, 1.0 / mx.maximum(depth, 0.5), 0.0)
    if flow is None:
        flow = mx.zeros(depth.shape + (2,), dtype=mx.float32)
    w = depth.shape[-1]
    fl = mx.clip(flow / (0.05 * w), -3, 3)
    return mx.concatenate([rgb, inv_d[..., None], alpha[..., None], fl], axis=-1)


def flow_matching_loss(model: RefinerUNet, x0: mx.array, x1: mx.array, cond: mx.array, key=None,
                       x1_weight: float = 0.5) -> mx.array:
    """Rectified-flow velocity MSE + an L1 on the implied clean image x̂1 = x_t + (1 - t) v (sharpens the
    few-step sampler and keeps in-distribution views close to identity)."""
    b = x0.shape[0]
    # logit-normal timestep sampling (as in SD3) concentrates on the middle of the path
    t = mx.sigmoid(mx.random.normal((b,), key=key))
    tb = t[:, None, None, None]
    xt = (1 - tb) * x0 + tb * x1
    v = model(xt, cond, t)
    loss = mx.mean(mx.square(v - (x1 - x0)))
    if x1_weight > 0:
        loss = loss + x1_weight * mx.mean(mx.abs(xt + (1 - tb) * v - x1))
    return loss


class Refiner:
    def __init__(self, model: RefinerUNet, steps: int = 2, dtype=mx.float16):
        self.model = model
        self.steps = steps
        self.dtype = dtype
        if dtype != mx.float32:
            self.model.set_dtype(dtype)
        self.model.eval()
        self._compiled = mx.compile(self._run)

    def _run(self, x0, cond):
        x = x0
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t = mx.full((x.shape[0],), i * dt, dtype=x.dtype)
            x = x + dt * self.model(x, cond, t)
        return mx.clip(x, 0.0, 1.0)

    def __call__(self, rgb: mx.array, depth: mx.array, alpha: mx.array, flow: mx.array | None = None) -> mx.array:
        squeeze = rgb.ndim == 3
        cond = make_condition(rgb, depth, alpha, flow)
        x0, cond = (rgb[None], cond[None]) if squeeze else (rgb, cond)
        out = self._compiled(x0.astype(self.dtype), cond.astype(self.dtype)).astype(mx.float32)
        return out[0] if squeeze else out

    def refine(self, render_out) -> mx.array:
        return self(render_out.rgb, render_out.depth, render_out.alpha, render_out.flow)

    # ------------------------------------------------------------------ io
    @staticmethod
    def load(path: str | Path, steps: int | None = None, dtype=mx.float16) -> "Refiner":
        weights, meta = mx.load(str(path), return_metadata=True)
        cfg = json.loads(meta["config"])
        model = RefinerUNet(**{k: v for k, v in cfg.items() if k in ("in_ch", "cond_ch", "out_ch", "base", "mults", "tdim", "patch")})
        model.load_weights(list(weights.items()))
        return Refiner(model, steps or int(cfg.get("steps", 2)), dtype)


def save_refiner(model: RefinerUNet, path: str | Path, steps: int = 2, extra: dict | None = None, dtype=mx.float16):
    from mlx.utils import tree_flatten

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {k: v.astype(dtype) for k, v in tree_flatten(model.parameters())}
    cfg = {**model.config, "steps": steps, **(extra or {})}
    mx.save_safetensors(str(path), weights, metadata={"format": "m4dgs-refiner", "config": json.dumps(cfg)})


class IdentityRefiner:
    """Pass-through used when no refiner checkpoint is given."""

    steps = 0

    def refine(self, render_out) -> mx.array:
        return mx.clip(render_out.rgb, 0, 1)


def count_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return int(sum(np.prod(v.shape) for _, v in tree_flatten(model.parameters())))
