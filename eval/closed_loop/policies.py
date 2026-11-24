"""Driving policies that run inside the closed loop.

* `expert`   — privileged pure-pursuit + IDM on simulator state (labels for imitation, sanity baseline)
* `pilotnet` — compact end-to-end visuomotor CNN in MLX (image + speed -> steer, accel)
* `openpilot`— adapter for comma.ai's supercombo ONNX model via onnxruntime (experimental)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from eval.metrics.safety import ReferencePath

from .dynamics import Action, EgoState


@dataclass
class Observation:
    image: np.ndarray                 # (H,W,3) float32 [0,1] — refined camera view
    speed: float
    t: float
    # privileged simulator state (only the expert may use these)
    ego: EgoState | None = None
    actors: list = field(default_factory=list)   # [(name, pose 4x4, size, velocity)]
    route: ReferencePath | None = None
    depth: np.ndarray | None = None


class Policy:
    name = "policy"

    def reset(self):
        pass

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError


# ------------------------------------------------------------------------------------------ expert
class ExpertPolicy(Policy):
    name = "expert"

    def __init__(self, target_speed: float = 8.0, lookahead: float = 8.0, wheelbase: float = 2.8,
                 time_headway: float = 1.5, min_gap: float = 3.0, a_max: float = 2.0, b_comf: float = 3.0):
        self.v0, self.L, self.wb = target_speed, lookahead, wheelbase
        self.T, self.s0, self.a, self.b = time_headway, min_gap, a_max, b_comf

    def act(self, obs: Observation) -> Action:
        e, route = obs.ego, obs.route
        s, _, _ = route.project([e.x, e.y])
        tgt = route.point_at(s + max(self.L, 0.8 * e.v))
        dx, dy = tgt[0] - e.x, tgt[1] - e.y
        alpha = np.arctan2(dy, dx) - e.yaw
        ld = max(np.hypot(dx, dy), 1e-3)
        steer = float(np.arctan2(2 * self.wb * np.sin(alpha), ld))
        # IDM w.r.t. the closest actor in the ego corridor
        ego_pose = e.pose()
        inv = np.linalg.inv(ego_pose)
        gap, dv = np.inf, 0.0
        for _, pose, size, vel in obs.actors:
            rel = inv @ pose
            lon, lat = rel[0, 3], rel[1, 3]
            if lon > 0 and abs(lat) < (1.9 + size[1]) / 2 + 0.4:
                g = lon - (4.5 + size[0]) / 2
                if g < gap:
                    gap, dv = g, e.v - float(np.dot(vel, ego_pose[:3, 0]))
        acc = self.a * (1 - (e.v / self.v0) ** 4)
        if np.isfinite(gap):
            s_star = self.s0 + max(0.0, e.v * self.T + e.v * dv / (2 * np.sqrt(self.a * self.b)))
            acc -= self.a * (s_star / max(gap, 0.1)) ** 2
        return Action(steer, float(np.clip(acc, -8.0, 3.0)))


# ------------------------------------------------------------------------------------------ pilotnet
POLICY_INPUT = (128, 72)   # (W, H)


def preprocess(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, POLICY_INPUT, interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) - 0.5) / 0.25


class PilotNet(nn.Module):
    """NVIDIA-PilotNet-style CNN (Bojarski et al. 2016), resized for 128x72 input + speed."""

    def __init__(self):
        super().__init__()
        self.convs = [nn.Conv2d(3, 24, 5, stride=2), nn.Conv2d(24, 36, 5, stride=2), nn.Conv2d(36, 48, 3, stride=2),
                      nn.Conv2d(48, 64, 3), nn.Conv2d(64, 64, 3)]
        self.fc1 = nn.Linear(64 * 3 * 10 + 1, 100)
        self.fc2 = nn.Linear(100, 50)
        self.out = nn.Linear(50, 2)

    def __call__(self, img: mx.array, speed: mx.array) -> mx.array:
        h = img
        for c in self.convs:
            h = nn.elu(c(h))
        h = mx.concatenate([h.reshape(h.shape[0], -1), speed.reshape(-1, 1) / 10.0], axis=-1)
        h = nn.elu(self.fc2(nn.elu(self.fc1(h))))
        y = self.out(h)
        return mx.stack([0.5 * mx.tanh(y[:, 0]), 8.0 * mx.tanh(y[:, 1])], axis=-1)   # steer rad, accel m/s^2


class PilotNetPolicy(Policy):
    name = "pilotnet"

    def __init__(self, model: PilotNet | None = None):
        self.model = model or PilotNet()
        self._fwd = mx.compile(lambda x, v: self.model(x, v))

    def act(self, obs: Observation) -> Action:
        x = mx.array(preprocess(obs.image)[None])
        y = self._fwd(x, mx.array([obs.speed], dtype=mx.float32))
        y = np.array(y)[0]
        return Action(float(y[0]), float(y[1]))

    def save(self, path):
        from mlx.utils import tree_flatten

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path), dict(tree_flatten(self.model.parameters())),
                            metadata={"format": "m4dgs-policy", "arch": "pilotnet",
                                      "input": json.dumps(list(POLICY_INPUT))})

    @staticmethod
    def load(path) -> "PilotNetPolicy":
        m = PilotNet()
        m.load_weights(str(path))
        m.eval()
        return PilotNetPolicy(m)


# ------------------------------------------------------------------------------------------ openpilot
class OpenpilotPolicy(Policy):
    """EXPERIMENTAL adapter for openpilot's `supercombo.onnx` (driving model) via onnxruntime.

    It warps the camera view into the model's 256x128 road-camera frame, packs two consecutive frames as
    YUV420 (6 channels each, 12 total), zero-fills recurrent/aux inputs, and converts the predicted
    desired curvature + plan acceleration into (steer, accel) through the bicycle model. Output-tensor
    layouts differ between openpilot releases: set `curvature_index` / `accel_index` for your model.
    Not validated in this repository (no model weights are bundled).
    """

    name = "openpilot"

    def __init__(self, onnx_path: str, wheelbase: float = 2.8, curvature_index: int | None = None,
                 accel_index: int | None = None):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("openpilot policy needs `pip install onnxruntime` and a supercombo.onnx path") from e
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"{onnx_path} not found. Download supercombo.onnx from the openpilot repo "
                                    "(selfdrive/modeld/models) and pass --policy_ckpt.")
        self.sess = ort.InferenceSession(onnx_path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        self.wb = wheelbase
        self.ci, self.ai = curvature_index, accel_index
        self.prev = None

    @staticmethod
    def _yuv(img: np.ndarray) -> np.ndarray:
        rgb = (cv2.resize(img, (512, 256)) * 255).astype(np.uint8)
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)          # (384, 512)
        Y, U, V = yuv[:256], yuv[256:320].reshape(128, 256), yuv[320:].reshape(128, 256)
        return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], U, V])

    def reset(self):
        self.prev = None

    def act(self, obs: Observation) -> Action:
        cur = self._yuv(obs.image)
        prev = cur if self.prev is None else self.prev
        self.prev = cur
        feeds = {}
        for name, meta in self.inputs.items():
            shape = [d if isinstance(d, int) else 1 for d in meta.shape]
            dtype = np.float16 if "float16" in meta.type else np.float32
            if name in ("input_imgs", "big_input_imgs"):
                feeds[name] = np.concatenate([prev, cur])[None].astype(dtype if "uint8" not in meta.type else np.uint8)
            elif name == "traffic_convention":
                feeds[name] = np.array([[1, 0]], dtype=dtype)   # right-hand traffic
            else:
                feeds[name] = np.zeros(shape, dtype=dtype)
        out = self.sess.run(None, feeds)[0].reshape(-1)
        curv = float(out[self.ci]) if self.ci is not None else 0.0
        acc = float(out[self.ai]) if self.ai is not None else 0.0
        return Action(float(np.arctan(curv * self.wb)), acc)


def make_policy(name: str, ckpt: str | None = None, **kw) -> Policy:
    if name == "expert":
        return ExpertPolicy(**kw)
    if name == "pilotnet":
        return PilotNetPolicy.load(ckpt) if ckpt else PilotNetPolicy()
    if name == "openpilot":
        return OpenpilotPolicy(ckpt or "checkpoints/supercombo.onnx")
    raise ValueError(f"unknown policy {name!r} (expert | pilotnet | openpilot)")
