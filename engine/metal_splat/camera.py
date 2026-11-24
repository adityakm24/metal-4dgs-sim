"""Pinhole camera in OpenCV convention (+x right, +y down, +z forward)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray  # (4,4) camera-to-world, OpenCV axes

    def __post_init__(self):
        # plain Python scalars: numpy scalars would hijack arithmetic with MLX arrays
        self.width, self.height = int(self.width), int(self.height)
        self.fx, self.fy, self.cx, self.cy = float(self.fx), float(self.fy), float(self.cx), float(self.cy)
        self.c2w = np.asarray(self.c2w, dtype=np.float64)

    @property
    def w2c(self) -> np.ndarray:
        return invert_se3(self.c2w)

    @property
    def center(self) -> np.ndarray:
        return self.c2w[:3, 3]

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def scaled(self, width: int, height: int) -> "Camera":
        sx, sy = width / self.width, height / self.height
        return replace(self, width=width, height=height, fx=self.fx * sx, fy=self.fy * sy,
                       cx=self.cx * sx, cy=self.cy * sy)

    def with_pose(self, c2w: np.ndarray) -> "Camera":
        return replace(self, c2w=np.asarray(c2w, dtype=np.float64))

    def project(self, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (M,3) world points -> (M,2) pixels and (M,) depth (numpy, non-differentiable)."""
        w2c = self.w2c
        p = pts_world @ w2c[:3, :3].T + w2c[:3, 3]
        z = p[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        uv = np.stack([self.fx * p[:, 0] / zs + self.cx, self.fy * p[:, 1] / zs + self.cy], axis=1)
        return uv, z

    def to_json(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy}


def invert_se3(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """OpenCV camera-to-world looking from eye to target (world z-up)."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w
