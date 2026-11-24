"""Closed-loop ego-motion integrator: kinematic bicycle model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from engine.scene_graph.tracks import pose_matrix, yaw_quat


@dataclass
class Action:
    steer: float = 0.0   # front-wheel angle (rad), + = left
    accel: float = 0.0   # longitudinal acceleration (m/s^2)


@dataclass
class EgoState:
    x: float
    y: float
    yaw: float
    v: float
    z: float = 0.0

    def pose(self) -> np.ndarray:
        return pose_matrix([self.x, self.y, self.z], yaw_quat(self.yaw))


@dataclass
class BicycleModel:
    wheelbase: float = 2.8
    max_steer: float = 0.5
    max_accel: float = 3.0
    max_decel: float = 8.0
    max_speed: float = 30.0

    def step(self, s: EgoState, a: Action, dt: float) -> EgoState:
        steer = float(np.clip(a.steer, -self.max_steer, self.max_steer))
        acc = float(np.clip(a.accel, -self.max_decel, self.max_accel))
        v = float(np.clip(s.v + acc * dt, 0.0, self.max_speed))
        vm = 0.5 * (s.v + v)
        yaw = s.yaw + vm / self.wheelbase * np.tan(steer) * dt
        ym = 0.5 * (s.yaw + yaw)
        return replace(s, x=s.x + vm * np.cos(ym) * dt, y=s.y + vm * np.sin(ym) * dt, yaw=float(yaw), v=v)
