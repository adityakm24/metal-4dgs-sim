"""Rigid-body tracks (pose over time) and the programmatic trajectory editor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------- quaternion utils
def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(np.asarray(q, dtype=np.float64))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R: np.ndarray) -> np.ndarray:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_yaw(q: np.ndarray) -> float:
    R = quat_to_R(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0, q1 = quat_normalize(q0), quat_normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return quat_normalize(q0 + u * (q1 - q0))
    th = np.arccos(d)
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


def pose_matrix(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(quat)
    T[:3, 3] = pos
    return T


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ----------------------------------------------------------------------------- track
@dataclass
class Track:
    times: np.ndarray   # (K,)
    pos: np.ndarray     # (K,3)
    quat: np.ndarray    # (K,4) wxyz

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(-1, 3)
        self.quat = quat_normalize(np.asarray(self.quat, dtype=np.float64).reshape(-1, 4))
        order = np.argsort(self.times, kind="stable")
        self.times, self.pos, self.quat = self.times[order], self.pos[order], self.quat[order]

    # -- construction / io
    @staticmethod
    def static(pos=(0, 0, 0), quat=(1, 0, 0, 0)) -> "Track":
        return Track(np.array([0.0]), np.array([pos]), np.array([quat]))

    @staticmethod
    def from_json(items: list[dict]) -> "Track":
        return Track([d["t"] for d in items], [d["pos"] for d in items], [d["quat"] for d in items])

    def to_json(self) -> list[dict]:
        return [{"t": float(t), "pos": [float(v) for v in p], "quat": [float(v) for v in q]}
                for t, p, q in zip(self.times, self.pos, self.quat)]

    def copy(self) -> "Track":
        return Track(self.times.copy(), self.pos.copy(), self.quat.copy())

    @property
    def t_range(self) -> tuple[float, float]:
        return float(self.times[0]), float(self.times[-1])

    # -- evaluation
    def _locate(self, t: float):
        ts = self.times
        if t <= ts[0]:
            return 0, 0, 0.0
        if t >= ts[-1]:
            k = len(ts) - 1
            return k, k, 0.0
        i = int(np.searchsorted(ts, t, side="right") - 1)
        u = (t - ts[i]) / max(ts[i + 1] - ts[i], 1e-9)
        return i, i + 1, float(u)

    def position(self, t: float) -> np.ndarray:
        i, j, u = self._locate(t)
        return (1 - u) * self.pos[i] + u * self.pos[j]

    def rotation_quat(self, t: float) -> np.ndarray:
        i, j, u = self._locate(t)
        return slerp(self.quat[i], self.quat[j], u)

    def pose(self, t: float) -> np.ndarray:
        return pose_matrix(self.position(t), self.rotation_quat(t))

    def velocity(self, t: float, h: float = 0.05) -> np.ndarray:
        return (self.position(t + h) - self.position(t - h)) / (2 * h)

    def heading(self, t: float) -> float:
        return quat_yaw(self.rotation_quat(t))

    def resample(self, dt: float, t0: float | None = None, t1: float | None = None) -> "Track":
        a, b = self.t_range
        t0 = a if t0 is None else t0
        t1 = b if t1 is None else t1
        ts = np.arange(t0, t1 + 1e-9, dt)
        return Track(ts, np.stack([self.position(t) for t in ts]), np.stack([self.rotation_quat(t) for t in ts]))

    # -- editing (all return new tracks) ---------------------------------------------------
    def _dense(self, dt: float = 0.05) -> "Track":
        if len(self.times) < 2:
            return self.copy()
        return self.resample(dt)

    def reorient_along_path(self, min_speed: float = 0.5) -> "Track":
        """Re-derive yaw from the direction of travel where the actor is moving (keeps roll/pitch=0)."""
        out = self.copy()
        for k, t in enumerate(out.times):
            v = out.velocity(t)
            if np.linalg.norm(v[:2]) > min_speed:
                out.quat[k] = yaw_quat(np.arctan2(v[1], v[0]))
        return out

    def lateral_offset(self, offset: float, t_start: float, ramp: float = 1.5, t_end: float | None = None,
                       reorient: bool = True) -> "Track":
        """Shift the actor along its own left (+y) axis, blending in over `ramp` seconds from t_start.
        A negative offset moves it to its right. With t_end, blends back out after t_end."""
        out = self._dense()
        for k, t in enumerate(out.times):
            w = smoothstep((t - t_start) / max(ramp, 1e-6))
            if t_end is not None:
                w *= 1 - smoothstep((t - t_end) / max(ramp, 1e-6))
            left = quat_to_R(self.rotation_quat(t))[:, 1]
            out.pos[k] = self.position(t) + w * offset * left
        return out.reorient_along_path() if reorient else out

    def longitudinal_offset(self, offset: float, t_start: float = -1e9, ramp: float = 0.0) -> "Track":
        out = self._dense()
        for k, t in enumerate(out.times):
            w = smoothstep((t - t_start) / ramp) if ramp > 0 else float(t >= t_start)
            fwd = quat_to_R(self.rotation_quat(t))[:, 0]
            out.pos[k] = self.position(t) + w * offset * fwd
        return out

    def time_shift(self, dt: float) -> "Track":
        out = self.copy()
        out.times = out.times + dt
        return out

    def speed_scale(self, factor: float, t_start: float = 0.0) -> "Track":
        """Replay the track `factor`x faster after t_start (position continuity at t_start)."""
        out = self._dense()
        new_pos, new_q = [], []
        for t in out.times:
            src = t if t < t_start else t_start + (t - t_start) * factor
            new_pos.append(self.position(src))
            new_q.append(self.rotation_quat(src))
        out.pos, out.quat = np.stack(new_pos), np.stack(new_q)
        return out

    def velocity_spike(self, delta_v: float, t_start: float, duration: float = 1.0) -> "Track":
        """Add `delta_v` m/s along the heading for `duration` s (integrated displacement persists)."""
        out = self._dense()
        disp = 0.0
        prev_t = out.times[0]
        for k, t in enumerate(out.times):
            dt = t - prev_t
            prev_t = t
            if t_start <= t <= t_start + duration:
                disp += delta_v * dt
            fwd = quat_to_R(self.rotation_quat(t))[:, 0]
            out.pos[k] = self.position(t) + disp * fwd
        return out

    def hard_brake(self, t_start: float, decel: float = 8.0) -> "Track":
        """Decelerate to standstill along the original path from t_start."""
        out = self._dense()
        v0 = float(np.linalg.norm(self.velocity(t_start)[:2]))
        # arc-length along the original track
        ts = out.times
        s_orig = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.stack([self.position(t) for t in ts]), axis=0), axis=1))])
        s_start = float(np.interp(t_start, ts, s_orig))
        new_pos, new_q = [], []
        for t in ts:
            if t <= t_start:
                s = float(np.interp(t, ts, s_orig))
            else:
                tau = min(t - t_start, v0 / max(decel, 1e-6))
                s = s_start + v0 * tau - 0.5 * decel * tau * tau
            src_t = float(np.interp(s, s_orig, ts))
            new_pos.append(self.position(src_t))
            new_q.append(self.rotation_quat(src_t))
        out.pos, out.quat = np.stack(new_pos), np.stack(new_q)
        return out
