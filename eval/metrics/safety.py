"""Collision / time-to-collision / route metrics."""

from __future__ import annotations

import numpy as np


def box_corners_2d(pose: np.ndarray, size) -> np.ndarray:
    l, w = size[0] / 2, size[1] / 2
    local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]])
    return local @ pose[:2, :2].T + pose[:2, 3]


def obb_overlap(pose_a, size_a, pose_b, size_b) -> bool:
    """Separating-axis test for two yawed rectangles (bird's-eye view)."""
    ca, cb = box_corners_2d(pose_a, size_a), box_corners_2d(pose_b, size_b)
    for poly in (ca, cb):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = ca @ n, cb @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def time_to_collision(ego_pose, ego_size, ego_v, actor_pose, actor_size, actor_vel) -> float:
    """Constant-velocity TTC to an actor inside the ego's driving corridor (inf if none)."""
    rel = np.linalg.inv(ego_pose) @ actor_pose
    lon, lat = rel[0, 3], rel[1, 3]
    if lon <= 0 or abs(lat) > (ego_size[1] + actor_size[1]) / 2 + 0.3:
        return float("inf")
    gap = lon - (ego_size[0] + actor_size[0]) / 2
    fwd = ego_pose[:3, 0]
    closing = ego_v - float(np.dot(actor_vel, fwd))
    if gap <= 0:
        return 0.0
    return gap / closing if closing > 1e-3 else float("inf")


class ReferencePath:
    """Polyline from the logged ego track; lateral error and progress queries."""

    def __init__(self, points: np.ndarray):
        pts = np.asarray(points, dtype=np.float64)[:, :2]
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3])
        self.pts = pts[keep]
        seg = np.diff(self.pts, axis=0)
        self.seglen = np.linalg.norm(seg, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.seglen)])
        self.length = float(self.s[-1])

    def project(self, xy) -> tuple[float, float, int]:
        """Returns (arc length s, signed lateral offset (+left), segment index)."""
        p = np.asarray(xy, dtype=np.float64)
        a, b = self.pts[:-1], self.pts[1:]
        ab = b - a
        u = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(self.seglen ** 2, 1e-9), 0, 1)
        proj = a + u[:, None] * ab
        d = np.linalg.norm(proj - p, axis=1)
        i = int(np.argmin(d))
        cross = ab[i, 0] * (p[1] - a[i, 1]) - ab[i, 1] * (p[0] - a[i, 0])
        return float(self.s[i] + u[i] * self.seglen[i]), float(np.sign(cross) * d[i]), i

    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0, self.length))
        return np.array([np.interp(s, self.s, self.pts[:, 0]), np.interp(s, self.s, self.pts[:, 1])])
