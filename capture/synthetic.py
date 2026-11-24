"""Procedural ground-truth driving world built directly from Gaussians.

Used for the bundled sample scene, tests, and for generating perfectly paired
(degraded reconstruction, ground truth) frames to train the refiner.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

from engine.scene_graph import Ego, Node, SceneGraph, Track, default_camera_to_ego, make_gaussians, yaw_quat
from engine.scene_graph import gaussians as G

LANE_W = 3.5
SKY = (0.62, 0.76, 0.92)


def _plane(rng, x0, x1, y0, y1, z, spacing, base_rgb, noise=0.04, thickness=0.01):
    xs = np.arange(x0, x1, spacing)
    ys = np.arange(y0, y1, spacing)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    n = X.size
    pts = np.stack([X.ravel(), Y.ravel(), np.full(n, z)], 1)
    pts[:, :2] += rng.uniform(-0.3, 0.3, (n, 2)) * spacing
    rgb = np.clip(np.asarray(base_rgb) + rng.normal(0, noise, (n, 3)), 0, 1)
    scales = np.stack([np.full(n, spacing * 0.75), np.full(n, spacing * 0.75), np.full(n, thickness)], 1)
    return pts, rgb, scales


def _box_surface(rng, center, size, spacing, rgb_fn, yaw=0.0, open_bottom=True):
    """Gaussians on the faces of an axis-aligned (then yawed) box."""
    l, w, h = size
    pts, normals = [], []
    for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        if axis == 2 and sign < 0 and open_bottom:
            continue
        dims = [l, w, h]
        u_ax, v_ax = [a for a in range(3) if a != axis]
        us = np.arange(-dims[u_ax] / 2, dims[u_ax] / 2 + 1e-6, spacing)
        vs = np.arange(-dims[v_ax] / 2, dims[v_ax] / 2 + 1e-6, spacing)
        U, V = np.meshgrid(us, vs, indexing="ij")
        p = np.zeros((U.size, 3))
        p[:, u_ax], p[:, v_ax] = U.ravel(), V.ravel()
        p[:, axis] = sign * dims[axis] / 2
        pts.append(p)
        nrm = np.zeros_like(p)
        nrm[:, axis] = sign
        normals.append(nrm)
    pts, normals = np.concatenate(pts), np.concatenate(normals)
    rgb = rgb_fn(pts, normals)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    world = pts @ Rz.T + np.asarray(center)
    scales = np.full((len(pts), 3), spacing * 0.7)
    # flatten along the face normal
    scales[np.abs(normals[:, 0]) > 0, 0] = spacing * 0.08
    scales[np.abs(normals[:, 1]) > 0, 1] = spacing * 0.08
    scales[np.abs(normals[:, 2]) > 0, 2] = spacing * 0.08
    return world, rgb, scales, normals


def make_car(rng, body_rgb, size=(4.5, 1.9, 1.5), spacing=0.07, sh_degree=1):
    """A car in its local frame (x fwd, y left, z up, box centred at origin)."""
    l, w, h = size
    body_h = h * 0.55
    cabin_h = h - body_h

    def body_color(p, n):
        rgb = np.tile(np.asarray(body_rgb, dtype=np.float64), (len(p), 1))
        shade = 0.85 + 0.15 * n[:, 2]          # top lighter
        rgb *= shade[:, None]
        # head/tail lights
        front = (n[:, 0] > 0.5) & (np.abs(np.abs(p[:, 1]) - w * 0.35) < 0.15) & (p[:, 2] > 0.05)
        back = (n[:, 0] < -0.5) & (np.abs(np.abs(p[:, 1]) - w * 0.35) < 0.15) & (p[:, 2] > 0.05)
        rgb[front] = [0.95, 0.95, 0.85]
        rgb[back] = [0.85, 0.08, 0.05]
        return np.clip(rgb + rng.normal(0, 0.02, rgb.shape), 0, 1)

    def cabin_color(p, n):
        rgb = np.tile([0.10, 0.12, 0.16], (len(p), 1))
        top = n[:, 2] > 0.5
        rgb[top] = np.asarray(body_rgb) * 0.9
        return np.clip(rgb + rng.normal(0, 0.02, rgb.shape), 0, 1)

    p1, c1, s1, _ = _box_surface(rng, (0, 0, -h / 2 + 0.25 + (body_h - 0.25) / 2), (l, w, body_h - 0.25), spacing, body_color)
    p2, c2, s2, _ = _box_surface(rng, (-0.3, 0, -h / 2 + body_h + cabin_h / 2), (l * 0.55, w * 0.9, cabin_h), spacing, cabin_color)
    wheels = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * l * 0.32, sy * (w / 2 - 0.05)
            th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            for rr in np.linspace(0.05, 0.33, 5):
                wheels.append(np.stack([cx + rr * np.cos(th), np.full_like(th, cy), -h / 2 + 0.33 + rr * np.sin(th)], 1))
    p3 = np.concatenate(wheels)
    c3 = np.full((len(p3), 3), 0.05)
    s3 = np.full((len(p3), 3), 0.05)
    pts = np.concatenate([p1, p2, p3])
    rgb = np.concatenate([c1, c2, c3])
    sc = np.concatenate([s1, s2, s3])
    return make_gaussians(pts, rgb, sc, sh_degree=sh_degree, opacity=0.95)


def straight_track(x0, y, speed, t0=0.0, t1=10.0, dt=0.1, z=0.0) -> Track:
    ts = np.arange(t0, t1 + 1e-9, dt)
    pos = np.stack([x0 + speed * (ts - t0), np.full_like(ts, y), np.full_like(ts, z)], 1)
    return Track(ts, pos, np.tile(yaw_quat(0.0), (len(ts), 1)))


def build_world(seed: int = 0, length: float = 170.0, duration: float = 10.0, sh_degree: int = 1,
                width: int = 512, height: int = 288, density: float = 1.0) -> SceneGraph:
    rng = np.random.default_rng(seed)
    sp = lambda s: s / np.sqrt(density)
    x0, x1 = -15.0, length
    parts = []
    # asphalt: ego lane centred on y=0, left lane at y=+3.5
    road_y0, road_y1 = -LANE_W / 2, LANE_W * 1.5
    p, c, s = _plane(rng, x0, x1, road_y0, road_y1, 0.0, sp(0.22), (0.27, 0.27, 0.29), noise=0.03)
    parts.append((p, c, s))
    # lane markings: solid edges + dashed centre
    for ym, dashed in [(road_y0 + 0.15, False), (road_y1 - 0.15, False), (LANE_W / 2, True)]:
        xs = np.arange(x0, x1, sp(0.12))
        if dashed:
            xs = xs[(np.mod(xs, 9.0) < 3.0)]
        ys = np.linspace(ym - 0.06, ym + 0.06, 2)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        pm = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.01)], 1)
        cm = np.clip(np.full((len(pm), 3), 0.92) + rng.normal(0, 0.02, (len(pm), 3)), 0, 1)
        parts.append((pm, cm, np.tile([sp(0.09), 0.06, 0.005], (len(pm), 1))))
    # curbs + sidewalks
    for y_a, y_b in [(road_y0 - 2.5, road_y0), (road_y1, road_y1 + 2.5)]:
        p, c, s = _plane(rng, x0, x1, y_a, y_b, 0.15, sp(0.3), (0.62, 0.6, 0.57), noise=0.04)
        parts.append((p, c, s))
    # grass verges
    for y_a, y_b in [(road_y0 - 9.0, road_y0 - 2.5), (road_y1 + 2.5, road_y1 + 9.0)]:
        p, c, s = _plane(rng, x0, x1, y_a, y_b, 0.05, sp(0.4), (0.28, 0.45, 0.2), noise=0.06, thickness=0.05)
        parts.append((p, c, s))
    # buildings
    palette = [(0.75, 0.68, 0.58), (0.55, 0.38, 0.3), (0.82, 0.8, 0.76), (0.45, 0.5, 0.58), (0.7, 0.55, 0.4)]
    for side, yb in [(-1, road_y0 - 9.0 - 4.0), (1, road_y1 + 9.0 + 4.0)]:
        x = x0
        while x < x1:
            L = rng.uniform(8, 16)
            Wd = rng.uniform(6, 9)
            Hh = rng.uniform(5, 16)
            base = np.asarray(palette[rng.integers(len(palette))])

            def bcol(pt, n, base=base, Hh=Hh):
                rgb = np.tile(base, (len(pt), 1)) * (0.8 + 0.2 * np.abs(n[:, 0:1]))
                win = (np.mod(pt[:, 2] + Hh / 2, 3.0) > 1.2) & (np.mod(pt[:, 2] + Hh / 2, 3.0) < 2.3) & \
                      (np.mod(pt[:, 0] + pt[:, 1], 2.5) < 1.3) & (np.abs(n[:, 2]) < 0.5)
                rgb[win] = [0.2, 0.25, 0.32]
                return np.clip(rgb + rng.normal(0, 0.025, rgb.shape), 0, 1)

            p, c, s, _ = _box_surface(rng, (x + L / 2, yb + side * Wd / 2, Hh / 2), (L, Wd, Hh), sp(0.35), bcol)
            parts.append((p, c, s))
            x += L + rng.uniform(1, 4)
    # trees on the verges
    for side, yt in [(-1, road_y0 - 5.5), (1, road_y1 + 5.5)]:
        for xt in np.arange(x0 + 5, x1, 14.0) + rng.uniform(-3, 3, len(np.arange(x0 + 5, x1, 14.0))):
            n_leaf = int(500 * density)
            d = rng.normal(size=(n_leaf, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            r = 1.8 * np.cbrt(rng.uniform(0.3, 1, n_leaf))
            pl = np.array([xt, yt, 4.2]) + d * r[:, None] * np.array([1, 1, 1.2])
            cl = np.clip(np.array([0.18, 0.38, 0.14]) + rng.normal(0, 0.05, (n_leaf, 3)), 0, 1)
            parts.append((pl, cl, np.full((n_leaf, 3), 0.22)))
            zt = np.linspace(0.1, 3.0, 30)
            pt = np.stack([np.full_like(zt, xt), np.full_like(zt, yt), zt], 1)
            parts.append((pt, np.tile([0.35, 0.24, 0.15], (len(zt), 1)), np.tile([0.12, 0.12, 0.1], (len(zt), 1))))
    pts = np.concatenate([q[0] for q in parts])
    rgb = np.concatenate([q[1] for q in parts])
    sc = np.concatenate([q[2] for q in parts])
    bg = make_gaussians(pts, rgb, sc, sh_degree=sh_degree, opacity=0.97)
    nodes = [Node("background", "background", bg)]

    car_size = [4.5, 1.9, 1.5]
    z_car = car_size[2] / 2
    actors = [
        ("actor_1", (0.75, 0.1, 0.08), straight_track(22.0, 0.0, 7.0, t1=duration, z=z_car)),      # lead car
        ("actor_2", (0.1, 0.25, 0.7), straight_track(4.0, LANE_W, 9.5, t1=duration, z=z_car)),     # left lane
        ("actor_3", (0.85, 0.85, 0.85), straight_track(70.0, LANE_W, 6.0, t1=duration, z=z_car)),  # slow, left lane
    ]
    for name, col, tr in actors:
        nodes.append(Node(name, "actor", make_car(rng, col, tuple(car_size), spacing=sp(0.08), sh_degree=sh_degree),
                          track=tr, size=car_size, category="vehicle.car"))
    ego_track = straight_track(0.0, 0.0, 8.0, t1=duration, z=0.0)
    fx = 300.0 * width / 512
    ego = Ego(ego_track, {"width": width, "height": height, "fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2},
              default_camera_to_ego(1.5, 1.0), [4.5, 1.9, 1.6])
    return SceneGraph(nodes, ego, sh_degree=sh_degree, time_range=(0.0, duration), background_color=SKY)


def write_sample_dataset(root: str | Path, scene: SceneGraph, fps: float = 10.0, lidar_points: int = 60_000,
                         seed: int = 0):
    """Render ground-truth frames along the ego track and write a generic m4dgs log."""
    from engine.scene_graph import render

    from .dataset import ActorAnnotation, Dataset, Frame, save_log

    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    t0, t1 = scene.time_range
    frames, pts = [], []
    for i, t in enumerate(np.arange(t0, t1 + 1e-9, 1.0 / fps)):
        cam = scene.ego_camera(float(t))
        out = render(scene, cam, float(t))
        rgb = np.clip(np.array(out.rgb), 0, 1)
        depth = np.array(out.depth)
        ip, dp = root / "images" / f"{i:06d}.png", root / "depth" / f"{i:06d}.npy"
        cv2.imwrite(str(ip), cv2.cvtColor((rgb * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        np.save(dp, depth.astype(np.float32))
        frames.append(Frame(float(t), cam, ip, dp))
        # sparse "LiDAR/VIO" points: back-project a random subset of valid depth pixels
        v, u = np.nonzero((depth > 0.5) & (depth < 60))
        if len(v):
            k = rng.choice(len(v), size=min(len(v), lidar_points // int((t1 - t0) * fps + 1)), replace=False)
            v, u = v[k], u[k]
            z = depth[v, u]
            xc = np.stack([(u + 0.5 - cam.cx) / cam.fx * z, (v + 0.5 - cam.cy) / cam.fy * z, z], 1)
            xw = xc @ cam.c2w[:3, :3].T + cam.c2w[:3, 3]
            pts.append(np.concatenate([xw, rgb[v, u], np.full((len(z), 1), t)], 1))
    actors = [ActorAnnotation(n.name, n.track, n.size, n.category, n.visible_range) for n in scene.actors]
    ds = Dataset(frames, source="synthetic", ego_track=scene.ego.track, camera_to_ego=scene.ego.camera_to_ego,
                 actors=actors, points=np.concatenate(pts), background_color=scene.background_color, root=root)
    save_log(ds, root)
    return ds
