#!/usr/bin/env python3
"""Writes a synthetic 4DGS scene checkpoint (docs/formats.md §3) + a matching rollout (§4).

numpy only. Usage (from the repo root):
    .venv/bin/python apps/macbook/Tools/make_test_scene.py [--out apps/macbook/Fixtures/test_scene.safetensors]
                                                          [--gaussians 200000] [--seed 0]

Scene (world z-up, +x = ego heading, +y = left), time range [0, 8] s:
  * background: ground (asphalt road |y|<7, sidewalks, grass), lane markings (yellow centre line y=0,
    dashed white y=±3.5, solid white edges y=±7), box buildings (left side y>0 BLUE-ish, right side y<0
    ORANGE-ish), a sky dome, a ball rolling across the road (per-Gaussian velocity) and a short-lived
    "flash" cluster (temporal opacity).
  * actor_1: red car, lead vehicle in the ego lane (y=-1.75), 7 m/s, rear tail-lights.
  * actor_2: blue car, oncoming lane (y=+1.75), heading -x at 8 m/s, front head-lights, visible_range [0, 7].
  * ego: drives +x at 8 m/s in lane y=-1.75; camera 512x288, fx=fy=300, 1.5 m up, looking along +x.
"""
import argparse
import json
import math
import os
import struct

import numpy as np

C0 = 0.28209479177387814
SH_DEGREE = 1
K = (SH_DEGREE + 1) ** 2


def rgb_to_sh0(rgb):
    return (np.asarray(rgb, np.float64) - 0.5) / C0


def yaw_quat(yaw):
    yaw = np.asarray(yaw, np.float64)
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], -1)


class Node:
    def __init__(self, name):
        self.name = name
        self.parts = []

    def add(self, means, sigma, rgb, opacity=0.95, quats=None, velocity=None, t_center=4.0, t_sigma=1e4,
            sh_rest=None, rng=None):
        means = np.asarray(means, np.float64).reshape(-1, 3)
        n = len(means)
        sigma = np.broadcast_to(np.asarray(sigma, np.float64), (n, 3))
        rgb = np.broadcast_to(np.asarray(rgb, np.float64), (n, 3))
        if quats is None:
            quats = yaw_quat(rng.uniform(0, 2 * np.pi, n)) if rng is not None else np.tile([1.0, 0, 0, 0], (n, 1))
        quats = np.broadcast_to(np.asarray(quats, np.float64), (n, 4))
        vel = np.zeros((n, 3)) if velocity is None else np.broadcast_to(np.asarray(velocity, np.float64), (n, 3))
        sh = np.zeros((n, K, 3))
        sh[:, 0, :] = rgb_to_sh0(rgb)
        if sh_rest is not None:
            sh[:, 1:, :] = np.broadcast_to(sh_rest, (n, K - 1, 3))
        op = np.clip(np.broadcast_to(np.asarray(opacity, np.float64), (n,)), 1e-4, 1 - 1e-4)
        self.parts.append(dict(
            means=means, log_scales=np.log(sigma), quats=quats,
            opacity_logits=np.log(op / (1 - op))[:, None], sh=sh, velocity=vel,
            t_center=np.broadcast_to(np.asarray(t_center, np.float64), (n,))[:, None].copy(),
            t_log_scale=np.full((n, 1), math.log(t_sigma))))

    def tensors(self):
        out = {}
        for k in self.parts[0]:
            out[f"{self.name}.{k}"] = np.concatenate([p[k] for p in self.parts]).astype(np.float32)
        return out

    @property
    def count(self):
        return sum(len(p["means"]) for p in self.parts)


def box_surface(lo, hi, spacing, rng, faces=("+x", "-x", "+y", "-y", "+z", "-z")):
    """Points on the faces of an axis-aligned box, returned with their face normal axis/sign and (u,v) coords."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts, info = [], []
    for f in faces:
        ax = "xyz".index(f[1])
        sgn = 1 if f[0] == "+" else -1
        a, b = [i for i in range(3) if i != ax]
        na = max(2, int(round((hi[a] - lo[a]) / spacing)))
        nb = max(2, int(round((hi[b] - lo[b]) / spacing)))
        ua = (np.arange(na) + 0.5) / na
        ub = (np.arange(nb) + 0.5) / nb
        UA, UB = np.meshgrid(ua, ub, indexing="ij")
        P = np.zeros((UA.size, 3))
        P[:, a] = lo[a] + UA.ravel() * (hi[a] - lo[a])
        P[:, b] = lo[b] + UB.ravel() * (hi[b] - lo[b])
        P[:, ax] = hi[ax] if sgn > 0 else lo[ax]
        P[:, [a, b]] += rng.uniform(-0.2, 0.2, (len(P), 2)) * spacing
        pts.append(P)
        info.append(np.column_stack([np.full(len(P), ax), np.full(len(P), sgn),
                                     P[:, a] - lo[a], P[:, b] - lo[b]]))
    return np.concatenate(pts), np.concatenate(info)


def flat_sigma(info, spacing, thin=0.02):
    """Disc-shaped sigmas lying in each face plane."""
    s = np.full((len(info), 3), 0.6 * spacing)
    s[np.arange(len(info)), info[:, 0].astype(int)] = thin
    return s


def build_car(node, body_rgb, rng, size=(4.5, 1.9, 1.5), front_lights=False, rear_lights=True):
    L, Wd, H = size
    sp = 0.09
    # lower body
    P, info = box_surface([-L / 2, -Wd / 2, -H / 2], [L / 2, Wd / 2, -H / 2 + 0.9], sp, rng)
    rgb = np.tile(body_rgb, (len(P), 1)).astype(float)
    rgb *= rng.uniform(0.9, 1.05, (len(P), 1))
    bottom = (P[:, 2] < -H / 2 + 0.25)
    rgb[bottom] = [0.05, 0.05, 0.05]                      # tyres / underbody band
    if rear_lights:
        rear = (info[:, 0] == 0) & (info[:, 1] < 0) & (np.abs(np.abs(P[:, 1]) - 0.7) < 0.2) & (np.abs(P[:, 2] + 0.2) < 0.12)
        rgb[rear] = [1.0, 0.05, 0.05]
    if front_lights:
        front = (info[:, 0] == 0) & (info[:, 1] > 0) & (np.abs(np.abs(P[:, 1]) - 0.7) < 0.2) & (np.abs(P[:, 2] + 0.2) < 0.12)
        rgb[front] = [1.0, 1.0, 0.85]
    sh_rest = np.zeros((K - 1, 3))
    sh_rest[2] = -0.08                                    # slight view-dependent sheen (degree-1 x term)
    node.add(P, flat_sigma(info, sp, 0.015), np.clip(rgb, 0, 1), opacity=0.98, quats=[1, 0, 0, 0], sh_rest=sh_rest)
    # cabin (windows dark, roof body colour)
    P, info = box_surface([-1.3, -Wd / 2 + 0.1, -H / 2 + 0.9], [1.0, Wd / 2 - 0.1, H / 2], sp, rng)
    rgb = np.tile([0.08, 0.1, 0.13], (len(P), 1))
    rgb[info[:, 0] == 2] = body_rgb
    node.add(P, flat_sigma(info, sp, 0.015), rgb, opacity=0.98, quats=[1, 0, 0, 0])


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "Fixtures", "test_scene.safetensors"))
    ap.add_argument("--gaussians", type=int, default=200_000, help="approximate total Gaussian count")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    T0, T1 = 0.0, 8.0
    EGO_Y, EGO_V = -1.75, 8.0
    bg = Node("background")

    # --- ground: budget ≈ 65% of the Gaussians
    X0, X1, Y0, Y1 = -30.0, 170.0, -30.0, 30.0
    ground_budget = max(20_000, int(args.gaussians * 0.66))
    sp = math.sqrt((X1 - X0) * (Y1 - Y0) / ground_budget)
    xs = np.arange(X0, X1, sp)
    ys = np.arange(Y0, Y1, sp)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    G = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
    G[:, :2] += rng.uniform(-0.3, 0.3, (len(G), 2)) * sp
    ay = np.abs(G[:, 1])
    rgb = np.empty((len(G), 3))
    road = ay < 7.2
    walk = (ay >= 7.2) & (ay < 10)
    grass = ay >= 10
    rgb[road] = [0.23, 0.23, 0.25]
    rgb[walk] = [0.55, 0.54, 0.52]
    rgb[grass] = [0.24, 0.42, 0.18]
    rgb *= rng.uniform(0.85, 1.12, (len(G), 1))
    sig = np.column_stack([np.full(len(G), 0.65 * sp), np.full(len(G), 0.65 * sp), np.full(len(G), 0.01)])
    bg.add(G, sig, np.clip(rgb, 0, 1), opacity=0.97, rng=rng)

    # --- lane markings (slightly above the ground)
    def line(y, colour, dashed=False, width=0.15):
        step = 0.06
        x = np.arange(X0, X1, step)
        if dashed:
            x = x[np.mod(x, 12.0) < 3.0]
        P = []
        for dy in (-width / 4, width / 4):
            P.append(np.column_stack([x, np.full(len(x), y + dy), np.full(len(x), 0.02)]))
        P = np.concatenate(P)
        bg.add(P, [0.05, width / 3.5, 0.005], colour, opacity=0.99, quats=[1, 0, 0, 0])

    line(0.0, [0.95, 0.75, 0.1])
    line(3.5, [0.95, 0.95, 0.95], dashed=True)
    line(-3.5, [0.95, 0.95, 0.95], dashed=True)
    line(7.0, [0.95, 0.95, 0.95])
    line(-7.0, [0.95, 0.95, 0.95])

    # --- buildings: left (+y) blue-ish, right (-y) orange-ish
    bsp = 0.45
    for side in (+1, -1):
        x = -10.0
        while x < 160:
            lx = rng.uniform(9, 16)
            ly = rng.uniform(7, 11)
            h = rng.uniform(6, 18)
            y_near = 13.0 + rng.uniform(0, 2)
            lo = [x, y_near if side > 0 else -y_near - ly, 0.0]
            hi = [x + lx, y_near + ly if side > 0 else -y_near, h]
            P, info = box_surface(lo, hi, bsp, rng, faces=("+x", "-x", "+y", "-y", "+z"))
            base = np.array([0.35, 0.5, 0.85]) if side > 0 else np.array([0.9, 0.55, 0.25])
            base = base * rng.uniform(0.75, 1.1)
            c = np.tile(base, (len(P), 1))
            # windows: dark grid on vertical faces
            vert = info[:, 0] != 2
            u, v = info[:, 2], info[:, 3]
            win = vert & (np.mod(u, 3.0) > 1.0) & (np.mod(v, 3.0) > 1.2) & (np.mod(v, 3.0) < 2.4)
            c[win] = [0.12, 0.14, 0.2]
            c[info[:, 0] == 2] = base * 0.6               # roofs
            sh_rest = np.zeros((len(P), K - 1, 3))
            sh_rest[win] = 0.06 * rng.standard_normal((int(win.sum()), K - 1, 1))
            bg.add(P, flat_sigma(info, bsp, 0.03), np.clip(c, 0, 1), opacity=0.98, quats=[1, 0, 0, 0], sh_rest=sh_rest)
            x += lx + rng.uniform(3, 8)

    # --- sky dome (upper hemisphere, Fibonacci sampling)
    n_sky = 2500
    i = np.arange(n_sky)
    zc = 1 - (i + 0.5) / n_sky * 1.3            # cos(polar) in (−0.3, 1]: dips below the horizon
    r = np.sqrt(np.clip(1 - zc ** 2, 0, 1))
    phi = i * math.pi * (3 - math.sqrt(5))
    R = 400.0
    sky = np.column_stack([70 + R * r * np.cos(phi), R * r * np.sin(phi), R * zc])
    elev = np.clip(zc, 0, 1)[:, None]
    col = (1 - elev) * np.array([0.78, 0.86, 0.95]) + elev * np.array([0.33, 0.55, 0.88])
    bg.add(sky, R * math.sqrt(2 * math.pi / n_sky) * 0.7, col, opacity=0.99, rng=rng)

    # --- dynamic background bits: rolling ball (velocity) + short flash (temporal opacity)
    n_ball = 400
    d = rng.standard_normal((n_ball, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ball = d * 0.45 + np.array([30.0, 2.0, 0.45])            # position at t_center = 3 s
    bg.add(ball, 0.08, [0.9, 0.9, 0.1], opacity=0.98, velocity=[0.0, -2.0, 0.0], t_center=3.0, t_sigma=1e4, rng=rng)
    flash = rng.standard_normal((300, 3)) * [0.3, 0.3, 0.3] + [45.0, -9.0, 3.0]
    bg.add(flash, 0.12, [1.0, 0.2, 0.9], opacity=0.99, t_center=2.0, t_sigma=0.6, rng=rng)

    # --- actors
    a1 = Node("actor_1")
    build_car(a1, [0.8, 0.08, 0.06], rng, rear_lights=True)
    a2 = Node("actor_2")
    build_car(a2, [0.12, 0.28, 0.85], rng, front_lights=True, rear_lights=False)

    ts = np.arange(T0, T1 + 1e-9, 0.5)
    a1_track = [{"t": float(t), "pos": [20.0 + 7.0 * t, EGO_Y, 0.75], "quat": [1.0, 0.0, 0.0, 0.0]} for t in ts]
    a2_track = []
    for t in ts:                                   # oncoming with a gentle weave (exercises slerp)
        y = 1.75 + 0.25 * math.sin(0.8 * t)
        dy = 0.25 * 0.8 * math.cos(0.8 * t)
        yaw = math.atan2(dy, -8.0)
        a2_track.append({"t": float(t), "pos": [56.0 - 8.0 * t, y, 0.75], "quat": yaw_quat(yaw).tolist()})
    ego_track = [{"t": float(t), "pos": [EGO_V * t, EGO_Y, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]} for t in ts]

    # OpenCV camera in ego frame: cam z → ego +x, cam x → ego −y, cam y → ego −z; 1.5 m up.
    cam_to_ego = [[0.0, 0.0, 1.0, 0.0],
                  [-1.0, 0.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0, 1.5],
                  [0.0, 0.0, 0.0, 1.0]]
    scene_graph = {
        "version": 1,
        "sh_degree": SH_DEGREE,
        "time_range": [T0, T1],
        "background_color": [0.62, 0.72, 0.84],
        "nodes": [
            {"name": "background", "kind": "background"},
            {"name": "actor_1", "kind": "actor", "category": "vehicle.car", "size": [4.5, 1.9, 1.5], "track": a1_track},
            {"name": "actor_2", "kind": "actor", "category": "vehicle.car", "size": [4.5, 1.9, 1.5], "track": a2_track,
             "visible_range": [0.0, 7.0]},
        ],
        "ego": {
            "size": [4.5, 1.9, 1.6],
            "camera_to_ego": cam_to_ego,
            "camera": {"width": 512, "height": 288, "fx": 300.0, "fy": 300.0, "cx": 256.0, "cy": 144.0},
            "track": ego_track,
        },
    }

    tensors = {}
    for n in (bg, a1, a2):
        tensors.update(n.tensors())
    write_safetensors(out, tensors, {"format": "m4dgs-scene", "version": "1", "scene_graph": json.dumps(scene_graph)})
    total = sum(n.count for n in (bg, a1, a2))
    print(f"wrote {out}: {total} gaussians (background {bg.count}, actor_1 {a1.count}, actor_2 {a2.count}), "
          f"{os.path.getsize(out) / 1e6:.1f} MB")

    rollout_path = os.path.splitext(out)[0] + ".rollout.json"
    write_rollout(rollout_path, os.path.basename(out), a1_track, a2_track, rng)
    print(f"wrote {rollout_path}")


def write_safetensors(path, tensors, metadata):
    header = {"__metadata__": metadata}
    offset = 0
    names = sorted(tensors)
    for k in names:
        a = np.ascontiguousarray(tensors[k], dtype="<f4")
        header[k] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [offset, offset + a.nbytes]}
        offset += a.nbytes
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for k in names:
            f.write(np.ascontiguousarray(tensors[k], dtype="<f4").tobytes())


def write_rollout(path, scene_name, a1_track, a2_track, rng):
    """Lead car (actor_1) brakes hard at t=2.5 s; the ego reacts late → rear-end collision."""
    dt = 0.1
    steps = []
    ex, ev, ey = 0.0, 8.0, -1.75
    ax, av = 20.0, 7.0
    a1_sim = []
    min_ttc = float("inf")
    collided = False
    yaw = 0.0
    for i in range(81):
        t = i * dt
        # actor_1 dynamics
        a_acc = -4.0 if t >= 2.5 else 0.0
        # ego policy: brake 3 m/s² from 3.6 s, small steering wiggle
        accel = -3.0 if t >= 3.6 and ev > 0 else 0.0
        steer = 0.02 * math.sin(1.3 * t) + (0.05 if 3.6 <= t < 4.6 else 0.0)
        gap = ax - ex - 4.5
        closing = ev - av
        if closing > 1e-3:
            min_ttc = min(min_ttc, max(gap, 0.0) / closing)
        events = []
        if gap <= 0.0:
            events.append("collision:actor_1")
            collided = True
        lat = {"raster": round(float(3.1 + 0.25 * rng.standard_normal()), 3),
               "refine": round(float(18.0 + 1.2 * rng.standard_normal()), 3),
               "policy": round(float(1.2 + 0.1 * rng.standard_normal()), 3)}
        steps.append({"t": round(t, 3), "ego": {"pos": [ex, ey, 0.0], "quat": yaw_quat(yaw).tolist()},
                      "speed": round(ev, 4), "action": {"steer": round(steer, 4), "accel": accel},
                      "latency_ms": lat, "events": events})
        a1_sim.append({"t": round(t, 3), "pos": [ax, -1.75, 0.75], "quat": [1.0, 0.0, 0.0, 0.0]})
        # integrate
        if gap > 0.0:
            ex += ev * dt * math.cos(yaw)
            ey += ev * dt * math.sin(yaw)
            yaw += ev * dt * steer * 0.1
            ev = max(0.0, ev + accel * dt)
            ax += av * dt
            av = max(0.0, av + a_acc * dt)
        else:
            ev, av = 0.0, 0.0
    rollout = {
        "format": "m4dgs-rollout", "version": 1,
        "scene": scene_name,
        "policy": "pilotnet",
        "dt": dt,
        "perturbations": ["actor_1:speed_scale=0.0,t=2.5s (hard brake -4 m/s^2)"],
        "actor_tracks": {"actor_1": a1_sim, "actor_2": a2_track},
        "steps": steps,
        "metrics": {"collision": collided, "min_ttc": round(min_ttc, 3) if min_ttc < float("inf") else None,
                    "max_lateral_error": 0.12, "route_completion": round(ex / 64.0, 3)},
    }
    with open(path, "w") as f:
        json.dump(rollout, f, indent=1)


if __name__ == "__main__":
    main()
