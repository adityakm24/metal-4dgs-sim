"""Programmatic scenario perturbations, parsed from compact spec strings.

Grammar:  <target>:<kind>=<value>[,<param>=<value>...]      (units m, s, m/s, deg are optional suffixes)

  actor_2:lateral_offset=-1.8m,t=2.5s[,ramp=1.0s][,until=6s]   shift along the actor's left axis (cut-in)
  actor_2:cut_in=-3.5m,t=2.5s                                   alias with a sharper 1 s ramp
  actor_1:hard_brake=8,t=3s                                     brake to standstill at 8 m/s^2
  actor_1:velocity_spike=4m/s,t=2s,duration=1s                  sudden speed change
  actor_3:speed_scale=1.5,t=0s     actor_3:time_shift=-1s     actor_3:longitudinal_offset=5m
  actor_2:remove=1                                              delete the actor
  ego:lateral_offset=1.0m   ego:yaw=5deg   ego:speed=12          initial-state perturbation of the ego
  obstacle_1:spawn=30m,t=2s[,lateral=0m]                        static box spawned 30 m ahead of the ego
  camera:occlusion=0.35,t=3s,duration=1s                        synthetic lens occlusion (fraction of view)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from engine.scene_graph import Node, SceneGraph
from engine.scene_graph.tracks import Track, quat_to_R, yaw_quat


@dataclass
class Perturbation:
    target: str
    kind: str
    value: float
    params: dict[str, float] = field(default_factory=dict)
    spec: str = ""

    def p(self, key, default=None):
        return self.params.get(key, default)


_NUM = re.compile(r"^\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*(m/s|mps|m|s|deg|%)?\s*$", re.I)


def _num(s: str) -> float:
    m = _NUM.match(s)
    if not m:
        raise ValueError(f"cannot parse number {s!r}")
    v = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit == "deg":
        v = np.deg2rad(v)
    elif unit == "%":
        v /= 100
    return v


def parse(spec: str) -> list[Perturbation]:
    out = []
    for part in [p for p in re.split(r";", spec) if p.strip()]:
        target, _, rest = part.strip().partition(":")
        if not rest:
            raise ValueError(f"perturbation {part!r} must look like target:kind=value,...")
        items = [kv.strip() for kv in rest.split(",") if kv.strip()]
        kind, _, val = items[0].partition("=")
        params = {}
        for kv in items[1:]:
            k, _, v = kv.partition("=")
            params[k.strip()] = _num(v)
        out.append(Perturbation(target.strip(), kind.strip(), _num(val or "1"), params, part.strip()))
    return out


@dataclass
class EgoInit:
    lateral: float = 0.0
    yaw: float = 0.0
    speed: float | None = None


@dataclass
class ImageEffect:
    kind: str
    t0: float
    t1: float
    amount: float
    seed: int = 0

    def apply(self, img: np.ndarray, t: float) -> np.ndarray:
        if not (self.t0 <= t <= self.t1):
            return img
        h, w = img.shape[:2]
        rng = np.random.default_rng(self.seed)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        mask = np.zeros((h, w), np.float32)
        r = np.sqrt(self.amount * h * w / np.pi)
        for _ in range(3):
            cx, cy = rng.uniform(0.25, 0.75) * w, rng.uniform(0.3, 0.8) * h
            mask = np.maximum(mask, np.clip(1.2 - np.hypot(xx - cx, yy - cy) / (r * rng.uniform(0.5, 0.8)), 0, 1))
        dirt = np.array([0.18, 0.15, 0.12], np.float32)
        return img * (1 - mask[..., None]) + dirt * mask[..., None]


def make_box_node(name: str, size=(1.2, 1.2, 1.0), color=(0.55, 0.42, 0.25), sh_degree: int = 1, spacing=0.08) -> Node:
    from capture.synthetic import _box_surface
    from engine.scene_graph import make_gaussians

    rng = np.random.default_rng(abs(hash(name)) % 2 ** 31)
    pts, rgb, sc, n = _box_surface(rng, (0, 0, 0), size, spacing,
                                   lambda p, nn: np.clip(np.asarray(color) * (0.8 + 0.2 * nn[:, 2:3]) + rng.normal(0, 0.03, p.shape), 0, 1),
                                   open_bottom=False)
    return Node(name, "actor", make_gaussians(pts, rgb, sc, sh_degree=sh_degree, opacity=0.98),
                size=list(size), category="static_object")


def apply(scene: SceneGraph, perts: list[Perturbation]) -> tuple[SceneGraph, EgoInit, list[ImageEffect]]:
    sc = scene.fork()
    ego = EgoInit()
    effects: list[ImageEffect] = []
    for p in perts:
        t = p.p("t", sc.time_range[0])
        if p.target == "ego":
            if p.kind == "lateral_offset":
                ego.lateral = p.value
            elif p.kind == "yaw":
                ego.yaw = p.value
            elif p.kind == "speed":
                ego.speed = p.value
            else:
                raise ValueError(f"unknown ego perturbation {p.kind}")
            continue
        if p.target == "camera":
            if p.kind != "occlusion":
                raise ValueError(f"unknown camera perturbation {p.kind}")
            effects.append(ImageEffect("occlusion", t, t + p.p("duration", 1.0), p.value, seed=len(effects)))
            continue
        if p.kind == "spawn":
            ego_pose = sc.ego.track.pose(t)
            node = make_box_node(p.target, sh_degree=sc.sh_degree)
            pos = ego_pose[:3, 3] + p.value * ego_pose[:3, 0] + p.p("lateral", 0.0) * ego_pose[:3, 1]
            pos[2] = node.size[2] / 2 + p.p("z", ego_pose[2, 3])
            node.track = Track.static(pos, yaw_quat(np.arctan2(ego_pose[1, 0], ego_pose[0, 0])))
            node.visible_range = (t, 1e9)
            sc.nodes.append(node)
            continue
        node = sc.node(p.target)
        tr = node.track
        if p.kind in ("lateral_offset", "cut_in"):
            ramp = p.p("ramp", 1.0 if p.kind == "cut_in" else 1.5)
            node.track = tr.lateral_offset(p.value, t, ramp=ramp, t_end=p.p("until"))
        elif p.kind == "longitudinal_offset":
            node.track = tr.longitudinal_offset(p.value, t, p.p("ramp", 0.0))
        elif p.kind == "speed_scale":
            node.track = tr.speed_scale(p.value, t)
        elif p.kind == "velocity_spike":
            node.track = tr.velocity_spike(p.value, t, p.p("duration", 1.0))
        elif p.kind == "hard_brake":
            node.track = tr.hard_brake(t, p.value)
        elif p.kind == "time_shift":
            node.track = tr.time_shift(p.value)
        elif p.kind == "remove":
            node.visible_range = (1e9, 1e9)
        else:
            raise ValueError(f"unknown actor perturbation {p.kind}")
    return sc, ego, effects
