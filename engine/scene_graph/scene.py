"""Editable 4DGS scene graph: a static/4D background node plus rigid actor nodes on tracks.

Serialised as a single safetensors checkpoint (see docs/formats.md §3).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from engine.metal_splat.camera import Camera
from engine.metal_splat.project import eval_sh, normalize, quat_multiply

from . import gaussians as G
from .tracks import Track, R_to_quat


@dataclass
class Node:
    name: str
    kind: str                                  # "background" | "actor"
    params: dict[str, mx.array]
    track: Track | None = None                 # actor_to_world over time (None for background)
    size: list[float] | None = None            # box l, w, h
    category: str = ""
    visible_range: tuple[float, float] | None = None   # optional: actor only exists in [t0, t1]

    @property
    def num_gaussians(self) -> int:
        return G.count(self.params)

    def pose(self, t: float) -> np.ndarray:
        return np.eye(4) if self.track is None else self.track.pose(t)

    def visible(self, t: float) -> bool:
        return self.visible_range is None or (self.visible_range[0] <= t <= self.visible_range[1])


@dataclass
class Ego:
    track: Track
    camera: dict                                # width, height, fx, fy, cx, cy
    camera_to_ego: np.ndarray                   # (4,4) OpenCV camera in ego frame
    size: list[float] = field(default_factory=lambda: [4.5, 1.9, 1.6])

    def camera_at(self, ego_to_world: np.ndarray) -> Camera:
        c = self.camera
        return Camera(int(c["width"]), int(c["height"]), float(c["fx"]), float(c["fy"]), float(c["cx"]),
                      float(c["cy"]), ego_to_world @ self.camera_to_ego)


def default_camera_to_ego(height: float = 1.5, forward: float = 0.0) -> np.ndarray:
    """Forward-looking camera: cam z -> ego +x, cam x -> ego -y, cam y -> ego -z."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    T[:3, 3] = [forward, 0.0, height]
    return T


def _eval_node_impl(degree: int, with_color: bool):
    def f(means, velocity, t_center, t_log_scale, opacity_logits, quats, sh, R, p, qn, cam, t):
        dt = t - t_center[:, 0]
        mu = means + velocity * dt[:, None]
        op = mx.sigmoid(opacity_logits[:, 0]) * mx.exp(-0.5 * mx.square(dt / mx.exp(t_log_scale[:, 0])))
        mu = mu @ R.T + p
        q = quat_multiply(qn[None], normalize(quats))
        if not with_color:
            return mu, q, op
        dirs = normalize(mu - cam[None]) @ R      # world -> node-local: R^T d (row-vector form)
        return mu, q, op, eval_sh(sh, dirs, degree)

    return f


_EVAL_CACHE: dict = {}


def _eval_node(degree: int, with_color: bool):
    key = (degree, with_color)
    if key not in _EVAL_CACHE:
        _EVAL_CACHE[key] = mx.compile(_eval_node_impl(degree, with_color))
    return _EVAL_CACHE[key]


class SceneGraph:
    def __init__(self, nodes: list[Node], ego: Ego, sh_degree: int = 1, time_range=(0.0, 1.0),
                 background_color=(0.0, 0.0, 0.0)):
        self.nodes = nodes
        self.ego = ego
        self.sh_degree = sh_degree
        self.time_range = tuple(float(v) for v in time_range)
        self.background_color = tuple(float(v) for v in background_color)

    # ------------------------------------------------------------------ access
    def node(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"no node named {name!r}; have {[n.name for n in self.nodes]}")

    @property
    def actors(self) -> list[Node]:
        return [n for n in self.nodes if n.kind == "actor"]

    @property
    def num_gaussians(self) -> int:
        return sum(n.num_gaussians for n in self.nodes)

    def ego_camera(self, t: float | None = None, ego_to_world: np.ndarray | None = None) -> Camera:
        if ego_to_world is None:
            ego_to_world = self.ego.track.pose(self.time_range[0] if t is None else t)
        return self.ego.camera_at(ego_to_world)

    def fork(self) -> "SceneGraph":
        """Copy of the graph (tracks, metadata) that shares Gaussian tensors — cheap per-rollout edits."""
        out = copy.copy(self)
        out.nodes = [copy.copy(n) for n in self.nodes]
        for n in out.nodes:
            n.track = None if n.track is None else n.track.copy()
        out.ego = copy.copy(self.ego)
        out.ego.track = self.ego.track.copy()
        return out

    # ------------------------------------------------------------------ 4D evaluation
    def evaluate(self, t: float, cam_center: np.ndarray, params: dict[str, dict] | None = None,
                 with_color: bool = True) -> dict[str, mx.array]:
        """World-space Gaussians at time t (differentiable w.r.t. params)."""
        out = {k: [] for k in ("means", "quats", "log_scales", "opacities", "rgb")}
        c = mx.array(np.asarray(cam_center, dtype=np.float32))
        tt = mx.array(float(t), dtype=mx.float32)
        fn = _eval_node(self.sh_degree, with_color)
        for node in self.nodes:
            if not node.visible(t):
                continue
            g = params[node.name] if params is not None and node.name in params else node.params
            if G.count(g) == 0:
                continue
            T = node.pose(t)
            R = mx.array(T[:3, :3].astype(np.float32))
            p = mx.array(T[:3, 3].astype(np.float32))
            qn = mx.array(R_to_quat(T[:3, :3]).astype(np.float32))
            res = fn(g["means"], g["velocity"], g["t_center"], g["t_log_scale"], g["opacity_logits"], g["quats"],
                     g["sh"], R, p, qn, c, tt)
            out["means"].append(res[0])
            out["quats"].append(res[1])
            out["log_scales"].append(g["log_scales"])
            out["opacities"].append(res[2])
            if with_color:
                out["rgb"].append(res[3])
        if not out["means"]:
            raise ValueError("scene has no visible Gaussians at t=%.3f" % t)
        return {k: mx.concatenate(v, axis=0) for k, v in out.items() if v}

    def positions(self, t: float, params: dict[str, dict] | None = None, visible_at: float | None = None) -> mx.array:
        """World means at t, in the same order as `evaluate(visible_at)` (for optical flow)."""
        res = []
        vt = t if visible_at is None else visible_at
        for node in self.nodes:
            if not node.visible(vt):
                continue
            g = params[node.name] if params is not None and node.name in params else node.params
            if G.count(g) == 0:
                continue
            mu = g["means"] + g["velocity"] * (t - g["t_center"][:, 0])[:, None]
            if node.track is not None:
                T = node.track.pose(t)
                mu = mu @ mx.array(T[:3, :3].astype(np.float32)).T + mx.array(T[:3, 3].astype(np.float32))
            res.append(mu)
        return mx.concatenate(res, axis=0)

    # ------------------------------------------------------------------ serialisation
    def graph_json(self) -> dict:
        nodes = []
        for n in self.nodes:
            d = {"name": n.name, "kind": n.kind}
            if n.kind == "actor":
                d.update({"category": n.category, "size": list(map(float, n.size or [4.5, 1.9, 1.6])),
                          "track": n.track.to_json() if n.track is not None else Track.static().to_json()})
                if n.visible_range is not None:
                    d["visible_range"] = list(map(float, n.visible_range))
            nodes.append(d)
        return {
            "version": 1,
            "sh_degree": self.sh_degree,
            "time_range": list(self.time_range),
            "background_color": list(self.background_color),
            "nodes": nodes,
            "ego": {
                "size": list(map(float, self.ego.size)),
                "camera_to_ego": self.ego.camera_to_ego.tolist(),
                "camera": self.ego.camera,
                "track": self.ego.track.to_json(),
            },
        }

    def apply_graph_json(self, g: dict) -> None:
        """Apply (possibly edited) graph metadata; tensors are kept."""
        self.sh_degree = int(g.get("sh_degree", self.sh_degree))
        self.time_range = tuple(g.get("time_range", self.time_range))
        self.background_color = tuple(g.get("background_color", self.background_color))
        by_name = {n.name: n for n in self.nodes}
        for d in g["nodes"]:
            n = by_name.get(d["name"])
            if n is None:
                continue
            if d.get("kind") == "actor":
                n.track = Track.from_json(d["track"])
                n.size = d.get("size", n.size)
                n.category = d.get("category", n.category)
                vr = d.get("visible_range")
                n.visible_range = tuple(vr) if vr else None
        e = g["ego"]
        self.ego = Ego(Track.from_json(e["track"]), dict(e["camera"]), np.array(e["camera_to_ego"], dtype=np.float64),
                       list(e.get("size", self.ego.size if self.ego else [4.5, 1.9, 1.6])))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for n in self.nodes:
            p = G.pad_sh(n.params, self.sh_degree)
            for k in G.PARAM_KEYS:
                arrays[f"{n.name}.{k}"] = p[k].astype(mx.float32)
        meta = {"format": "m4dgs-scene", "version": "1", "scene_graph": json.dumps(self.graph_json())}
        mx.save_safetensors(str(path), arrays, metadata=meta)
        return path

    @staticmethod
    def edits_path(path: str | Path) -> Path:
        path = Path(path)
        return path.with_name(path.name.removesuffix(".safetensors") + ".edits.json")

    @staticmethod
    def load(path: str | Path, edits: str | Path | bool | None = None) -> "SceneGraph":
        """Load a checkpoint. edits=None: auto-apply `<scene>.edits.json` if present; False: ignore."""
        arrays, meta = mx.load(str(path), return_metadata=True)
        if meta.get("format") != "m4dgs-scene":
            raise ValueError(f"{path} is not an m4dgs scene checkpoint")
        g = json.loads(meta["scene_graph"])
        nodes = []
        for d in g["nodes"]:
            params = {k: arrays[f"{d['name']}.{k}"] for k in G.PARAM_KEYS}
            nodes.append(Node(d["name"], d["kind"], params))
        scene = SceneGraph(nodes, None, int(g["sh_degree"]), g["time_range"], g.get("background_color", (0, 0, 0)))
        scene.apply_graph_json(g)
        if edits is None:
            ep = SceneGraph.edits_path(path)
            edits = ep if ep.exists() else False
        if edits:
            scene.apply_graph_json(json.loads(Path(edits).read_text()))
        return scene
