"""4DGS optimisation: static/4D background + rigid actor nodes, Adam, adaptive density control."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from capture.dataset import Dataset
from engine.scene_graph import Ego, Node, SceneGraph, default_camera_to_ego, make_gaussians, render
from engine.scene_graph import gaussians as G
from engine.scene_graph.gaussians import estimate_scales_knn
from engine.scene_graph.segmentation import segment_points
from engine.scene_graph.tracks import Track

from .init import initial_points, scene_extent
from .losses import inverse_depth_l1, photometric, psnr


@dataclass
class TrainConfig:
    iterations: int = 7000
    sh_degree: int = 1
    max_width: int = 640
    lr_means: float = 1.6e-4          # x scene extent, decays 100x
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacity: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    lr_velocity: float = 1e-4
    lr_time: float = 1e-3
    lr_bg: float = 1e-2
    ssim_lambda: float = 0.2
    depth_weight: float = 0.1
    velocity_l1: float = 1e-3
    densify_from: int = 500
    densify_until_frac: float = 0.5
    densify_every: int = 100
    densify_grad: float = 2e-4         # NDC-space positional gradient threshold
    opacity_reset_every: int = 3000
    max_gaussians: int = 800_000
    sh_up_every: int = 1000
    holdout_every: int = 8
    max_init_points: int = 300_000
    mono_depth: bool = False
    seed: int = 0
    learn_background: bool = True
    dynamic_background: bool = True    # per-Gaussian velocity / temporal opacity on the background
    log_every: int = 100


class Adam:
    """Adam over a dict-of-dicts of arrays whose rows can be re-indexed (densify/prune)."""

    def __init__(self, b1=0.9, b2=0.999, eps=1e-15):
        self.b1, self.b2, self.eps = b1, b2, eps
        self.state: dict[tuple, tuple[mx.array, mx.array]] = {}
        self.t = 0

    def step(self, params: dict, grads: dict, lrs: dict) -> dict:
        self.t += 1
        out = {}
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for node, g in params.items():
            out[node] = {}
            for k, p in g.items():
                lr = lrs.get((node, k), lrs.get(k, 0.0))
                gr = grads.get(node, {}).get(k)
                if lr == 0.0 or gr is None:
                    out[node][k] = p
                    continue
                m, v = self.state.get((node, k), (mx.zeros_like(p), mx.zeros_like(p)))
                m = self.b1 * m + (1 - self.b1) * gr
                v = self.b2 * v + (1 - self.b2) * gr * gr
                self.state[(node, k)] = (m, v)
                out[node][k] = p - lr * (m / bc1) / (mx.sqrt(v / bc2) + self.eps)
        return out

    def remap(self, node: str, idx: np.ndarray, n_new_zero: int = 0):
        """Keep rows `idx` (and append n_new_zero zeroed rows) for every state of `node`."""
        ix = mx.array(idx.astype(np.int32))
        for (nd, k), (m, v) in list(self.state.items()):
            if nd != node:
                continue
            m, v = m[ix], v[ix]
            if n_new_zero:
                z = mx.zeros((n_new_zero,) + m.shape[1:], dtype=m.dtype)
                m, v = mx.concatenate([m, z]), mx.concatenate([v, z])
            self.state[(nd, k)] = (m, v)


def _image_size(ds: Dataset, max_width: int) -> tuple[int, int]:
    c = ds.frames[0].camera
    if c.width <= max_width:
        return c.width, c.height
    s = max_width / c.width
    return int(round(c.width * s)), int(round(c.height * s))


def build_initial_scene(ds: Dataset, cfg: TrainConfig, size: tuple[int, int]) -> SceneGraph:
    pts = initial_points(ds, cfg.max_init_points, cfg.mono_depth, cfg.seed, image_size=size)
    t_obs = pts[:, 6]
    actors = [(a.name, a.track, a.size) for a in ds.actors]
    if actors:
        tt = np.where(np.isnan(t_obs), ds.time_range[0], t_obs)
        labels, local = segment_points(pts[:, :3], tt, actors)
    else:
        labels, local = np.full(len(pts), -1), pts[:, :3]
    t0, t1 = ds.time_range
    nodes = []
    bg = labels < 0
    xyz = pts[bg, :3]
    sc = estimate_scales_knn(xyz)
    nodes.append(Node("background", "background",
                      make_gaussians(xyz, np.clip(pts[bg, 3:6], 0, 1), sc[:, None].repeat(3, 1), cfg.sh_degree,
                                     opacity=0.1, t_center=(t0 + t1) / 2)))
    rng = np.random.default_rng(cfg.seed)
    for ai, a in enumerate(ds.actors):
        sel = labels == ai
        lp, col = local[sel], np.clip(pts[sel, 3:6], 0, 1)
        if len(lp) < 500:   # too few observed points: fill the box
            n_fill = 2000
            fill = (rng.uniform(-0.5, 0.5, (n_fill, 3)) * np.asarray(a.size))
            lp = np.concatenate([lp, fill])
            col = np.concatenate([col, np.full((n_fill, 3), 0.5)])
        scl = estimate_scales_knn(lp)
        nodes.append(Node(a.name, "actor", make_gaussians(lp, col, scl[:, None].repeat(3, 1), cfg.sh_degree, opacity=0.1),
                          track=a.track, size=list(a.size), category=a.category, visible_range=a.visible_range))
    cam0 = ds.frames[0].camera.scaled(*size)
    ego_track = ds.ego_track
    c2e = ds.camera_to_ego
    if ego_track is None:   # handheld capture: ego == camera rig
        ego_track = Track([f.t for f in ds.frames], [f.camera.center for f in ds.frames],
                          [np.array([1, 0, 0, 0])] * len(ds.frames))
        from engine.scene_graph.tracks import R_to_quat
        ego_track.quat = np.stack([R_to_quat(f.camera.c2w[:3, :3]) for f in ds.frames])
        c2e = np.eye(4)
    ego = Ego(ego_track, cam0.to_json(), c2e if c2e is not None else default_camera_to_ego())
    bgc = ds.background_color or (0.5, 0.5, 0.5)
    return SceneGraph(nodes, ego, sh_degree=cfg.sh_degree, time_range=(t0, t1), background_color=bgc)


class Trainer:
    def __init__(self, ds: Dataset, cfg: TrainConfig):
        self.ds, self.cfg = ds, cfg
        self.size = _image_size(ds, cfg.max_width)
        self.train_frames, self.test_frames = ds.split(cfg.holdout_every)
        self.scene = build_initial_scene(ds, cfg, self.size)
        self.center, self.extent = scene_extent(ds)
        self.params = {n.name: dict(n.params) for n in self.scene.nodes}
        self.bg = mx.array(np.asarray(self.scene.background_color, dtype=np.float32))
        self.opt = Adam()
        self.active_sh = 0
        self.rng = np.random.default_rng(cfg.seed)
        self._reset_stats()
        self._cache = {}
        print(f"[train] {len(self.train_frames)} train / {len(self.test_frames)} test frames @ {self.size}, "
              f"{self.scene.num_gaussians} initial Gaussians, extent {self.extent:.1f} m")

    # ------------------------------------------------------------------ helpers
    def _reset_stats(self):
        self.grad_accum = {k: np.zeros(G.count(v)) for k, v in self.params.items()}
        self.grad_denom = {k: np.zeros(G.count(v)) for k, v in self.params.items()}

    def _lrs(self, it: int) -> dict:
        c = self.cfg
        frac = min(it / max(c.iterations, 1), 1.0)
        lr_mean = c.lr_means * self.extent * math.exp(math.log(0.01) * frac)
        lrs = {"means": lr_mean, "log_scales": c.lr_scales, "quats": c.lr_quats, "opacity_logits": c.lr_opacity,
               "sh": c.lr_sh0, "velocity": c.lr_velocity if c.dynamic_background else 0.0,
               "t_center": c.lr_time if c.dynamic_background else 0.0,
               "t_log_scale": c.lr_time if c.dynamic_background else 0.0}
        for n in self.scene.nodes:
            if n.kind == "actor":   # rigid actors: no per-Gaussian motion
                for k in ("velocity", "t_center", "t_log_scale"):
                    lrs[(n.name, k)] = 0.0
        return lrs

    def _target(self, frame):
        key = id(frame)
        if key not in self._cache:
            img = mx.array(frame.image(self.size))
            d = frame.depth(self.size)
            self._cache[key] = (img, mx.array(d) if d is not None else None)
        return self._cache[key]

    def _visible_nodes(self, t):
        return [n.name for n in self.scene.nodes if n.visible(t) and G.count(self.params[n.name]) > 0]

    def _sh_mask(self, sh: mx.array) -> mx.array:
        k = (self.active_sh + 1) ** 2
        if k >= sh.shape[1]:
            return sh
        mask = mx.array((np.arange(sh.shape[1]) < k).astype(np.float32))[None, :, None]
        return sh * mask

    # ------------------------------------------------------------------ step
    def _loss_fn(self, params, bg_logit, offset, frame, gt, gt_depth):
        p = {k: {**v, "sh": self._sh_mask(v["sh"])} for k, v in params.items()}
        bg = mx.sigmoid(bg_logit) if self.cfg.learn_background else self.bg
        cam = frame.camera.scaled(*self.size)
        out = render(self.scene, cam, frame.t, params=p, background=bg, means2d_offset=offset)
        loss = photometric(out.rgb, gt, self.cfg.ssim_lambda)
        if gt_depth is not None and self.cfg.depth_weight > 0:
            loss = loss + self.cfg.depth_weight * inverse_depth_l1(out.depth, gt_depth)
        if self.cfg.velocity_l1 > 0 and "background" in params:
            loss = loss + self.cfg.velocity_l1 * mx.mean(mx.abs(params["background"]["velocity"]))
        return loss, (out.rgb, out.radii)

    def train(self, callback=None) -> SceneGraph:
        c = self.cfg
        bg_logit = mx.array(np.log(np.clip(self.bg, 1e-3, 1 - 1e-3) / (1 - np.clip(self.bg, 1e-3, 1 - 1e-3))))
        bg_m = mx.zeros(3)
        bg_v = mx.zeros(3)
        grad_fn = mx.value_and_grad(self._loss_fn, argnums=(0, 1, 2))
        t_start = time.perf_counter()
        densify_until = int(c.iterations * c.densify_until_frac)
        order = []
        for it in range(1, c.iterations + 1):
            if not order:
                order = list(self.rng.permutation(len(self.train_frames)))
            frame = self.train_frames[order.pop()]
            gt, gt_depth = self._target(frame)
            vis = self._visible_nodes(frame.t)
            n_vis = sum(G.count(self.params[v]) for v in vis)
            offset = mx.zeros((n_vis, 2))
            (loss, (pred, radii)), (g_params, g_bg, g_off) = grad_fn(self.params, bg_logit, offset, frame, gt, gt_depth)
            self.params = self.opt.step(self.params, g_params, self._lrs(it))
            if c.learn_background:
                bg_m = 0.9 * bg_m + 0.1 * g_bg
                bg_v = 0.999 * bg_v + 0.001 * g_bg * g_bg
                bg_logit = bg_logit - c.lr_bg * (bg_m / (1 - 0.9 ** it)) / (mx.sqrt(bg_v / (1 - 0.999 ** it)) + 1e-15)
            mx.eval(self.params, bg_logit, loss, g_off, radii)

            if it < densify_until:
                gn = np.linalg.norm(np.array(g_off), axis=1) * 0.5 * max(self.size)
                rr = np.array(radii) > 0
                o = 0
                for name in vis:
                    n = G.count(self.params[name])
                    self.grad_accum[name] += np.where(rr[o:o + n], gn[o:o + n], 0)
                    self.grad_denom[name] += rr[o:o + n]
                    o += n
                if it >= c.densify_from and it % c.densify_every == 0:
                    self.densify_and_prune()
                if it % c.opacity_reset_every == 0:
                    self.reset_opacity()
            if it % c.sh_up_every == 0 and self.active_sh < c.sh_degree:
                self.active_sh += 1
            if it % c.log_every == 0 or it == 1:
                n_tot = sum(G.count(v) for v in self.params.values())
                print(f"[train] it {it:5d}  loss {loss.item():.4f}  psnr {psnr(pred, gt):5.2f}  "
                      f"gaussians {n_tot}  {(time.perf_counter() - t_start) / it * 1e3:.0f} ms/it", flush=True)
            if callback is not None:
                callback(it, self)
        self.bg = mx.sigmoid(bg_logit) if c.learn_background else self.bg
        return self.export()

    # ------------------------------------------------------------------ density control
    def densify_and_prune(self):
        c = self.cfg
        total = sum(G.count(v) for v in self.params.values())
        for name, g in list(self.params.items()):
            n = G.count(g)
            if n == 0:
                continue
            avg = self.grad_accum[name] / np.maximum(self.grad_denom[name], 1)
            scales = np.exp(np.array(g["log_scales"]))
            smax = scales.max(1)
            ext = self.extent if name == "background" else max(self.scene.node(name).size or [5.0])
            big = smax > 0.01 * ext
            hot = avg >= c.densify_grad
            if total >= c.max_gaussians:
                hot[:] = False
            clone = hot & ~big
            split = hot & big
            opac = 1 / (1 + np.exp(-np.array(g["opacity_logits"])[:, 0]))
            prune = (opac < 0.005) | (smax > 0.5 * ext) | split
            keep = np.where(~prune)[0]
            new = {}
            # clones: same parameters
            ci = np.where(clone)[0]
            # splits: 2 samples inside each, scale / 1.6
            si = np.where(split)[0]
            parts = []
            if len(ci):
                parts.append(G.take(g, ci))
            if len(si):
                s = G.take(g, np.repeat(si, 2))
                from engine.metal_splat.project import quat_to_rotmat

                R = np.array(quat_to_rotmat(s["quats"]))
                sc = np.exp(np.array(s["log_scales"]))
                samp = self.rng.normal(size=sc.shape) * sc
                offs = np.einsum("nij,nj->ni", R, samp)
                s["means"] = s["means"] + mx.array(offs.astype(np.float32))
                s["log_scales"] = s["log_scales"] - math.log(1.6)
                parts.append(s)
            kept = G.take(g, keep)
            new = G.concat(kept, *parts) if parts else kept
            n_add = G.count(new) - len(keep)
            self.opt.remap(name, keep, n_add)
            self.params[name] = new
            total += G.count(new) - n
        self._reset_stats()

    def reset_opacity(self):
        cap = float(np.log(0.01 / 0.99))
        for name, g in self.params.items():
            g["opacity_logits"] = mx.minimum(g["opacity_logits"], cap)
            for (nd, k) in list(self.opt.state):
                if nd == name and k == "opacity_logits":
                    m, v = self.opt.state[(nd, k)]
                    self.opt.state[(nd, k)] = (mx.zeros_like(m), mx.zeros_like(v))

    # ------------------------------------------------------------------ output
    def export(self) -> SceneGraph:
        for n in self.scene.nodes:
            n.params = self.params[n.name]
        self.scene.background_color = tuple(float(v) for v in np.array(self.bg))
        return self.scene

    def evaluate(self, frames=None) -> dict:
        frames = self.test_frames if frames is None else frames
        if not frames:
            return {}
        scene = self.export()
        vals = []
        for f in frames:
            gt, _ = self._target(f)
            out = render(scene, f.camera.scaled(*self.size), f.t)
            vals.append(psnr(out.rgb, gt))
        return {"psnr": float(np.mean(vals)), "frames": len(frames)}
