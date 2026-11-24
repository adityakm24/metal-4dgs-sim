"""Closed-loop rollout engine:  4DGS render -> refiner -> policy -> bicycle model -> new ego pose."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from engine.scene_graph import SceneGraph, render
from engine.scene_graph.tracks import R_to_quat, quat_yaw
from eval.metrics.geometry import warping_error
from eval.metrics.safety import ReferencePath, obb_overlap, time_to_collision
from models.refiner.refiner import IdentityRefiner

from . import perturbations as P
from .dynamics import Action, BicycleModel, EgoState
from .policies import Observation, Policy


@dataclass
class StepRecord:
    t: float
    ego: EgoState
    action: Action
    latency_ms: dict
    events: list
    min_ttc: float
    lateral_error: float
    frames: dict | None = None      # raw/refined/depth/flow numpy (only when keep_frames)


@dataclass
class RolloutResult:
    steps: list[StepRecord]
    metrics: dict
    perturbations: list[str]
    scene: SceneGraph
    policy: str
    dt: float
    scene_path: str = ""
    actor_tracks: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "format": "m4dgs-rollout", "version": 1, "scene": self.scene_path, "policy": self.policy, "dt": self.dt,
            "perturbations": self.perturbations,
            "actor_tracks": {n.name: n.track.resample(self.dt).to_json() for n in self.scene.actors if n.track is not None},
            "steps": [{
                "t": s.t,
                "ego": {"pos": [s.ego.x, s.ego.y, s.ego.z], "quat": R_to_quat(s.ego.pose()[:3, :3]).tolist()},
                "speed": s.ego.v,
                "action": {"steer": s.action.steer, "accel": s.action.accel},
                "latency_ms": s.latency_ms, "events": s.events,
                "min_ttc": None if not np.isfinite(s.min_ttc) else s.min_ttc,
                "lateral_error": s.lateral_error,
            } for s in self.steps],
            "metrics": self.metrics,
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))
        return path


class ClosedLoopSim:
    def __init__(self, scene: SceneGraph, policy: Policy, refiner=None, dt: float = 0.1,
                 dynamics: BicycleModel | None = None, offroad_threshold: float = 2.5):
        self.base_scene = scene
        self.policy = policy
        self.refiner = refiner or IdentityRefiner()
        self.dt = dt
        self.dyn = dynamics or BicycleModel()
        self.offroad = offroad_threshold
        tr = scene.ego.track.resample(0.1)
        self.route = ReferencePath(tr.pos)

    def run(self, perturbations: str | list[P.Perturbation] = "", duration: float | None = None,
            keep_frames: bool = False, on_step=None, stop_on_collision: bool = True,
            measure_warp: bool = False) -> RolloutResult:
        perts = P.parse(perturbations) if isinstance(perturbations, str) else list(perturbations)
        scene, ego_init, effects = P.apply(self.base_scene, perts)
        t0, t1 = scene.time_range
        if duration is not None:
            t1 = min(t1, t0 + duration)
        # initial ego state from the logged track (+ perturbation)
        E = scene.ego.track.pose(t0)
        v0 = float(np.linalg.norm(scene.ego.track.velocity(t0 + 0.05)[:2]))
        yaw = quat_yaw(R_to_quat(E[:3, :3]))
        left = E[:3, 1]
        state = EgoState(E[0, 3] + ego_init.lateral * left[0], E[1, 3] + ego_init.lateral * left[1],
                         yaw + ego_init.yaw, ego_init.speed if ego_init.speed is not None else v0, E[2, 3])
        self.policy.reset()
        ego_size = scene.ego.size
        steps: list[StepRecord] = []
        prev_cam, prev_t, prev_img = None, None, None
        collided, offroad, min_ttc_all, warp_errs = None, False, np.inf, []
        t = t0
        while t <= t1 + 1e-9:
            ego_pose = state.pose()
            cam = scene.ego_camera(ego_to_world=ego_pose)
            lat = {}
            tic = time.perf_counter()
            out = render(scene, cam, t, flow_ref=(prev_cam, prev_t) if prev_cam is not None else None, realtime=True)
            mx.eval(out.rgb, out.depth, out.alpha, *( [out.flow] if out.flow is not None else []))
            lat["raster"] = (time.perf_counter() - tic) * 1e3
            tic = time.perf_counter()
            refined = self.refiner.refine(out)
            mx.eval(refined)
            lat["refine"] = (time.perf_counter() - tic) * 1e3
            img = np.array(refined)
            for fx in effects:
                img = fx.apply(img, t)
            # privileged state for the expert + safety monitors
            actors, events, step_ttc = [], [], np.inf
            for n in scene.actors:
                if not n.visible(t) or n.track is None:
                    continue
                pose, vel = n.track.pose(t), n.track.velocity(t)
                actors.append((n.name, pose, n.size, vel))
                if obb_overlap(ego_pose, ego_size, pose, n.size):
                    events.append(f"collision:{n.name}")
                    collided = collided or n.name
                step_ttc = min(step_ttc, time_to_collision(ego_pose, ego_size, state.v, pose, n.size, vel))
            s_route, lat_err, _ = self.route.project([state.x, state.y])
            if abs(lat_err) > self.offroad:
                events.append("offroad")
                offroad = True
            min_ttc_all = min(min_ttc_all, step_ttc)
            if measure_warp and prev_img is not None and out.flow is not None:
                warp_errs.append(warping_error(img, prev_img, np.array(out.flow), np.array(out.alpha) > 0.5))
            obs = Observation(img, state.v, t, state, actors, self.route, np.array(out.depth))
            tic = time.perf_counter()
            action = self.policy.act(obs)
            lat["policy"] = (time.perf_counter() - tic) * 1e3
            lat["total"] = lat["raster"] + lat["refine"] + lat["policy"]
            frames = None
            if keep_frames or on_step is not None:
                frames = {"raw": np.clip(np.array(out.rgb), 0, 1), "refined": img, "depth": np.array(out.depth),
                          "flow": None if out.flow is None else np.array(out.flow)}
            rec = StepRecord(t, state, action, lat, events, step_ttc, lat_err, frames if keep_frames else None)
            steps.append(rec)
            if on_step is not None:
                on_step(rec, frames, scene, self)
            if collided and stop_on_collision:
                break
            prev_cam, prev_t, prev_img = cam, t, img
            state = self.dyn.step(state, action, self.dt)
            state.z = float(scene.ego.track.position(t + self.dt)[2])
            t = round(t + self.dt, 6)
        s_end, _, _ = self.route.project([state.x, state.y])
        s_start, _, _ = self.route.project([steps[0].ego.x, steps[0].ego.y])
        lat_all = [s.latency_ms for s in steps[1:]] or [s.latency_ms for s in steps]
        metrics = {
            "collision": bool(collided), "collided_with": collided or None, "offroad": offroad,
            "min_ttc": None if not np.isfinite(min_ttc_all) else float(min_ttc_all),
            "max_lateral_error": float(max(abs(s.lateral_error) for s in steps)),
            "route_completion": float(np.clip((s_end - s_start) / max(self.route.length - s_start, 1e-6), 0, 1)),
            "duration": float(steps[-1].t - steps[0].t),
            "latency_ms_p50": {k: float(np.median([l[k] for l in lat_all])) for k in lat_all[0]},
        }
        if warp_errs:
            metrics["warp_error"] = float(np.nanmean(warp_errs))
        return RolloutResult(steps, metrics, [p.spec for p in perts], scene, self.policy.name, self.dt)
