#!/usr/bin/env python
"""Latency benchmark of each closed-loop stage at the scene's camera resolution (P50 / P90 over N runs)."""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import mlx.core as mx
import numpy as np

from engine.scene_graph import SceneGraph, render
from eval.closed_loop.policies import Observation, make_policy
from eval.closed_loop.simulator import ClosedLoopSim
from models.refiner.refiner import Refiner


def timed(fn, n, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return {"p50": float(np.median(ts)), "p90": float(np.percentile(ts, 90))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--refiner", default=None)
    ap.add_argument("--policy", default="pilotnet")
    ap.add_argument("--policy_ckpt", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="outputs/benchmark.json")
    a = ap.parse_args()
    scene = SceneGraph.load(a.scene)
    c = scene.ego.camera
    t = 0.5 * sum(scene.time_range)
    cam, prev = scene.ego_camera(t), scene.ego_camera(t - 0.1)
    res = {"resolution": f"{c['width']}x{c['height']}", "gaussians": scene.num_gaussians, "device": str(mx.default_device())}

    def r():
        o = render(scene, cam, t, flow_ref=(prev, t - 0.1))
        mx.eval(o.rgb, o.depth, o.alpha, o.flow)
        return o

    res["raster_rgb_depth_flow_ms"] = timed(r, a.n)
    out = r()
    res["rendered_tile_pairs"] = out.num_rendered
    if a.refiner:
        ref = Refiner.load(a.refiner)
        res["refiner_steps"] = ref.steps
        res["refine_ms"] = timed(lambda: mx.eval(ref.refine(out)), a.n)
    pol = make_policy(a.policy, a.policy_ckpt)
    obs = Observation(np.clip(np.array(out.rgb), 0, 1), 8.0, t)
    res["policy_ms"] = timed(lambda: pol.act(obs), a.n)
    sim = ClosedLoopSim(scene, make_policy("pilotnet", a.policy_ckpt) if a.policy == "pilotnet" else pol,
                        Refiner.load(a.refiner) if a.refiner else None)
    roll = sim.run("", duration=5.0)
    res["closed_loop_step_ms_p50"] = roll.metrics["latency_ms_p50"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
