#!/usr/bin/env python
"""Run one closed-loop adversarial simulation with the 3-panel telemetry visualizer.

Example:
  python eval/closed_loop/run_sim.py --scene checkpoints/scene_01.safetensors \
      --refiner checkpoints/flow_refiner.safetensors --policy pilotnet --policy_ckpt checkpoints/pilotnet.safetensors \
      --perturbation "actor_2:lateral_offset=-1.8m,t=2.5s" --record demo_out.mp4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.scene_graph import SceneGraph  # noqa: E402
from eval.closed_loop.policies import make_policy  # noqa: E402
from eval.closed_loop.simulator import ClosedLoopSim  # noqa: E402
from models.refiner.refiner import Refiner  # noqa: E402
from ui.telemetry import TelemetryUI  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--refiner", default=None, help="flow-matching refiner checkpoint (omit for raw splats)")
    ap.add_argument("--refiner_steps", type=int, default=None)
    ap.add_argument("--policy", default="expert", help="expert | pilotnet | openpilot")
    ap.add_argument("--policy_ckpt", default=None)
    ap.add_argument("--perturbation", action="append", default=[], help="repeatable; see eval/closed_loop/perturbations.py")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--record", default=None, help="write the 3-panel video (.mp4)")
    ap.add_argument("--show", action="store_true", help="open a live OpenCV window")
    ap.add_argument("--rollout", default=None, help="rollout JSON path (default next to --record or outputs/)")
    ap.add_argument("--no_edits", action="store_true", help="ignore <scene>.edits.json from the Mac app")
    a = ap.parse_args()

    scene = SceneGraph.load(a.scene, edits=False if a.no_edits else None)
    refiner = Refiner.load(a.refiner, steps=a.refiner_steps) if a.refiner else None
    policy = make_policy(a.policy, a.policy_ckpt)
    sim = ClosedLoopSim(scene, policy, refiner, dt=a.dt)
    ui = TelemetryUI(a.record, a.show, fps=1.0 / a.dt) if (a.record or a.show) else None
    res = sim.run(";".join(a.perturbation), duration=a.duration, on_step=ui, measure_warp=True)
    if ui:
        ui.close()
    res.scene_path = str(a.scene)
    out = a.rollout or (str(Path(a.record).with_suffix(".rollout.json")) if a.record else "outputs/last.rollout.json")
    res.save(out)
    print(json.dumps(res.metrics, indent=1))
    print(f"rollout -> {out}" + (f", video -> {a.record}" if a.record else ""))


if __name__ == "__main__":
    main()
