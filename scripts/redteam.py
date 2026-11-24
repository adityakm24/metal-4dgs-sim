#!/usr/bin/env python
"""Adversarial red-teaming: CEM search over cut-ins, hard brakes, velocity spikes, occlusions and spawned
obstacles to find closed-loop failures of a policy. Failures are logged to JSONL."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from engine.scene_graph import SceneGraph
from eval.closed_loop.policies import make_policy
from eval.closed_loop.redteam import RedTeamConfig, RedTeamer
from eval.closed_loop.simulator import ClosedLoopSim
from models.refiner.refiner import Refiner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--refiner", default=None)
    ap.add_argument("--policy", default="pilotnet")
    ap.add_argument("--policy_ckpt", default="checkpoints/pilotnet.safetensors")
    ap.add_argument("--families", default="cut_in,hard_brake,velocity_spike,occlusion,obstacle")
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--log", default="outputs/redteam.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true", help="keep existing log entries (summary covers all)")
    a = ap.parse_args()
    if not a.append:
        Path(a.log).unlink(missing_ok=True)
    sim = ClosedLoopSim(SceneGraph.load(a.scene), make_policy(a.policy, a.policy_ckpt),
                        Refiner.load(a.refiner) if a.refiner else None)
    rt = RedTeamer(sim, RedTeamConfig(tuple(a.families.split(",")), a.population, max(1, a.population // 3),
                                      a.iterations, a.seed), a.log)
    if a.append and Path(a.log).exists():
        rt.results = [json.loads(l) for l in Path(a.log).read_text().splitlines() if l.strip()]
    rt.search()
    summary = rt.summary()
    Path(a.log).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
