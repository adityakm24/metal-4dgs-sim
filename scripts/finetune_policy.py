#!/usr/bin/env python
"""Counterfactual Intervention Rate: DAgger fine-tuning of the policy on red-team failures.

1. Evaluate the base policy on the failure scenarios from a red-team log (+ random held-out scenarios).
2. Roll the *student* out on those scenarios while recording expert labels on the states it visits (DAgger),
   aggregate with the original BC data, fine-tune.
3. Re-evaluate on the same failure scenarios and on unseen held-out scenarios; report collision rates.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from engine.scene_graph import SceneGraph
from eval.closed_loop.data import collect, random_scenario
from eval.closed_loop.policies import PilotNetPolicy
from eval.closed_loop.redteam import FAMILIES, to_spec
from eval.closed_loop.simulator import ClosedLoopSim
from eval.closed_loop.train_bc import train_pilotnet
from models.refiner.refiner import Refiner


def collision_rate(sim, specs):
    res = [sim.run(s) for s in specs]
    return float(np.mean([r.metrics["collision"] or r.metrics["offroad"] for r in res]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--refiner", default=None)
    ap.add_argument("--policy_ckpt", default="checkpoints/pilotnet.safetensors")
    ap.add_argument("--redteam_log", default="outputs/redteam.jsonl")
    ap.add_argument("--bc_data", default="outputs/bc_data.npz")
    ap.add_argument("--heldout", type=int, default=20, help="fresh adversarial scenarios not used for training")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--output", default="checkpoints/pilotnet_dagger.safetensors")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    scene = SceneGraph.load(a.scene)
    refiner = Refiner.load(a.refiner) if a.refiner else None
    recs = [json.loads(l) for l in Path(a.redteam_log).read_text().splitlines() if l.strip()]
    fails = sorted({r["spec"] for r in recs if r["metrics"]["collision"] or r["metrics"]["offroad"]})
    if not fails:
        raise SystemExit("no failures in the red-team log: nothing to fine-tune on")
    # held-out: freshly sampled adversarial scenarios from the same families
    actors = [n.name for n in scene.actors]
    heldout = []
    for _ in range(a.heldout):
        fam = rng.choice(["cut_in", "hard_brake", "velocity_spike"])
        x = {k: float(rng.uniform(lo, hi)) for k, lo, hi in FAMILIES[fam]}
        heldout.append(to_spec(fam, rng.choice(actors), x))

    base = PilotNetPolicy.load(a.policy_ckpt)
    sim = ClosedLoopSim(scene, base, refiner)
    before = {"failures": collision_rate(sim, fails), "heldout": collision_rate(sim, heldout)}
    print(f"[dagger] before: {before}", flush=True)

    student = PilotNetPolicy.load(a.policy_ckpt)
    imgs, spd, lab = collect(sim, fails * 2, driver=student, seed=a.seed)
    # also some fresh randomised scenarios with the student driving
    extra = [random_scenario(scene, rng) for _ in range(len(fails))]
    i2, s2, l2 = collect(sim, extra, driver=student, seed=a.seed + 100)
    d = np.load(a.bc_data)
    imgs = np.concatenate([d["imgs"], imgs, i2])
    spd = np.concatenate([d["speeds"], spd, s2])
    lab = np.concatenate([d["labels"], lab, l2])
    train_pilotnet(student.model, imgs, spd, lab, epochs=a.epochs, lr=1e-4, seed=a.seed)
    student.save(a.output)
    sim.policy = PilotNetPolicy.load(a.output)
    after = {"failures": collision_rate(sim, fails), "heldout": collision_rate(sim, heldout)}
    report = {"failure_scenarios": len(fails), "heldout_scenarios": len(heldout),
              "intervention_rate_before": before, "intervention_rate_after": after, "checkpoint": a.output}
    Path(a.output.removesuffix(".safetensors") + ".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
