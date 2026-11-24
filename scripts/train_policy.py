#!/usr/bin/env python
"""Train the compact PilotNet visuomotor policy by imitating the privileged expert inside the simulator
(DART noise injection over randomised long-tail scenarios), on refined frames."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from engine.scene_graph import SceneGraph
from eval.closed_loop.data import collect, random_scenario
from eval.closed_loop.policies import ExpertPolicy, PilotNetPolicy
from eval.closed_loop.simulator import ClosedLoopSim
from eval.closed_loop.train_bc import train_pilotnet
from models.refiner.refiner import Refiner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--refiner", default=None)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--output", default="checkpoints/pilotnet.safetensors")
    ap.add_argument("--data", default="outputs/bc_data.npz", help="cache of collected samples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if Path(a.data).exists():
        d = np.load(a.data)
        imgs, spd, lab = d["imgs"], d["speeds"], d["labels"]
        print(f"[bc] loaded {len(imgs)} cached samples from {a.data}")
    else:
        scene = SceneGraph.load(a.scene)
        sim = ClosedLoopSim(scene, ExpertPolicy(), Refiner.load(a.refiner) if a.refiner else None)
        specs = [random_scenario(scene, rng) for _ in range(a.episodes)]
        imgs, spd, lab = collect(sim, specs, seed=a.seed)
        Path(a.data).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.data, imgs=imgs, speeds=spd, labels=lab, specs=np.array(specs))
    pol = PilotNetPolicy()
    stats = train_pilotnet(pol.model, imgs, spd, lab, epochs=a.epochs, seed=a.seed)
    pol.save(a.output)
    Path(a.output.removesuffix(".safetensors") + ".metrics.json").write_text(json.dumps(stats, indent=1))
    print(f"[bc] {stats} -> {a.output}")


if __name__ == "__main__":
    main()
