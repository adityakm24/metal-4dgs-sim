"""Programmatic red-teamer: cross-entropy search over long-tail perturbations to find policy failures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .simulator import ClosedLoopSim

# family -> list of (param, low, high)
FAMILIES = {
    "cut_in": [("offset", -4.5, -1.0), ("t", 0.5, 6.0), ("ramp", 0.5, 2.0)],
    "hard_brake": [("decel", 3.0, 9.0), ("t", 0.5, 7.0)],
    "velocity_spike": [("dv", -7.0, 3.0), ("t", 0.5, 6.0), ("duration", 0.5, 2.0)],
    "occlusion": [("amount", 0.1, 0.5), ("t", 0.5, 7.0), ("duration", 0.5, 2.0)],
    "obstacle": [("dist", 15.0, 45.0), ("t", 0.5, 5.0), ("lateral", -1.0, 1.0)],
}


def to_spec(family: str, actor: str | None, x: dict) -> str:
    if family == "cut_in":
        return f"{actor}:cut_in={x['offset']:.2f}m,t={x['t']:.2f}s,ramp={x['ramp']:.2f}s"
    if family == "hard_brake":
        return f"{actor}:hard_brake={x['decel']:.2f},t={x['t']:.2f}s"
    if family == "velocity_spike":
        return f"{actor}:velocity_spike={x['dv']:.2f}m/s,t={x['t']:.2f}s,duration={x['duration']:.2f}s"
    if family == "occlusion":
        return f"camera:occlusion={x['amount']:.2f},t={x['t']:.2f}s,duration={x['duration']:.2f}s"
    if family == "obstacle":
        return f"obstacle_rt:spawn={x['dist']:.1f}m,t={x['t']:.2f}s,lateral={x['lateral']:.2f}m"
    raise ValueError(family)


def severity(m: dict) -> float:
    """Higher = worse for the policy."""
    s = 0.0
    if m["collision"]:
        s += 100.0
    if m["offroad"]:
        s += 50.0
    ttc = m["min_ttc"]
    s += 10.0 / max(ttc if ttc is not None else 1e9, 0.1)
    s += 5.0 * m["max_lateral_error"]
    return s


@dataclass
class RedTeamConfig:
    families: tuple = ("cut_in", "hard_brake", "velocity_spike", "occlusion", "obstacle")
    population: int = 10
    elites: int = 3
    iterations: int = 3
    seed: int = 0


class RedTeamer:
    def __init__(self, sim: ClosedLoopSim, cfg: RedTeamConfig, log_path: str | Path | None = None):
        self.sim, self.cfg = sim, cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []

    def _evaluate(self, family: str, actor: str | None, x: dict) -> dict:
        spec = to_spec(family, actor, x)
        res = self.sim.run(spec, stop_on_collision=True)
        rec = {"family": family, "actor": actor, "params": x, "spec": spec, "metrics": res.metrics,
               "severity": severity(res.metrics), "policy": self.sim.policy.name}
        self.results.append(rec)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        flag = "FAIL" if (res.metrics["collision"] or res.metrics["offroad"]) else "ok  "
        print(f"[redteam] {flag} sev={rec['severity']:7.2f}  {spec}", flush=True)
        return rec

    def search(self) -> list[dict]:
        c = self.cfg
        actors = [n.name for n in self.sim.base_scene.actors]
        for family in c.families:
            space = FAMILIES[family]
            targets = actors if family in ("cut_in", "hard_brake", "velocity_spike") else [None]
            for actor in targets:
                mu = np.array([(lo + hi) / 2 for _, lo, hi in space])
                sd = np.array([(hi - lo) / 3 for _, lo, hi in space])
                for _ in range(c.iterations):
                    pop = []
                    for _ in range(c.population):
                        v = np.clip(self.rng.normal(mu, sd), [lo for _, lo, _ in space], [hi for _, _, hi in space])
                        x = {name: float(val) for (name, _, _), val in zip(space, v)}
                        pop.append((v, self._evaluate(family, actor, x)))
                    pop.sort(key=lambda p: -p[1]["severity"])
                    el = np.stack([p[0] for p in pop[: c.elites]])
                    mu, sd = el.mean(0), np.maximum(el.std(0), 0.05 * np.array([hi - lo for _, lo, hi in space]))
        return self.results

    def summary(self) -> dict:
        fails = [r for r in self.results if r["metrics"]["collision"] or r["metrics"]["offroad"]]
        by_family = {}
        for r in self.results:
            d = by_family.setdefault(r["family"], {"rollouts": 0, "failures": 0})
            d["rollouts"] += 1
            d["failures"] += int(r["metrics"]["collision"] or r["metrics"]["offroad"])
        return {"rollouts": len(self.results), "failures": len(fails),
                "failure_rate": len(fails) / max(len(self.results), 1), "by_family": by_family,
                "worst": [r["spec"] for r in sorted(self.results, key=lambda r: -r["severity"])[:10]]}
