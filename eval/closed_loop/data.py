"""Imitation data from the simulator: DART-style noisy expert rollouts and DAgger on student rollouts."""

from __future__ import annotations

import numpy as np

from .dynamics import Action
from .policies import ExpertPolicy, Observation, Policy, preprocess
from .simulator import ClosedLoopSim


class LabelingPolicy(Policy):
    """Executes `driver` (expert + noise, or the student) while recording expert labels on visited states."""

    name = "labeling"

    def __init__(self, expert: ExpertPolicy, driver: Policy | None = None, steer_noise: float = 0.0,
                 accel_noise: float = 0.0, seed: int = 0):
        self.expert, self.driver = expert, driver
        self.sn, self.an = steer_noise, accel_noise
        self.rng = np.random.default_rng(seed)
        self.images, self.speeds, self.labels = [], [], []
        self._noise = np.zeros(2)

    def act(self, obs: Observation) -> Action:
        lab = self.expert.act(obs)
        self.images.append(preprocess(obs.image).astype(np.float16))
        self.speeds.append(obs.speed)
        self.labels.append([lab.steer, lab.accel])
        if self.driver is not None:
            return self.driver.act(obs)
        # temporally-correlated noise (Ornstein-Uhlenbeck) to visit recovery states
        self._noise = 0.8 * self._noise + self.rng.normal(0, 1, 2) * [self.sn, self.an]
        return Action(lab.steer + self._noise[0], lab.accel + self._noise[1])

    def arrays(self):
        return (np.stack(self.images), np.asarray(self.speeds, np.float32), np.asarray(self.labels, np.float32))


def random_scenario(scene, rng) -> str:
    """Sample a training scenario: nominal, cut-ins, hard brakes, speed changes, ego offsets."""
    actors = [n.name for n in scene.actors]
    parts = []
    r = rng.random()
    if actors and r < 0.35:
        a = rng.choice(actors)
        parts.append(f"{a}:cut_in={rng.uniform(-4.0, -1.0):.2f}m,t={rng.uniform(0.5, 6):.2f}s,ramp={rng.uniform(0.8, 2.5):.2f}s")
    elif actors and r < 0.6:
        a = rng.choice(actors)
        parts.append(f"{a}:hard_brake={rng.uniform(2, 8):.1f},t={rng.uniform(0.5, 7):.2f}s")
    elif actors and r < 0.75:
        a = rng.choice(actors)
        parts.append(f"{a}:velocity_spike={rng.uniform(-5, 5):.2f}m/s,t={rng.uniform(0.5, 6):.2f}s,duration=1.5s")
    if rng.random() < 0.4:
        parts.append(f"ego:lateral_offset={rng.uniform(-1.2, 1.2):.2f}m")
    if rng.random() < 0.3:
        parts.append(f"ego:yaw={rng.uniform(-6, 6):.1f}deg")
    if rng.random() < 0.3:
        parts.append(f"ego:speed={rng.uniform(4, 12):.1f}")
    return ";".join(parts)


def collect(sim: ClosedLoopSim, specs: list[str], driver: Policy | None = None, steer_noise=0.06, accel_noise=0.6,
            seed: int = 0, verbose: bool = True):
    imgs, spd, lab = [], [], []
    base_policy = sim.policy
    for i, spec in enumerate(specs):
        lp = LabelingPolicy(ExpertPolicy(), driver, steer_noise, accel_noise, seed + i)
        sim.policy = lp
        res = sim.run(spec, stop_on_collision=True)
        a, b, c = lp.arrays()
        imgs.append(a), spd.append(b), lab.append(c)
        if verbose:
            print(f"[collect] {i + 1}/{len(specs)} {len(a)} samples  collision={res.metrics['collision']}  {spec!r}",
                  flush=True)
    sim.policy = base_policy
    return np.concatenate(imgs), np.concatenate(spd), np.concatenate(lab)
