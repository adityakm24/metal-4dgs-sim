#!/usr/bin/env python
"""Generate the bundled synthetic sample: a ground-truth Gaussian driving world, rendered into a
posed RGB + depth log (data/sample_scene) that train_4dgs.py reconstructs from."""
import argparse

import _bootstrap  # noqa: F401

from capture.synthetic import build_world, write_sample_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/sample_scene")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    world = build_world(seed=a.seed, duration=a.duration, width=a.width, height=a.height)
    world.save(f"{a.out}/gt_scene.safetensors")
    ds = write_sample_dataset(a.out, world, fps=a.fps, seed=a.seed)
    print(f"wrote {len(ds.frames)} frames + ground-truth scene ({world.num_gaussians} Gaussians) to {a.out}")


if __name__ == "__main__":
    main()
