#!/usr/bin/env python
"""Reconstruct a 4DGS scene graph from an iPhone ARKit capture, a nuScenes scene, or an m4dgs log."""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from capture.dataset import load_any
from core.reconstruct.trainer import TrainConfig, Trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True, help="m4dgs log dir, ARKit capture dir/.zip, or nuScenes root")
    ap.add_argument("--output", default="checkpoints/scene_01.safetensors")
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--sh_degree", type=int, default=1)
    ap.add_argument("--max_width", type=int, default=640)
    ap.add_argument("--holdout_every", type=int, default=8)
    ap.add_argument("--max_gaussians", type=int, default=800_000)
    ap.add_argument("--mono_depth", action="store_true", help="Depth Anything V2 init (needs transformers+torch)")
    ap.add_argument("--static", action="store_true", help="disable per-Gaussian velocity/temporal opacity")
    ap.add_argument("--nuscenes_scene", default=None, help="scene name when --data_dir is a nuScenes root")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    kw = {"scene_name": a.nuscenes_scene} if a.nuscenes_scene else {}
    ds = load_any(a.data_dir, **kw)
    cfg = TrainConfig(iterations=a.iterations, sh_degree=a.sh_degree, max_width=a.max_width,
                      holdout_every=a.holdout_every, max_gaussians=a.max_gaussians, mono_depth=a.mono_depth,
                      dynamic_background=not a.static, seed=a.seed,
                      densify_until_frac=0.5 if a.iterations >= 2000 else 0.6,
                      densify_from=min(500, a.iterations // 6))
    tr = Trainer(ds, cfg)
    t0 = time.perf_counter()
    scene = tr.train()
    print(f"[train] done in {time.perf_counter() - t0:.0f} s")
    metrics = tr.evaluate()
    out = scene.save(a.output)
    Path(str(out).removesuffix(".safetensors") + ".metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[train] held-out {metrics}  ->  {out}  ({scene.num_gaussians} Gaussians)")


if __name__ == "__main__":
    main()
