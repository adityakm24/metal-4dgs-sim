#!/usr/bin/env python
"""Validate the no-LiDAR initialisation path: Depth Anything V2 relative depth, scale/shift-aligned to a
*sparse* set of metric points (stand-in for ARKit VIO feature points), scored against held-out LiDAR."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from capture.dataset import load_any
from core.reconstruct.mono_depth import MonoDepth
from eval.metrics.geometry import abs_rel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data/kitti_raw")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--sparse", type=int, default=150, help="metric points per frame used for alignment (VIO-like)")
    ap.add_argument("--out", default="outputs/mono_init.json")
    a = ap.parse_args()
    ds = load_any(a.data_dir)
    rng = np.random.default_rng(0)
    est = MonoDepth()
    rows = []
    for f in ds.frames[:: max(1, len(ds.frames) // a.frames)][: a.frames]:
        pts = ds.points[np.isclose(ds.points[:, 6], f.t)]
        idx = rng.permutation(len(pts))
        fit, held = pts[idx[: a.sparse]], pts[idx[a.sparse:]]
        img = f.image()
        d = est.metric_depth(img, f.camera, fit)
        if d is None:
            continue
        uv, z = f.camera.project(held[:, :3])
        H, W = d.shape
        ok = (z > 1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        pred = d[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
        gt = z[ok]
        m = pred > 0
        rows.append({"t": f.t, "absrel": abs_rel(pred[m], gt[m]), "delta1": float(np.mean(np.maximum(pred[m] / gt[m], gt[m] / pred[m]) < 1.25))})
        print(f"t={f.t:5.2f}  AbsRel {rows[-1]['absrel']:.3f}  δ<1.25 {rows[-1]['delta1']:.3f}", flush=True)
    res = {"frames": len(rows), "sparse_points_per_frame": a.sparse,
           "absrel_mean": float(np.mean([r["absrel"] for r in rows])),
           "delta1_mean": float(np.mean([r["delta1"] for r in rows])), "per_frame": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "per_frame"})


if __name__ == "__main__":
    main()
