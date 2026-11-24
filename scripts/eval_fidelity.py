#!/usr/bin/env python
"""Geometric-fidelity evaluation at shifted viewpoints (lateral ±2.5 m) against a ground-truth scene.

Reports, per lateral shift: PSNR of raw splats and refined frames vs ground truth, metric depth AbsRel of the
4DGS depth buffer vs ground-truth depth, and temporal warping error E_warp of raw vs refined sequences.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import mlx.core as mx
import numpy as np

from engine.scene_graph import SceneGraph, render
from eval.metrics.geometry import abs_rel, warping_error
from eval.metrics.image import psnr
from models.refiner.degrade import shifted_ego_camera
from models.refiner.refiner import Refiner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gt_scene", required=True)
    ap.add_argument("--refiner", default=None)
    ap.add_argument("--shifts", default="0,1,2.5,-2.5")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default="outputs/fidelity.json")
    a = ap.parse_args()
    recon, gt = SceneGraph.load(a.scene, edits=False), SceneGraph.load(a.gt_scene, edits=False)
    ref = Refiner.load(a.refiner) if a.refiner else None
    t0, t1 = recon.time_range
    ts = np.linspace(t0 + 1, t1 - 1, a.frames)
    report = {}
    for lat in [float(s) for s in a.shifts.split(",")]:
        m = {"psnr_raw": [], "psnr_refined": [], "depth_absrel": [], "warp_raw": [], "warp_refined": []}
        prev = None
        for t in ts:
            cam = shifted_ego_camera(recon, float(t), lat)
            pc = shifted_ego_camera(recon, float(t - 0.1), lat)
            o = render(recon, cam, float(t), flow_ref=(pc, float(t - 0.1)))
            g = render(gt, cam, float(t))
            o_prev = render(recon, pc, float(t - 0.1))
            raw, gimg = np.clip(np.array(o.rgb), 0, 1), np.clip(np.array(g.rgb), 0, 1)
            m["psnr_raw"].append(psnr(raw, gimg))
            m["depth_absrel"].append(abs_rel(np.array(o.depth), np.array(g.depth), np.array(g.alpha) > 0.5))
            flow, valid = np.array(o.flow), np.array(o.alpha) > 0.5
            m["warp_raw"].append(warping_error(raw, np.clip(np.array(o_prev.rgb), 0, 1), flow, valid))
            if ref is not None:
                rf = np.array(ref.refine(o))
                rp = np.array(ref.refine(o_prev))
                m["psnr_refined"].append(psnr(rf, gimg))
                m["warp_refined"].append(warping_error(rf, rp, flow, valid))
        report[f"lateral_{lat:+.1f}m"] = {k: float(np.nanmean(v)) for k, v in m.items() if v}
        print(f"lateral {lat:+.1f} m: {report[f'lateral_{lat:+.1f}m']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
