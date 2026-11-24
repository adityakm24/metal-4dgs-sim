#!/usr/bin/env python
"""Train the 2-step flow-matching refiner on paired (degraded splat render, ground truth) frames."""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from core.reconstruct.losses import psnr
from engine.scene_graph import SceneGraph
from models.refiner.degrade import holdout_pairs, synthetic_pairs
from models.refiner.refiner import COND_CH, Refiner, count_params, flow_matching_loss, make_condition, save_refiner
from models.refiner.unet import RefinerUNet


def to_inputs(buf: np.ndarray):
    b = mx.array(buf.astype(np.float32))
    rgb, depth, alpha, flow = b[..., :3], b[..., 3], b[..., 4], b[..., 5:7]
    return rgb, make_condition(rgb, depth, alpha, flow)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="reconstructed scene checkpoint")
    ap.add_argument("--gt_scene", help="ground-truth scene (synthetic mode)")
    ap.add_argument("--data_dir", help="capture/log for holdout mode (real data)")
    ap.add_argument("--output", default="checkpoints/flow_refiner.safetensors")
    ap.add_argument("--pairs", type=int, default=480)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, nargs=2, default=[128, 256], metavar=("H", "W"))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--sample_steps", type=int, default=2)
    ap.add_argument("--noise", type=float, default=0.02, help="std of noise added to x0 during training")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    recon = SceneGraph.load(a.scene, edits=False)
    t0 = time.perf_counter()
    if a.gt_scene:
        gt = SceneGraph.load(a.gt_scene, edits=False)
        conds, tgts = synthetic_pairs(recon, gt, a.pairs, seed=a.seed)
        mode = "synthetic"
    else:
        from capture.dataset import load_any

        ds = load_any(a.data_dir)
        c = recon.ego.camera
        conds, tgts = holdout_pairs(recon, ds.frames, (int(c["width"]), int(c["height"])))
        mode = "holdout"
    n_val = max(8, len(conds) // 10)
    vc, vt, conds, tgts = conds[:n_val], tgts[:n_val], conds[n_val:], tgts[n_val:]
    print(f"[refiner] {len(conds)} train / {n_val} val {mode} pairs in {time.perf_counter() - t0:.0f} s")

    model = RefinerUNet(cond_ch=COND_CH, base=a.base)
    print(f"[refiner] {count_params(model) / 1e6:.2f} M params")
    sched = optim.cosine_decay(a.lr, a.steps, a.lr * 0.05)
    opt = optim.AdamW(learning_rate=sched, weight_decay=1e-4)
    loss_and_grad = nn.value_and_grad(model, flow_matching_loss)
    H, W = conds.shape[1:3]
    ch, cw = a.crop

    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, len(conds), a.batch)
        ys = rng.integers(0, H - ch + 1, a.batch)
        xs = rng.integers(0, W - cw + 1, a.batch)
        cb = np.stack([conds[i, y:y + ch, x:x + cw] for i, y, x in zip(idx, ys, xs)])
        tb = np.stack([tgts[i, y:y + ch, x:x + cw] for i, y, x in zip(idx, ys, xs)])
        if rng.random() < 0.5:   # horizontal flip (flip u-flow sign)
            cb, tb = cb[:, :, ::-1].copy(), tb[:, :, ::-1].copy()
            cb[..., 5] *= -1
        rgb, cond = to_inputs(cb)
        x0 = rgb + a.noise * mx.random.normal(rgb.shape)
        x1 = mx.array(tb.astype(np.float32))
        loss, grads = loss_and_grad(model, x0, x1, cond)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if step % 100 == 0 or step == 1:
            print(f"[refiner] step {step:5d} loss {loss.item():.5f}  {(time.perf_counter() - t0) / step * 1e3:.0f} ms/step",
                  flush=True)

    save_refiner(model, a.output, steps=a.sample_steps, extra={"mode": mode, "noise": a.noise})
    ref = Refiner.load(a.output)
    raw, refd, lat = [], [], []
    for i in range(n_val):
        rgb, cond = to_inputs(vc[i:i + 1])
        b = vc[i].astype(np.float32)
        tt = time.perf_counter()
        out = ref(mx.array(b[..., :3]), mx.array(b[..., 3]), mx.array(b[..., 4]), mx.array(b[..., 5:7]))
        mx.eval(out)
        lat.append((time.perf_counter() - tt) * 1e3)
        gt = mx.array(vt[i].astype(np.float32))
        raw.append(psnr(mx.array(b[..., :3]), gt))
        refd.append(psnr(out, gt))
    metrics = {"val_psnr_raw": float(np.mean(raw)), "val_psnr_refined": float(np.mean(refd)),
               "latency_ms_p50": float(np.median(lat[2:] if len(lat) > 4 else lat)), "mode": mode}
    Path(a.output.removesuffix(".safetensors") + ".metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[refiner] {metrics} -> {a.output}")


if __name__ == "__main__":
    main()
