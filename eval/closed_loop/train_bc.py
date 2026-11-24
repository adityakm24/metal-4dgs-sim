"""Behaviour cloning for the PilotNet visuomotor policy."""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .policies import PilotNet

LABEL_SCALE = np.array([10.0, 1.0], np.float32)   # weight steering errors ~ accel errors


def train_pilotnet(model: PilotNet, imgs, speeds, labels, epochs: int = 30, batch: int = 64, lr: float = 3e-4,
                   seed: int = 0, val_frac: float = 0.1, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    n = len(imgs)
    perm = rng.permutation(n)
    nv = max(1, int(n * val_frac))
    vi, ti = perm[:nv], perm[nv:]
    steps = epochs * max(1, len(ti) // batch)
    opt = optim.Adam(learning_rate=optim.cosine_decay(lr, steps, lr * 0.05))
    scale = mx.array(LABEL_SCALE)

    def loss_fn(m, x, v, y):
        return mx.mean(mx.square((m(x, v) - y) * scale))

    lg = nn.value_and_grad(model, loss_fn)
    t0 = time.perf_counter()

    def batch_arrays(idx, flip):
        x = imgs[idx].astype(np.float32)
        y = labels[idx].copy()
        if flip:
            x = x[:, :, ::-1]
            y[:, 0] *= -1
        return mx.array(np.ascontiguousarray(x)), mx.array(speeds[idx]), mx.array(y)

    for ep in range(epochs):
        rng.shuffle(ti)
        model.train()
        tot = 0.0
        for k in range(0, len(ti) - batch + 1, batch):
            x, v, y = batch_arrays(ti[k:k + batch], rng.random() < 0.5)
            loss, g = lg(model, x, v, y)
            g, _ = optim.clip_grad_norm(g, 1.0)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state, loss)
            tot += loss.item()
        model.eval()
        x, v, y = batch_arrays(vi, False)
        val = loss_fn(model, x, v, y).item()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"[bc] epoch {ep + 1:3d}  train {tot / max(1, len(ti) // batch):.4f}  val {val:.4f}  "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return {"val_loss": val, "samples": n}
