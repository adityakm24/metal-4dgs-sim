"""Compact UNet velocity field with a ControlNet-style, zero-initialised geometry adapter.

The denoiser trunk sees the current flow state x_t. A separate control encoder sees the splat
conditioning (raw render RGB, inverse metric depth, alpha, optical flow) and injects features into every
trunk resolution through zero-initialised 1x1 convs, so training starts as an identity-ish refiner and
geometry guidance is learnt as a residual — the ControlNet recipe, at a size that runs in real time.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def space_to_depth(x: mx.array, p: int) -> mx.array:
    b, h, w, c = x.shape
    x = x.reshape(b, h // p, p, w // p, p, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h // p, w // p, p * p * c)


def depth_to_space(x: mx.array, p: int) -> mx.array:
    b, h, w, c = x.shape
    c0 = c // (p * p)
    x = x.reshape(b, h, w, p, p, c0).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h * p, w * p, c0)


def timestep_embedding(t: mx.array, dim: int) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
    args = (t.reshape(-1, 1) * 1000.0) * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, cin), cin, pytorch_compatible=True)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout * 2)
        self.n2 = nn.GroupNorm(min(8, cout), cout, pytorch_compatible=True)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def __call__(self, x, temb):
        h = self.c1(nn.silu(self.n1(x)))
        scale, shift = mx.split(self.t(temb)[:, None, None, :], 2, axis=-1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(nn.silu(h))
        return h + (x if self.skip is None else self.skip(x))


class ZeroConv(nn.Conv2d):
    def __init__(self, c: int):
        super().__init__(c, c, 1)
        self.weight = mx.zeros_like(self.weight)
        self.bias = mx.zeros_like(self.bias)


class RefinerUNet(nn.Module):
    def __init__(self, in_ch: int = 3, cond_ch: int = 7, out_ch: int = 3, base: int = 32, mults=(1, 2, 4), tdim: int = 128,
                 patch: int = 2):
        super().__init__()
        self.tdim = tdim
        self.patch = patch
        self.config = {"in_ch": in_ch, "cond_ch": cond_ch, "out_ch": out_ch, "base": base, "mults": list(mults),
                       "tdim": tdim, "patch": patch}
        in_ch, cond_ch, out_ch = in_ch * patch * patch, cond_ch * patch * patch, out_ch * patch * patch
        chs = [base * m for m in mults]
        self.t_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        # trunk
        self.stem = nn.Conv2d(in_ch + cond_ch, chs[0], 3, padding=1)
        self.down = [ResBlock(chs[max(i - 1, 0)] if i else chs[0], c, tdim) for i, c in enumerate(chs)]
        self.pool = [nn.Conv2d(c, c, 3, stride=2, padding=1) for c in chs[:-1]]
        self.mid = ResBlock(chs[-1], chs[-1], tdim)
        self.up = [ResBlock(chs[i] * 2, chs[max(i - 1, 0)] if i else chs[0], tdim) for i in reversed(range(len(chs)))]
        self.out_norm = nn.GroupNorm(8, chs[0], pytorch_compatible=True)
        self.out = nn.Conv2d(chs[0], out_ch, 3, padding=1)
        # control adapter (ControlNet-style): encoder on conditioning + zero convs per level
        self.c_stem = nn.Conv2d(cond_ch, chs[0], 3, padding=1)
        self.c_blocks = [nn.Conv2d(chs[max(i - 1, 0)] if i else chs[0], c, 3, padding=1) for i, c in enumerate(chs)]
        self.c_pool = [nn.Conv2d(c, c, 3, stride=2, padding=1) for c in chs[:-1]]
        self.c_zero = [ZeroConv(c) for c in chs]

    def __call__(self, x_t: mx.array, cond: mx.array, t: mx.array) -> mx.array:
        p = self.patch
        H, W = x_t.shape[1:3]
        ph, pw = (-H) % (4 * p), (-W) % (4 * p)
        if ph or pw:   # pad to a multiple of the total downsampling factor
            x_t = mx.pad(x_t, [(0, 0), (0, ph), (0, pw), (0, 0)], mode="edge")
            cond = mx.pad(cond, [(0, 0), (0, ph), (0, pw), (0, 0)], mode="edge")
        if p > 1:
            x_t, cond = space_to_depth(x_t, p), space_to_depth(cond, p)
        temb = self.t_mlp(timestep_embedding(t, self.tdim))
        # control features
        ctrl = []
        c = nn.silu(self.c_stem(cond))
        for i, blk in enumerate(self.c_blocks):
            c = nn.silu(blk(c))
            ctrl.append(self.c_zero[i](c))
            if i < len(self.c_pool):
                c = self.c_pool[i](c)
        # trunk
        h = self.stem(mx.concatenate([x_t, cond], axis=-1))
        skips = []
        for i, blk in enumerate(self.down):
            h = blk(h, temb) + ctrl[i]
            skips.append(h)
            if i < len(self.pool):
                h = self.pool[i](h)
        h = self.mid(h, temb)
        for j, blk in enumerate(self.up):
            i = len(self.down) - 1 - j
            if h.shape[1:3] != skips[i].shape[1:3]:
                h = mx.repeat(mx.repeat(h, 2, axis=1), 2, axis=2)[:, : skips[i].shape[1], : skips[i].shape[2]]
            h = blk(mx.concatenate([h, skips[i]], axis=-1), temb)
        out = self.out(nn.silu(self.out_norm(h)))
        if p > 1:
            out = depth_to_space(out, p)
        return out[:, :H, :W]
