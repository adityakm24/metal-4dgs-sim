"""Differentiable tile-based Gaussian rasterizer: MLX binning + custom Metal compositing.

Pipeline per frame:
  1. (MLX) tile bounding boxes + prefix sum of per-Gaussian tile counts
  2. (Metal) emit (tile << 32 | depth) keys, (MLX) 64-bit radix/merge argsort
  3. (Metal) per-tile [start, end) ranges
  4. (Metal) front-to-back compositing of C feature channels; custom VJP runs the backward kernel
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

from .kernels import TILE, backward_kernel, duplicate_keys_kernel, forward_kernel, tile_ranges_kernel


@dataclass
class Binning:
    ranges: mx.array      # (num_tiles, 2) uint32
    point_ids: mx.array   # (T,) uint32
    num_entries: int


def tile_grid(width: int, height: int) -> tuple[int, int]:
    return math.ceil(width / TILE), math.ceil(height / TILE)


_CAPACITY: dict = {}
_LAST_TOTAL: dict = {}


def bin_gaussians(means2d: mx.array, radii: mx.array, depths: mx.array, width: int, height: int,
                  exact: bool = True) -> Binning:
    """Assign Gaussians to the 16x16 tiles they overlap and sort each tile's list by depth.

    exact=True synchronises once to size the key buffers (training). exact=False (real-time sim) reuses the
    previous frame's size + 30% headroom so the whole frame is encoded without a CPU<->GPU round trip;
    `num_entries` is then -1 (unknown until evaluated)."""
    means2d = mx.stop_gradient(means2d)
    depths = mx.stop_gradient(depths)
    radii = mx.stop_gradient(radii)
    tiles_x, tiles_y = tile_grid(width, height)
    num_tiles = tiles_x * tiles_y
    n = means2d.shape[0]
    r = radii.astype(mx.float32)
    valid = radii > 0
    tx0 = mx.clip(mx.floor((means2d[:, 0] - r) / TILE), 0, tiles_x).astype(mx.int32)
    tx1 = mx.clip(mx.ceil((means2d[:, 0] + r) / TILE), 0, tiles_x).astype(mx.int32)
    ty0 = mx.clip(mx.floor((means2d[:, 1] - r) / TILE), 0, tiles_y).astype(mx.int32)
    ty1 = mx.clip(mx.ceil((means2d[:, 1] + r) / TILE), 0, tiles_y).astype(mx.int32)
    valid = valid & (tx1 > tx0) & (ty1 > ty0)
    rects = mx.stack([tx0, ty0, tx1, ty1], axis=1) * valid[:, None].astype(mx.int32)
    counts = ((rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])).astype(mx.uint32)
    incl = mx.cumsum(counts)
    if n == 0:
        return Binning(mx.zeros((num_tiles, 2), dtype=mx.uint32), mx.zeros((1,), dtype=mx.uint32), 0)
    total_arr = incl[-1:]
    key = (width, height)
    if exact or key not in _CAPACITY:
        # synchronise once to size the buffers exactly
        total = int(total_arr.item())
        capacity = max(total, 1)
    else:
        # sync-free: size from the previous frame (grown if it overflowed), mask the unused tail
        prev = _LAST_TOTAL.get(key)
        if prev is not None:
            last = int(prev.item())        # already computed by the previous frame: no stall
            if last > _CAPACITY[key] or last < _CAPACITY[key] // 3:
                _CAPACITY[key] = int(last * 1.3) + 1024
        capacity = _CAPACITY[key]
        total = -1
    _LAST_TOTAL[key] = total_arr
    if exact or key not in _CAPACITY:
        _CAPACITY[key] = int(capacity * 1.3) + 1024
    offsets = incl - counts
    keys, values = duplicate_keys_kernel()(
        inputs=[offsets, rects, depths.astype(mx.float32), mx.array([capacity], dtype=mx.uint32)],
        template=[("TILES_X", tiles_x)],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(capacity,), (capacity,)],
        output_dtypes=[mx.uint64, mx.uint32],
    )
    if total < 0:   # mask the unused tail with sentinel keys that sort last
        valid = mx.arange(capacity, dtype=mx.uint32) < total_arr
        keys = mx.where(valid, keys, (mx.array(0xFFFFFFFF, dtype=mx.uint64) << 32))
    order = mx.stop_gradient(mx.argsort(keys))
    keys = mx.stop_gradient(keys[order])
    point_ids = mx.stop_gradient(values[order])
    (ranges,) = tile_ranges_kernel()(
        inputs=[keys, total_arr],
        grid=(capacity, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(num_tiles, 2)],
        output_dtypes=[mx.uint32],
        init_value=0,
    )
    return Binning(mx.stop_gradient(ranges), point_ids, total)


@lru_cache(maxsize=64)
def _composite_fn(width: int, height: int, channels: int):
    tiles_x, tiles_y = tile_grid(width, height)
    template = [("TILE", TILE), ("W", width), ("H", height), ("TILES_X", tiles_x), ("C", channels)]
    grid = (tiles_x * TILE, tiles_y * TILE, 1)
    tg = (TILE, TILE, 1)

    @mx.custom_function
    def composite(means2d, conic_op, feats, ranges, point_ids):
        out_feat, out_T, n_contrib = forward_kernel()(
            inputs=[ranges, point_ids, means2d, conic_op, feats],
            template=template,
            grid=grid,
            threadgroup=tg,
            output_shapes=[(height, width, channels), (height, width), (height, width)],
            output_dtypes=[mx.float32, mx.float32, mx.uint32],
        )
        return out_feat, out_T, n_contrib

    @composite.vjp
    def composite_vjp(primals, cotangents, outputs):
        means2d, conic_op, feats, ranges, point_ids = primals
        d_feat, d_T = cotangents[0], cotangents[1]
        _, out_T, n_contrib = outputs
        if d_feat is None:
            d_feat = mx.zeros((height, width, channels), dtype=mx.float32)
        if d_T is None:
            d_T = mx.zeros((height, width), dtype=mx.float32)
        n = means2d.shape[0]
        d_means2d, d_conic_op, d_feats = backward_kernel()(
            inputs=[ranges, point_ids, means2d, conic_op, feats, out_T, n_contrib,
                    d_feat.astype(mx.float32), d_T.astype(mx.float32)],
            template=template,
            grid=grid,
            threadgroup=tg,
            output_shapes=[(n, 2), (n, 4), (n, channels)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            init_value=0,
        )
        return d_means2d, d_conic_op, d_feats, mx.zeros_like(ranges), mx.zeros_like(point_ids)

    return composite


def rasterize(
    means2d: mx.array,
    conics: mx.array,
    opacities: mx.array,
    feats: mx.array,
    radii: mx.array,
    depths: mx.array,
    width: int,
    height: int,
    binning: Binning | None = None,
    exact: bool = True,
) -> tuple[mx.array, mx.array, Binning]:
    """Composite per-Gaussian features. Returns (features (H,W,C), alpha (H,W), binning).

    Differentiable w.r.t. means2d, conics, opacities and feats.
    """
    if binning is None:
        binning = bin_gaussians(means2d, radii, depths, width, height, exact=exact)
    channels = feats.shape[1]
    conic_op = mx.concatenate([conics, opacities.reshape(-1, 1)], axis=1).astype(mx.float32)
    fn = _composite_fn(width, height, channels)
    out_feat, out_T, _ = fn(means2d.astype(mx.float32), conic_op, feats.astype(mx.float32),
                            binning.ranges, binning.point_ids)
    return out_feat, 1.0 - out_T, binning
