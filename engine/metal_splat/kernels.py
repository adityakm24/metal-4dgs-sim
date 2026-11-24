"""JIT-compiled Metal kernels (via ``mx.fast.metal_kernel``) operating directly on MLX arrays.

MLX arrays live in unified memory, so these kernels read/write the same buffers the rest of the
MLX graph uses: there are no host<->device copies anywhere in the rasterization path.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

SHADER_DIR = Path(__file__).parent / "shaders"
TILE = 16


def _src(name: str) -> str:
    return (SHADER_DIR / f"{name}.metal").read_text()


@lru_cache(maxsize=None)
def _header() -> str:
    return _src("common")


@lru_cache(maxsize=None)
def duplicate_keys_kernel():
    return mx.fast.metal_kernel(
        name="m4dgs_duplicate_keys",
        input_names=["offsets", "rects", "depths", "capacity"],
        output_names=["keys", "values"],
        source=_src("duplicate_keys"),
        header=_header(),
    )


@lru_cache(maxsize=None)
def tile_ranges_kernel():
    return mx.fast.metal_kernel(
        name="m4dgs_tile_ranges",
        input_names=["keys", "total"],
        output_names=["ranges"],
        source=_src("tile_ranges"),
        header=_header(),
    )


@lru_cache(maxsize=None)
def forward_kernel():
    return mx.fast.metal_kernel(
        name="m4dgs_rasterize_forward",
        input_names=["ranges", "point_ids", "means2d", "conic_op", "feats"],
        output_names=["out_feat", "out_T", "n_contrib"],
        source=_src("rasterize_forward"),
        header=_header(),
    )


@lru_cache(maxsize=None)
def backward_kernel():
    return mx.fast.metal_kernel(
        name="m4dgs_rasterize_backward",
        input_names=[
            "ranges", "point_ids", "means2d", "conic_op", "feats",
            "out_T", "n_contrib", "dL_dfeat", "dL_dT",
        ],
        output_names=["d_means2d", "d_conic_op", "d_feats"],
        source=_src("rasterize_backward"),
        header=_header(),
        atomic_outputs=True,
    )


def all_kernel_sources() -> dict[str, str]:
    """Full kernel sources, used by scripts/build_metal.sh for validation."""
    return {n: _src(n) for n in ["common", "duplicate_keys", "tile_ranges", "rasterize_forward", "rasterize_backward"]}
