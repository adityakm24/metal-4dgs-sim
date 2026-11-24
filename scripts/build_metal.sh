#!/usr/bin/env bash
# Build/validate the custom Metal 4DGS kernels.
#
# The rasterizer kernels are JIT-compiled by MLX (mx.fast.metal_kernel) from engine/metal_splat/shaders/*.metal
# the first time they run, against the same unified-memory buffers MLX uses. This script compiles every kernel
# for each template configuration used at runtime, runs the forward/backward correctness tests, and — when the
# offline Metal toolchain is installed — also syntax-checks the shader sources with `xcrun metal`.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}

if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  echo "[build_metal] offline Metal toolchain found: syntax-checking shaders"
  tmp=$(mktemp -d)
  for k in duplicate_keys tile_ranges rasterize_forward rasterize_backward; do
    "$PY" - "$k" "$tmp" <<'PYEOF'
import sys, re
from pathlib import Path
k, tmp = sys.argv[1], Path(sys.argv[2])
sh = Path("engine/metal_splat/shaders")
hdr = (sh / "common.metal").read_text()
body = (sh / f"{k}.metal").read_text()
# wrap the MLX kernel body into a standalone kernel with representative template constants
consts = "constant constexpr int TILE=16, W=512, H=288, TILES_X=32, C=6;\n"
sig = {
 "duplicate_keys": "device const uint* offsets [[buffer(0)]], constant int* offsets_shape [[buffer(1)]], device const int* rects [[buffer(2)]], device const float* depths [[buffer(3)]], device ulong* keys [[buffer(4)]], device uint* values [[buffer(5)]], uint3 thread_position_in_grid [[thread_position_in_grid]]",
 "tile_ranges": "device const ulong* keys [[buffer(0)]], constant int* keys_shape [[buffer(1)]], device uint* ranges [[buffer(2)]], uint3 thread_position_in_grid [[thread_position_in_grid]]",
 "rasterize_forward": "device const uint* ranges [[buffer(0)]], device const uint* point_ids [[buffer(1)]], device const float* means2d [[buffer(2)]], device const float* conic_op [[buffer(3)]], device const float* feats [[buffer(4)]], device float* out_feat [[buffer(5)]], device float* out_T [[buffer(6)]], device uint* n_contrib [[buffer(7)]], uint3 thread_position_in_grid [[thread_position_in_grid]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]",
 "rasterize_backward": "device const uint* ranges [[buffer(0)]], device const uint* point_ids [[buffer(1)]], device const float* means2d [[buffer(2)]], device const float* conic_op [[buffer(3)]], device const float* feats [[buffer(4)]], device const float* out_T [[buffer(5)]], device const uint* n_contrib [[buffer(6)]], device const float* dL_dfeat [[buffer(7)]], device const float* dL_dT [[buffer(8)]], device atomic_float* d_means2d [[buffer(9)]], device atomic_float* d_conic_op [[buffer(10)]], device atomic_float* d_feats [[buffer(11)]], uint3 thread_position_in_grid [[thread_position_in_grid]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint thread_index_in_threadgroup [[thread_index_in_threadgroup]], uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
}[k]
(tmp / f"{k}.metal").write_text(hdr + consts + f"kernel void {k}({sig}) {{\n{body}\n}}\n")
PYEOF
    xcrun -sdk macosx metal -c "$tmp/$k.metal" -o "$tmp/$k.air"
  done
  mkdir -p build && xcrun -sdk macosx metallib "$tmp"/*.air -o build/metal_splat.metallib
  echo "[build_metal] wrote build/metal_splat.metallib"
else
  echo "[build_metal] offline Metal toolchain not installed (optional: xcodebuild -downloadComponent MetalToolchain);"
  echo "              kernels will be JIT-compiled by MLX."
fi

echo "[build_metal] JIT-compiling MLX kernels and running rasterizer correctness tests"
"$PY" -m pytest -q tests/test_rasterizer.py
echo "[build_metal] OK"
