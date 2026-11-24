// Shared helpers for the metal-4dgs-sim splat kernels.
//
// These sources are JIT-compiled through `mx.fast.metal_kernel` (see kernels.py), so each
// kernel file is a *function body*; this file is prepended as the header. Template constants
// available inside bodies: TILE (tile edge in px), W, H (image size), TILES_X, C (feature channels).

#include <metal_stdlib>
using namespace metal;

constant constexpr float SPLAT_ALPHA_MIN = 1.0f / 255.0f;
constant constexpr float SPLAT_ALPHA_MAX = 0.99f;
constant constexpr float SPLAT_T_MIN = 1e-4f;
constant constexpr uint SPLAT_BATCH = 256;   // == TILE * TILE for TILE = 16

// Gaussian exponent for pixel offset d = pixel - mean and conic (a, b, c) = inverse 2D covariance.
inline float splat_power(float dx, float dy, float a, float b, float c) {
    return -0.5f * (a * dx * dx + c * dy * dy) - b * dx * dy;
}
