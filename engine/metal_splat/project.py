"""EWA projection of 3D Gaussians and spherical-harmonics colour, written as plain MLX ops so
autodiff flows through them; only compositing needs a hand-written backward (rasterize.py)."""

from __future__ import annotations

import mlx.core as mx

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def normalize(v: mx.array, axis: int = -1, eps: float = 1e-12) -> mx.array:
    return v / mx.sqrt(mx.sum(v * v, axis=axis, keepdims=True) + eps)


def quat_to_rotmat(q: mx.array) -> mx.array:
    """(N,4) wxyz -> (N,3,3)."""
    q = normalize(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r0 = mx.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1)
    r1 = mx.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1)
    r2 = mx.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1)
    return mx.stack([r0, r1, r2], axis=1)


def quat_multiply(a: mx.array, b: mx.array) -> mx.array:
    """Hamilton product a ⊗ b, wxyz, broadcasting."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return mx.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def eval_sh(sh: mx.array, dirs: mx.array, degree: int) -> mx.array:
    """sh: (N,K,3), dirs: (N,3) unit -> rgb (N,3) using the 3DGS convention (+0.5, clamp >= 0)."""
    res = SH_C0 * sh[:, 0]
    if degree >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        res = res - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
        if degree >= 2:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            res = (res + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5]
                   + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6] + SH_C2[3] * xz * sh[:, 7]
                   + SH_C2[4] * (xx - yy) * sh[:, 8])
            if degree >= 3:
                res = (res + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9] + SH_C3[1] * xy * z * sh[:, 10]
                       + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                       + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                       + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
                       + SH_C3[5] * z * (xx - yy) * sh[:, 14] + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return mx.maximum(res + 0.5, 0.0)


def rgb_to_sh0(rgb):
    return (rgb - 0.5) / SH_C0


def project_points(means: mx.array, w2c: mx.array, fx, fy, cx, cy) -> tuple[mx.array, mx.array]:
    """World points -> (N,2) pixels, (N,) depth."""
    p = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = p[:, 2]
    zs = mx.where(z > 1e-4, z, 1e-4)
    uv = mx.stack([fx * p[:, 0] / zs + cx, fy * p[:, 1] / zs + cy], axis=1)
    return uv, z


def _cov2d_conic(means, quats, log_scales, w2c, intr, lim, blur):
    """Elementwise (fusable) EWA projection; avoids slow batched 3x3 GEMMs."""
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    W = [[w2c[r, c] for c in range(3)] for r in range(3)]
    mx_, my_, mz_ = means[:, 0], means[:, 1], means[:, 2]
    x = W[0][0] * mx_ + W[0][1] * my_ + W[0][2] * mz_ + w2c[0, 3]
    y = W[1][0] * mx_ + W[1][1] * my_ + W[1][2] * mz_ + w2c[1, 3]
    z = W[2][0] * mx_ + W[2][1] * my_ + W[2][2] * mz_ + w2c[2, 3]
    zs = mx.where(z > 1e-4, z, 1.0)
    tx = mx.clip(x / zs, -lim[0], lim[0]) * zs
    ty = mx.clip(y / zs, -lim[1], lim[1]) * zs
    j00 = fx / zs
    j02 = -fx * tx / (zs * zs)
    j11 = fy / zs
    j12 = -fy * ty / (zs * zs)
    # T = J W  (2x3)
    T0 = [j00 * W[0][k] + j02 * W[2][k] for k in range(3)]
    T1 = [j11 * W[1][k] + j12 * W[2][k] for k in range(3)]
    # rotation from quaternion
    qn = quats / mx.sqrt(mx.sum(quats * quats, axis=1, keepdims=True) + 1e-12)
    qw, qx, qy, qz = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    Rq = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ]
    s = mx.exp(log_scales)
    # A = T R diag(s)  (2x3); cov = A A^T
    A0 = [(T0[0] * Rq[0][k] + T0[1] * Rq[1][k] + T0[2] * Rq[2][k]) * s[:, k] for k in range(3)]
    A1 = [(T1[0] * Rq[0][k] + T1[1] * Rq[1][k] + T1[2] * Rq[2][k]) * s[:, k] for k in range(3)]
    a = A0[0] * A0[0] + A0[1] * A0[1] + A0[2] * A0[2] + blur
    b = A0[0] * A1[0] + A0[1] * A1[1] + A0[2] * A1[2]
    c = A1[0] * A1[0] + A1[1] * A1[1] + A1[2] * A1[2] + blur
    u = fx * x / zs + cx
    v = fy * y / zs + cy
    return u, v, z, a, b, c


_cov2d_conic_compiled = mx.compile(_cov2d_conic)


def project_gaussians(
    means: mx.array,        # (N,3) world
    quats: mx.array,        # (N,4) world wxyz
    log_scales: mx.array,   # (N,3)
    w2c: mx.array,          # (4,4)
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    near: float = 0.2,
    far: float = 1e4,
    blur: float = 0.3,
):
    """Returns means2d (N,2), conics (N,3), depths (N,), radii (N,) int32, valid mask (N,)."""
    intr = mx.array([fx, fy, cx, cy], dtype=mx.float32)
    lim = mx.array([1.3 * 0.5 * width / fx, 1.3 * 0.5 * height / fy], dtype=mx.float32)
    u, v, z, a, b, c = _cov2d_conic_compiled(means, quats, log_scales, w2c, intr, lim, mx.array(blur, dtype=mx.float32))
    det = a * c - b * b
    valid = (z > near) & (z < far) & (det > 1e-12)
    det_s = mx.where(valid, det, 1.0)
    conics = mx.stack([c / det_s, -b / det_s, a / det_s], axis=1)
    mid = 0.5 * mx.stop_gradient(a + c)
    lam = mid + mx.sqrt(mx.maximum(mid * mid - mx.stop_gradient(det_s), 0.1))
    radius = mx.ceil(3.0 * mx.sqrt(lam))
    means2d = mx.stack([u, v], axis=1)
    us, vs = mx.stop_gradient(u), mx.stop_gradient(v)
    valid = valid & (us + radius > 0) & (us - radius < width) & (vs + radius > 0) & (vs - radius < height)
    radii = mx.where(valid, radius, 0).astype(mx.int32)
    return means2d, conics, z, radii, valid
