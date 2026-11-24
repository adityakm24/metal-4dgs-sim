import numpy as np
import mlx.core as mx

from engine.metal_splat import rasterize


def reference_composite(means2d, conics, opac, feats, W, H):
    """Brute-force numpy compositing with identical cut-offs (depth order = array order)."""
    C = feats.shape[1]
    out = np.zeros((H, W, C))
    Tout = np.ones((H, W))
    for yy in range(H):
        for xx in range(W):
            px, py = xx + 0.5, yy + 0.5
            T = 1.0
            acc = np.zeros(C)
            for i in range(len(opac)):
                dx, dy = px - means2d[i, 0], py - means2d[i, 1]
                a, b, c = conics[i]
                power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
                if power > 0:
                    continue
                alpha = min(0.99, opac[i] * np.exp(power))
                if alpha < 1 / 255:
                    continue
                tT = T * (1 - alpha)
                if tT < 1e-4:
                    break
                acc += feats[i] * alpha * T
                T = tT
            out[yy, xx] = acc
            Tout[yy, xx] = T
    return out, 1 - Tout


def make_case(n=12, W=40, H=24, C=4, seed=0):
    rng = np.random.default_rng(seed)
    means2d = np.stack([rng.uniform(0, W, n), rng.uniform(0, H, n)], 1).astype(np.float32)
    s = rng.uniform(1.5, 5.0, (n, 2))
    rho = rng.uniform(-0.6, 0.6, n)
    cov = np.stack([s[:, 0] ** 2, rho * s[:, 0] * s[:, 1], s[:, 1] ** 2], 1)
    det = cov[:, 0] * cov[:, 2] - cov[:, 1] ** 2
    conics = np.stack([cov[:, 2] / det, -cov[:, 1] / det, cov[:, 0] / det], 1).astype(np.float32)
    opac = rng.uniform(0.3, 0.95, n).astype(np.float32)
    feats = rng.uniform(0, 1, (n, C)).astype(np.float32)
    depths = np.arange(1, n + 1, dtype=np.float32)
    radii = np.ceil(3 * s.max(1) * 1.3).astype(np.int32)
    return means2d, conics, opac, feats, depths, radii


def test_forward_matches_reference():
    W, H = 40, 24
    m, co, op, f, d, r = make_case(W=W, H=H)
    out, alpha, _ = rasterize(mx.array(m), mx.array(co), mx.array(op), mx.array(f), mx.array(r), mx.array(d), W, H)
    ref, ref_a = reference_composite(m, co, op, f, W, H)
    assert np.abs(np.array(out) - ref).max() < 1e-4
    assert np.abs(np.array(alpha) - ref_a).max() < 1e-4


def dense_reference_mlx(means2d, conics, opac, feats, W, H):
    """Dense differentiable MLX compositing with the same cut-offs (no early termination)."""
    ys, xs = mx.meshgrid(mx.arange(H, dtype=mx.float32) + 0.5, mx.arange(W, dtype=mx.float32) + 0.5, indexing="ij")
    px, py = xs.reshape(-1, 1), ys.reshape(-1, 1)
    dx, dy = px - means2d[None, :, 0], py - means2d[None, :, 1]
    a, b, c = conics[None, :, 0], conics[None, :, 1], conics[None, :, 2]
    power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
    raw = opac[None] * mx.exp(power)
    alpha = mx.minimum(raw, 0.99)
    alpha = mx.where((power <= 0) & (alpha >= 1 / 255), alpha, 0.0)
    T = mx.cumprod(1 - alpha, axis=1)
    T_excl = mx.concatenate([mx.ones_like(T[:, :1]), T[:, :-1]], axis=1)
    out = (alpha * T_excl) @ feats
    return out.reshape(H, W, -1), (1 - T[:, -1]).reshape(H, W)


def test_backward_matches_autodiff_reference():
    W, H = 40, 24
    m, co, op, f, d, _ = make_case(n=10, W=W, H=H, seed=3)
    r = np.full_like(d, 200, dtype=np.int32)  # cover the whole image: no tile truncation
    rng = np.random.default_rng(1)
    wf = mx.array(rng.normal(size=(H, W, f.shape[1])).astype(np.float32))
    wa = mx.array(rng.normal(size=(H, W)).astype(np.float32))

    def loss_metal(*a):
        out, alpha, _ = rasterize(*a, mx.array(r), mx.array(d), W, H)
        return mx.sum(out * wf) + mx.sum(alpha * wa)

    def loss_ref(*a):
        out, alpha = dense_reference_mlx(*a, W, H)
        return mx.sum(out * wf) + mx.sum(alpha * wa)

    args = [mx.array(x) for x in (m, co, op, f)]
    assert abs(loss_metal(*args).item() - loss_ref(*args).item()) < 1e-2
    g1 = mx.grad(loss_metal, argnums=(0, 1, 2, 3))(*args)
    g2 = mx.grad(loss_ref, argnums=(0, 1, 2, 3))(*args)
    for a, b in zip(g1, g2):
        a, b = np.array(a), np.array(b)
        assert np.abs(a - b).max() <= 1e-3 * max(1.0, np.abs(b).max()), (np.abs(a - b).max(), np.abs(b).max())


def test_sync_free_binning_matches_exact():
    from capture.synthetic import build_world
    from engine.scene_graph import render

    w = build_world(length=40, duration=3.0, density=0.15, width=128, height=72)
    for t in (0.5, 0.6, 0.7):     # first call sizes buffers, later ones reuse capacity
        cam = w.ego_camera(t)
        a = render(w, cam, t).numpy()["rgb"]
        b = render(w, cam, t, realtime=True).numpy()["rgb"]
        assert np.abs(a - b).max() < 1e-5
