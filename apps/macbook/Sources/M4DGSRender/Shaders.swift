// Metal shader source, compiled at runtime with MTLDevice.makeLibrary(source:options:)
// (the offline Metal toolchain is not required).
//
// Pipeline per frame:
//   1. preprocess   (compute) 4D eval → world → camera → EWA 2D covariance → oriented 3σ quad, SH colour,
//                   depth key; visible splats are compacted with an atomic counter.
//   2. radix sort   (compute) 4 × 8-bit stable LSD passes (hist → scan → scatter) on 32-bit depth keys,
//                   ascending key = descending depth = back-to-front.
//   3. draw_args    (compute) writes the indirect draw arguments (instanceCount = visible count).
//   4. raster       (render)  instanced quads, gaussian falloff, premultiplied "over" blending.
//   5. lines        (render)  overlay segments as screen-space quads.

enum ShaderSource {
    static let sortThreadgroup = 256
    static let sortItemsPerThread = 8
    static var sortTile: Int { sortThreadgroup * sortItemsPerThread }

    static let source = #"""
#include <metal_stdlib>
using namespace metal;

constant uint RADIX      = 256;
constant uint SORT_TG    = 256;
constant uint SORT_ITEMS = 8;
constant uint SORT_TILE  = SORT_TG * SORT_ITEMS;
constant uint SORT_SG    = SORT_TG / 32;

struct FrameUniforms {
    float4x4 world_to_cam;
    float4   cam_center;
    float4   intr;        // fx fy cx cy (target pixels)
    float4   clip_rect;   // x0 y0 x1 y1 (target pixels)
    float2   viewport;    // target W H
    float    time;
    float    near_z;
    uint     num_gaussians;
    uint     sh_degree;
    uint     sh_stride;
    float    low_pass;
    float4   fill;
};

struct GaussianIn {
    float4 mean_tc;   // xyz mean (node local), w = t_center
    float4 vel_its;   // xyz velocity (node local), w = 1/sigma_t
    float4 quat;      // w x y z (normalized)
    float4 scale_op;  // xyz scale, w = opacity
};

struct NodePose {
    float4 c0; float4 c1; float4 c2;  // rotation columns
    float4 pos;                       // w = visible (1/0)
    float4 quat;                      // w x y z
};

struct Splat {
    float4 center_axis1;   // xy center px, zw = 3σ major axis (px)
    float4 axis2_depth;    // xy = 3σ minor axis (px), z = depth
    float4 color;          // rgb, a = alpha(t)
};

struct LineSeg { float4 a; float4 b; float4 color; };   // a.w = width px

inline float4 qmul(float4 a, float4 b) {   // Hamilton, (w,x,y,z) stored in (x,y,z,w) slots
    return float4(a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
                  a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
                  a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
                  a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x);
}

inline float3x3 qmat(float4 q) {
    q = normalize(q);
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(float3(1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y)),
                    float3(2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x)),
                    float3(2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x*x + y*y)));
}

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[5] = { 1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                            -1.0925484305920792f, 0.5462742152960396f };
constant float SH_C3[7] = { -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                            0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                            -0.5900435899266435f };

inline float3 shc(device const float* sh, uint base, uint k) {
    return float3(sh[base + 3*k], sh[base + 3*k + 1], sh[base + 3*k + 2]);
}

inline float3 eval_sh(device const float* sh, uint base, uint deg, float3 d) {
    float3 r = SH_C0 * shc(sh, base, 0);
    if (deg >= 1) {
        float x = d.x, y = d.y, z = d.z;
        r += -SH_C1 * y * shc(sh, base, 1) + SH_C1 * z * shc(sh, base, 2) - SH_C1 * x * shc(sh, base, 3);
        if (deg >= 2) {
            float xx = x*x, yy = y*y, zz = z*z, xy = x*y, yz = y*z, xz = x*z;
            r += SH_C2[0] * xy * shc(sh, base, 4) + SH_C2[1] * yz * shc(sh, base, 5)
               + SH_C2[2] * (2*zz - xx - yy) * shc(sh, base, 6)
               + SH_C2[3] * xz * shc(sh, base, 7) + SH_C2[4] * (xx - yy) * shc(sh, base, 8);
            if (deg >= 3) {
                r += SH_C3[0] * y * (3*xx - yy) * shc(sh, base, 9)
                   + SH_C3[1] * xy * z * shc(sh, base, 10)
                   + SH_C3[2] * y * (4*zz - xx - yy) * shc(sh, base, 11)
                   + SH_C3[3] * z * (2*zz - 3*xx - 3*yy) * shc(sh, base, 12)
                   + SH_C3[4] * x * (4*zz - xx - yy) * shc(sh, base, 13)
                   + SH_C3[5] * z * (xx - yy) * shc(sh, base, 14)
                   + SH_C3[6] * x * (xx - 3*yy) * shc(sh, base, 15);
            }
        }
    }
    return r;
}

// ---------------------------------------------------------------------------------------------
// 1. Preprocess: 4D transform + EWA projection + SH colour + depth key
// ---------------------------------------------------------------------------------------------
kernel void preprocess(device const GaussianIn* gin     [[buffer(0)]],
                       device const float*      sh      [[buffer(1)]],
                       device const uint*       node_id [[buffer(2)]],
                       device const NodePose*   nodes   [[buffer(3)]],
                       constant FrameUniforms&  U       [[buffer(4)]],
                       device Splat*            splats  [[buffer(5)]],
                       device uint*             keys    [[buffer(6)]],
                       device uint*             vals    [[buffer(7)]],
                       device atomic_uint*      counter [[buffer(8)]],
                       uint i [[thread_position_in_grid]])
{
    if (i >= U.num_gaussians) return;
    GaussianIn G = gin[i];
    NodePose N = nodes[node_id[i]];
    if (N.pos.w < 0.5) return;                                  // node outside visible_range

    // temporal opacity
    float dt = U.time - G.mean_tc.w;
    float tz = dt * G.vel_its.w;
    float alpha = G.scale_op.w * exp(-0.5f * tz * tz);
    if (alpha < 1.0f / 255.0f) return;

    // mean at t, node -> world -> camera (OpenCV)
    float3x3 Rn = float3x3(N.c0.xyz, N.c1.xyz, N.c2.xyz);
    float3 ml = G.mean_tc.xyz + G.vel_its.xyz * dt;
    float3 mw = Rn * ml + N.pos.xyz;
    float3 pc = (U.world_to_cam * float4(mw, 1.0f)).xyz;
    if (pc.z < U.near_z) return;

    float fx = U.intr.x, fy = U.intr.y, cx = U.intr.z, cy = U.intr.w;
    float z = pc.z;
    float2 px = float2(fx * pc.x / z + cx, fy * pc.y / z + cy);

    // Σ_world = M Mᵀ with M = R(q_node ⊗ q) S ; camera-space T = W M
    float3x3 Rw = qmat(qmul(N.quat, G.quat));
    float3 s = G.scale_op.xyz;
    float3x3 M = float3x3(Rw[0] * s.x, Rw[1] * s.y, Rw[2] * s.z);
    float3x3 Wr = float3x3(U.world_to_cam[0].xyz, U.world_to_cam[1].xyz, U.world_to_cam[2].xyz);
    float3x3 T = Wr * M;

    // EWA Jacobian with the usual frustum clamp of x/z, y/z (≈ 3DGS 1.3·tan(fov/2))
    float lox = (U.clip_rect.x - cx) / fx, hix = (U.clip_rect.z - cx) / fx;
    float loy = (U.clip_rect.y - cy) / fy, hiy = (U.clip_rect.w - cy) / fy;
    float mx = 0.15f * (hix - lox), my = 0.15f * (hiy - loy);
    float txz = clamp(pc.x / z, lox - mx, hix + mx);
    float tyz = clamp(pc.y / z, loy - my, hiy + my);
    float3 J0 = float3(fx / z, 0.0f, -fx * txz / z);
    float3 J1 = float3(0.0f, fy / z, -fy * tyz / z);
    float3 A0 = float3(dot(J0, T[0]), dot(J0, T[1]), dot(J0, T[2]));
    float3 A1 = float3(dot(J1, T[0]), dot(J1, T[1]), dot(J1, T[2]));
    float a = dot(A0, A0) + U.low_pass;     // Σ2D = J W Σ Wᵀ Jᵀ + 0.3 I
    float b = dot(A0, A1);
    float c = dot(A1, A1) + U.low_pass;

    // eigen-decomposition → oriented 3σ quad
    float mid = 0.5f * (a + c);
    float disc = sqrt(max(0.25f * (a - c) * (a - c) + b * b, 0.0f));
    float l1 = mid + disc;
    float l2 = max(mid - disc, 1e-6f);
    float2 v1 = (fabs(b) > 1e-9f) ? normalize(float2(b, l1 - a)) : ((a >= c) ? float2(1, 0) : float2(0, 1));
    float2 v2 = float2(-v1.y, v1.x);
    float r1 = 3.0f * sqrt(l1);
    float r2 = 3.0f * sqrt(l2);
    float ex = sqrt(r1 * r1 * v1.x * v1.x + r2 * r2 * v2.x * v2.x);
    float ey = sqrt(r1 * r1 * v1.y * v1.y + r2 * r2 * v2.y * v2.y);
    if (px.x + ex < U.clip_rect.x || px.x - ex > U.clip_rect.z ||
        px.y + ey < U.clip_rect.y || px.y - ey > U.clip_rect.w) return;

    // view-dependent colour: direction in the node-local frame
    float3 dir = normalize(mw - U.cam_center.xyz);
    dir = transpose(Rn) * dir;
    float3 rgb = max(eval_sh(sh, i * U.sh_stride, U.sh_degree, dir) + 0.5f, 0.0f);

    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    Splat o;
    o.center_axis1 = float4(px, v1 * r1);
    o.axis2_depth  = float4(v2 * r2, z, 0.0f);
    o.color        = float4(rgb, alpha);
    splats[slot] = o;
    keys[slot] = 0xFFFFFFFFu - as_type<uint>(z);   // ascending key == descending depth (z > 0)
    vals[slot] = slot;
}

// ---------------------------------------------------------------------------------------------
// 2. Stable LSD radix sort (8-bit digits). Every kernel reads the element count from `counter`.
// hist layout: digit-major [RADIX][numTiles] so an exclusive scan yields global scatter offsets.
// ---------------------------------------------------------------------------------------------
kernel void radix_hist(device const uint* keys    [[buffer(0)]],
                       device uint*       hist    [[buffer(1)]],
                       device const uint* counter [[buffer(2)]],
                       constant uint&     shift   [[buffer(3)]],
                       uint tg  [[threadgroup_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]])
{
    threadgroup atomic_uint lh[RADIX];
    uint n = counter[0];
    uint tiles = (n + SORT_TILE - 1) / SORT_TILE;
    if (tg >= tiles) return;
    atomic_store_explicit(&lh[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = tg * SORT_TILE;
    for (uint k = 0; k < SORT_ITEMS; ++k) {
        uint idx = base + k * SORT_TG + lid;
        if (idx < n) atomic_fetch_add_explicit(&lh[(keys[idx] >> shift) & 0xFFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hist[lid * tiles + tg] = atomic_load_explicit(&lh[lid], memory_order_relaxed);
}

kernel void radix_scan(device uint*       hist    [[buffer(0)]],
                       device const uint* counter [[buffer(1)]],
                       uint lid  [[thread_index_in_threadgroup]],
                       uint tpg  [[threads_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sgid [[simdgroup_index_in_threadgroup]],
                       uint nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup uint sgSums[32];
    uint n = counter[0];
    uint tiles = (n + SORT_TILE - 1) / SORT_TILE;
    uint total = RADIX * tiles;
    uint per = (total + tpg - 1) / tpg;
    uint start = min(lid * per, total);
    uint end = min(start + per, total);
    uint sum = 0;
    for (uint i = start; i < end; ++i) sum += hist[i];
    uint pre = simd_prefix_exclusive_sum(sum);
    if (lane == 31) sgSums[sgid] = pre + sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        uint v = lane < nsg ? sgSums[lane] : 0u;
        uint p = simd_prefix_exclusive_sum(v);
        if (lane < nsg) sgSums[lane] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint off = sgSums[sgid] + pre;
    for (uint i = start; i < end; ++i) { uint v = hist[i]; hist[i] = off; off += v; }
}

kernel void radix_scatter(device const uint* keys_in  [[buffer(0)]],
                          device const uint* vals_in  [[buffer(1)]],
                          device uint*       keys_out [[buffer(2)]],
                          device uint*       vals_out [[buffer(3)]],
                          device const uint* hist     [[buffer(4)]],
                          device const uint* counter  [[buffer(5)]],
                          constant uint&     shift    [[buffer(6)]],
                          uint tg   [[threadgroup_position_in_grid]],
                          uint lid  [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint digitBase[RADIX];
    threadgroup uint sgCounts[SORT_SG][RADIX];
    uint n = counter[0];
    uint tiles = (n + SORT_TILE - 1) / SORT_TILE;
    if (tg >= tiles) return;
    digitBase[lid] = hist[lid * tiles + tg];
    uint ltMask = (1u << lane) - 1u;
    for (uint chunk = 0; chunk < SORT_ITEMS; ++chunk) {
        for (uint s = 0; s < SORT_SG; ++s) sgCounts[s][lid] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint idx = tg * SORT_TILE + chunk * SORT_TG + lid;
        bool valid = idx < n;
        uint key = valid ? keys_in[idx] : 0u;
        uint val = valid ? vals_in[idx] : 0u;
        uint d = (key >> shift) & 0xFFu;

        // lanes of this simdgroup holding the same digit (match via 8 ballots)
        uint peers = uint(simd_vote::vote_t(simd_ballot(valid)));
        for (uint bit = 0; bit < 8; ++bit) {
            bool on = ((d >> bit) & 1u) != 0u;
            uint vb = uint(simd_vote::vote_t(simd_ballot(on)));
            peers &= on ? vb : ~vb;
        }
        uint lower = peers & ltMask;
        uint rank = popcount(lower);
        if (valid && lower == 0u) sgCounts[sgid][d] = popcount(peers);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            uint pre = 0;
            for (uint s = 0; s < sgid; ++s) pre += sgCounts[s][d];
            uint dst = digitBase[d] + pre + rank;
            keys_out[dst] = key;
            vals_out[dst] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tot = 0;
        for (uint s = 0; s < SORT_SG; ++s) tot += sgCounts[s][lid];
        digitBase[lid] += tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void make_draw_args(device const uint* counter [[buffer(0)]],
                           device uint*       args    [[buffer(1)]],
                           uint i [[thread_position_in_grid]])
{
    // MTLDrawIndexedPrimitivesIndirectArguments: indexCount, instanceCount, indexStart, baseVertex, baseInstance
    if (i == 0) { args[0] = 6u * counter[0]; args[1] = 1u; args[2] = 0u; args[3] = 0u; args[4] = 0u; }
}

// ---------------------------------------------------------------------------------------------
// 4. Splat raster
// ---------------------------------------------------------------------------------------------
struct SplatVOut {
    float4 pos [[position]];
    float2 uv;                  // in σ units: |uv|² = Mahalanobis distance²
    float4 color [[flat]];
};

// Non-instanced indexed quads (vertex 4k+c = corner c of the k-th splat in sorted order): on Apple GPUs this
// is much cheaper than instancing a 4-vertex strip per splat; the index buffer lets the 2 shared corners reuse.
vertex SplatVOut splat_vs(uint vid [[vertex_id]],
                          device const Splat*     splats [[buffer(0)]],
                          device const uint*      order  [[buffer(1)]],
                          constant FrameUniforms& U      [[buffer(2)]])
{
    Splat s = splats[order[vid >> 2]];
    float2 corner = float2((vid & 1u) ? 1.0f : -1.0f, (vid & 2u) ? 1.0f : -1.0f);
    float2 p = s.center_axis1.xy + corner.x * s.center_axis1.zw + corner.y * s.axis2_depth.xy;
    SplatVOut o;
    o.pos = float4(p.x / U.viewport.x * 2.0f - 1.0f, 1.0f - p.y / U.viewport.y * 2.0f, 0.0f, 1.0f);
    o.uv = corner * 3.0f;
    o.color = s.color;
    return o;
}

fragment float4 splat_fs(SplatVOut in [[stage_in]])
{
    float d2 = dot(in.uv, in.uv);
    float a = min(0.99f, in.color.a * exp(-0.5f * d2));
    if (d2 > 9.0f || a < 1.0f / 255.0f) return float4(0.0f);   // no discard: keeps the TBDR fast path
    return float4(in.color.rgb * a, a);                         // premultiplied
}

// ---------------------------------------------------------------------------------------------
// 5. Overlays
// ---------------------------------------------------------------------------------------------
struct LineVOut { float4 pos [[position]]; float4 color; };

vertex LineVOut line_vs(uint vid [[vertex_id]], uint iid [[instance_id]],
                        device const LineSeg*   segs [[buffer(0)]],
                        constant FrameUniforms& U    [[buffer(1)]])
{
    LineSeg s = segs[iid];
    LineVOut o;
    o.color = float4(s.color.rgb * s.color.a, s.color.a);
    float3 ca = (U.world_to_cam * float4(s.a.xyz, 1.0f)).xyz;
    float3 cb = (U.world_to_cam * float4(s.b.xyz, 1.0f)).xyz;
    float nz = U.near_z;
    if (ca.z < nz && cb.z < nz) { o.pos = float4(0.0f, 0.0f, -1.0f, 1.0f); return o; }
    if (ca.z < nz) ca = mix(ca, cb, (nz - ca.z) / (cb.z - ca.z));
    if (cb.z < nz) cb = mix(cb, ca, (nz - cb.z) / (ca.z - cb.z));
    float2 pa = float2(U.intr.x * ca.x / ca.z + U.intr.z, U.intr.y * ca.y / ca.z + U.intr.w);
    float2 pb = float2(U.intr.x * cb.x / cb.z + U.intr.z, U.intr.y * cb.y / cb.z + U.intr.w);
    float2 d = pb - pa;
    float len = length(d);
    d = len > 1e-4f ? d / len : float2(1.0f, 0.0f);
    float2 nrm = float2(-d.y, d.x);
    float hw = 0.5f * s.a.w;
    float along = (vid & 1u) ? 1.0f : 0.0f;
    float side = (vid & 2u) ? 1.0f : -1.0f;
    float2 p = mix(pa, pb, along) + nrm * side * hw + d * (along * 2.0f - 1.0f) * hw;
    o.pos = float4(p.x / U.viewport.x * 2.0f - 1.0f, 1.0f - p.y / U.viewport.y * 2.0f, 0.0f, 1.0f);
    return o;
}

fragment float4 line_fs(LineVOut in [[stage_in]]) { return in.color; }

struct FillVOut { float4 pos [[position]]; };
vertex FillVOut fill_vs(uint vid [[vertex_id]]) {
    FillVOut o;
    o.pos = float4(vid == 1u ? 3.0f : -1.0f, vid == 2u ? 3.0f : -1.0f, 0.0f, 1.0f);
    return o;
}
fragment float4 fill_fs(constant FrameUniforms& U [[buffer(0)]]) { return U.fill; }
"""#
}
