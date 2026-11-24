// Backward pass of rasterize_forward: back-to-front traversal per pixel, gradients reduced
// across the SIMD group (all lanes visit the same Gaussian in lockstep) before one atomic add.
// Inputs:  ranges, point_ids, means2d, conic_op, feats   (as forward)
//          out_T (H,W), n_contrib (H,W)                 (forward outputs)
//          dL_dfeat (H,W,C), dL_dT (H,W)                (cotangents)
// Outputs (atomic, zero-initialised): d_means2d (N,2), d_conic_op (N,4), d_feats (N,C)
uint2 gid = thread_position_in_grid.xy;
uint2 tg = threadgroup_position_in_grid.xy;
uint lid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint tile = tg.y * TILES_X + tg.x;
bool inside = gid.x < W && gid.y < H;
float px = (float)gid.x + 0.5f;
float py = (float)gid.y + 0.5f;
uint pix = inside ? gid.y * W + gid.x : 0;

uint start = ranges[2 * tile];
uint end = ranges[2 * tile + 1];

threadgroup uint s_id[SPLAT_BATCH];
threadgroup float2 s_xy[SPLAT_BATCH];
threadgroup float4 s_co[SPLAT_BATCH];
threadgroup float s_f[SPLAT_BATCH * C];

float T_final = inside ? out_T[pix] : 1.0f;
float T = T_final;
uint last = inside ? n_contrib[pix] : 0;
float dL_dpix[C];
float accum[C];
float last_f[C];
for (int ch = 0; ch < C; ++ch) {
    dL_dpix[ch] = inside ? dL_dfeat[pix * C + ch] : 0.0f;
    accum[ch] = 0.0f;
    last_f[ch] = 0.0f;
}
float dL_dTf = inside ? dL_dT[pix] : 0.0f;
float last_alpha = 0.0f;

for (int base = (int)end; base > (int)start; base -= (int)SPLAT_BATCH) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int idx = base - 1 - (int)lid;
    if (idx >= (int)start) {
        uint g = point_ids[idx];
        s_id[lid] = g;
        s_xy[lid] = float2(means2d[2 * g], means2d[2 * g + 1]);
        s_co[lid] = float4(conic_op[4 * g], conic_op[4 * g + 1], conic_op[4 * g + 2], conic_op[4 * g + 3]);
        for (int ch = 0; ch < C; ++ch) s_f[lid * C + ch] = feats[g * C + ch];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cnt = min(SPLAT_BATCH, (uint)(base - (int)start));
    for (uint j = 0; j < cnt; ++j) {
        uint rank = (uint)(base - 1 - (int)j) - start + 1;
        bool active = inside && rank <= last;
        float g_mx = 0.0f, g_my = 0.0f, g_a = 0.0f, g_b = 0.0f, g_c = 0.0f, g_op = 0.0f;
        float g_f[C];
        for (int ch = 0; ch < C; ++ch) g_f[ch] = 0.0f;

        float4 co = s_co[j];
        float2 xy = s_xy[j];
        float dx = px - xy.x;
        float dy = py - xy.y;
        float power = splat_power(dx, dy, co.x, co.y, co.z);
        float G = exp(power);
        float raw = co.w * G;
        float alpha = min(SPLAT_ALPHA_MAX, raw);
        active = active && power <= 0.0f && alpha >= SPLAT_ALPHA_MIN;

        if (active) {
            T = T / (1.0f - alpha);
            float w = alpha * T;
            float dL_dalpha = 0.0f;
            for (int ch = 0; ch < C; ++ch) {
                float f = s_f[j * C + ch];
                accum[ch] = last_alpha * last_f[ch] + (1.0f - last_alpha) * accum[ch];
                last_f[ch] = f;
                dL_dalpha += (f - accum[ch]) * dL_dpix[ch];
                g_f[ch] = w * dL_dpix[ch];
            }
            dL_dalpha *= T;
            last_alpha = alpha;
            dL_dalpha += (-T_final / (1.0f - alpha)) * dL_dTf;
            if (raw < SPLAT_ALPHA_MAX) {
                g_op = G * dL_dalpha;
                float dL_dpower = dL_dalpha * alpha;
                g_mx = dL_dpower * (co.x * dx + co.y * dy);
                g_my = dL_dpower * (co.z * dy + co.y * dx);
                g_a = -0.5f * dx * dx * dL_dpower;
                g_b = -dx * dy * dL_dpower;
                g_c = -0.5f * dy * dy * dL_dpower;
            }
        }

        if (simd_any(active)) {
            g_mx = simd_sum(g_mx);
            g_my = simd_sum(g_my);
            g_a = simd_sum(g_a);
            g_b = simd_sum(g_b);
            g_c = simd_sum(g_c);
            g_op = simd_sum(g_op);
            for (int ch = 0; ch < C; ++ch) g_f[ch] = simd_sum(g_f[ch]);
            if (lane == 0) {
                uint g = s_id[j];
                atomic_fetch_add_explicit(&d_means2d[2 * g], g_mx, memory_order_relaxed);
                atomic_fetch_add_explicit(&d_means2d[2 * g + 1], g_my, memory_order_relaxed);
                atomic_fetch_add_explicit(&d_conic_op[4 * g], g_a, memory_order_relaxed);
                atomic_fetch_add_explicit(&d_conic_op[4 * g + 1], g_b, memory_order_relaxed);
                atomic_fetch_add_explicit(&d_conic_op[4 * g + 2], g_c, memory_order_relaxed);
                atomic_fetch_add_explicit(&d_conic_op[4 * g + 3], g_op, memory_order_relaxed);
                for (int ch = 0; ch < C; ++ch)
                    atomic_fetch_add_explicit(&d_feats[g * C + ch], g_f[ch], memory_order_relaxed);
            }
        }
    }
}
