// Tile-based front-to-back alpha compositing of C-channel features (RGB, depth, flow, ...).
// One threadgroup per TILE x TILE tile, one thread per pixel.
// Inputs:  ranges    (num_tiles, 2) uint32  [start, end) into point_ids
//          point_ids (T,)  uint32  Gaussian index, sorted by (tile, depth)
//          means2d   (N,2) float32 pixel-space centers
//          conic_op  (N,4) float32 [a, b, c, opacity]
//          feats     (N,C) float32
// Outputs: out_feat (H,W,C) float32, out_T (H,W) float32 final transmittance,
//          n_contrib (H,W) uint32 number of entries traversed up to the last contributor.
uint2 gid = thread_position_in_grid.xy;
uint2 tg = threadgroup_position_in_grid.xy;
uint lid = thread_index_in_threadgroup;
uint tile = tg.y * TILES_X + tg.x;
bool inside = gid.x < W && gid.y < H;
float px = (float)gid.x + 0.5f;
float py = (float)gid.y + 0.5f;

uint start = ranges[2 * tile];
uint end = ranges[2 * tile + 1];

threadgroup uint s_id[SPLAT_BATCH];
threadgroup float2 s_xy[SPLAT_BATCH];
threadgroup float4 s_co[SPLAT_BATCH];
threadgroup atomic_int s_done;

float T = 1.0f;
float acc[C];
for (int ch = 0; ch < C; ++ch) acc[ch] = 0.0f;
uint contrib = 0;
uint last = 0;
bool done = !inside;

for (uint base = start; base < end; base += SPLAT_BATCH) {
    if (lid == 0) atomic_store_explicit(&s_done, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (done) atomic_fetch_add_explicit(&s_done, 1, memory_order_relaxed);
    uint idx = base + lid;
    if (idx < end) {
        uint g = point_ids[idx];
        s_id[lid] = g;
        s_xy[lid] = float2(means2d[2 * g], means2d[2 * g + 1]);
        s_co[lid] = float4(conic_op[4 * g], conic_op[4 * g + 1], conic_op[4 * g + 2], conic_op[4 * g + 3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&s_done, memory_order_relaxed) == (int)SPLAT_BATCH) break;

    uint cnt = min(SPLAT_BATCH, end - base);
    for (uint j = 0; j < cnt && !done; ++j) {
        contrib++;
        float2 xy = s_xy[j];
        float4 co = s_co[j];
        float dx = px - xy.x;
        float dy = py - xy.y;
        float power = splat_power(dx, dy, co.x, co.y, co.z);
        if (power > 0.0f) continue;
        float alpha = min(SPLAT_ALPHA_MAX, co.w * exp(power));
        if (alpha < SPLAT_ALPHA_MIN) continue;
        float test_T = T * (1.0f - alpha);
        if (test_T < SPLAT_T_MIN) { done = true; continue; }
        uint g = s_id[j];
        float w = alpha * T;
        for (int ch = 0; ch < C; ++ch) acc[ch] += feats[g * C + ch] * w;
        T = test_T;
        last = contrib;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (inside) {
    uint pix = gid.y * W + gid.x;
    for (int ch = 0; ch < C; ++ch) out_feat[pix * C + ch] = acc[ch];
    out_T[pix] = T;
    n_contrib[pix] = last;
}
