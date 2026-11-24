// One thread per Gaussian: emit one (tile << 32 | depth_bits) key per overlapped tile.
// Inputs:  offsets  (N,) uint32 exclusive prefix sum of tile counts
//          rects    (N,4) int32 [tx0, ty0, tx1, ty1) tile-space bounding box (empty when culled)
//          depths   (N,) float32 camera-space z (> 0 for visible Gaussians)
//          capacity (1,) uint32 size of the output buffers (entries beyond it are dropped)
// Outputs: keys (capacity,) uint64, values (capacity,) uint32 Gaussian index
uint i = thread_position_in_grid.x;
if (i >= offsets_shape[0]) return;
uint cap = capacity[0];
int x0 = rects[4 * i + 0];
int y0 = rects[4 * i + 1];
int x1 = rects[4 * i + 2];
int y1 = rects[4 * i + 3];
uint off = offsets[i];
// Positive IEEE floats sort correctly as unsigned integers.
ulong dbits = (ulong)as_type<uint>(depths[i]);
for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
        if (off >= cap) return;
        ulong tile = (ulong)(y * TILES_X + x);
        keys[off] = (tile << 32) | dbits;
        values[off] = i;
        off++;
    }
}
