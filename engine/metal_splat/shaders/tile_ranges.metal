// One thread per sorted key: write [start, end) of each tile's run into ranges (num_tiles, 2).
// Only the first `total[0]` keys are valid (the rest are sentinels sorted to the end).
// ranges must be zero-initialised (tiles without Gaussians keep start == end == 0).
uint k = thread_position_in_grid.x;
uint n = min((uint)keys_shape[0], total[0]);
if (k >= n) return;
uint tile = (uint)(keys[k] >> 32);
if (k == 0 || (uint)(keys[k - 1] >> 32) != tile) {
    ranges[2 * tile] = k;
}
if (k == n - 1 || (uint)(keys[k + 1] >> 32) != tile) {
    ranges[2 * tile + 1] = k + 1;
}
