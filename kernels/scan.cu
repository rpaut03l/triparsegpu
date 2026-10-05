#include "common.cuh"

// Work-efficient (Blelloch) exclusive scan over a block's tile, plus the
// three-kernel pattern that combines per-block results into a full-array scan.
//   1. block_scan      - scan each tile, write the tile's total to block_sums
//   2. scan_block_sums - scan block_sums (single block, assumes <= 2*BS entries)
//   3. add_offsets     - add each tile's exclusive prefix back into its tile
// ITEMS_PER_BLOCK = 2 * BLOCK_SIZE: Blelloch handles two elements per thread.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#define ITEMS (2 * BLOCK_SIZE)

// Exclusive scan of `n` int64 values. Each block handles ITEMS elements.
extern "C" __global__ void block_scan(
        const long long* __restrict__ in, long long* __restrict__ out,
        long long n, long long* __restrict__ block_sums)
{
    __shared__ long long s[ITEMS];
    const int t = threadIdx.x;
    const long long base = blockIdx.x * (long long)ITEMS;

    s[2*t]     = (base + 2*t     < n) ? in[base + 2*t]     : 0LL;
    s[2*t + 1] = (base + 2*t + 1 < n) ? in[base + 2*t + 1] : 0LL;
    __syncthreads();

    // up-sweep (reduce)
    int offset = 1;
    for (int d = ITEMS >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (t < d) {
            int ai = offset * (2*t + 1) - 1;
            int bi = offset * (2*t + 2) - 1;
            s[bi] += s[ai];
        }
        offset <<= 1;
    }

    // record the tile total, then clear the root for the down-sweep
    if (t == 0) {
        if (block_sums) block_sums[blockIdx.x] = s[ITEMS - 1];
        s[ITEMS - 1] = 0LL;
    }

    // down-sweep
    for (int d = 1; d < ITEMS; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            int ai = offset * (2*t + 1) - 1;
            int bi = offset * (2*t + 2) - 1;
            long long tmp = s[ai];
            s[ai] = s[bi];
            s[bi] += tmp;
        }
    }
    __syncthreads();

    if (base + 2*t     < n) out[base + 2*t]     = s[2*t];
    if (base + 2*t + 1 < n) out[base + 2*t + 1] = s[2*t + 1];
}

// Add each block's exclusive prefix (from the scanned block_sums) to its tile.
extern "C" __global__ void add_offsets(
        long long* __restrict__ data, long long n,
        const long long* __restrict__ scanned_sums)
{
    const long long i = blockIdx.x * (long long)ITEMS + threadIdx.x;
    const long long add = scanned_sums[blockIdx.x];
    if (i < n)                 data[i]              += add;
    if (i + BLOCK_SIZE < n &&
        (i + BLOCK_SIZE) / ITEMS == blockIdx.x) data[i + BLOCK_SIZE] += add;
}

// Stream compaction: write the index of every non-zero flag, in order.
// `pos` must already hold the exclusive scan of the flags.
extern "C" __global__ void compact_indices(
        const unsigned char* __restrict__ flag,
        const long long* __restrict__ pos,
        long long n, long long* __restrict__ out)
{
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n && flag[i]) out[pos[i]] = i;
}
