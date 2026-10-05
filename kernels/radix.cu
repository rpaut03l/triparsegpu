#include "common.cuh"

// Least-significant-digit radix sort over uint64 keys, carrying an int64 value.
// One pass per RADIX_BITS: histogram -> exclusive scan -> stable scatter.
// Histogram is stored bucket-major ([digit][block]) so a single scan over it
// gives every block the global offset for each of its digits.

#ifndef RADIX_BITS
#define RADIX_BITS 8
#endif
#define RADIX (1 << RADIX_BITS)
#define RMASK (RADIX - 1)

#ifndef TPB
#define TPB 256
#endif

extern "C" __global__ void radix_histogram(
        const unsigned long long* __restrict__ keys, long long n, int shift,
        long long* __restrict__ hist)          // [RADIX][gridDim.x], bucket-major
{
    __shared__ unsigned int h[RADIX];
    for (int i = threadIdx.x; i < RADIX; i += blockDim.x) h[i] = 0u;
    __syncthreads();

    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&h[(keys[i] >> shift) & RMASK], 1u);
    __syncthreads();

    for (int d = threadIdx.x; d < RADIX; d += blockDim.x)
        hist[(long long)d * gridDim.x + blockIdx.x] = (long long)h[d];
}

// Stable scatter. Each block re-counts its own digits, derives its local
// starting offset from the scanned histogram, and ranks equal digits by their
// position within the block so the sort stays stable.
// Stable scatter with warp-level ranking.
// Rank = (equal-digit threads earlier in my warp) + (equal-digit threads in
// earlier warps of this tile). The first comes from a ballot; the second from
// a small scan over per-warp digit counts held in shared memory.
#define WARPS (TPB / 32)

extern "C" __global__ void radix_scatter(
        const unsigned long long* __restrict__ keys_in,
        const long long* __restrict__ vals_in,
        long long n, int shift,
        const long long* __restrict__ offsets,  // scanned hist, bucket-major
        unsigned long long* __restrict__ keys_out,
        long long* __restrict__ vals_out)
{
    __shared__ unsigned int warp_cnt[WARPS][RADIX];   // per-warp digit counts

    const int t    = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const long long i = blockIdx.x * (long long)blockDim.x + t;
    const bool active = (i < n);

    const unsigned long long key = active ? keys_in[i] : 0ULL;
    const long long val = active ? vals_in[i] : 0LL;
    const unsigned int d = active ? (unsigned int)((key >> shift) & RMASK) : 0u;

    for (int q = t; q < WARPS * RADIX; q += blockDim.x)
        ((unsigned int*)warp_cnt)[q] = 0u;
    __syncthreads();

    // --- rank within my warp, among threads sharing my digit ---
    unsigned int peers = 0xffffffffu;
    #pragma unroll
    for (int b = 0; b < RADIX_BITS; ++b) {
        const unsigned int bit = (d >> b) & 1u;
        const unsigned int mask = __ballot_sync(0xffffffffu, bit);
        peers &= bit ? mask : ~mask;
    }
    const unsigned int act = __ballot_sync(0xffffffffu, active);
    peers &= act;
    const unsigned int lane_rank = __popc(peers & ((1u << lane) - 1u));
    const unsigned int peer_total = __popc(peers);
    const int leader = __ffs(peers) - 1;

    if (active && lane == leader) warp_cnt[warp][d] = peer_total;
    __syncthreads();

    // --- offset of my warp among earlier warps with the same digit ---
    unsigned int warp_off = 0u;
    if (active) for (int w = 0; w < warp; ++w) warp_off += warp_cnt[w][d];

    if (active) {
        const long long dst = offsets[(long long)d * gridDim.x + blockIdx.x]
                            + warp_off + lane_rank;
        keys_out[dst] = key;
        vals_out[dst] = val;
    }
}
