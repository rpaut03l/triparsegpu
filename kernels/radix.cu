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
extern "C" __global__ void radix_scatter(
        const unsigned long long* __restrict__ keys_in,
        const long long* __restrict__ vals_in,
        long long n, int shift,
        const long long* __restrict__ offsets,  // scanned hist, bucket-major
        unsigned long long* __restrict__ keys_out,
        long long* __restrict__ vals_out)
{
    __shared__ unsigned long long sk[TPB];
    __shared__ long long sv[TPB];

    const int t = threadIdx.x;
    const long long i = blockIdx.x * (long long)blockDim.x + t;
    const bool active = (i < n);

    sk[t] = active ? keys_in[i] : 0ULL;
    sv[t] = active ? vals_in[i] : 0LL;
    __syncthreads();

    if (!active) return;

    const unsigned int d = (unsigned int)((sk[t] >> shift) & RMASK);

    // stable rank: how many earlier threads in this tile share my digit
    unsigned int rank = 0u;
    for (int q = 0; q < t; ++q) {
        const long long j = blockIdx.x * (long long)blockDim.x + q;
        if (j < n && (unsigned int)((sk[q] >> shift) & RMASK) == d) ++rank;
    }

    const long long dst = offsets[(long long)d * gridDim.x + blockIdx.x] + rank;
    keys_out[dst] = sk[t];
    vals_out[dst] = sv[t];
}
