#include "common.cuh"

// Connected-component labelling by hooking + pointer jumping (Problem 3).
// label[i] is i's parent; a root satisfies label[r] == r.
// Iterate hook->compress until no label changes. Convergence is a single
// int read back to the host per round: no array transfer.

extern "C" __global__ void ccl_init(long long* __restrict__ label, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) label[i] = i;
}

__device__ __forceinline__ long long find_root(const long long* label, long long x) {
    while (label[x] != x) x = label[x];
    return x;
}

// For each edge, attach the higher root to the lower one.
extern "C" __global__ void ccl_hook(
        const long long* __restrict__ eu, const long long* __restrict__ ev,
        long long ne, long long* __restrict__ label, int* __restrict__ changed)
{
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= ne) return;
    long long ru = find_root(label, eu[e]);
    long long rv = find_root(label, ev[e]);
    if (ru == rv) return;
    long long hi = ru > rv ? ru : rv;
    long long lo = ru < rv ? ru : rv;
    // atomicMin on the higher root; if it moved, we changed something
    unsigned long long old = atomicMin((unsigned long long*)&label[hi],
                                       (unsigned long long)lo);
    if ((long long)old != lo) *changed = 1;
}

// Path compression: every node points straight at its root.
extern "C" __global__ void ccl_compress(long long* __restrict__ label, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long r = find_root(label, i);
    if (label[i] != r) label[i] = r;
}
