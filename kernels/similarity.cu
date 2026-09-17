#include "common.cuh"

// One warp per sorted position. Compare line at `pos` against up to WINDOW
// earlier lines inside the SAME segment. Drain similarity:
//     sim = (# positions where token hashes are equal) / token_count
// Wildcards (hash 0) match other wildcards. Emit an edge when sim >= st.
// The CCL stage transitively closes clusters, so a bounded window still
// connects everything Drain's sequential scan would.
extern "C" __global__ void similarity_edges(
        const unsigned long long* __restrict__ tok_hash,
        const long long* __restrict__ line_tok_offs,
        const int* __restrict__ line_tok_count,
        const long long* __restrict__ order,
        const long long* __restrict__ seg_of_pos,
        const long long* __restrict__ seg_start,
        long long nlines, float st, int window,
        long long* __restrict__ edge_u,
        long long* __restrict__ edge_v,
        unsigned long long* __restrict__ edge_cnt,
        unsigned long long edge_cap)
{
    const long long gtid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long pos  = gtid >> 5;              // one warp per position
    const int lane = threadIdx.x & 31;
    if (pos >= nlines) return;

    const long long s0 = seg_start[seg_of_pos[pos]];
    const long long li = order[pos];
    const int len = line_tok_count[li];
    if (len == 0) return;
    const long long oi = line_tok_offs[li];

    const long long jlo = max(s0, pos - (long long)window);
    for (long long q = jlo; q < pos; ++q) {
        const long long lj = order[q];
        const long long oj = line_tok_offs[lj];

        int match = 0;
        for (int k = lane; k < len; k += 32) {
            const unsigned long long a = tok_hash[oi + k];
            const unsigned long long b = tok_hash[oj + k];
            match += (a == b) ? 1 : 0;             // wildcards are both 0 -> match
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            match += __shfl_down_sync(0xffffffffu, match, off);

        if (lane == 0 && (float)match / (float)len >= st) {
            const unsigned long long e = atomicAdd(edge_cnt, 1ULL);
            if (e < edge_cap) { edge_u[e] = li; edge_v[e] = lj; }
        }
    }
}
