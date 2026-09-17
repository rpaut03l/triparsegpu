#include "common.cuh"

// One block per component. A token position keeps its hash only if EVERY line
// in the component agrees there; otherwise it becomes a wildcard (0).
// comp_lines is the line ids grouped by component; comp_start/comp_end delimit
// each group. All lines in a component share a token count (same bucket key).
extern "C" __global__ void consensus(
        const unsigned long long* __restrict__ tok_hash,
        const long long* __restrict__ line_tok_offs,
        const long long* __restrict__ comp_lines,
        const long long* __restrict__ comp_start,
        const long long* __restrict__ comp_end,
        const int* __restrict__ comp_len,        // token count per component
        int max_len,
        unsigned long long* __restrict__ tpl_hash)   // [ncomp][max_len]
{
    const long long c = blockIdx.x;
    const long long a = comp_start[c], b = comp_end[c];
    const int len = comp_len[c];
    const long long o0 = line_tok_offs[comp_lines[a]];

    for (int k = threadIdx.x; k < len; k += blockDim.x) {
        unsigned long long h = tok_hash[o0 + k];
        bool same = (h != 0ULL);                 // already a wildcard -> stays
        for (long long q = a + 1; q < b && same; ++q)
            same = (tok_hash[line_tok_offs[comp_lines[q]] + k] == h);
        tpl_hash[c * max_len + k] = same ? h : 0ULL;
    }
}
