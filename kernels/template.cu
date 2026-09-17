#include "common.cuh"

// Initialise each component's template from its representative line.
extern "C" __global__ void consensus_init(
        const unsigned long long* __restrict__ tok_hash,
        const long long* __restrict__ line_tok_offs,
        const long long* __restrict__ comp_lines,
        const long long* __restrict__ comp_start,
        const int* __restrict__ comp_len,
        long long ncomp, int max_len,
        unsigned long long* __restrict__ tpl_hash)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long c = i / max_len;
    int k = (int)(i % max_len);
    if (c >= ncomp) return;
    if (k >= comp_len[c]) { tpl_hash[c * max_len + k] = 0ULL; return; }
    const long long o0 = line_tok_offs[comp_lines[comp_start[c]]];
    tpl_hash[c * max_len + k] = tok_hash[o0 + k];
}

// One thread per (line, token position). If this line disagrees with the
// template at position k, clear it to a wildcard. Races are benign: every
// writer writes the same value (0).
extern "C" __global__ void consensus_reduce(
        const unsigned long long* __restrict__ tok_hash,
        const long long* __restrict__ line_tok_offs,
        const long long* __restrict__ comp_lines,        // lines sorted by component
        const long long* __restrict__ comp_of_sorted,    // component id per sorted position
        const int* __restrict__ comp_len,
        long long nlines, int max_len,
        unsigned long long* __restrict__ tpl_hash)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long p = i / max_len;                 // sorted position
    int k = (int)(i % max_len);
    if (p >= nlines) return;

    const long long c = comp_of_sorted[p];
    if (k >= comp_len[c]) return;

    const long long o = line_tok_offs[comp_lines[p]];
    const unsigned long long mine = tok_hash[o + k];
    unsigned long long* slot = &tpl_hash[c * max_len + k];
    if (*slot != 0ULL && *slot != mine) *slot = 0ULL;
}
