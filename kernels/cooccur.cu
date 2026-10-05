#include "common.cuh"

// Problem 2. Build a template co-occurrence graph, then BFS over it in CSR.
// Two templates co-occur if they appear within WINDOW lines of each other.
// BFS answers: which events are reachable from a given event, and how far.

// Count co-occurrence edges per source template (CSR row lengths).
extern "C" __global__ void cooccur_count(
        const long long* __restrict__ line_tpl, long long nlines,
        int window, int ntpl, int* __restrict__ row_count)
{
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= nlines) return;
    const long long a = line_tpl[i];
    const long long hi = min(nlines, i + (long long)window + 1);
    for (long long j = i + 1; j < hi; ++j) {
        const long long b = line_tpl[j];
        if (a != b) atomicAdd(&row_count[a], 1);
    }
}

// Fill CSR column indices using a per-row cursor.
extern "C" __global__ void cooccur_fill(
        const long long* __restrict__ line_tpl, long long nlines,
        int window, const long long* __restrict__ row_ptr,
        int* __restrict__ cursor, int* __restrict__ col_idx)
{
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= nlines) return;
    const long long a = line_tpl[i];
    const long long hi = min(nlines, i + (long long)window + 1);
    for (long long j = i + 1; j < hi; ++j) {
        const long long b = line_tpl[j];
        if (a == b) continue;
        const int slot = atomicAdd(&cursor[a], 1);
        col_idx[row_ptr[a] + slot] = (int)b;
    }
}

// One BFS level: expand the current frontier, mark newly reached vertices.
extern "C" __global__ void bfs_step(
        const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
        int* __restrict__ depth, int level, int ntpl, int* __restrict__ changed)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= ntpl || depth[v] != level) return;
    for (long long e = row_ptr[v]; e < row_ptr[v + 1]; ++e) {
        const int u = col_idx[e];
        if (depth[u] == -1) { depth[u] = level + 1; *changed = 1; }
    }
}
