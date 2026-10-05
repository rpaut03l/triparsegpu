"""Stage 8 (Problem 2): template co-occurrence graph in CSR + BFS.

Two templates co-occur if their lines appear within `window` of each other.
BFS from a template answers "which events follow from this one, and how far" —
the structure an on-call engineer needs when one alert fires.
"""
import cupy as cp
from triparsegpu.compile import kernel
from triparsegpu import scan as _scan

_TPB = 256


def build_csr(line_tpl, ntpl, window=10):
    """Returns (row_ptr int64[ntpl+1], col_idx int32[nnz])."""
    nlines = int(line_tpl.size)
    row_count = cp.zeros(ntpl, cp.int32)
    grid = ((nlines + _TPB - 1) // _TPB,)

    k_cnt = kernel("cooccur", "cooccur_count")
    k_cnt(grid, (_TPB,), (line_tpl, cp.int64(nlines), cp.int32(window),
                          cp.int32(ntpl), row_count))

    row_ptr = cp.empty(ntpl + 1, cp.int64)
    row_ptr[:-1] = _scan.exclusive_scan(row_count)        # our own scan
    row_ptr[-1] = row_ptr[-2] + row_count[-1]
    nnz = int(row_ptr[-1].item())

    col_idx = cp.empty(max(nnz, 1), cp.int32)
    cursor = cp.zeros(ntpl, cp.int32)
    k_fill = kernel("cooccur", "cooccur_fill")
    k_fill(grid, (_TPB,), (line_tpl, cp.int64(nlines), cp.int32(window),
                           row_ptr, cursor, col_idx))
    return row_ptr, col_idx, nnz


def bfs(row_ptr, col_idx, ntpl, source, max_levels=1000):
    """Level-synchronous BFS. Returns depth[ntpl]; -1 means unreachable."""
    depth = cp.full(ntpl, -1, cp.int32)
    depth[source] = 0
    changed = cp.zeros(1, cp.int32)
    k = kernel("cooccur", "bfs_step")
    grid = ((ntpl + _TPB - 1) // _TPB,)
    for level in range(max_levels):
        changed.fill(0)
        k(grid, (_TPB,), (row_ptr, col_idx, depth, cp.int32(level),
                          cp.int32(ntpl), changed))
        if int(changed.get()[0]) == 0:
            return depth, level + 1
    return depth, max_levels
