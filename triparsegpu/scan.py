"""Hand-written exclusive scan and stream compaction (Problem 7).

Replaces cupy.cumsum and cupy.flatnonzero with the classic three-kernel
pattern: scan each block's tile, scan the block totals, add the offsets back.
"""
import cupy as cp
from triparsegpu.compile import kernel

BLOCK_SIZE = 512
ITEMS = 2 * BLOCK_SIZE


def exclusive_scan(x):
    """Exclusive prefix sum over an int64 array. Returns a new array."""
    x = x.astype(cp.int64, copy=False)
    n = int(x.size)
    if n == 0:
        return cp.empty(0, cp.int64)

    out = cp.empty(n, cp.int64)
    nblocks = (n + ITEMS - 1) // ITEMS
    block_sums = cp.empty(nblocks, cp.int64)

    k_scan = kernel("scan", "block_scan")
    k_add = kernel("scan", "add_offsets")

    k_scan((nblocks,), (BLOCK_SIZE,), (x, out, cp.int64(n), block_sums))

    if nblocks > 1:
        # recursively scan the block totals, then fold them back in
        scanned = exclusive_scan(block_sums)
        k_add((nblocks,), (BLOCK_SIZE,), (out, cp.int64(n), scanned))

    return out


def inclusive_scan(x):
    return exclusive_scan(x) + x.astype(cp.int64, copy=False)


def compact(flag):
    """Indices of non-zero entries, in order — a hand-written flatnonzero."""
    n = int(flag.size)
    if n == 0:
        return cp.empty(0, cp.int64)
    f64 = flag.astype(cp.int64)
    pos = exclusive_scan(f64)
    total = int(pos[-1].item() + f64[-1].item())
    out = cp.empty(total, cp.int64)
    k = kernel("scan", "compact_indices")
    tpb = 256
    k(((n + tpb - 1) // tpb,), (tpb,),
      (flag.astype(cp.uint8, copy=False), pos, cp.int64(n), out))
    return out
