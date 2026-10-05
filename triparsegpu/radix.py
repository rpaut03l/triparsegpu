"""Hand-written LSD radix sort over uint64 keys (Problem 7).

Replaces cupy.argsort in bucket.py. Each pass: histogram -> exclusive scan
(our own, from scan.py) -> stable scatter.
"""
import cupy as cp
from triparsegpu.compile import kernel
from triparsegpu import scan as _scan

TPB = 256


def _defines(bits):
    return (f"-DRADIX_BITS={bits}", f"-DTPB={TPB}")


def sort_pairs(keys, vals, bits_per_pass=8, key_bits=64):
    """Stable sort of (uint64 keys, int64 vals) by key. Returns new arrays."""
    n = int(keys.size)
    if n == 0:
        return keys.copy(), vals.copy()

    radix = 1 << bits_per_pass
    k_hist = kernel("radix", "radix_histogram", _defines(bits_per_pass))
    k_scat = kernel("radix", "radix_scatter", _defines(bits_per_pass))

    blocks = max(1, (n + TPB - 1) // TPB)   # one contiguous tile per block
    kin = keys.astype(cp.uint64, copy=True)
    vin = vals.astype(cp.int64, copy=True)
    kout = cp.empty_like(kin)
    vout = cp.empty_like(vin)
    hist = cp.empty(radix * blocks, cp.int64)

    for shift in range(0, key_bits, bits_per_pass):
        hist.fill(0)
        k_hist((blocks,), (TPB,), (kin, cp.int64(n), cp.int32(shift), hist))
        offsets = _scan.exclusive_scan(hist)
        k_scat((blocks,), (TPB,), (kin, vin, cp.int64(n), cp.int32(shift),
                                   offsets, kout, vout))
        kin, kout = kout, kin
        vin, vout = vout, vin

    return kin, vin


def argsort(keys, bits_per_pass=8, key_bits=64):
    """Stable argsort of a uint64 array — the drop-in for cupy.argsort."""
    idx = cp.arange(keys.size, dtype=cp.int64)
    _, order = sort_pairs(keys, idx, bits_per_pass, key_bits)
    return order
