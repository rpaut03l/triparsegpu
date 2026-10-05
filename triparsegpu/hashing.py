"""Stage 2: hash every token; numeric/IP/hex tokens become wildcards (hash 0)."""
import cupy as cp
from triparsegpu.compile import kernel

_TPB = 256


def hash_tokens(buf, tok_start, tok_end):
    ntok = int(tok_start.size)
    out_hash = cp.empty(ntok, cp.uint64)
    out_wild = cp.empty(ntok, cp.uint8)
    out_bhash = cp.empty(ntok, cp.uint64)
    k = kernel("hash", "hash_tokens")
    grid = (ntok + _TPB - 1) // _TPB
    k((grid,), (_TPB,), (buf, tok_start, tok_end,
                         cp.int64(ntok), out_hash, out_wild, out_bhash))
    return out_hash, out_wild, out_bhash
