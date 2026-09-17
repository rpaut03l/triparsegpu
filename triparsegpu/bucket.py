"""Stages 3-4: build a bucket key per line, sort lines by it, find segments.

Drain's fixed-depth tree routes a line by (token count, first token, second
token). We encode the same routing into one 64-bit key and sort — lines that
Drain would put in the same leaf land in the same contiguous segment.
"""
import cupy as cp


def bucket_keys(tok_hash, line_tok_offs, line_tok_count, depth_tokens=2):
    """key = [count:16][first:24][second:24], so sorting groups by count then prefix."""
    nlines = int(line_tok_count.size)
    starts = line_tok_offs[:-1]
    ntok = int(tok_hash.size)

    first = cp.where(line_tok_count > 0,
                     tok_hash[cp.minimum(starts, ntok - 1)],
                     cp.uint64(0))
    if depth_tokens >= 2:
        second = cp.where(line_tok_count > 1,
                          tok_hash[cp.minimum(starts + 1, ntok - 1)],
                          cp.uint64(0))
    else:
        second = cp.zeros(nlines, cp.uint64)

    cnt = cp.minimum(line_tok_count.astype(cp.uint64), cp.uint64(0xFFFF))
    key = (cnt << cp.uint64(48)) \
        | ((first & cp.uint64(0xFFFFFF)) << cp.uint64(24)) \
        | (second & cp.uint64(0xFFFFFF))
    return key


def sort_lines(key, tier=1):
    """Sort line indices by key. Returns (order, seg_start, seg_end, seg_of_pos)."""
    if tier == 1:
        order = cp.argsort(key, kind="stable").astype(cp.int64)   # CUB radix sort
    else:
        from triparsegpu.radix import radix_argsort_u64           # tier 2, hand-written
        order = radix_argsort_u64(key)

    sk = key[order]
    new_seg = cp.empty(sk.size, cp.bool_)
    new_seg[0] = True
    new_seg[1:] = sk[1:] != sk[:-1]

    seg_start = cp.flatnonzero(new_seg).astype(cp.int64)
    seg_end = cp.concatenate([seg_start[1:], cp.array([sk.size], cp.int64)])
    seg_of_pos = (cp.cumsum(new_seg, dtype=cp.int64) - 1)          # segment id per sorted position
    return order, seg_start, seg_end, seg_of_pos
