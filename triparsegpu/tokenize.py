"""Stage 1: find token boundaries in the flat byte buffer.

Tier 1 uses CuPy primitives (cumsum = scan, flatnonzero = compaction).
Tier 2 will swap cumsum for kernels/scan.cu.
"""
import cupy as cp


def tokenize(buf, line_offs, tier=1):
    """buf: uint8 [nbytes], line_offs: int64 [nlines+1] (offsets into buf,
    each line terminated by '\\n').

    Returns:
      tok_start      int64 [ntok]  first byte of each token
      tok_end        int64 [ntok]  one past last byte
      tok_line       int64 [ntok]  which line each token belongs to
      line_tok_count int32 [nlines]
      line_tok_offs  int64 [nlines+1]  prefix sum of line_tok_count
    """
    nlines = line_offs.size - 1
    is_delim = (buf == 32) | (buf == 10) | (buf == 9)
    prev_delim = cp.empty_like(is_delim)
    prev_delim[0] = True
    prev_delim[1:] = is_delim[:-1]

    tok_start_flag = (~is_delim) & prev_delim          # first byte of a token
    tok_end_flag = is_delim & (~prev_delim)            # first delim after a token

    tok_start = cp.flatnonzero(tok_start_flag).astype(cp.int64)
    tok_end = cp.flatnonzero(tok_end_flag).astype(cp.int64)
    if tok_end.size < tok_start.size:                  # buffer not delim-terminated
        tok_end = cp.concatenate([tok_end, cp.array([buf.size], cp.int64)])
    tok_end = tok_end[: tok_start.size]

    # which line owns each token: lines are contiguous, so searchsorted on starts
    tok_line = (cp.searchsorted(line_offs, tok_start, side="right") - 1).astype(cp.int64)

    line_tok_count = cp.bincount(tok_line, minlength=nlines).astype(cp.int32)
    line_tok_offs = cp.empty(nlines + 1, cp.int64)
    line_tok_offs[0] = 0
    cp.cumsum(line_tok_count, dtype=cp.int64, out=line_tok_offs[1:])   # SCAN

    return tok_start, tok_end, tok_line, line_tok_count, line_tok_offs
