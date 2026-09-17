"""GPU-side log loading: raw bytes in, Content buffer + offsets out.

Replaces the per-line Python regex in io.load_contents for the fixed-column
LogHub formats. The header is stripped by skipping K space-delimited fields.
"""
import numpy as np
import cupy as cp
from triparsegpu.compile import kernel

_TPB = 256

# number of header fields before <Content> for each dataset
HEADER_FIELDS = {
    "BGL": 9,          # Label Timestamp Date Node Time NodeRepeat Type Component Level
    "HDFS": 5,         # Date Time Pid Level Component:
    "OpenStack": 7,    # Logrecord Date Time Pid Level Component [ADDR]
}


def read_raw(path, limit_bytes=None):
    """Read the file as raw bytes (no Python per-line work)."""
    with open(path, "rb") as f:
        data = f.read(limit_bytes) if limit_bytes else f.read()
    return np.frombuffer(data, np.uint8)


def to_gpu_pinned(arr):
    """Copy a host array to the GPU through pinned memory so the transfer
    can DMA instead of going through a staging copy."""
    mem = cp.cuda.alloc_pinned_memory(arr.nbytes)
    staged = np.frombuffer(mem, arr.dtype, arr.size)
    staged[:] = arr
    out = cp.empty(arr.shape, arr.dtype)
    out.set(staged)
    return out


def split_lines(gbuf):
    """Newline positions -> line_start / line_end arrays (end excludes '\\n')."""
    n = int(gbuf.size)
    flag = cp.empty(n, cp.uint8)
    k = kernel("split", "mark_newlines")
    k(((n + _TPB - 1) // _TPB,), (_TPB,), (gbuf, cp.int64(n), flag))

    nl = cp.flatnonzero(flag).astype(cp.int64)        # index of each '\n'
    line_start = cp.concatenate([cp.zeros(1, cp.int64), nl + 1])
    line_end = cp.concatenate([nl, cp.array([n], cp.int64)])
    # drop a trailing empty line when the file DOES end in '\n'
    if int(line_start.size) and int(line_start[-1]) >= n:
        line_start, line_end = line_start[:-1], line_end[:-1]
    return line_start, line_end


# datasets whose Content begins after a bracketed field
BRACKET_FORMATS = {"OpenStack"}


def strip_header(gbuf, line_start, line_end, dataset):
    """Return content_start per line (content_end == line_end)."""
    nlines = int(line_start.size)
    out = cp.empty(nlines, cp.int64)
    grid = ((nlines + _TPB - 1) // _TPB,)
    if dataset in BRACKET_FORMATS:
        k = kernel("split", "content_start_bracket")
        k(grid, (_TPB,), (gbuf, line_start, line_end, cp.int64(nlines), out))
    else:
        k = kernel("split", "content_start")
        k(grid, (_TPB,), (gbuf, line_start, line_end,
                          cp.int64(nlines), cp.int32(HEADER_FIELDS[dataset]), out))
    return out


def load_gpu(path, dataset, limit_lines=None):
    """Full GPU path: returns (gbuf, content_start, content_end)."""
    raw = read_raw(path)
    gbuf = to_gpu_pinned(raw)
    ls, le = split_lines(gbuf)
    if limit_lines:
        ls, le = ls[:limit_lines], le[:limit_lines]
    cs = strip_header(gbuf, ls, le, dataset)
    return gbuf, cs, le
