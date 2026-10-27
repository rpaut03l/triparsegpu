"""Stages 5-6: similarity edges inside buckets, then connected components. - Problem Statement - 3"""
import cupy as cp
from triparsegpu.compile import kernel

_TPB = 256


def similarity_edges(tok_hash, line_tok_offs, line_tok_count,
                     order, seg_of_pos, seg_start,
                     st=0.5, window=64, edge_cap=None):
    nlines = int(order.size)
    if edge_cap is None:
        edge_cap = max(1 << 20, nlines * min(window, 64))
    edge_u = cp.empty(edge_cap, cp.int64)
    edge_v = cp.empty(edge_cap, cp.int64)
    edge_cnt = cp.zeros(1, cp.uint64)

    k = kernel("similarity", "similarity_edges")
    warps_needed = nlines
    threads = warps_needed * 32
    grid = (threads + _TPB - 1) // _TPB
    k((grid,), (_TPB,), (tok_hash, line_tok_offs, line_tok_count,
                         order, seg_of_pos, seg_start,
                         cp.int64(nlines), cp.float32(st), cp.int32(window),
                         edge_u, edge_v, edge_cnt, cp.uint64(edge_cap)))

    n = int(edge_cnt.get()[0])
    if n > edge_cap:
        raise RuntimeError(f"edge overflow: {n} edges > cap {edge_cap}; "
                           f"raise edge_cap or lower window")
    return edge_u[:n], edge_v[:n], n


def ccl(edge_u, edge_v, nlines, max_rounds=100):
    """Connected components over the edge list. Returns label[nlines]."""
    label = cp.empty(nlines, cp.int64)
    changed = cp.zeros(1, cp.int32)
    ne = int(edge_u.size)

    k_init = kernel("ccl", "ccl_init")
    k_hook = kernel("ccl", "ccl_hook")
    k_comp = kernel("ccl", "ccl_compress")

    g_n = (nlines + _TPB - 1) // _TPB
    k_init((g_n,), (_TPB,), (label, cp.int64(nlines)))
    if ne == 0:
        return label, 0

    g_e = (ne + _TPB - 1) // _TPB
    rounds = 0
    for _ in range(max_rounds):
        changed.fill(0)
        k_hook((g_e,), (_TPB,), (edge_u, edge_v, cp.int64(ne), label, changed))
        k_comp((g_n,), (_TPB,), (label, cp.int64(nlines)))
        rounds += 1
        if int(changed.get()[0]) == 0:      # 4-byte D->H read, nothing else
            break
    return label, rounds
