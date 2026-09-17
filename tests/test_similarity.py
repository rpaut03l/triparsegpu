import numpy as np, cupy as cp, pytest
from triparsegpu import io, tokenize, hashing, bucket, cluster


def _stages(contents):
    buf_np, offs_np = io.to_buffer(contents)
    buf, offs = cp.asarray(buf_np), cp.asarray(offs_np)
    ts, te, tl, cnt, toffs = tokenize.tokenize(buf, offs)
    h, w = hashing.hash_tokens(buf, ts, te)
    key = bucket.bucket_keys(h, toffs, cnt)
    order, s0, s1, sop = bucket.sort_lines(key)
    return h, toffs, cnt, order, sop, s0


def ref_edges(h_np, toffs_np, cnt_np, order_np, sop_np, s0_np, st, window):
    """CPU reference for the same bounded-window comparison."""
    out = set()
    for pos in range(len(order_np)):
        li = order_np[pos]
        ln = cnt_np[li]
        if ln == 0:
            continue
        oi = toffs_np[li]
        seg0 = s0_np[sop_np[pos]]
        for q in range(max(seg0, pos - window), pos):
            lj = order_np[q]
            oj = toffs_np[lj]
            m = sum(1 for k in range(ln) if h_np[oi + k] == h_np[oj + k])
            if m / ln >= st:
                out.add((int(li), int(lj)))
    return out


def test_identical_lines_produce_edge():
    h, toffs, cnt, order, sop, s0 = _stages(
        ["connected to 10.0.0.7 port 22", "connected to 10.0.0.9 port 22"])
    eu, ev, n = cluster.similarity_edges(h, toffs, cnt, order, sop, s0, st=0.5, window=64)
    assert n == 1


def test_unrelated_lines_no_edge():
    h, toffs, cnt, order, sop, s0 = _stages(["alpha beta gamma", "one two three"])
    eu, ev, n = cluster.similarity_edges(h, toffs, cnt, order, sop, s0, st=0.9, window=64)
    assert n == 0


@pytest.mark.parametrize("D", ["BGL", "HDFS", "Thunderbird", "OpenStack"])
def test_matches_cpu_reference(D):
    contents, _ = io.load_contents(f"data/loghub/{D}/{D}_2k.log", D)
    contents = contents[:400]
    h, toffs, cnt, order, sop, s0 = _stages(contents)
    ST, W = 0.5, 16
    eu, ev, n = cluster.similarity_edges(h, toffs, cnt, order, sop, s0, st=ST, window=W)
    got = {(int(a), int(b)) for a, b in zip(eu.get(), ev.get())}
    exp = ref_edges(h.get(), toffs.get(), cnt.get(), order.get(), sop.get(), s0.get(), ST, W)
    print(f"\n{D}: {n} edges (ref {len(exp)})")
    assert got == exp
