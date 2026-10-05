"""End-to-end GPU parse: returns line->template ids, template strings, stage times."""
import time
import cupy as cp
from triparsegpu import io, tokenize, hashing, bucket, cluster, template


def parse(contents, tier=1, st=0.5, window=64, depth_tokens=2, timing=False):
    T = {}
    sync = cp.cuda.Device().synchronize
    t0 = time.perf_counter()

    def mark(name):
        nonlocal t0
        if timing:
            sync()
            now = time.perf_counter()
            T[name] = now - t0
            t0 = now

    buf_np, offs_np = io.to_buffer(contents)
    mark("to_buffer")

    buf, offs = cp.asarray(buf_np), cp.asarray(offs_np)
    mark("h2d")

    ts, te, tl, cnt, toffs = tokenize.tokenize(buf, offs, tier)
    mark("tokenize")

    h, w, bh = hashing.hash_tokens(buf, ts, te)
    mark("hash")

    key = bucket.bucket_keys(bh, toffs, cnt, depth_tokens)
    order, s0, s1, sop = bucket.sort_lines(key, tier)
    mark("sort")

    eu, ev, ne = cluster.similarity_edges(h, toffs, cnt, order, sop, s0, st, window)
    mark("similarity")

    label, rounds = cluster.ccl(eu, ev, len(contents))
    mark("ccl")

    ids, templates = template.consensus(h, toffs, cnt, label, contents)
    mark("template")

    meta = {"edges": ne, "ccl_rounds": rounds, "templates": len(templates)}
    return ids.get(), templates, (T if timing else None), meta
