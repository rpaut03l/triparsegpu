import numpy as np, cupy as cp, pytest
from triparsegpu import io, tokenize, hashing


def _run(lines):
    buf_np, offs_np = io.to_buffer(lines)
    buf, offs = cp.asarray(buf_np), cp.asarray(offs_np)
    ts, te, tl, cnt, toffs = tokenize.tokenize(buf, offs)
    h, w = hashing.hash_tokens(buf, ts, te)
    toks = [bytes(buf_np[s:e]).decode("utf-8", "ignore")
            for s, e in zip(ts.get(), te.get())]
    return toks, h.get(), w.get()


def test_wildcard_classification():
    toks, h, w = _run(["abc 123 10.0.0.7 0xdeadbeef blk_42 hello"])
    got = dict(zip(toks, w))
    assert got["abc"] == 0
    assert got["hello"] == 0
    assert got["123"] == 1, "plain integer must be wildcard"
    assert got["10.0.0.7"] == 1, "IPv4 must be wildcard"
    assert got["0xdeadbeef"] == 1, "hex must be wildcard"


def test_same_token_same_hash():
    toks, h, w = _run(["connected port", "connected port"])
    assert h[0] == h[2] and h[1] == h[3]


def test_wildcards_all_hash_zero():
    toks, h, w = _run(["1 22 333 4444"])
    assert (h == 0).all() and (w == 1).all()


def test_no_collisions_on_real_data():
    contents, _ = io.load_contents("data/loghub/HDFS/HDFS_2k.log", "HDFS")
    toks, h, w = _run(contents)
    seen = {}
    coll = 0
    for t, hv, wv in zip(toks, h, w):
        if wv:
            continue
        if hv in seen and seen[hv] != t:
            coll += 1
        seen[hv] = t
    uniq = len({t for t, wv in zip(toks, w) if not wv})
    print(f"\nHDFS: {len(toks):,} tokens, {uniq:,} unique non-wild, {coll} collisions, "
          f"{w.sum():,} wildcards ({w.mean():.1%})")
    assert coll == 0
