import numpy as np, cupy as cp, pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from triparsegpu import io, tokenize, hashing, bucket, cluster


def same_partition(a, b):
    """True if two labelings induce the same partition (ids may differ)."""
    a, b = np.asarray(a), np.asarray(b)
    return len(set(zip(a, b))) == len(set(a)) == len(set(b))


def test_chain():
    eu = cp.array([0, 1, 2], cp.int64)
    ev = cp.array([1, 2, 3], cp.int64)
    lab, r = cluster.ccl(eu, ev, 4)
    assert len(set(lab.get())) == 1


def test_two_components():
    eu = cp.array([0, 3], cp.int64)
    ev = cp.array([1, 4], cp.int64)
    lab, r = cluster.ccl(eu, ev, 6)      # 2 is isolated, 5 is isolated
    assert len(set(lab.get())) == 4


def test_no_edges():
    lab, r = cluster.ccl(cp.empty(0, cp.int64), cp.empty(0, cp.int64), 5)
    assert len(set(lab.get())) == 5


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_vs_scipy(seed):
    rng = np.random.default_rng(seed)
    n, m = 5000, 8000
    u = rng.integers(0, n, m)
    v = rng.integers(0, n, m)
    lab, rounds = cluster.ccl(cp.asarray(u.astype(np.int64)),
                              cp.asarray(v.astype(np.int64)), n)
    g = coo_matrix((np.ones(m), (u, v)), shape=(n, n))
    ncc, ref = connected_components(g, directed=False)
    assert same_partition(lab.get(), ref)
    assert len(set(lab.get())) == ncc


@pytest.mark.parametrize("D", ["BGL", "HDFS", "Thunderbird", "OpenStack"])
def test_real_pipeline_vs_scipy(D):
    contents, _ = io.load_contents(f"data/loghub/{D}/{D}_2k.log", D)
    buf_np, offs_np = io.to_buffer(contents)
    buf, offs = cp.asarray(buf_np), cp.asarray(offs_np)
    ts, te, tl, cnt, toffs = tokenize.tokenize(buf, offs)
    h, w = hashing.hash_tokens(buf, ts, te)
    key = bucket.bucket_keys(h, toffs, cnt)
    order, s0, s1, sop = bucket.sort_lines(key)
    eu, ev, ne = cluster.similarity_edges(h, toffs, cnt, order, sop, s0, st=0.5, window=64)
    lab, rounds = cluster.ccl(eu, ev, len(contents))

    u, v = eu.get(), ev.get()
    g = coo_matrix((np.ones(ne), (u, v)), shape=(len(contents),) * 2)
    ncc, ref = connected_components(g, directed=False)
    print(f"\n{D}: {ne:,} edges -> {len(set(lab.get()))} components "
          f"(scipy {ncc}), {rounds} rounds")
    assert same_partition(lab.get(), ref)
