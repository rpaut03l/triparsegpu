import numpy as np, cupy as cp, pytest
from triparsegpu import scan


@pytest.mark.parametrize("n", [0, 1, 7, 1023, 1024, 1025, 4096, 100_000, 5_000_000])
def test_exclusive_scan(n):
    rng = np.random.default_rng(n)
    x = rng.integers(0, 100, n).astype(np.int64)
    got = scan.exclusive_scan(cp.asarray(x)).get()
    exp = np.concatenate([[0], np.cumsum(x)[:-1]]).astype(np.int64) if n else np.empty(0, np.int64)
    assert np.array_equal(got, exp), f"n={n}"


@pytest.mark.parametrize("n", [1, 1000, 1_000_000])
def test_inclusive_scan(n):
    x = np.ones(n, np.int64)
    got = scan.inclusive_scan(cp.asarray(x)).get()
    assert np.array_equal(got, np.cumsum(x))


@pytest.mark.parametrize("n,p", [(1000, 0.5), (100_000, 0.1), (1_000_000, 0.9)])
def test_compact_matches_flatnonzero(n, p):
    rng = np.random.default_rng(0)
    flag = (rng.random(n) < p).astype(np.uint8)
    g = cp.asarray(flag)
    got = scan.compact(g).get()
    exp = cp.flatnonzero(g).get()
    assert np.array_equal(got, exp)


def test_matches_cupy_cumsum_on_real_sizes():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, 11_175_629).astype(np.int64)   # HDFS line count
    g = cp.asarray(x)
    got = scan.exclusive_scan(g).get()
    exp = (cp.cumsum(g) - g).get()
    assert np.array_equal(got, exp)
