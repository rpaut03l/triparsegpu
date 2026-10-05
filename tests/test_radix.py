import numpy as np, cupy as cp, pytest
from triparsegpu import radix


@pytest.mark.parametrize("n", [1, 255, 256, 1000, 100_000, 1_000_000])
def test_sort_matches_numpy(n):
    rng = np.random.default_rng(n)
    k = rng.integers(0, 2**63, n, dtype=np.uint64)
    gk, gv = radix.sort_pairs(cp.asarray(k), cp.arange(n, dtype=cp.int64))
    assert np.array_equal(gk.get(), np.sort(k))


@pytest.mark.parametrize("n", [1000, 100_000])
def test_stability_with_duplicates(n):
    """Equal keys must keep their original relative order."""
    rng = np.random.default_rng(0)
    k = rng.integers(0, 16, n, dtype=np.uint64)          # lots of duplicates
    idx = np.arange(n, dtype=np.int64)
    gk, gv = radix.sort_pairs(cp.asarray(k), cp.asarray(idx))
    exp = np.argsort(k, kind="stable")
    assert np.array_equal(gk.get(), k[exp])
    assert np.array_equal(gv.get(), exp), "sort is not stable"


def test_argsort_matches_cupy():
    rng = np.random.default_rng(7)
    k = cp.asarray(rng.integers(0, 2**40, 500_000, dtype=np.uint64))
    got = radix.argsort(k).get()
    exp = cp.argsort(k, kind="stable").get()
    assert np.array_equal(k.get()[got], k.get()[exp])     # same sorted order
    assert np.array_equal(got, exp), "stable order differs"


def test_all_equal_keys():
    n = 10_000
    k = cp.full(n, 42, cp.uint64)
    order = radix.argsort(k).get()
    assert np.array_equal(order, np.arange(n))            # stability edge case
