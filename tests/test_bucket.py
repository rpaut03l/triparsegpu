import numpy as np, cupy as cp, pytest
from triparsegpu import io, tokenize, hashing, bucket


def _pipeline(D, n=None):
    contents, _ = io.load_contents(f"data/loghub/{D}/{D}_2k.log", D)
    if n:
        contents = contents[:n]
    buf_np, offs_np = io.to_buffer(contents)
    buf, offs = cp.asarray(buf_np), cp.asarray(offs_np)
    ts, te, tl, cnt, toffs = tokenize.tokenize(buf, offs)
    h, w, bh = hashing.hash_tokens(buf, ts, te)
    key = bucket.bucket_keys(h, toffs, cnt)
    return contents, cnt, key, h, toffs


@pytest.mark.parametrize("D", ["BGL", "HDFS", "Thunderbird", "OpenStack"])
def test_key_encodes_token_count(D):
    contents, cnt, key, _, _ = _pipeline(D)
    assert np.array_equal((key.get() >> 48).astype(np.int32), cnt.get())


def test_identical_lines_get_identical_keys():
    contents = ["connected to 10.0.0.7 port 22", "connected to 10.0.0.9 port 22"]
    buf_np, offs_np = io.to_buffer(contents)
    buf, offs = cp.asarray(buf_np), cp.asarray(offs_np)
    ts, te, tl, cnt, toffs = tokenize.tokenize(buf, offs)
    h, w, bh = hashing.hash_tokens(buf, ts, te)
    key = bucket.bucket_keys(h, toffs, cnt)
    assert key[0] == key[1], "same template, different params -> same bucket"


def test_different_lengths_differ():
    contents = ["a b c", "a b c d"]
    buf_np, offs_np = io.to_buffer(contents)
    buf, offs = cp.asarray(buf_np), cp.asarray(offs_np)
    ts, te, tl, cnt, toffs = tokenize.tokenize(buf, offs)
    h, w, bh = hashing.hash_tokens(buf, ts, te)
    key = bucket.bucket_keys(h, toffs, cnt)
    assert key[0] != key[1]


@pytest.mark.parametrize("D", ["BGL", "HDFS", "Thunderbird", "OpenStack"])
def test_segments_are_correct(D):
    contents, cnt, key, _, _ = _pipeline(D)
    order, s0, s1, seg_of_pos = bucket.sort_lines(key)
    k = key.get()[order.get()]

    # sorted
    assert (np.diff(k) >= 0).all()
    # segment boundaries match value changes
    ref_starts = np.flatnonzero(np.concatenate([[True], k[1:] != k[:-1]]))
    assert np.array_equal(s0.get(), ref_starts)
    assert s1.get()[-1] == len(k)
    # every position maps to the segment containing it
    sop = seg_of_pos.get()
    for seg in range(min(20, len(ref_starts))):
        a, b = s0.get()[seg], s1.get()[seg]
        assert (sop[a:b] == seg).all()

    sizes = (s1 - s0).get()
    print(f"\n{D}: {len(contents):,} lines -> {len(ref_starts):,} buckets, "
          f"largest {sizes.max():,}, mean {sizes.mean():.1f}")
