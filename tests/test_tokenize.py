import numpy as np, cupy as cp, pytest
from triparsegpu import io, tokenize


def ref_tokens(contents):
    """CPU reference: (line_index, token_string) for every token."""
    out = []
    for i, c in enumerate(contents):
        for t in c.split():
            out.append((i, t))
    return out


@pytest.mark.parametrize("D", ["BGL", "HDFS", "Thunderbird", "OpenStack"])
def test_tokenize_matches_python_split(D):
    contents, _ = io.load_contents(f"data/loghub/{D}/{D}_2k.log", D)
    buf_np, offs_np = io.to_buffer(contents)
    buf, offs = cp.asarray(buf_np), cp.asarray(offs_np)

    ts, te, tl, cnt, toffs = tokenize.tokenize(buf, offs)
    ref = ref_tokens(contents)

    assert int(ts.size) == len(ref), f"token count {int(ts.size)} vs {len(ref)}"

    # per-line counts
    ref_counts = np.zeros(len(contents), np.int32)
    for i, _ in ref:
        ref_counts[i] += 1
    assert np.array_equal(cnt.get(), ref_counts)

    # offsets are a correct prefix sum
    assert np.array_equal(toffs.get()[1:], np.cumsum(ref_counts))

    # spot-check actual token strings
    ts_h, te_h, tl_h = ts.get(), te.get(), tl.get()
    for k in list(range(50)) + list(range(len(ref) - 50, len(ref))):
        got = bytes(buf_np[ts_h[k]:te_h[k]]).decode("utf-8", "ignore")
        assert got == ref[k][1], f"token {k}: {got!r} != {ref[k][1]!r}"
        assert tl_h[k] == ref[k][0]

    print(f"\n{D}: {int(ts.size):,} tokens over {len(contents):,} lines — OK")
