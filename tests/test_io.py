import pandas as pd, pytest
from triparsegpu import io

@pytest.mark.parametrize("D", ["BGL", "HDFS", "Thunderbird", "OpenStack"])
def test_header_strip_matches_ground_truth(D):
    raw = f"data/loghub/{D}/{D}_2k.log"
    gt = pd.read_csv(f"data/loghub/{D}/{D}_2k.log_structured.csv").Content.astype(str).tolist()
    got, misses = io.load_contents(raw, D)
    assert len(got) == len(gt), f"line count {len(got)} vs {len(gt)}"
    exact = sum(a.strip() == b.strip() for a, b in zip(got, gt))
    print(f"\n{D}: {exact}/{len(gt)} exact ({exact/len(gt):.4f}), {misses} format misses")
    if exact < len(gt):
        for a, b in zip(got, gt):
            if a.strip() != b.strip():
                print(f"  got: {a!r}\n  exp: {b!r}")
                break
    assert exact / len(gt) > 0.99

@pytest.mark.parametrize("D", ["HDFS"])
def test_buffer_roundtrip(D):
    got, _ = io.load_contents(f"data/loghub/{D}/{D}_2k.log", D)
    buf, offs = io.to_buffer(got)
    assert offs.size == len(got) + 1
    first = bytes(buf[offs[0]:offs[1]-1]).decode()
    assert first == got[0]
