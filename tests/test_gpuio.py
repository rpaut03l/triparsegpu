import cupy as cp, pytest
from triparsegpu import io, gpuio


@pytest.mark.parametrize("D", ["BGL", "HDFS", "OpenStack"])
def test_gpu_header_strip_matches_cpu(D):
    path = f"data/loghub/{D}/{D}_2k.log"
    cpu, _ = io.load_contents(path, D)

    gbuf, cs, ce = gpuio.load_gpu(path, D)
    raw = gbuf.get()
    cs_h, ce_h = cs.get(), ce.get()

    assert len(cs_h) == len(cpu), f"{len(cs_h)} lines vs {len(cpu)}"
    exact = 0
    first_bad = None
    for i in range(len(cpu)):
        got = bytes(raw[cs_h[i]:ce_h[i]]).decode("utf-8", "ignore")
        if got.strip() == cpu[i].strip():
            exact += 1
        elif first_bad is None:
            first_bad = (i, got, cpu[i])
    print(f"\n{D}: {exact}/{len(cpu)} exact ({exact/len(cpu):.4f})")
    if first_bad:
        i, got, exp = first_bad
        print(f"  line {i}\n   got: {got!r}\n   exp: {exp!r}")
    assert exact / len(cpu) > 0.99
