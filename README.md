# TriParseGPU

GPU-native log template mining for large-scale system logs.

Log parsing (Drain and its variants) is the first step in every log-analysis
pipeline and runs single-threaded on CPU at roughly 10k–55k lines/s.
TriParseGPU reformulates it as three data-parallel problems — radix sort/scan,
a similarity graph with connected-component labelling, and BFS over a template
graph — and runs it on the GPU.

**IIT Jodhpur · M.Tech (AI) · GPU Programming · Prof. Binod Kumar**
Rohit Patel (G25AIT2089) · Jeenal Chaudhary (G25AIT2027) ·
Kosuru Yuvaraj (G25AIT2054) · Sharvan Vittala (G25AIT2099)

## Status

Tier 1 (GPU pipeline on library primitives) complete and validated.

| Dataset | GA (GPU) | GA (Drain3) | PA (GPU) | Templates | GT |
|---|---|---|---|---|---|
| BGL | 0.9685 | 0.9625 | 0.8115 | 105 | 120 |
| HDFS | 0.9975 | 0.9975 | 0.8085 | 16 | 14 |
| Thunderbird | 0.9425 | 0.9550 | 0.1770 | 203 | 149 |
| OpenStack | 0.7325 | 0.7325 | 0.0185 | 278 | 43 |

Drain3 CPU baseline: BGL 45,418 lines/s · HDFS 54,545 · OpenStack 28,259 ·
Thunderbird 8,475.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install cupy-cuda13x numpy pandas matplotlib pyyaml pytest drain3 regex scikit-learn scipy
```

Requires CUDA 13.x and a GPU of compute capability 12.0 (set `ARCH` in
`triparsegpu/compile.py` for others).

## Use

```python
from triparsegpu import io, pipeline

contents, _ = io.load_contents("path/to/HDFS.log", "HDFS")
contents = io.preprocess(contents, "HDFS")
ids, templates, timings, meta = pipeline.parse(contents, st=0.4, depth_tokens=2)
```

## Pipeline

1. `io` — strip headers per the logparser log_format, apply parameter regexes
2. `tokenize` — token boundaries via flags + scan
3. `hashing` — FNV-1a per token, wildcard classification *(CUDA)*
4. `bucket` — 64-bit bucket key, sort lines, find segments
5. `cluster.similarity_edges` — warp-per-line similarity inside a bucket *(CUDA)*
6. `cluster.ccl` — connected components, hook + pointer jumping *(CUDA)*
7. `template.consensus` — per-component token consensus *(CUDA)*

## Tests

```bash
pytest -q tests/
```

Every stage is checked against a CPU reference (NumPy, scipy, Drain3).

## Docs

- `docs/DESIGN.md` — full design, kernel listings, paper plan
- `docs/RUNBOOK.md` — step-by-step execution guide, screen sessions, GPU sharing

## License

Apache-2.0
