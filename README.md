# TriParseGPU

GPU-native log template mining for large-scale system logs.

Log parsing (Drain and its variants) is the first step in every log-analysis
pipeline and runs single-threaded on CPU at roughly 8k–80k lines/s.
TriParseGPU reformulates it as three data-parallel problems — radix sort/scan,
a similarity graph with connected-component labelling, and BFS over a template
graph — and runs it entirely on the GPU.

**IIT Jodhpur · M.Tech (AI) · GPU Programming · Prof. Binod Kumar**
Rohit Patel (G25AIT2089) · Jeenal Chaudhary (G25AIT2027) ·
Kosuru Yuvaraj (G25AIT2054) · Sharvan Vittala (G25AIT2099)

## Status

Complete. Tier 1 (GPU pipeline on library primitives) and tier 2 (hand-written
CUDA kernels) both validated; tier 2 is the course deliverable.

### Accuracy (LogHub-2k, tier 2, window 16)

| Dataset | GA | PA | FGA | ARI vs Drain3 | Drain3 GA |
|---|---|---|---|---|---|
| BGL | 0.9685 | 0.8115 | 0.8444 | 1.0000 | 0.9625 |
| HDFS | 0.9975 | 0.8085 | 0.8667 | 1.0000 | 0.9975 |
| Thunderbird | 0.9425 | 0.1770 | 0.7045 | 0.9999 | 0.9550 |
| OpenStack | 0.8070 | 0.0720 | 0.6933 | 0.9777 | 0.7325 |

At or above the CPU reference on three of four datasets. On OpenStack we exceed
it: separating the bucket hash from the text hash avoids a structural weakness
in Drain's first-token routing (15,791 templates / 6,679 singletons → 47 / 3).

### Throughput (full datasets, tier 2, locked clocks, median of 5)

| Dataset | Lines | GPU kernels | GPU +H2D | Speedup | Tier-1 kernels |
|---|---|---|---|---|---|
| OpenStack | 137,074 | 26.3 M/s | 22.1 M/s | 826× | 28.4 M/s |
| BGL | 4,747,963 | 63.5 M/s | 50.8 M/s | 1,398× | 64.6 M/s |
| Thunderbird | 2,000,000 | 23.7 M/s | 21.5 M/s | 2,791× | 23.8 M/s |
| HDFS | 11,175,629 | 65.5 M/s | 49.2 M/s | 820× | 66.5 M/s |

The hand-written tier costs 1.5–7.4% against library primitives end to end.

### Course problems implemented

- **P7** — radix sort over the 64-bit bucket key + Blelloch work-efficient scan
- **P3** — connected-component labelling (hook + pointer jumping, `atomicMin`)
- **P2** — level-synchronous BFS over the template co-occurrence graph in CSR

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB, CC 12.0, CUDA 13.2,
driver 595.58.03. 84 tests, every stage verified against a CPU reference.

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

## Reproduce

```bash
bash scripts/lock_clocks.sh          # required for stable numbers
export TRIPARSE_CLOCKS_LOCKED=1

# one dataset, tier 2
python bench/parsebench.py --dataset BGL --tier 2 --limit 1000000 \
  --window 16 --reps 5 --warmup 1 --out results/bgl_1m.json

# full sweep: tier comparison, scaling curve, window sensitivity
bash bench/run_all.sh

# figures from the result JSONs
python model/stages.py
```

Profiling (Nsight Compute, needs sudo for hardware counters; replays each
kernel 10–40×, so run it on an otherwise idle GPU):

```bash
sudo env "PATH=$PATH" bash scripts/profile.sh BGL 2 1000000 > results/ncu/bgl.csv
```

## Pipeline

1. `io` / `gpuio` — header strip and parameter regexes; `gpuio` does newline
   marking and header stripping on-device from pinned host memory
   (13.2 GB/s H2D, 10.9 M lines/s end to end from disk) *(CUDA)*
2. `tokenize` — token boundaries via flags + scan
3. `hashing` — FNV-1a per token, wildcard classification, separate bucket hash *(CUDA)*
4. `bucket` — 64-bit bucket key, radix sort, segment boundaries *(CUDA)*
5. `cluster.similarity_edges` — warp-per-line similarity inside a bucket *(CUDA)*
6. `cluster.ccl` — connected components, hook + pointer jumping *(CUDA)*
7. `template.consensus` — per-component token consensus *(CUDA)*
8. `graph` — template co-occurrence graph in CSR + level-synchronous BFS *(CUDA)*

## Tests

```bash
pytest -q tests/
```

Every stage is checked against a CPU reference (NumPy, scipy, Drain3).

## Docs

- `docs/DESIGN.md` — full design, kernel listings, paper plan
- `docs/RUNBOOK.md` — step-by-step execution guide, screen sessions, GPU sharing
- `docs/TriParseGPU_HANDBOOK.md` — team handbook, diagrams, cheatsheet

## License

Apache-2.0
