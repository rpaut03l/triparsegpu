# TriParseGPU

GPU-native log template mining for large-scale system logs.

Log parsing (Drain and its variants) is the first step in every log-analysis
pipeline and runs single-threaded on CPU at roughly 8k to 80k lines/s.
TriParseGPU reformulates it as three data-parallel problems: radix sort and
scan, a similarity graph with connected-component labelling, and BFS over a
template graph. The whole pipeline runs on the GPU.

**IIT Jodhpur · M.Tech (AI) · GPU Programming · Prof. Binod Kumar**

---

## The team

| Member | Roll number | Owns |
|---|---|---|
| **Rohit Patel** | G25AIT2089 | Tokenisation, hashing, bucket key, GPU I/O, profiling, paper |
| **Jeenal Chaudhary** | G25AIT2027 | Similarity kernel, connected components, template consensus |
| **Kosuru Yuvaraj** | G25AIT2054 | Radix sort, Blelloch scan, co-occurrence graph and BFS |
| **Sharvan Vittala** | G25AIT2099 | Accuracy metrics, test suite, benchmark harness, figures |

We split the work by pipeline stage so each of us owns kernels end to end:
the CUDA source, the Python wrapper, the tests against a CPU reference, and the
numbers in the results table.

### Who built what

**Rohit Patel (G25AIT2089)**
Stages 1 to 4 and the measurement stack.
- `kernels/split.cu`, `kernels/hash.cu`: newline marking, header stripping,
  FNV-1a token hashing, wildcard classification
- `triparsegpu/io.py`, `gpuio.py`, `tokenize.py`, `hashing.py`, `bucket.py`
- The two-hash design: separating the bucket hash from the text hash, which
  took OpenStack from 15,791 templates to 47 and grouping accuracy from 0.7325
  to 0.8070, above the CPU reference
- On-device file reading with pinned memory: 13.2 GB/s transfer, 10.9 M lines/s
  from disk
- The three-layer profiling stack: Nsight Compute, eBPF probes, DCGM, including
  `scripts/profile.sh`, `scripts/ncu_summary.py` and `scripts/ebpf/`
- The bottleneck diagnosis and the paper draft

**Jeenal Chaudhary (G25AIT2027)**
Stages 5 to 7, the core of the reformulation.
- `kernels/similarity.cu`: one warp per line, 32 lanes over 32 token positions,
  shuffle reduction with no shared memory
- `kernels/ccl.cu`: connected components by hook and pointer jumping, with a
  four-byte convergence flag instead of a full label-array readback
- `kernels/template.cu`: per-component token consensus. Rewriting the thread
  mapping from one block per component to one thread per (line, token) pair
  took this from 113 ms to 1.1 ms
- `triparsegpu/cluster.py`, `template.py`
- Validation of components against `scipy.sparse.csgraph`

**Kosuru Yuvaraj (G25AIT2054)**
The Problem 7 primitives and the graph traversal.
- `kernels/radix.cu`: LSD radix sort, stable with duplicates, 8 bits per pass.
  Replacing the serial digit-ranking loop with a warp ballot took the sort from
  7.77 ms to 5.22 ms on 11.2 M keys
- `kernels/scan.cu`: Blelloch work-efficient scan, up-sweep and down-sweep.
  Reaches parity with CuPy at 100 M elements (2.19 ms against 2.18 ms)
- `kernels/cooccur.cu`: co-occurrence graph construction in CSR and
  level-synchronous BFS
- `triparsegpu/radix.py`, `scan.py`, `graph.py`
- The hand-written vs library comparison against CUB

**Sharvan Vittala (G25AIT2099)**
Correctness and evidence.
- `triparsegpu/metrics.py`: GA, PA, FGA and ARI
- `triparsegpu/ref_drain3.py`: the Drain3 baseline using the official logparser
  benchmark settings, reproducing the published numbers exactly. Without this
  no speed claim would mean anything
- `tests/`: 84 tests, every stage checked against a CPU reference
- `bench/parsebench.py`, `bench/run_all.sh`, `triparsegpu/timing.py`: locked
  clocks, warm-up at full input size, median of five, three reported columns
- `model/stages.py`: the five figures, generated from the result JSONs

---

## Status

Complete. Tier 1 (GPU pipeline on library primitives) and tier 2 (hand-written
CUDA kernels) both validated. Tier 2 is the course deliverable.

### Accuracy (LogHub-2k, tier 2, window 16)

| Dataset | GA | PA | FGA | ARI vs Drain3 | Drain3 GA |
|---|---|---|---|---|---|
| BGL | 0.9685 | 0.8115 | 0.8444 | 1.0000 | 0.9625 |
| HDFS | 0.9975 | 0.8085 | 0.8667 | 1.0000 | 0.9975 |
| Thunderbird | 0.9425 | 0.1770 | 0.7045 | 0.9999 | 0.9550 |
| OpenStack | **0.8070** | 0.0720 | 0.6933 | 0.9777 | 0.7325 |

At or above the CPU reference on three of four datasets. On OpenStack we exceed
it: separating the bucket hash from the text hash avoids a structural weakness
in Drain's first-token routing (15,791 templates and 6,679 singletons become
47 and 3).

### Throughput (full datasets, tier 2, locked clocks, median of 5)

| Dataset | Lines | GPU kernels | GPU +H2D | Speedup | Tier-1 kernels |
|---|---|---|---|---|---|
| OpenStack | 137,074 | 26.3 M/s | 22.1 M/s | 826x | 28.4 M/s |
| BGL | 4,747,963 | 63.5 M/s | 50.8 M/s | 1,398x | 64.6 M/s |
| Thunderbird | 2,000,000 | 23.7 M/s | 21.5 M/s | 2,791x | 23.8 M/s |
| HDFS | 11,175,629 | 65.5 M/s | 49.2 M/s | 820x | 66.5 M/s |

The hand-written tier costs 1.5% to 7.4% against library primitives end to end.

### Per-kernel profile (BGL 1M, tier 2, idle GPU)

| Kernel | Total us | % | Mean SM % |
|---|---|---|---|
| similarity_edges | 8,300 | 75.3 | 14.7 |
| ccl_hook | 1,773 | 16.1 | 8.0 |
| consensus_reduce | 401 | 3.6 | 22.0 |
| block_scan | 160 | 1.5 | 19.7 |
| radix_scatter | 112 | 1.0 | 30.3 |
| radix_histogram | 95 | 0.9 | 22.0 |
| hash_tokens | 87 | 0.8 | 69.4 |
| add_offsets | 64 | 0.6 | 6.3 |
| ccl_compress | 23 | 0.2 | 9.4 |

11.0 ms over 54 launches. Measured twice, on a busy and an idle GPU, agreeing
within 0.3%.

### Course problems implemented

- **P7** radix sort over the 64-bit bucket key plus Blelloch work-efficient scan
- **P3** connected-component labelling (hook and pointer jumping, `atomicMin`)
- **P2** level-synchronous BFS over the template co-occurrence graph in CSR

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB, CC 12.0, CUDA 13.2,
driver 595.58.03. 84 tests, every stage verified against a CPU reference.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install cupy-cuda13x numpy pandas matplotlib pyyaml pytest drain3 regex scikit-learn scipy
```

Requires CUDA 13.x and a GPU of compute capability 12.0. Set `ARCH` in
`triparsegpu/compile.py` for other architectures.

---

## Use

```python
from triparsegpu import io, pipeline

contents, _ = io.load_contents("path/to/HDFS.log", "HDFS")
contents = io.preprocess(contents, "HDFS")
ids, templates, timings, meta = pipeline.parse(contents, st=0.4, depth_tokens=2)
```

---

## Reproduce

```bash
bash scripts/lock_clocks.sh          # required for stable numbers
export TRIPARSE_CLOCKS_LOCKED=1

# one dataset, tier 2 (dataset names are case sensitive)
python bench/parsebench.py --dataset BGL --tier 2 --limit 1000000 \
  --window 16 --reps 5 --warmup 1 --out results/bgl_1m.json

# full sweep: tier comparison, scaling curve, window sensitivity
bash bench/run_all.sh

# figures from the result JSONs
python model/stages.py
```

Profiling (Nsight Compute, needs sudo for hardware counters, replays each
kernel 10 to 40 times, so run it on an otherwise idle GPU):

```bash
sudo env "PATH=$PATH" bash scripts/profile.sh BGL 2 1000000 > results/ncu/bgl.csv
python scripts/ncu_summary.py results/ncu/bgl.csv
```

---

## Pipeline

1. `io` / `gpuio`: header strip and parameter regexes. `gpuio` does newline
   marking and header stripping on-device from pinned host memory
   (13.2 GB/s H2D, 10.9 M lines/s end to end from disk) *(CUDA)*
2. `tokenize`: token boundaries via flags and scan
3. `hashing`: FNV-1a per token, wildcard classification, separate bucket hash *(CUDA)*
4. `bucket`: 64-bit bucket key, radix sort, segment boundaries *(CUDA)*
5. `cluster.similarity_edges`: warp-per-line similarity inside a bucket *(CUDA)*
6. `cluster.ccl`: connected components, hook and pointer jumping *(CUDA)*
7. `template.consensus`: per-component token consensus *(CUDA)*
8. `graph`: template co-occurrence graph in CSR plus level-synchronous BFS *(CUDA)*

---

## Tests

```bash
pytest -q
```

Every stage is checked against a CPU reference: Python `str.split` for
tokenisation, NumPy stable `argsort` for the sort, scipy for components and
BFS, Drain3 for end-to-end clustering.

---

## Docs

| File | What it covers |
|---|---|
| `docs/README.md` | Documentation index |
| `docs/PROJECT_STORY.md` | Why we chose this problem, the gap, and how we built it |
| `docs/MASTER_GUIDE.md` | Concepts, mnemonics, cheatsheets, commands, open items |
| `docs/CODE_WALKTHROUGH.md` | Python harness and CUDA kernels, module by module |
| `docs/PROBLEM_MAPPING.md` | P7, P3 and P2, and why we chose those three |
| `docs/VIVA_CHEATSHEET.md` | Fast answers |
| `docs/DESIGN.md` | Design rationale and kernel listings |
| `docs/RUNBOOK.md` | Step-by-step execution on the GPU box |

---

## License

Apache-2.0
