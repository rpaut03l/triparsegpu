# TriParseGPU - Viva Cheatsheet

Fast answers. Every number here is measured and in the repo.
Scan the **bold** line; the sentences under it are the backup if pressed.

---

## 30-second pitch

> Log parsing turns raw log lines into templates. It is the first step in every
> log pipeline and it runs single-threaded on CPU at 8k-80k lines/s. We
> reformulated the standard algorithm so it is order-independent, ran it on a
> GPU at 23.7-65.5 million lines/s, and matched or beat the CPU reference on
> accuracy. No GPU implementation of log template mining existed before this.

---

## Numbers you must know cold

| Question | Answer |
|---|---|
| Speedup | **820x to 2,791x** (tier 2, full datasets) |
| Peak throughput | **65.5 M lines/s** (HDFS, kernels only) |
| Accuracy vs CPU | **at or above on 3 of 4** datasets |
| Best result | **OpenStack GA 0.8070 vs Drain3 0.7325** |
| Tests | **84**, every stage vs a CPU reference |
| Kernels | **9 .cu files, 7 hand-written stages** |
| Bottleneck | **similarity_edges, 75.3% of kernel time** |
| Total kernel time | **11.0 ms** over 54 launches (BGL 1M) |
| Hardware | RTX PRO 6000 Blackwell Max-Q, 96 GB, CC 12.0, CUDA 13.2 |
| Course problems | **P7 sort/scan, P3 components, P2 BFS** |

---

## The likely questions

### "What exactly is the contribution?"

**Five things.** (1) Batch-Drain, an order-independent reformulation of Drain.
(2) The first GPU implementation of log template mining. (3) Accuracy evidence
at or above the CPU reference. (4) A hand-written-vs-library primitive study
answered with measurements. (5) A three-layer observability recipe including
two negative results.

---

### "Why has nobody done this before?"

**Drain is sequential by construction, and the two communities do not overlap.**
Drain's "first match wins" makes line N depend on lines 1..N-1. Beyond that:
GPU researchers do not work on logs, log researchers do not write CUDA. The
closest prior GPU work (RAPIDS, 2019) accelerates regex *filtering*, which is a
different problem. LILAC (2024) uses a GPU only to run an LLM.

---

### "How can a reformulation be MORE accurate than the thing it reformulates?"

**Because order dependence was itself a source of error.**
Drain routes on the first token. When a UUID sits in that position, Drain splits
clusters that belong together. We separate the bucket hash from the text hash:
any token containing a wildcard marker maps to zero for bucketing only, while
still printing correctly in the template.

OpenStack: 15,791 templates / 6,679 singletons -> **47 / 3**. GA 0.7325 -> **0.8070**.

---

### "Prove the bounded window is correct."

**This is our weakest point and we say so.**
Empirically: grouping accuracy is identical at w=16 and w=128 on BGL while the
edge count grows fivefold. Connected components closes every gap the window
left - if A~B and B~C, A and C land in one cluster without ever being compared.
ARI vs Drain3 is 1.0000 on BGL and HDFS, 0.9999 on Thunderbird.

We have a proof sketch, not a formal proof. It is named in future work.

---

### "Why is PA so low on Thunderbird and OpenStack?"

**Because the ground truth masks inside tokens, and published Drain PA is
similarly low on these datasets.** GA is the headline metric in the benchmark
literature; we report PA alongside it rather than hiding it. Our HDFS PA is
0.8085 once we kept literal prefixes (`blk_<*>`) and punctuation in the
template - that fix took PA from 0.0000 to 0.8085.

---

### "Is the speedup real, or did you move work off the clock?"

**We report three columns, and the unflattering one is in the paper.**

| Column | HDFS 1M | What it includes |
|---|---|---|
| Kernels only | 44 M lines/s | GPU kernel time alone |
| Plus H2D | 12.5 M lines/s | + host-to-device copy |
| End to end | 127 k lines/s | + CPU file read and regex |

End to end is 2.4x over Drain3, not 230x. That is why we moved file reading
onto the GPU too (10.9 M lines/s from disk, 13.2 GB/s pinned transfer).

---

### "Which of the professor's problems did you use, and how deeply?"

**Three, each above the tier the statement asked for.**

| Problem | What the statement asked | What we did |
|---|---|---|
| P7 | prefix scan | radix sort over a 64-bit key + Blelloch scan underneath it |
| P3 | connected-component labelling | iterative hook + pointer jumping to convergence on a 16M-edge graph |
| P2 | BFS | build the co-occurrence graph on-device in CSR, then level-synchronous BFS |

---

### "Hand-written or library - which wins?"

**Neither, and that is the honest answer.**
- Scan: parity (2.19 ms vs CuPy 2.18 ms at 100 M).
- Radix sort: we are 2.4x behind CUB (5.22 vs 2.20 ms) and can account for the
  whole gap - CUB does 8-16 items per thread, we do one.
- Where no library exists (components, similarity, consensus, co-occurrence) -
  hand-written in both tiers, and those are 95% of kernel time.
- End to end the hand-written tier costs 1.5-7.4%.

---

### "What is the bottleneck, and how do you know?"

**similarity_edges at 75.3%, and it is latency-bound on scattered writes, not
atomics.**

We suspected atomic contention: 15.9 M atomic sectors on one global counter,
1.9% DRAM, 14.7% SM. So we designed a test - if it is contention, cost per edge
rises with window size; if it is the writes, it stays flat.

Measured edges/ms at w = 1, 4, 16: **1,670 / 1,917 / 1,971 - flat.**

Two scattered 8-byte stores per edge: 16 M x 16 B = 256 MB, against 268 MB
Nsight measured. The fix is to emit fewer edges, not to write them faster.

---

### "Why three profiling layers?"

**Because Nsight will never be allowed on a production GPU.**

| Layer | Tool | Overhead | Prod-safe | Answers |
|---|---|---|---|---|
| B | DCGM / nvidia-smi dmon | ~0% | yes | is it busy, throttling, neighbours |
| A | eBPF (bpftrace, BCC) | <3% | yes | launch counts, latency, host stalls |
| C | Nsight Compute | 10-40x replay | no | per-kernel DRAM, stalls, occupancy |

Start at the cheapest layer that answers the question. We used Nsight once,
after eBPF had already ruled out launch overhead (13% of GPU time).

---

### "Any negative results?"

**Two, and we are publishing both.**
- 167 eBPF probes on memory transfer and sync attached and never fired - CuPy
  links the CUDA runtime statically, so there is no `libcudart.so` to probe.
- 1 Hz DCGM sampling caught activity in 6 of 62 samples; the pipeline does
  ~75 ms of work per invocation, so most samples fall between launches.

---

### "What was your biggest optimisation?"

**Template consensus: 113 ms -> 1.1 ms, a 100x, from changing thread mapping
alone.** One block per component meant 36 blocks on a 188-SM GPU - the device
was 80% idle. One thread per (line, token) pair fixed it. The algorithm did
not change.

Runner-up: pinned host memory, 1.2 -> 13.2 GB/s for one line of code.

---

### "What did you get wrong?"

Say these plainly; they land better than pretending otherwise.

- First Drain3 baseline was wrong - our guessed regex masks gave OpenStack GA
  0.3095 against a published 0.733. Fixed by reading the `logparser` benchmark
  source: per-dataset regexes, st=0.5, depth 4 or 5.
- PA was 0.0000 on HDFS until we kept literal prefixes and punctuation.
- Radix sort broke above 262,144 keys - a strided multi-chunk scatter loop.
  Fixed with one contiguous tile per block.
- A warp-aggregated atomic patch did nothing and was reverted - only lane 0
  evaluates the emit condition, so the ballot aggregated nothing.
- A BFS test failed and the **test data** was wrong, not the kernel.
- We nearly published a wrong-by-1000x profile number. Nsight auto-scales units
  per row; summing the raw value column mixes ns, us and ms. Caught by a second
  independent capture; the fix is committed as `scripts/ncu_summary.py`.

---

### "What is left to do?"

| Priority | Item |
|---|---|
| High | Course report and paper prose |
| Medium | Replace `cp.bincount` / `cp.searchsorted` in `tokenize.py`, or scope it explicitly |
| Medium | Streaming mode for the full 211 M-line Thunderbird set |
| Medium | LogHub-2.0, plus Spell and LILAC as comparison parsers |
| Low | TikZ pipeline figure for the paper |

Future work from the profile: two-pass clustering emitting fewer edges,
decoupled-lookback single-pass scan, multi-item-per-thread radix.

---

## If you only remember one thing per teammate

| | Owns | One-liner |
|---|---|---|
| **Rohit** (G25AIT2089) | tokenise, hash, bucket, GPU I/O, profiling, paper | "The two-hash split is why we beat Drain on OpenStack." |
| **Jeenal** (G25AIT2027) | similarity, components, consensus | "Consensus went 113 ms to 1.1 ms by changing thread mapping, not the algorithm." |
| **Kosuru Yuvaraj** (G25AIT2054) | radix sort, scan, co-occurrence BFS | "Our scan matches CuPy; our radix is 2.4x behind CUB and we know exactly why." |
| **Sharvan Vittala** (G25AIT2099) | metrics, tests, benchmarks, figures | "84 tests, every stage against a CPU reference, locked clocks, median of five." |

---

## Commands you may be asked to run live

```bash
# prove it works
pytest -q                                    # 84 passed

# one benchmark
bash scripts/lock_clocks.sh && export TRIPARSE_CLOCKS_LOCKED=1
python bench/parsebench.py --dataset BGL --tier 2 --limit 1000000 \
  --window 16 --reps 5 --warmup 1 --out results/bgl_1m.json

# the whole sweep + figures
bash bench/run_all.sh && python model/stages.py

# per-kernel profile
sudo env "PATH=$PATH" bash scripts/profile.sh BGL 2 1000000 > results/ncu/bgl.csv
python scripts/ncu_summary.py results/ncu/bgl.csv
```

---

## Three sentences that work as a closing

1. "A nine-year-old algorithm with thousands of citations, used in every log
   pipeline in production, had never been moved onto parallel hardware."
2. "Where our order-independent method disagrees with the sequential one, it is
   the sequential one that is wrong."
3. "Every item in our future-work list came from a measurement, not a wish list."
