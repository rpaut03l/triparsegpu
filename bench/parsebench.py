#!/usr/bin/env python3
"""ClusterBench for TriParseGPU: one run -> one JSON in results/<gpu>/.

Usage:
  python bench/parsebench.py --dataset HDFS --tier 2 --reps 5 \
      --out results/<gpu>/HDFS_t2.json
  python bench/parsebench.py --dataset HDFS --tier 2 --limit 1000000 ...
"""
import argparse, json, pathlib, subprocess, sys, time
import numpy as np
import cupy as cp
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from triparsegpu import io, pipeline, timing

CFG = yaml.safe_load(open(pathlib.Path(__file__).parent / "datasets.yaml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--tier", type=int, default=2, choices=[1, 2])
    ap.add_argument("--limit", type=int, default=None, help="max lines")
    ap.add_argument("--st", type=float, default=0.4)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    c = CFG[a.dataset]

    # CPU-side load + preprocess, timed once (not part of the GPU figure)
    t0 = time.perf_counter()
    contents, misses = io.load_contents(c["path"], a.dataset, limit=a.limit)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    pre = io.preprocess(contents, a.dataset)
    t_regex = time.perf_counter() - t0
    n = len(pre)

    def run():
        _, tpls, T, meta = pipeline.parse(
            pre, tier=a.tier, st=a.st, window=a.window,
            depth_tokens=c["depth_tokens"], timing=True)
        run.meta = meta
        run.ntpl = len(tpls)
        return T

    med, runs = timing.repeat(run, warmup=a.warmup, reps=a.reps)

    kernel_stages = ("tokenize", "hash", "sort", "similarity", "ccl", "template")
    kernel_s = sum(med[s] for s in kernel_stages if s in med)
    gpu_s = sum(v for k, v in med.items() if k != "to_buffer")
    e2e_s = t_read + t_regex + sum(med.values())
    base = c["drain3_lines_per_s"]

    res = dict(
        dataset=a.dataset, tier=a.tier, lines=n, limit=a.limit,
        templates=run.ntpl, format_misses=misses,
        st=a.st, window=a.window, depth_tokens=c["depth_tokens"],
        edges=run.meta["edges"], ccl_rounds=run.meta["ccl_rounds"],
        stage_s=med,
        kernel_s=kernel_s, gpu_s=gpu_s, e2e_s=e2e_s,
        cpu_read_s=t_read, cpu_regex_s=t_regex,
        kernel_lines_per_s=n / kernel_s,
        gpu_lines_per_s=n / gpu_s,
        e2e_lines_per_s=n / e2e_s,
        drain3_lines_per_s=base,
        speedup_kernel=n / kernel_s / base,
        speedup_gpu=n / gpu_s / base,
        speedup_e2e=n / e2e_s / base,
        reps=a.reps, warmup=a.warmup,
        clocks_locked=timing.clocks_locked(),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).strip(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        **timing.gpu_info(),
    )

    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p, "w"), indent=1)

    warn = "" if res["clocks_locked"] else "  [CLOCKS NOT LOCKED]"
    print(f"{a.dataset:12s} t{a.tier}  {n:>11,} lines  "
          f"kernels {res['kernel_lines_per_s']:>12,.0f} l/s ({res['speedup_kernel']:>6.0f}x)  "
          f"gpu {res['gpu_lines_per_s']:>11,.0f} l/s ({res['speedup_gpu']:>5.0f}x)  "
          f"e2e {res['e2e_lines_per_s']:>9,.0f} l/s{warn}")


if __name__ == "__main__":
    main()
