#!/usr/bin/env python3
"""Figures from the benchmark sweep. Usage: python model/stages.py results/<gpu>"""
import json, glob, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rd = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                  glob.glob("results/NVIDIA*")[0])
out = pathlib.Path("figures"); out.mkdir(exist_ok=True)
runs = [json.load(open(f)) for f in rd.glob("*.json")]
plt.rcParams.update({"figure.dpi": 160, "font.size": 9})

# --- Figure 1: throughput vs Drain3, per dataset -------------------------
full = {r["dataset"]: r for r in runs
        if r["limit"] is None and r["tier"] == 2 and r["window"] == 16}
if full:
    ds = ["OpenStack", "BGL", "Thunderbird", "HDFS"]
    ds = [d for d in ds if d in full]
    x = np.arange(len(ds)); w = 0.27
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(x - w, [full[d]["drain3_lines_per_s"] for d in ds], w, label="Drain3 (CPU)")
    ax.bar(x,     [full[d]["gpu_lines_per_s"] for d in ds], w, label="GPU incl. H2D")
    ax.bar(x + w, [full[d]["kernel_lines_per_s"] for d in ds], w, label="GPU kernels")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(ds)
    ax.set_ylabel("lines / second (log)")
    ax.set_title("TriParseGPU vs Drain3, full datasets, locked clocks")
    for i, d in enumerate(ds):
        ax.text(i + w, full[d]["kernel_lines_per_s"] * 1.15,
                f"{full[d]['speedup_kernel']:.0f}x", ha="center", fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig1_throughput.png"); plt.close(fig)
    print("fig1_throughput.png")

# --- Figure 2: stage breakdown -------------------------------------------
if "BGL" in full:
    st = full["BGL"]["stage_s"]
    order = ["tokenize", "hash", "sort", "similarity", "ccl", "template"]
    vals = [st[k] * 1000 for k in order if k in st]
    labs = [k for k in order if k in st]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(labs[::-1], vals[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:.1f} ms ({v/sum(vals)*100:.0f}%)", va="center", fontsize=8)
    ax.set_xlabel("milliseconds"); ax.set_title(f"Stage breakdown, BGL {full['BGL']['lines']:,} lines")
    ax.set_xlim(0, max(vals) * 1.35); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig2_stages.png"); plt.close(fig)
    print("fig2_stages.png")

# --- Figure 3: scaling ----------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.2))
for d in ("BGL", "HDFS"):
    pts = sorted([(r["lines"], r["kernel_lines_per_s"]) for r in runs
                  if r["dataset"] == d and r["tier"] == 2 and r["window"] == 16])
    if len(pts) > 1:
        ax.plot(*zip(*pts), "o-", label=d)
ax.set_xscale("log"); ax.set_xlabel("input lines (log)")
ax.set_ylabel("kernel lines / second")
ax.set_title("Throughput scales with input size")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(out / "fig3_scaling.png"); plt.close(fig)
print("fig3_scaling.png")

# --- Figure 4: window cost ------------------------------------------------
wr = sorted([(r["window"], r["kernel_lines_per_s"], r["edges"]) for r in runs
             if r["dataset"] == "HDFS" and r["limit"] == 1_000_000])
if len(wr) > 1:
    w_, tp, ed = zip(*wr)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(w_, tp, "o-", color="tab:blue", label="throughput")
    ax.set_xscale("log", base=2); ax.set_xlabel("similarity window")
    ax.set_ylabel("kernel lines / s", color="tab:blue")
    ax2 = ax.twinx(); ax2.plot(w_, np.array(ed)/1e6, "s--", color="tab:red", label="edges")
    ax2.set_ylabel("edges (millions)", color="tab:red")
    ax.axvline(16, color="green", ls=":", lw=1.5)
    ax.text(16, max(tp)*.6, " accuracy\n saturates\n here", fontsize=8, color="green")
    ax.set_title("Window cost vs benefit (HDFS, 1M lines)")
    fig.tight_layout(); fig.savefig(out / "fig4_window.png"); plt.close(fig)
    print("fig4_window.png")

# --- Figure 5: roofline from ncu -----------------------------------------
import csv, io as _io
ncu = pathlib.Path("results/ncu/BGL_t2_mem.csv")
if ncu.exists():
    import pandas as pd
    raw = ncu.read_text().splitlines()
    h = next(i for i, l in enumerate(raw) if l.startswith('"ID"'))
    df = pd.read_csv(_io.StringIO("\n".join([raw[h]] + raw[h+2:])), thousands=",")
    n = lambda c: pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    k = pd.DataFrame({"kernel": df["Kernel Name"], "us": n("gpu__time_duration.sum")/1000,
                      "mb": n("dram__bytes.sum")/1e6}).groupby("kernel").sum()
    k["gbs"] = (k.mb/1e3)/(k.us/1e6)
    PEAK = 1790
    fig, ax = plt.subplots(figsize=(6, 3.4))
    y = np.arange(len(k)); k = k.sort_values("gbs")
    ax.barh(y, k.gbs, color=["tab:red" if v < 200 else "tab:orange" if v < 800 else "tab:green" for v in k.gbs])
    ax.axvline(PEAK, color="k", ls="--", lw=1); ax.text(PEAK*.97, len(k)-.5, "peak", ha="right", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(k.index, fontsize=8)
    ax.set_xlabel("achieved DRAM bandwidth (GB/s)")
    ax.set_title("Per-kernel bandwidth vs hardware peak")
    fig.tight_layout(); fig.savefig(out / "fig5_bandwidth.png"); plt.close(fig)
    print("fig5_bandwidth.png")

print(f"\nall figures -> {out}/")
