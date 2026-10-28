#!/usr/bin/env python3
"""Per-kernel time summary from an ncu --csv export.

ncu auto-scales units PER ROW (ns/us/ms). Summing the raw Metric Value
column mixes scales and silently produces wrong percentages — always
normalise via the Metric Unit column, as below.

usage: python scripts/ncu_summary.py results/ncu/bgl_t2_idle.csv
"""
import sys
import pandas as pd

d = pd.read_csv(sys.argv[1], thousands=",")
d.columns = [c.strip('"') for c in d.columns]
d = d[d["Metric Name"] == "gpu__time_duration.sum"].copy()
scale = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
d["us"] = d["Metric Value"] * d["Metric Unit"].map(scale)
g = d.groupby("Kernel Name")["us"].agg(["sum", "count"]).sort_values("sum", ascending=False)
g["pct"] = 100 * g["sum"] / g["sum"].sum()
print(g.round(1).to_string())
print(f"\ntotal {g['sum'].sum():,.0f} us over {int(g['count'].sum())} launches")
