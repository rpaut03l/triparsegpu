"""Tier 0: Drain3 CPU reference. Produces baseline throughput + reference cluster ids."""
import time, sys, json, argparse, pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

# Official logparser benchmark settings (logpai/logparser, Drain/benchmark.py).
# regex -> "<*>" via mask_with="*"; st and depth per dataset.
SETTINGS = {
    "BGL":         {"regex": [r"core\.\d+"], "st": 0.5, "depth": 4},
    "HDFS":        {"regex": [r"blk_(|-)[0-9]+", r"(\d+\.){3}\d+(:\d+)?"], "st": 0.5, "depth": 4},
    "Thunderbird": {"regex": [r"(\d+\.){3}\d+"], "st": 0.5, "depth": 4},
    "OpenStack":   {"regex": [r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"], "st": 0.5, "depth": 5},
}


def run(contents, sim_th=None, depth=None, masking=True, report_every=1_000_000, dataset=None):
    cf = SETTINGS.get(dataset, {"regex": [], "st": 0.5, "depth": 4})
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = sim_th if sim_th is not None else cf["st"]
    cfg.drain_depth = depth if depth is not None else cf["depth"]
    cfg.profiling_enabled = False
    if masking:
        cfg.masking_instructions = [MaskingInstruction(r, "*") for r in cf["regex"]]
    tm = TemplateMiner(config=cfg)
    ids = []
    t0 = time.perf_counter()
    for n, line in enumerate(contents, 1):
        ids.append(tm.add_log_message(line)["cluster_id"])
        if n % report_every == 0:
            el = time.perf_counter() - t0
            print(f"  {n:,} lines  {n/el:,.0f} lines/s  {len(tm.drain.clusters)} templates", flush=True)
    dt = time.perf_counter() - t0
    return ids, {c.cluster_id: c.get_template() for c in tm.drain.clusters}, dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--raw", action="store_true", help="plain .log file instead of structured csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=None, help="BGL|HDFS|Thunderbird|OpenStack for per-dataset masks")
    a = ap.parse_args()

    if a.raw:
        with open(a.path, errors="ignore") as f:
            contents = [l.rstrip("\n") for i, l in enumerate(f) if not a.limit or i < a.limit]
    else:
        df = pd.read_csv(a.path)
        contents = df.Content.astype(str).tolist()[: a.limit]

    ids, tpl, dt = run(contents, dataset=a.dataset)
    res = dict(path=a.path, lines=len(contents), templates=len(tpl),
               seconds=dt, lines_per_s=len(contents) / dt)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump({"summary": res, "ids": ids, "templates": tpl}, open(a.out, "w"))

if __name__ == "__main__":
    main()
