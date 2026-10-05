"""Timing helpers: warm up at full size, then take the median of N reps."""
import time
import numpy as np
import cupy as cp


def repeat(fn, warmup=1, reps=5):
    """Run fn() warmup+reps times at full size. Returns (median_dict, all_runs).

    fn must return a dict of stage -> seconds. The warmup runs matter: the
    first allocation of a large buffer costs hundreds of ms of driver time.
    """
    runs = []
    for i in range(warmup + reps):
        out = fn()
        cp.cuda.Device().synchronize()
        if i >= warmup:
            runs.append(out)
    stages = list(runs[0].keys())
    med = {s: float(np.median([r[s] for r in runs])) for s in stages}
    return med, runs


def gpu_info():
    p = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "gpu": p["name"].decode(),
        "cc": f"{p['major']}.{p['minor']}",
        "total_mem_gb": round(p["totalGlobalMem"] / 1e9, 1),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cupy": cp.__version__,
    }


def clocks_locked():
    """True if graphics clocks are pinned (needed for paper-grade numbers)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks_throttle_reasons.sw_power_cap,"
             "clocks.applications.graphics", "--format=csv,noheader"],
            text=True).strip()
        return "[N/A]" not in out.split(",")[-1]
    except Exception:
        return False
