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


import os


def clock_state():
    """What the GPU clock is doing right now.

    `clocks.applications.graphics` is deprecated on recent drivers, so we read
    the current clock and the throttle reasons instead. When -lgc is active the
    driver reports the SW_POWER_CAP / APPLICATIONS_CLOCKS_SETTING throttle
    reason, which is the reliable signal.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=clocks.current.graphics,clocks.max.graphics,"
             "clocks_throttle_reasons.applications_clocks_setting",
             "--format=csv,noheader,nounits"],
            text=True).strip()
        cur, mx, reason = [x.strip() for x in out.split(",")]
        # The throttle reason only shows under load, and the applications-clock
        # field is deprecated on this driver. Record the raw numbers and let the
        # caller note the lock explicitly; a flag that guesses is worse than none.
        locked = reason.lower() in ("active", "1", "true")
        # TRIPARSE_CLOCKS_LOCKED=1 is set by scripts/lock_clocks.sh so the
        # recorded result states the methodology rather than inferring it.
        declared = os.environ.get("TRIPARSE_CLOCKS_LOCKED") == "1"
        return {"clock_mhz": int(cur) if cur.isdigit() else None,
                "clock_max_mhz": int(mx) if mx.isdigit() else None,
                "clocks_locked": declared or locked,
                "clocks_locked_declared": declared,
                "throttle_reason": reason}
    except Exception as e:
        return {"clock_mhz": None, "clock_max_mhz": None,
                "clocks_locked": False, "throttle_reason": f"unavailable: {e}"}
