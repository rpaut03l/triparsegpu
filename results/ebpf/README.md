# Layer A (eBPF) results and limits

## What worked

`cuda_launch_latency.bt` — uprobes on `cuLaunchKernel` in libcuda.so.

BGL, tier 2, 1M lines:
- 314 kernel launches
- median launch call ~5 us, long tail to 64 us
- total host-side launch cost: 2.83 ms
- against ~22 ms of kernel time: launch overhead is ~13% of GPU time

## What did not work, and why

`cuda_sync_stalls.bt` attaches but never fires. CuPy's wheel
(cupy-cuda13x 14.2.0) statically links the CUDA runtime — there is no
libcudart.so in the process map, only libcuda.so.595.58.03. Memcpy and
synchronisation therefore never cross a dynamically-linked symbol boundary that
a uprobe can attach to. Kernel launches are the exception: they funnel through
libcuda's `cuLaunchKernel`, which is why that probe works.

Implication for the paper's observability section: on a statically linked
runtime, eBPF sees launches but not transfers or syncs. Layer B (DCGM) and
Layer C (Nsight) cover what it cannot. A deployment that needs host-side
transfer visibility must use a dynamically linked CUDA runtime.
