# Layer B (device counters) results and resolution limit

Tool: `nvidia-smi dmon -s pucm -d 1` (DCGM not installed; dmon reads the same
counters). Minimum sampling interval is 1 second.

## Observation

BGL tier 2, 10 back-to-back invocations of 20 reps each:

- 6 of 62 samples recorded any SM activity at all (~10%)
- SM utilisation in those samples: 10, 23, 40, 99, 99% — a 10x spread
- Memory controller: 6-27%
- Peak power 79 W of a ~300 W budget

## Why

The pipeline does ~75 ms of GPU work per invocation on 4.7M lines. At 1 Hz, most
samples fall between kernel launches, and any sample that does land averages
kernel time together with host-side Python time in the same second.

## Implication for the paper

Coarse device counters tell you a GPU is busy; they do not tell you what it is
doing. For millisecond-scale kernels this layer has a resolution floor well above
the workload. It remains useful for one thing: confirming no other process is on
the card during a benchmark, and confirming clocks are where you set them
(pclk 2332-2340 MHz under the 2626 MHz lock, i.e. power-limited on this Max-Q
part).

Per-kernel attribution requires Layer C (Nsight Compute). Host-side launch and
transfer accounting requires Layer A (eBPF), with the static-runtime caveat in
results/ebpf/README.md.
