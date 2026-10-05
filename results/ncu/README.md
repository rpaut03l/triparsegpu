# Layer C (Nsight Compute) — BGL tier 2, 1M lines

Requires sudo (ERR_NVGPUCTRPERM otherwise; the permanent fix is
NVreg_RestrictProfilingToAdminUsers=0 plus a reboot).

## Per-kernel breakdown

kernel            calls      us   dram_mb  dram%   sm%    atom_sectors  GB/s
similarity_edges      1  8305.2    268.2    1.9   14.7      15,908,678    32
ccl_hook              3  1761.2    809.8   51.5    8.0      15,322,319   460
consensus_reduce      1   402.4     52.5    7.6   22.0               0   130
block_scan           18   160.8     72.2   17.2   19.6               0   449
radix_scatter         8   112.6    128.4   67.0   30.1               0  1141
radix_histogram       8    94.7     64.1   39.7   22.2               0   677
hash_tokens           1    87.5    100.7   67.4   69.5               0  1150
add_offsets           9    62.9     72.1   67.7    6.4               0  1147
ccl_compress          3    22.1     24.0   64.6    9.6               0  1089

Total: 11.0 ms over 54 launches. Card peak is about 1790 GB/s.

## Findings

1. similarity_edges is 75% of kernel time at 1.9% of peak DRAM and 14.7% SM.
   A window sweep shows edges-per-millisecond is nearly constant (1670, 1917,
   1971 M/ms at windows 1, 4, 16), so contention on the global edge counter is
   NOT the limiter despite 15.9M atomic sectors. The cost is the two scattered
   8-byte stores per edge: 16M edges x 16 bytes = 256 MB, matching the measured
   268 MB. Latency-bound on uncoalesced writes.
   The real fix is emitting fewer edges, not writing them faster - a two-pass
   scheme that runs CCL at window 4 and then compares only component
   representatives.

2. ccl_hook is 16% of time at 460 GB/s with a long-scoreboard stall ratio of
   120. This is the dependent-load pointer chase in find_root, memory-latency-
   bound as expected for union-find.

3. hash_tokens reaches 69.5% SM throughput and 1150 GB/s - the only kernel near
   the hardware limit, and it is 0.8% of total time.

4. add_offsets runs at 1147 GB/s but only 6.4% SM with a stall ratio of 168:
   pure bandwidth work with nothing to overlap. Evidence for replacing the
   three-kernel scan with a single-pass decoupled-lookback version.

## Caveat

One unrelated process (filenest) was resident on the GPU during these runs.
Numbers are indicative; re-run on an idle card before publication.
