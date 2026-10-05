#!/usr/bin/env bash
# Layer C: Nsight Compute. EXCLUSIVE GPU ONLY - replays each kernel 10-40x.
# usage: scripts/profile.sh BGL 2 1000000
set -e
cd "$(dirname "$0")/.."
D=${1:-BGL}; T=${2:-2}; L=${3:-1000000}
mkdir -p results/ncu results/nsys
TAG="${D}_t${T}_n${L}"

echo "--- checking the GPU is free ---"
nvidia-smi --query-compute-apps=pid,process_name --format=csv

echo "--- ncu (per-kernel counters) ---"
ncu --set full --csv \
  -k regex:"radix_|block_scan|add_offsets|compact_indices|ccl_|similarity_edges|consensus|hash_tokens|mark_newlines|content_start" \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum \
  --page raw -o "results/ncu/$TAG" \
  python bench/parsebench.py --dataset $D --tier $T --limit $L --reps 1 --warmup 0 --out /tmp/tpg/ncu.json \
  2>&1 | tail -20

echo "--- nsys (timeline) ---"
nsys profile --stats=true -o "results/nsys/$TAG" --force-overwrite true \
  python bench/parsebench.py --dataset $D --tier $T --limit $L --reps 3 --out /tmp/tpg/nsys.json \
  2>&1 | tail -30

echo "done: results/ncu/$TAG  results/nsys/$TAG"
