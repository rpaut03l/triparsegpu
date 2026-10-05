#!/usr/bin/env bash
# Full sweep. Lock clocks first (scripts/lock_clocks.sh) and make sure
# nothing else is on the GPU.
set -e
cd "$(dirname "$0")/.."
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_')
OUT="results/$GPU"
mkdir -p "$OUT"

echo "=== tier comparison, full datasets ==="
for D in OpenStack BGL Thunderbird HDFS; do
  for T in 1 2; do
    python bench/parsebench.py --dataset $D --tier $T --reps 5 \
      --out "$OUT/${D}_t${T}.json"
  done
done

echo "=== scaling curve ==="
for D in BGL HDFS; do
  for L in 100000 1000000 10000000; do
    python bench/parsebench.py --dataset $D --tier 2 --limit $L --reps 5 \
      --out "$OUT/${D}_t2_n${L}.json"
  done
done

echo "=== similarity window sensitivity ==="
for W in 4 8 16 32 64; do
  python bench/parsebench.py --dataset HDFS --tier 2 --limit 1000000 \
    --window $W --reps 3 --out "$OUT/HDFS_t2_w${W}.json"
done

echo "done -> $OUT"
