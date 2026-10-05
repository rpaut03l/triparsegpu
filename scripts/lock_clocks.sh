#!/usr/bin/env bash
# Lock clocks for paper-grade numbers. Run ONLY when you have the GPU to
# yourself - this setting is global. Always unlock afterwards.
set -e
case "${1:-lock}" in
  lock)
    sudo nvidia-smi -pm 1
    MAX=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits)
    LOCK=$(( MAX * 85 / 100 ))     # 85% of max: sustainable without throttling
    echo "locking graphics clock to ${LOCK} MHz (max ${MAX})"
    sudo nvidia-smi -lgc "${LOCK},${LOCK}"
    nvidia-smi --query-gpu=clocks.applications.graphics --format=csv
    ;;
  unlock)
    sudo nvidia-smi -rgc
    echo "clocks unlocked"
    ;;
  *) echo "usage: $0 [lock|unlock]"; exit 1 ;;
esac
