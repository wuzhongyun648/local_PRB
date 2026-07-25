#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 DATA_DIR OUTPUT_DIR LOG_DIR [ROUNDS] [SEEDS...]"
  exit 2
fi

DATA_DIR=$1
OUTPUT_DIR=$2
LOG_DIR=$3
shift 3
if [[ $# -gt 0 ]]; then
  ROUNDS=$1
  shift
else
  ROUNDS=1000
fi
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(100 101 102)
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

for INDEX in 0 1 2 3; do
  GROUP="group$((INDEX + 1))"
  CUDA_VISIBLE_DEVICES=$INDEX \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  nohup python -u "$SCRIPT_DIR/run_group.py" \
    --group "$GROUP" \
    --rounds "$ROUNDS" \
    --seeds "${SEEDS[@]}" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --continue-on-error \
    > "$LOG_DIR/${GROUP}_T${ROUNDS}.log" 2>&1 < /dev/null &
  echo "$GROUP $!"
done
