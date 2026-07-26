#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OUTPUT_DIR LOG_DIR"
  exit 2
fi

OUTPUT_DIR=$1
LOG_DIR=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  nohup python -u "$SCRIPT_DIR/run_queue.py" \
    --queue "$GPU" \
    --output-dir "$OUTPUT_DIR" \
    --continue-on-error \
    > "$LOG_DIR/gpu${GPU}.log" 2>&1 < /dev/null &
  echo "gpu${GPU} $!"
done
