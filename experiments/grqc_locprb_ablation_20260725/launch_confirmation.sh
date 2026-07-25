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
  ROUNDS=10000
fi
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(200 201 202 203 204)
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Two unique configurations per GPU, five paired validation seeds each.
QUEUE_NAMES=(confirm_gpu0 confirm_gpu1 confirm_gpu2 confirm_gpu3)
GROUP_HINTS=(group1 group2 group3 group4)
VARIANT_0=(g1_original_loc g1_current_loc)
VARIANT_1=(s3_shared_serving s4_no_positive_replacement)
VARIANT_2=(sf_all_fixed r0_eenet_direct)
VARIANT_3=(r2_power p4_exploit_only)

for INDEX in 0 1 2 3; do
  QUEUE_NAME=${QUEUE_NAMES[$INDEX]}
  GROUP_HINT=${GROUP_HINTS[$INDEX]}
  VARIANT_VAR="VARIANT_${INDEX}[@]"
  VARIANTS=("${!VARIANT_VAR}")
  CUDA_VISIBLE_DEVICES=$INDEX \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  nohup python -u "$SCRIPT_DIR/run_group.py" \
    --group "$GROUP_HINT" \
    --queue-name "$QUEUE_NAME" \
    --variants "${VARIANTS[@]}" \
    --rounds "$ROUNDS" \
    --seeds "${SEEDS[@]}" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --continue-on-error \
    > "$LOG_DIR/${QUEUE_NAME}_T${ROUNDS}.log" 2>&1 < /dev/null &
  echo "$QUEUE_NAME $!"
done
