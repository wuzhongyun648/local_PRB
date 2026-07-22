#!/usr/bin/env bash

set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

default_python="/inspire/ssd/project/fdu-aidake-cfff/public/wuzhongyun/envs/localprb/bin/python"
python_bin="${PYTHON_BIN:-$default_python}"
if [[ ! -x "$python_bin" ]]; then
  printf 'Python executable not found: %s\n' "$python_bin" >&2
  exit 1
fi

queue_stamp="${QUEUE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
log_dir="log/dynamic_queue_loc_dyn_a100_${queue_stamp}"
mkdir -p "$log_dir"
master_log="$log_dir/master.log"

read -r -a gpus <<< "${QUEUE_GPUS:-0 1}"
max_active="${QUEUE_MAX_ACTIVE:-2}"

# Run the large graphs first to reduce the chance that one long experiment is
# left running alone at the end. LocPRB is launched before dyn_locPRB.
task_names=(
  PPA_LocPRB Vessel_LocPRB Collab_LocPRB MovieLens_LocPRB Grqc_LocPRB Facebook_LocPRB Amazon_fashion_LocPRB
  PPA_dyn_locPRB Vessel_dyn_locPRB Collab_dyn_locPRB MovieLens_dyn_locPRB Grqc_dyn_locPRB Facebook_dyn_locPRB Amazon_fashion_dyn_locPRB
)

task_args=(
  "--graph_name PPA --method LocPRB --alpha 0.85 --appr_eps 1.74e-06 --T 5000 --lr1 0.01 --lr2 0.004"
  "--graph_name Vessel --method LocPRB --alpha 0.85 --appr_eps 2.86e-07 --T 5000 --lr1 0.01 --lr2 0.004"
  "--graph_name Collab --method LocPRB --alpha 0.85 --appr_eps 4.24e-06 --T 5000 --lr1 0.01 --lr2 0.004"
  "--graph_name MovieLens --method LocPRB --alpha 0.85 --appr_eps 8.33e-05 --T 10000 --lr1 0.0073 --lr2 0.0004"
  "--graph_name Grqc --method LocPRB --alpha 0.85 --appr_eps 1.91e-04 --T 10000 --lr1 0.01 --lr2 0.004"
  "--graph_name Facebook --method LocPRB --alpha 0.85 --appr_eps 2.48e-04 --T 10000 --lr1 0.06 --lr2 0.02"
  "--graph_name Amazon_fashion --method LocPRB --alpha 0.85 --appr_eps 1.25e-04 --T 5000 --lr1 0.1 --lr2 0.01"
  "--graph_name PPA --method dyn_locPRB --alpha 0.85 --appr_eps 1.74e-06 --T 5000 --lr1 0.01 --lr2 0.004"
  "--graph_name Vessel --method dyn_locPRB --alpha 0.85 --appr_eps 2.86e-07 --T 5000 --lr1 0.01 --lr2 0.004"
  "--graph_name Collab --method dyn_locPRB --alpha 0.85 --appr_eps 4.24e-06 --T 5000 --lr1 0.01 --lr2 0.004"
  "--graph_name MovieLens --method dyn_locPRB --alpha 0.85 --appr_eps 8.33e-05 --T 10000 --lr1 0.0073 --lr2 0.0004"
  "--graph_name Grqc --method dyn_locPRB --alpha 0.85 --appr_eps 1.91e-04 --T 10000 --lr1 0.01 --lr2 0.004"
  "--graph_name Facebook --method dyn_locPRB --alpha 0.85 --appr_eps 2.48e-04 --T 10000 --lr1 0.06 --lr2 0.02"
  "--graph_name Amazon_fashion --method dyn_locPRB --alpha 0.85 --appr_eps 1.25e-04 --T 5000 --lr1 0.1 --lr2 0.01"
)

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'python=%s tasks=%d max_active=%d gpus=%s\n' "$python_bin" "${#task_names[@]}" "$max_active" "${gpus[*]}"
  for index in "${!task_names[@]}"; do
    printf '%02d %s %s\n' "$((index + 1))" "${task_names[$index]}" "${task_args[$index]}"
  done
  exit 0
fi

declare -a active_pids=()
declare -A pid_gpu=()
declare -A pid_name=()
declare -A busy_gpu=()
declare -a failures=()
next_task=0
gpu_cursor=0
selected_gpu=""

log_master() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$master_log"
}

select_gpu() {
  local offset index candidate
  selected_gpu=""
  for ((offset = 0; offset < ${#gpus[@]}; offset++)); do
    index=$(((gpu_cursor + offset) % ${#gpus[@]}))
    candidate="${gpus[$index]}"
    if [[ -z "${busy_gpu[$candidate]:-}" ]]; then
      selected_gpu="$candidate"
      gpu_cursor=$(((index + 1) % ${#gpus[@]}))
      return 0
    fi
  done
  return 1
}

launch_next() {
  local index name gpu pid
  local -a args
  index="$next_task"
  name="${task_names[$index]}"
  select_gpu
  gpu="$selected_gpu"
  read -r -a args <<< "${task_args[$index]}"

  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  nohup "$python_bin" -u main.py \
    "${args[@]}" --runs 10 --workers 10 \
    > "$log_dir/${name}.log" 2>&1 < /dev/null &
  pid=$!

  active_pids+=("$pid")
  pid_gpu["$pid"]="$gpu"
  pid_name["$pid"]="$name"
  busy_gpu["$gpu"]="$pid"
  next_task=$((next_task + 1))
  log_master "START task=$((index + 1))/${#task_names[@]} name=${name} gpu=${gpu} pid=${pid}"
}

finish_one() {
  local finished_pid rc gpu name pid
  local -a remaining=()
  finished_pid=""
  while [[ -z "$finished_pid" ]]; do
    for pid in "${active_pids[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        finished_pid="$pid"
        break
      fi
    done
    [[ -n "$finished_pid" ]] || sleep 5
  done

  gpu="${pid_gpu[$finished_pid]}"
  name="${pid_name[$finished_pid]}"
  if wait "$finished_pid"; then
    rc=0
  else
    rc=$?
  fi
  unset 'busy_gpu[$gpu]' 'pid_gpu[$finished_pid]' 'pid_name[$finished_pid]'
  for pid in "${active_pids[@]}"; do
    [[ "$pid" == "$finished_pid" ]] || remaining+=("$pid")
  done
  active_pids=("${remaining[@]}")

  log_master "DONE name=${name} gpu=${gpu} pid=${finished_pid} status=${rc}"
  if ((rc != 0)); then
    failures+=("${name}:${rc}")
  fi
}

log_master "QUEUE_START tasks=${#task_names[@]} max_active=${max_active} gpus=${gpus[*]} runs=10 workers=10"

while ((next_task < ${#task_names[@]})) && ((${#active_pids[@]} < max_active)); do
  launch_next
done

while ((${#active_pids[@]} > 0)); do
  finish_one
  while ((next_task < ${#task_names[@]})) && ((${#active_pids[@]} < max_active)); do
    launch_next
  done
done

if ((${#failures[@]} > 0)); then
  log_master "QUEUE_DONE failures=${#failures[@]} details=${failures[*]}"
  exit 1
fi

log_master "QUEUE_DONE failures=0"
