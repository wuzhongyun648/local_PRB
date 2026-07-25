#!/usr/bin/env bash

set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-/inspire/ssd/project/fdu-aidake-cfff/public/wuzhongyun/envs/localprb/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  printf 'Python executable not found: %s\n' "$python_bin" >&2
  exit 1
fi

run_stamp="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
log_dir="$repo_dir/log/formal_a100_21_${run_stamp}"
result_root="$repo_dir/results/online_link_prediction"
mkdir -p "$log_dir" "$result_root"
master_log="$log_dir/master.log"

readonly runs=10
readonly workers=10
readonly evaluation_every=50
readonly common_args="--alpha 0.85 --runs $runs --workers $workers --seed 0 --n_neg 9 --init_hops 0 --init_topk 0 --evaluation_every $evaluation_every --hidden 100"
readonly -a gpus=(0 1)

readonly -a datasets=(
  PPA Vessel Collab MovieLens Grqc Facebook Amazon_fashion
)
readonly -a rounds=(
  5000 5000 5000 10000 10000 10000 5000
)
readonly -a epsilons=(
  1.74e-06 2.86e-07 4.24e-06 8.33e-05 1.91e-04 2.48e-04 1.25e-04
)
readonly -a lr1s=(
  0.01 0.01 0.01 0.0073 0.01 0.06 0.1
)
readonly -a lr2s=(
  0.004 0.004 0.004 0.0004 0.004 0.02 0.01
)
readonly -a phases=(
  numba-adaptive-dyn numba-locPRB PRB
)

log_master() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$master_log"
}

result_dir_from_log() {
  sed -n 's/^Output Dir: //p' "$1" | head -n 1
}

validate_result() {
  local task_log="$1"
  local dataset="$2"
  local method="$3"
  local task_rounds="$4"
  local result_dir
  result_dir="$(result_dir_from_log "$task_log")"
  if [[ -z "$result_dir" ]]; then
    printf 'No output directory recorded in %s\n' "$task_log" >&2
    return 1
  fi
  "$python_bin" "$repo_dir/scripts/validate_a100_result.py" \
    "$result_dir" \
    --dataset "$dataset" \
    --method "$method" \
    --rounds "$task_rounds" \
    --runs "$runs" \
    --evaluation-every "$evaluation_every"
}

preflight() {
  local test_dir="$result_root/.a100_write_test_${run_stamp}"
  mkdir "$test_dir"
  "$python_bin" -c \
    'import json, numpy as np, os, sys; p=sys.argv[1]; np.save(os.path.join(p, "probe.npy"), np.arange(5)); json.dump({"ok": True}, open(os.path.join(p, "probe.json"), "w")); assert np.load(os.path.join(p, "probe.npy")).tolist() == list(range(5))' \
    "$test_dir"
  rm "$test_dir/probe.npy" "$test_dir/probe.json"
  rmdir "$test_dir"

  local commit
  commit="$(/inspire/ssd/project/fdu-aidake-cfff/public/wuzhongyun/miniconda3/bin/git rev-parse HEAD)"
  if [[ "$commit" != "14438c444067f678b5eb283073d258d070c19aac" ]]; then
    printf 'Unexpected commit: %s\n' "$commit" >&2
    return 1
  fi
  "$python_bin" -m pip check
  log_master "PREFLIGHT_OK commit=$commit python=$python_bin runs=$runs workers=$workers evaluation_every=$evaluation_every"
}

run_phase() {
  local method="$1"
  local next_index=0
  local failed=0
  local pid index gpu dataset task_rounds epsilon lr1 lr2 task_name task_log
  local -a active_pids=()
  local -A pid_index=()
  local -A pid_gpu=()
  local -A pid_log=()
  local -A busy_gpu=()

  log_master "PHASE_START method=$method tasks=${#datasets[@]}"

  while ((next_index < ${#datasets[@]} || ${#active_pids[@]} > 0)); do
    while ((failed == 0 && next_index < ${#datasets[@]} && ${#active_pids[@]} < 2)); do
      index="$next_index"
      gpu=""
      local candidate_gpu
      for candidate_gpu in "${gpus[@]}"; do
        if [[ -z "${busy_gpu[$candidate_gpu]:-}" ]]; then
          gpu="$candidate_gpu"
          break
        fi
      done
      if [[ -z "$gpu" ]]; then
        log_master "INTERNAL_ERROR no free GPU for method=$method dataset=${datasets[$index]}"
        return 1
      fi
      dataset="${datasets[$index]}"
      task_rounds="${rounds[$index]}"
      epsilon="${epsilons[$index]}"
      lr1="${lr1s[$index]}"
      lr2="${lr2s[$index]}"
      task_name="${method}_${dataset}"
      task_log="$log_dir/${task_name}.log"

      local -a solver_args
      if [[ "$method" == "PRB" ]]; then
        solver_args=(--power_T 50)
      else
        solver_args=(--appr_eps "$epsilon")
      fi

      CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      PYTHONUNBUFFERED=1 \
      nohup "$python_bin" -u main.py \
        --graph_name "$dataset" \
        --method "$method" \
        "${solver_args[@]}" \
        --T "$task_rounds" \
        --lr1 "$lr1" \
        --lr2 "$lr2" \
        $common_args \
        > "$task_log" 2>&1 < /dev/null &
      pid=$!
      active_pids+=("$pid")
      pid_index["$pid"]="$index"
      pid_gpu["$pid"]="$gpu"
      pid_log["$pid"]="$task_log"
      busy_gpu["$gpu"]="$pid"
      next_index=$((next_index + 1))
      log_master "TASK_START method=$method dataset=$dataset gpu=$gpu pid=$pid rounds=$task_rounds log=$task_log"
    done

    local finished_pid=""
    while [[ -z "$finished_pid" ]]; do
      for pid in "${active_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          finished_pid="$pid"
          break
        fi
      done
      [[ -n "$finished_pid" ]] || sleep 10
    done

    index="${pid_index[$finished_pid]}"
    gpu="${pid_gpu[$finished_pid]}"
    task_log="${pid_log[$finished_pid]}"
    dataset="${datasets[$index]}"
    task_rounds="${rounds[$index]}"
    if wait "$finished_pid" && validate_result "$task_log" "$dataset" "$method" "$task_rounds" >> "$master_log" 2>&1; then
      log_master "TASK_VALIDATED method=$method dataset=$dataset gpu=$gpu pid=$finished_pid"
    else
      failed=1
      log_master "TASK_FAILED method=$method dataset=$dataset gpu=$gpu pid=$finished_pid log=$task_log"
    fi

    local -a remaining=()
    for pid in "${active_pids[@]}"; do
      [[ "$pid" == "$finished_pid" ]] || remaining+=("$pid")
    done
    active_pids=("${remaining[@]}")
    unset 'busy_gpu[$gpu]'
    unset 'pid_index[$finished_pid]' 'pid_gpu[$finished_pid]' 'pid_log[$finished_pid]'
  done

  if ((failed != 0)); then
    log_master "PHASE_FAILED method=$method"
    return 1
  fi
  log_master "PHASE_VALIDATED method=$method tasks=${#datasets[@]}"
}

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'commit=14438c444067f678b5eb283073d258d070c19aac phases=%d tasks=%d max_active=2 gpus=0,1 runs=%d workers=%d evaluation_every=%d\n' \
    "${#phases[@]}" "$((${#phases[@]} * ${#datasets[@]}))" "$runs" "$workers" "$evaluation_every"
  for method in "${phases[@]}"; do
    for index in "${!datasets[@]}"; do
      if [[ "$method" == "PRB" ]]; then
        solver="power_T=50"
      else
        solver="eps=${epsilons[$index]}"
      fi
      printf '%s %s T=%s %s lr1=%s lr2=%s\n' \
        "$method" "${datasets[$index]}" "${rounds[$index]}" "$solver" "${lr1s[$index]}" "${lr2s[$index]}"
    done
  done
  exit 0
fi

preflight || exit 1
for method in "${phases[@]}"; do
  run_phase "$method" || exit 1
done
log_master "ALL_21_VALIDATED"
