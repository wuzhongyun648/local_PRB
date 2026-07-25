#!/usr/bin/env bash

set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-$repo_dir/.venv-dyn-diagnosis/bin/python}"
run_stamp="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
log_dir="$repo_dir/log/formal_local_prb4_${run_stamp}"
result_root="$repo_dir/results/online_link_prediction"
mkdir -p "$log_dir" "$result_root"
master_log="$log_dir/master.log"

readonly runs=10
readonly workers=10
readonly evaluation_every=50
readonly -a datasets=(MovieLens Grqc Facebook Amazon_fashion)
readonly -a rounds=(10000 10000 10000 5000)
readonly -a lr1s=(0.0073 0.01 0.06 0.1)
readonly -a lr2s=(0.0004 0.004 0.02 0.01)
readonly -a gpus=(0 1 2 3)

log_master() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$master_log"
}

result_dir_from_log() {
  sed -n 's/^Output Dir: //p' "$1" | head -n 1
}

validate_result() {
  local task_log="$1" dataset="$2" task_rounds="$3" result_dir
  result_dir="$(result_dir_from_log "$task_log")"
  [[ -n "$result_dir" ]] || {
    printf 'No output directory recorded in %s\n' "$task_log" >&2
    return 1
  }
  "$python_bin" "$repo_dir/scripts/validate_a100_result.py" \
    "$result_dir" --dataset "$dataset" --method PRB \
    --rounds "$task_rounds" --runs "$runs" \
    --evaluation-every "$evaluation_every"
}

if [[ ! -x "$python_bin" ]]; then
  printf 'Python executable not found: %s\n' "$python_bin" >&2
  exit 1
fi

probe_dir="$result_root/.local_prb4_write_test_${run_stamp}"
mkdir "$probe_dir" || exit 1
"$python_bin" -c \
  'import json,numpy as np,os,sys; p=sys.argv[1]; np.save(os.path.join(p,"probe.npy"),np.arange(5)); json.dump({"ok":True},open(os.path.join(p,"probe.json"),"w")); assert np.load(os.path.join(p,"probe.npy")).tolist()==list(range(5))' \
  "$probe_dir" || exit 1
rm "$probe_dir/probe.npy" "$probe_dir/probe.json"
rmdir "$probe_dir"

commit="$(git rev-parse HEAD)"
if [[ "$commit" != "14438c444067f678b5eb283073d258d070c19aac" ]]; then
  printf 'Unexpected commit: %s\n' "$commit" >&2
  exit 1
fi

log_master "PREFLIGHT_OK commit=$commit python=$python_bin runs=$runs workers=$workers evaluation_every=$evaluation_every"

declare -a pids=()
declare -a task_logs=()
for index in "${!datasets[@]}"; do
  dataset="${datasets[$index]}"
  task_rounds="${rounds[$index]}"
  gpu="${gpus[$index]}"
  task_log="$log_dir/PRB_${dataset}.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  nohup "$python_bin" -u main.py \
    --graph_name "$dataset" --method PRB --power_T 50 \
    --T "$task_rounds" --lr1 "${lr1s[$index]}" --lr2 "${lr2s[$index]}" \
    --alpha 0.85 --runs "$runs" --workers "$workers" --seed 0 \
    --n_neg 9 --init_hops 0 --init_topk 0 \
    --evaluation_every "$evaluation_every" --hidden 100 \
    > "$task_log" 2>&1 < /dev/null &
  pid=$!
  pids+=("$pid")
  task_logs+=("$task_log")
  log_master "TASK_START method=PRB dataset=$dataset gpu=$gpu pid=$pid rounds=$task_rounds log=$task_log"
done

failed=0
declare -a done_flags=(0 0 0 0)
remaining=${#pids[@]}
while ((remaining > 0)); do
  for index in "${!pids[@]}"; do
    [[ "${done_flags[$index]}" == 0 ]] || continue
    pid="${pids[$index]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      done_flags[$index]=1
      remaining=$((remaining - 1))
      dataset="${datasets[$index]}"
      task_rounds="${rounds[$index]}"
      task_log="${task_logs[$index]}"
      if wait "$pid" && validate_result "$task_log" "$dataset" "$task_rounds" >> "$master_log" 2>&1; then
        log_master "TASK_VALIDATED method=PRB dataset=$dataset gpu=${gpus[$index]} pid=$pid"
      else
        failed=1
        log_master "TASK_FAILED method=PRB dataset=$dataset gpu=${gpus[$index]} pid=$pid log=$task_log"
      fi
    fi
  done
  ((remaining == 0)) || sleep 10
done

if ((failed != 0)); then
  log_master "LOCAL_PRB4_FAILED"
  exit 1
fi
log_master "LOCAL_PRB4_VALIDATED"
