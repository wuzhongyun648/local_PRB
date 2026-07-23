#!/usr/bin/env bash

set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

queue_stamp="${QUEUE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
log_dir="log/dyn_then_vessel_locprb_${queue_stamp}"
mkdir -p "$log_dir"
master_log="$log_dir/master.log"

ppa_gpu="${PPA_GPU:-0}"
vessel_gpu="${VESSEL_GPU:-2}"

log_master() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$master_log"
}

log_master "QUEUE_START ppa_gpu=${ppa_gpu} vessel_gpu=${vessel_gpu} runs=10 workers=10"

CUDA_VISIBLE_DEVICES="$ppa_gpu" nohup python -u main.py \
  --graph_name PPA --method numba-adaptive-dyn --alpha 0.85 --appr_eps 1.74e-06 \
  --T 5000 --lr1 0.01 --lr2 0.004 --runs 10 --workers 10 \
  > "$log_dir/PPA_dyn_locPRB.log" 2>&1 < /dev/null &
ppa_pid=$!
log_master "START name=PPA_dyn_locPRB gpu=${ppa_gpu} pid=${ppa_pid}"

CUDA_VISIBLE_DEVICES="$vessel_gpu" nohup python -u main.py \
  --graph_name Vessel --method numba-adaptive-dyn --alpha 0.85 --appr_eps 2.86e-07 \
  --T 5000 --lr1 0.01 --lr2 0.004 --runs 10 --workers 10 \
  > "$log_dir/Vessel_dyn_locPRB.log" 2>&1 < /dev/null &
vessel_dyn_pid=$!
log_master "START name=Vessel_dyn_locPRB gpu=${vessel_gpu} pid=${vessel_dyn_pid}"

if wait "$ppa_pid"; then
  ppa_status=0
else
  ppa_status=$?
fi
log_master "DONE name=PPA_dyn_locPRB gpu=${ppa_gpu} pid=${ppa_pid} status=${ppa_status}"

if wait "$vessel_dyn_pid"; then
  vessel_dyn_status=0
else
  vessel_dyn_status=$?
fi
log_master "DONE name=Vessel_dyn_locPRB gpu=${vessel_gpu} pid=${vessel_dyn_pid} status=${vessel_dyn_status}"

if ((ppa_status != 0 || vessel_dyn_status != 0)); then
  log_master "QUEUE_STOP dyn_failure ppa_status=${ppa_status} vessel_status=${vessel_dyn_status}"
  exit 1
fi

CUDA_VISIBLE_DEVICES="$vessel_gpu" nohup python -u main.py \
  --graph_name Vessel --method numba-locPRB --alpha 0.85 --appr_eps 2.86e-07 \
  --T 5000 --lr1 0.01 --lr2 0.004 --runs 10 --workers 10 \
  > "$log_dir/Vessel_LocPRB.log" 2>&1 < /dev/null &
vessel_loc_pid=$!
log_master "START name=Vessel_LocPRB gpu=${vessel_gpu} pid=${vessel_loc_pid}"

if wait "$vessel_loc_pid"; then
  vessel_loc_status=0
else
  vessel_loc_status=$?
fi
log_master "DONE name=Vessel_LocPRB gpu=${vessel_gpu} pid=${vessel_loc_pid} status=${vessel_loc_status}"

if ((vessel_loc_status != 0)); then
  log_master "QUEUE_DONE failures=1"
  exit 1
fi

log_master "QUEUE_DONE failures=0"
