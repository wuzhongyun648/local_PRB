#!/usr/bin/env bash

set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-$repo_dir/.venv-dyn-diagnosis/bin/python}"
rounds="${ROUNDS:-20}"
runs="${RUNS:-1}"
seed="${SEED:-0}"
cuda_device="${CUDA_DEVICE:-}"
stamp="${MATRIX_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
log_dir="log/backend_matrix_T${rounds}_${stamp}"
mkdir -p "$log_dir"
master_log="$log_dir/master.log"
resource_log="$log_dir/resources.log"

read -r -a datasets <<< "${DATASETS:-MovieLens Amazon_fashion Facebook Grqc Collab PPA Vessel}"
read -r -a methods <<< "${METHODS:-LocPRB dyn_locPRB}"
read -r -a backends <<< "${BACKENDS:-python numba}"
diagnostics="${DIAGNOSTICS:-0}"
diagnostic_every="${DIAGNOSTIC_EVERY:-1}"

declare -A eps=(
  [MovieLens]=8.33e-05 [Amazon_fashion]=1.25e-04
  [Facebook]=2.48e-04 [Grqc]=1.91e-04 [Collab]=4.24e-06
  [PPA]=1.74e-06 [Vessel]=2.86e-07
)
declare -A lr1=(
  [MovieLens]=0.0073 [Amazon_fashion]=0.1 [Facebook]=0.06
  [Grqc]=0.01 [Collab]=0.01 [PPA]=0.01 [Vessel]=0.01
)
declare -A lr2=(
  [MovieLens]=0.0004 [Amazon_fashion]=0.01 [Facebook]=0.02
  [Grqc]=0.004 [Collab]=0.004 [PPA]=0.004 [Vessel]=0.004
)
declare -A kernel=(
  [MovieLens]=40 [Amazon_fashion]=40 [Facebook]=40 [Grqc]=40
  [Collab]=40 [PPA]=40 [Vessel]=5
)

snapshot_resources() {
  {
    date -u +%Y-%m-%dT%H:%M:%SZ
    uptime
    vmstat 1 2 | tail -1
    ps -eo pid,stat,pcpu,pmem,etime,args --sort=-pcpu | head -12
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,pstate --format=csv,noheader 2>&1 || true
  } >> "$resource_log"
}

failures=()
printf 'matrix_start=%s rounds=%s runs=%s seed=%s python=%s cuda=%q\n' \
  "$stamp" "$rounds" "$runs" "$seed" "$python_bin" "$cuda_device" | tee "$master_log"

for dataset in "${datasets[@]}"; do
  for backend in "${backends[@]}"; do
    for method in "${methods[@]}"; do
      name="${dataset}_${method}_${backend}"
      extra_args=()
      if [[ "$diagnostics" == "1" && "$method" == "dyn_locPRB" ]]; then
        extra_args+=(--ppr_diagnostics --ppr_diagnostic_every "$diagnostic_every")
        name="${name}_diagnostics"
      fi
      snapshot_resources
      printf '[%s] START %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" | tee -a "$master_log"
      if CUDA_VISIBLE_DEVICES="$cuda_device" \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        "$python_bin" -u main.py \
          --graph_name "$dataset" \
          --method "$method" \
          --ppr_backend "$backend" \
          --alpha 0.85 \
          --appr_eps "${eps[$dataset]}" \
          --T "$rounds" \
          --lr1 "${lr1[$dataset]}" \
          --lr2 "${lr2[$dataset]}" \
          --kernel_size "${kernel[$dataset]}" \
          --runs "$runs" \
          --workers 1 \
          --seed "$seed" \
          --evaluation_every 0 \
          "${extra_args[@]}" \
          > "$log_dir/${name}.log" 2>&1; then
        printf '[%s] DONE %s status=0\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" | tee -a "$master_log"
      else
        rc=$?
        failures+=("${name}:${rc}")
        printf '[%s] DONE %s status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" "$rc" | tee -a "$master_log"
      fi
      snapshot_resources
    done
  done
done

if ((${#failures[@]})); then
  printf 'matrix_done failures=%s details=%s\n' "${#failures[@]}" "${failures[*]}" | tee -a "$master_log"
  exit 1
fi

printf 'matrix_done failures=0\n' | tee -a "$master_log"
