#!/usr/bin/env bash

set -uo pipefail

cd "$(dirname "$0")"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p log

run_batch() {
    local dataset="$1"
    local rounds="$2"
    local eps="$3"
    local lr1="$4"
    local lr2="$5"
    local -a labels=(Loc DynLoc)
    local -a methods=(LocPRB dyn_locPRB)
    local -a pids=()

    echo "[$(date -u +%FT%TZ)] START batch ${dataset} (2 tasks, 10 workers)"
    for index in "${!labels[@]}"; do
        local label="${labels[$index]}"
        local log_path="log/matrix_${dataset}_${label}_${RUN_TAG}.log"
        "$PYTHON_BIN" -u main.py \
            --graph_name "$dataset" \
            --method "${methods[$index]}" \
            --alpha 0.85 \
            --appr_eps "$eps" \
            --T "$rounds" \
            --lr1 "$lr1" \
            --lr2 "$lr2" \
            --runs 10 \
            --workers 5 \
            --seed 0 \
            --if_save true \
            >"$log_path" 2>&1 &
        pids+=("$!")
        echo "[$(date -u +%FT%TZ)]   ${label} pid=${pids[-1]} log=${log_path}"
    done

    local status=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[$(date -u +%FT%TZ)] DONE ${dataset} ${labels[$index]}"
        else
            echo "[$(date -u +%FT%TZ)] FAIL ${dataset} ${labels[$index]}" >&2
            status=1
        fi
    done
    if (( status != 0 )); then
        echo "[$(date -u +%FT%TZ)] STOP after failed batch ${dataset}" >&2
        return "$status"
    fi
    echo "[$(date -u +%FT%TZ)] COMPLETE batch ${dataset}"
}

run_batch MovieLens 10000 8.33e-05 0.0073 0.0004 || exit $?
run_batch Amazon_fashion 5000 0.000125 0.1 0.01 || exit $?
run_batch Collab 5000 4.24e-06 0.01 0.004 || exit $?
run_batch Vessel 5000 2.86e-07 0.01 0.004 || exit $?

echo "[$(date -u +%FT%TZ)] ALL 8 EXPERIMENTS COMPLETE"
"$PYTHON_BIN" -m src.plot_locprb_comparison \
    --result-root results/online_link_prediction \
    --output-dir results/plots/locprb_comparison
echo "[$(date -u +%FT%TZ)] ALL COMPARISON FIGURES COMPLETE"
