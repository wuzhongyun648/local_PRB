#!/usr/bin/env python3
"""Validate one completed formal A100 result directory."""

import argparse
import glob
import json
import os
import sys

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def close_enough(actual, expected):
    tolerance = max(1e-6, 1e-8 * max(abs(actual), abs(expected), 1.0))
    return abs(actual - expected) <= tolerance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--rounds", required=True, type=int)
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--evaluation-every", default=50, type=int)
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    require(os.path.isdir(result_dir), f"missing result directory: {result_dir}")

    final_path = os.path.join(result_dir, "final_results.npy")
    require(os.path.isfile(final_path), f"missing {final_path}")
    final_data = np.load(final_path, allow_pickle=False)
    expected_shape = (args.runs, args.rounds, 5)
    require(
        final_data.shape == expected_shape,
        f"final_results shape {final_data.shape}, expected {expected_shape}",
    )
    require(np.isfinite(final_data).all(), "final_results contains non-finite values")

    config_path = os.path.join(result_dir, "config.txt")
    summary_path = os.path.join(result_dir, "metrics_summary.json")
    require(os.path.isfile(config_path) and os.path.getsize(config_path) > 0, "bad config.txt")
    require(os.path.isfile(summary_path), "missing metrics_summary.json")
    summary = load_json(summary_path)
    require(summary["dataset"] == args.dataset, "summary dataset mismatch")
    require(summary["method"] == args.method, "summary method mismatch")
    require(summary["runs"] == args.runs, "summary runs mismatch")
    require(summary["rounds"] == args.rounds, "summary rounds mismatch")
    require(
        summary["config"]["evaluation_every"] == args.evaluation_every,
        "summary evaluation interval mismatch",
    )

    worker_metrics = sorted(
        glob.glob(os.path.join(result_dir, "worker_*_metrics.json"))
    )
    worker_timeacc = sorted(
        glob.glob(os.path.join(result_dir, "worker_*_TimeAcc.npy"))
    )
    require(
        len(worker_metrics) == args.runs,
        f"found {len(worker_metrics)} worker metrics, expected {args.runs}",
    )
    require(
        len(worker_timeacc) == args.runs,
        f"found {len(worker_timeacc)} TimeAcc files, expected {args.runs}",
    )

    expected_evaluations = (
        (args.rounds - 1) // args.evaluation_every + 1
        if args.evaluation_every > 0
        else 0
    )
    prediction_total = 0.0
    evaluation_total = 0.0
    for run_id, (metric_path, timeacc_path) in enumerate(
        zip(worker_metrics, worker_timeacc)
    ):
        metric = load_json(metric_path)
        require(metric["run_id"] == run_id, f"worker id mismatch in {metric_path}")
        require(metric["dataset"] == args.dataset, f"worker dataset mismatch: {metric_path}")
        require(metric["method"] == args.method, f"worker method mismatch: {metric_path}")
        require(metric["rounds"] == args.rounds, f"worker rounds mismatch: {metric_path}")
        require(metric["resolved_backend"] == ("scipy" if args.method == "PRB" else "numba"),
                f"worker backend mismatch: {metric_path}")

        timing = metric["timing"]
        expected_online = (
            timing["raw_online_trial_wall_time"]
            - timing["testset_build_time"]
            - timing["evaluation_time"]
            - timing["diagnostic_time"]
            - timing["adaptive_prediction_time"]
            - timing["cache_control_time"]
        )
        require(
            close_enough(timing["online_total_time"], expected_online),
            f"online timing exclusion mismatch: {metric_path}",
        )
        require(timing["evaluation_time"] > 0.0, f"evaluation did not run: {metric_path}")
        if args.method == "numba-adaptive-dyn":
            require(
                timing["adaptive_prediction_calls"] == args.rounds,
                f"adaptive predictor call count mismatch: {metric_path}",
            )
            require(
                timing["adaptive_prediction_time"] > 0.0,
                f"adaptive prediction time missing: {metric_path}",
            )
        else:
            require(
                timing["adaptive_prediction_time"] == 0.0,
                f"unexpected adaptive prediction time: {metric_path}",
            )
        prediction_total += timing["adaptive_prediction_time"]
        evaluation_total += timing["evaluation_time"]

        timeacc = np.load(timeacc_path, allow_pickle=False)
        require(
            timeacc.shape == (expected_evaluations, 2),
            f"TimeAcc shape {timeacc.shape}, expected {(expected_evaluations, 2)}",
        )
        require(np.isfinite(timeacc).all(), f"TimeAcc contains non-finite values: {timeacc_path}")

    report = {
        "status": "validated",
        "result_dir": result_dir,
        "dataset": args.dataset,
        "method": args.method,
        "rounds": args.rounds,
        "runs": args.runs,
        "final_shape": list(final_data.shape),
        "evaluation_files": len(worker_timeacc),
        "worker_metrics": len(worker_metrics),
        "evaluation_time_total": evaluation_total,
        "adaptive_prediction_time_total": prediction_total,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
        raise
