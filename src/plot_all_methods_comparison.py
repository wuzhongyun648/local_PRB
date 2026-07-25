#!/usr/bin/env python3
"""Compare the formal PRB-family runs with the active PRB-style baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = (
    REPO_ROOT / "results" / "baselines_prb_style" / "20260710_184342"
)
DEFAULT_FORMAL_DIR = (
    REPO_ROOT / "results" / "formal_21_completed_20260725" / "results"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "all_methods_comparison_20260725"

DATASETS = {
    "MovieLens": {
        "baseline_aliases": ["MovieLens"],
        "formal": "MovieLens",
        "horizon": 10000,
    },
    "AmazonFashion": {
        "baseline_aliases": ["AmazonFashion", "Amazon"],
        "formal": "Amazon_fashion",
        "horizon": 5000,
    },
    "Facebook": {
        "baseline_aliases": ["Facebook"],
        "formal": "Facebook",
        "horizon": 10000,
    },
    "GrQc": {
        "baseline_aliases": ["GrQc"],
        "formal": "Grqc",
        "horizon": 10000,
    },
    "Collab": {
        "baseline_aliases": ["Collab"],
        "formal": "Collab",
        "horizon": 5000,
    },
    "PPA": {
        "baseline_aliases": ["PPA"],
        "formal": "PPA",
        "horizon": 5000,
    },
    "Vessel": {
        "baseline_aliases": ["Vessel"],
        "formal": "Vessel",
        "horizon": 5000,
    },
}

BASELINE_METHODS = [
    "EE-Net",
    "NeuralUCB",
    "NeuralTS",
    "NeuralGreedy",
    "LinUCB",
    "KernelUCB",
]
FORMAL_METHOD_LABELS = {
    "numba-adaptive-dyn": "Adaptive DYN (Numba)",
    "numba-locPRB": "LocPRB (Numba)",
    "PRB": "PRB",
}
METHOD_ORDER = BASELINE_METHODS + list(FORMAL_METHOD_LABELS.values())

METHOD_STYLES = {
    "EE-Net": {"color": "#7f7f7f", "linestyle": "--"},
    "NeuralUCB": {"color": "#8c564b", "linestyle": "--"},
    "NeuralTS": {"color": "#9467bd", "linestyle": "--"},
    "NeuralGreedy": {"color": "#e377c2", "linestyle": "--"},
    "LinUCB": {"color": "#bcbd22", "linestyle": "--"},
    "KernelUCB": {"color": "#17becf", "linestyle": "--"},
    "Adaptive DYN (Numba)": {"color": "#d62728", "linestyle": "-"},
    "LocPRB (Numba)": {"color": "#1f77b4", "linestyle": "-"},
    "PRB": {"color": "#2ca02c", "linestyle": "-"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot regret for six PRB-style baselines plus three formal methods, "
            "and evaluation accuracy for the three formal methods."
        )
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--formal-dir", type=Path, default=DEFAULT_FORMAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_matrix(
    data: np.ndarray,
    path: Path,
    *,
    ndim: int,
    runs: int = 10,
) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    require(array.ndim == ndim, f"{path}: expected {ndim} dimensions, got {array.shape}")
    require(array.shape[0] == runs, f"{path}: expected {runs} runs, got {array.shape}")
    require(np.isfinite(array).all(), f"{path}: contains NaN or Inf")
    return array


def confidence_interval(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    if n <= 1:
        return np.zeros_like(np.mean(values, axis=axis))
    return 1.96 * np.std(values, axis=axis, ddof=1) / math.sqrt(n)


def final_stats(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1))


def load_baselines(baseline_dir: Path) -> tuple[dict, dict]:
    metadata_path = baseline_dir / "metadata.json"
    require(metadata_path.is_file(), f"Missing baseline metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    require(metadata.get("args", {}).get("T") == 10000, "Baseline horizon is not 10000")
    require(metadata.get("runs") == 10, "Baseline run count is not 10")
    require(not metadata.get("failures"), "Baseline metadata contains failures")
    require(len(metadata.get("successes", [])) == 420, "Baseline does not have 420 successes")
    metadata_datasets = set(metadata.get("datasets", []))

    records = {}
    for display_dataset, spec in DATASETS.items():
        horizon = spec["horizon"]
        candidates = [
            name
            for name in spec["baseline_aliases"]
            if name in metadata_datasets
        ]
        require(
            len(candidates) == 1,
            f"{display_dataset}: expected exactly one baseline dataset alias, got {candidates}",
        )
        baseline_dataset = candidates[0]
        for method in BASELINE_METHODS:
            stem = f"{baseline_dataset}_{method}"
            regret_path = baseline_dir / f"{stem}_regret.npy"
            time_path = baseline_dir / f"{stem}_time.npy"
            require(regret_path.is_file(), f"Missing baseline regret: {regret_path}")
            require(time_path.is_file(), f"Missing baseline time: {time_path}")
            regret = validate_matrix(
                np.load(regret_path, allow_pickle=False), regret_path, ndim=2
            )
            step_time = validate_matrix(
                np.load(time_path, allow_pickle=False), time_path, ndim=2
            )
            require(
                regret.shape[1] >= horizon and step_time.shape[1] >= horizon,
                f"{stem}: shorter than comparison horizon {horizon}",
            )
            records[(display_dataset, method)] = {
                "kind": "baseline",
                "regret": regret[:, :horizon],
                "step_time": step_time[:, :horizon],
                "time_accuracy": None,
                "source": str(baseline_dir),
            }
    return records, metadata


def worker_sort_key(path: Path) -> int:
    match = re.search(r"worker_(\d+)_TimeAcc\.npy$", path.name)
    require(match is not None, f"Cannot parse worker id: {path}")
    return int(match.group(1))


def load_time_accuracy(run_dir: Path, horizon: int) -> np.ndarray:
    paths = sorted(run_dir.glob("worker_*_TimeAcc.npy"), key=worker_sort_key)
    require(len(paths) == 10, f"{run_dir}: expected 10 TimeAcc files, found {len(paths)}")
    arrays = []
    expected_points = (horizon - 1) // 50 + 1
    for path in paths:
        data = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        require(
            data.shape == (expected_points, 2),
            f"{path}: expected {(expected_points, 2)}, got {data.shape}",
        )
        require(np.isfinite(data).all(), f"{path}: contains NaN or Inf")
        require(np.all(np.diff(data[:, 0]) >= 0), f"{path}: time is not monotonic")
        require(
            np.all((data[:, 1] >= 0.0) & (data[:, 1] <= 1.0)),
            f"{path}: accuracy outside [0, 1]",
        )
        arrays.append(data)
    return np.stack(arrays, axis=0)


def discover_formal_runs(formal_dir: Path) -> dict:
    expected = {
        (spec["formal"], method)
        for spec in DATASETS.values()
        for method in FORMAL_METHOD_LABELS
    }
    found = {}
    for summary_path in sorted(formal_dir.glob("*/metrics_summary.json")):
        summary = read_json(summary_path)
        key = (summary.get("dataset"), summary.get("method"))
        if key not in expected:
            continue
        require(key not in found, f"Duplicate formal result for {key}")
        found[key] = (summary_path.parent, summary)
    missing = sorted(expected - set(found))
    require(not missing, f"Missing formal results: {missing}")
    require(len(found) == 21, f"Expected 21 formal results, found {len(found)}")
    return found


def load_formal(formal_dir: Path) -> dict:
    found = discover_formal_runs(formal_dir)
    records = {}
    for display_dataset, spec in DATASETS.items():
        horizon = spec["horizon"]
        for internal_method, display_method in FORMAL_METHOD_LABELS.items():
            run_dir, summary = found[(spec["formal"], internal_method)]
            require(summary.get("runs") == 10, f"{run_dir}: run count is not 10")
            require(summary.get("rounds") == horizon, f"{run_dir}: horizon mismatch")
            result_path = run_dir / "final_results.npy"
            data = validate_matrix(
                np.load(result_path, allow_pickle=False), result_path, ndim=3
            )
            require(
                data.shape == (10, horizon, 5),
                f"{result_path}: expected {(10, horizon, 5)}, got {data.shape}",
            )
            step_time = data[:, :, 0]
            regret = data[:, :, 1]
            require(np.all(step_time >= 0.0), f"{result_path}: negative step time")
            time_accuracy = load_time_accuracy(run_dir, horizon)
            records[(display_dataset, display_method)] = {
                "kind": "formal",
                "regret": regret,
                "step_time": step_time,
                "time_accuracy": time_accuracy,
                "source": str(run_dir),
                "internal_method": internal_method,
            }
    return records


def configure_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", alpha=0.35)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_regret(dataset: str, records: dict, figures_dir: Path) -> None:
    horizon = DATASETS[dataset]["horizon"]
    rounds = np.arange(1, horizon + 1)
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    for method in METHOD_ORDER:
        regret = records[(dataset, method)]["regret"]
        mean = np.mean(regret, axis=0)
        ci = confidence_interval(regret)
        style = METHOD_STYLES[method]
        linewidth = 3.0 if method in FORMAL_METHOD_LABELS.values() else 2.0
        ax.plot(
            rounds,
            mean,
            label=method,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=linewidth,
        )
        ax.fill_between(
            rounds,
            mean - ci,
            mean + ci,
            color=style["color"],
            alpha=0.10,
            linewidth=0,
        )
    configure_axes(ax, dataset, "Round", "Cumulative Regret")
    ax.set_xlim(1, horizon)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=3, fontsize=10.5, frameon=True, loc="upper left")
    save_figure(fig, figures_dir / f"{dataset}_Regret_vs_Round")


def plot_evaluation_accuracy(dataset: str, records: dict, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    for method in FORMAL_METHOD_LABELS.values():
        time_accuracy = records[(dataset, method)]["time_accuracy"]
        mean_time = np.mean(time_accuracy[:, :, 0], axis=0)
        accuracy_percent = time_accuracy[:, :, 1] * 100.0
        mean_accuracy = np.mean(accuracy_percent, axis=0)
        ci = confidence_interval(accuracy_percent)
        style = METHOD_STYLES[method]
        ax.plot(
            mean_time,
            mean_accuracy,
            label=method,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=3.0,
        )
        ax.fill_between(
            mean_time,
            mean_accuracy - ci,
            mean_accuracy + ci,
            color=style["color"],
            alpha=0.12,
            linewidth=0,
        )
    configure_axes(
        ax,
        dataset,
        "Cumulative Online Time (s, evaluation excluded)",
        "Fixed-Test Evaluation Accuracy (%)",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11.5, frameon=True, loc="best")
    save_figure(fig, figures_dir / f"{dataset}_Evaluation_Accuracy_vs_Time")


def make_summary_rows(records: dict) -> list[dict]:
    rows = []
    for dataset, spec in DATASETS.items():
        horizon = spec["horizon"]
        for method in METHOD_ORDER:
            record = records[(dataset, method)]
            final_regret_mean, final_regret_std = final_stats(
                record["regret"][:, -1]
            )
            total_times = np.sum(record["step_time"], axis=1)
            total_time_mean, total_time_std = final_stats(total_times)
            per_round_ms = total_times / horizon * 1000.0
            per_round_mean, per_round_std = final_stats(per_round_ms)
            row = {
                "dataset": dataset,
                "method": method,
                "method_family": record["kind"],
                "runs": int(record["regret"].shape[0]),
                "horizon": horizon,
                "final_regret_mean": final_regret_mean,
                "final_regret_std": final_regret_std,
                "online_time_mean_seconds": total_time_mean,
                "online_time_std_seconds": total_time_std,
                "mean_round_time_mean_ms": per_round_mean,
                "mean_round_time_std_ms": per_round_std,
                "final_evaluation_round": "",
                "final_evaluation_accuracy_mean_percent": "",
                "final_evaluation_accuracy_std_percent": "",
                "final_evaluation_time_mean_seconds": "",
                "source": record["source"],
            }
            if record["time_accuracy"] is not None:
                time_accuracy = record["time_accuracy"]
                accuracy = time_accuracy[:, -1, 1] * 100.0
                accuracy_mean, accuracy_std = final_stats(accuracy)
                row.update(
                    {
                        "final_evaluation_round": horizon - 50,
                        "final_evaluation_accuracy_mean_percent": accuracy_mean,
                        "final_evaluation_accuracy_std_percent": accuracy_std,
                        "final_evaluation_time_mean_seconds": float(
                            np.mean(time_accuracy[:, -1, 0])
                        ),
                    }
                )
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_markdown(
    rows: list[dict],
    path: Path,
    baseline_dir: Path,
    formal_dir: Path,
) -> None:
    by_dataset = {
        dataset: [row for row in rows if row["dataset"] == dataset]
        for dataset in DATASETS
    }
    lines = [
        "# All-Method Comparison",
        "",
        "## Protocol",
        "",
        f"- Baselines: `{baseline_dir}`",
        f"- Formal PRB-family runs: `{formal_dir}`",
        "- Regret figures contain all six baselines and the three formal methods.",
        "- Evaluation Accuracy–Time figures contain only Adaptive DYN (Numba), "
        "LocPRB (Numba), and PRB. The baseline run did not perform fixed-test evaluation.",
        "- Curves show the mean of 10 runs; shaded regions are 95% confidence intervals.",
        "- Baseline arrays are truncated to each formal dataset horizon.",
        "- Online time is the sum of per-round times. Formal-method time excludes test-set "
        "construction, evaluation, diagnostics, adaptive prediction, and cache-control time. "
        "Baseline time is its recorded per-round wall time and has no evaluation component.",
        "- The source runs were produced on different machines/GPU configurations. Time is "
        "reported as observed wall-clock performance and is not a hardware-normalized benchmark.",
        "- The time table reports the mean total online-loop time of one run, not the sum "
        "of all 10 runs and not end-to-end batch runtime. Mean per-round milliseconds are "
        "shown separately.",
        "- Evaluation accuracy uses the final available fixed-test evaluation, which occurs "
        "50 rounds before the horizon because evaluation runs at rounds 0, 50, 100, ...",
        "",
    ]
    for dataset, dataset_rows in by_dataset.items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"![{dataset} regret](figures/{dataset}_Regret_vs_Round.png)",
                "",
                f"![{dataset} evaluation accuracy](figures/{dataset}_Evaluation_Accuracy_vs_Time.png)",
                "",
                "| Method | Horizon | Final cumulative regret | Online time (s) | "
                "Mean round time (ms) | Final evaluation round | Final evaluation accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in dataset_rows:
            if row["final_evaluation_accuracy_mean_percent"] == "":
                eval_round = "—"
                eval_accuracy = "—"
            else:
                eval_round = str(row["final_evaluation_round"])
                eval_accuracy = mean_std(
                    float(row["final_evaluation_accuracy_mean_percent"]),
                    float(row["final_evaluation_accuracy_std_percent"]),
                ) + "%"
            lines.append(
                f"| {row['method']} | {row['horizon']} | "
                f"{mean_std(row['final_regret_mean'], row['final_regret_std'])} | "
                f"{mean_std(row['online_time_mean_seconds'], row['online_time_std_seconds'])} | "
                f"{mean_std(row['mean_round_time_mean_ms'], row['mean_round_time_std_ms'], 3)} | "
                f"{eval_round} | {eval_accuracy} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_metadata(
    path: Path,
    baseline_metadata: dict,
    baseline_dir: Path,
    formal_dir: Path,
    output_dir: Path,
) -> None:
    payload = {
        "baseline_dir": str(baseline_dir),
        "formal_dir": str(formal_dir),
        "output_dir": str(output_dir),
        "datasets": DATASETS,
        "methods": METHOD_ORDER,
        "baseline_metadata": {
            "run_name": baseline_metadata.get("run_name"),
            "runner_style": baseline_metadata.get("runner_style"),
            "created_at": baseline_metadata.get("created_at"),
            "successes": len(baseline_metadata.get("successes", [])),
            "failures": len(baseline_metadata.get("failures", [])),
        },
        "regret_figure_methods": METHOD_ORDER,
        "evaluation_figure_methods": list(FORMAL_METHOD_LABELS.values()),
        "confidence_interval": "mean ± 1.96 * sample_std / sqrt(10)",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_dir = args.baseline_dir.expanduser().resolve()
    formal_dir = args.formal_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_records, baseline_metadata = load_baselines(baseline_dir)
    formal_records = load_formal(formal_dir)
    records = {**baseline_records, **formal_records}
    require(
        len(records) == len(DATASETS) * len(METHOD_ORDER),
        f"Expected 63 dataset/method records, found {len(records)}",
    )

    for dataset in DATASETS:
        plot_regret(dataset, records, figures_dir)
        plot_evaluation_accuracy(dataset, records, figures_dir)

    rows = make_summary_rows(records)
    write_csv(rows, output_dir / "summary.csv")
    write_markdown(
        rows,
        output_dir / "ALL_METHODS_COMPARISON.md",
        baseline_dir,
        formal_dir,
    )
    write_run_metadata(
        output_dir / "comparison_metadata.json",
        baseline_metadata,
        baseline_dir,
        formal_dir,
        output_dir,
    )
    print(f"Validated and summarized {len(rows)} dataset/method combinations.")
    print(f"Saved 14 PNG and 14 PDF figures under {figures_dir}")
    print(f"Saved {output_dir / 'summary.csv'}")
    print(f"Saved {output_dir / 'ALL_METHODS_COMPARISON.md'}")


if __name__ == "__main__":
    main()
