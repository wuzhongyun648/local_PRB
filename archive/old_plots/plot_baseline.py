import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = CURRENT_DIR / "results" / "baselines_new"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot baseline regret/time npy files.")
    parser.add_argument(
        "--result-dir",
        default=str(DEFAULT_RESULT_ROOT),
        help="Directory containing baseline npy files, or the baselines_new root.",
    )
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to plot")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to plot")
    parser.add_argument("--output-dir", default=None, help="Directory for generated plots")
    parser.add_argument("--plot-time", action="store_true", help="Also plot cumulative time curves")
    return parser.parse_args()


def resolve_result_dir(path):
    result_dir = Path(path).expanduser().resolve()
    if (result_dir / "metadata.json").exists():
        return result_dir

    candidates = [
        child
        for child in result_dir.iterdir()
        if child.is_dir() and (child / "metadata.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No metadata.json found in {result_dir} or its run subdirectories")
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_metadata(result_dir):
    metadata_path = result_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    extra_metadata = []
    for path in sorted(result_dir.glob("metadata_*.json")):
        with path.open("r", encoding="utf-8") as f:
            extra_metadata.append(json.load(f))
    if extra_metadata:
        metadata["extra_metadata"] = extra_metadata
    return metadata


def infer_names(result_dir, suffix):
    names = []
    for path in sorted(result_dir.glob(f"*_{suffix}.npy")):
        names.append(path.name[: -len(f"_{suffix}.npy")])
    return names


def choose_datasets_methods(result_dir, metadata, datasets_arg, methods_arg):
    datasets = datasets_arg or metadata.get("datasets")
    methods = methods_arg or metadata.get("methods")
    if datasets_arg is None:
        datasets = list(datasets or [])
        for item in metadata.get("extra_metadata", []):
            for dataset in item.get("datasets", []):
                if dataset not in datasets:
                    datasets.append(dataset)
    if datasets and methods:
        return datasets, methods

    names = infer_names(result_dir, "regret")
    inferred = []
    known_methods = methods or ["EE-Net", "NeuralUCB", "NeuralTS", "NeuralGreedy", "LinUCB", "KernelUCB"]
    for name in names:
        for method in known_methods:
            suffix = f"_{method}"
            if name.endswith(suffix):
                inferred.append((name[: -len(suffix)], method))
                break

    if datasets is None:
        datasets = sorted({dataset for dataset, _method in inferred})
    if methods is None:
        methods = [method for method in known_methods if any(m == method for _d, m in inferred)]
    return datasets, methods


def summarize_array(array):
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    final_values = arr[:, -1]
    return {
        "runs": arr.shape[0],
        "T": arr.shape[1],
        "final_mean": float(np.mean(final_values)),
        "final_std": float(np.std(final_values)),
        "curve_mean": np.mean(arr, axis=0),
        "curve_std": np.std(arr, axis=0),
    }


def print_table(rows):
    if not rows:
        print("No data found.")
        return

    headers = ["Dataset", "Method", "Runs", "T", "FinalRegretMean", "FinalRegretStd", "FinalTimeMeanSec"]
    table = [headers]
    for row in rows:
        table.append(
            [
                row["dataset"],
                row["method"],
                str(row["runs"]),
                str(row["T"]),
                f"{row['final_regret_mean']:.3f}",
                f"{row['final_regret_std']:.3f}",
                "NA" if row["final_time_mean"] is None else f"{row['final_time_mean']:.3f}",
            ]
        )

    widths = [max(len(item) for item in column) for column in zip(*table)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in table[1:]:
        print(fmt.format(*row))


def plot_metric(dataset, methods, summaries, output_dir, metric_name, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for method in methods:
        summary = summaries.get((dataset, method, metric_name))
        if summary is None:
            continue
        x = np.arange(summary["T"])
        mean = summary["curve_mean"]
        std = summary["curve_std"]
        ax.plot(x, mean, label=method)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(f"{dataset} {ylabel}")
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / f"{dataset}_{metric_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    result_dir = resolve_result_dir(args.result_dir)
    metadata = load_metadata(result_dir)
    datasets, methods = choose_datasets_methods(result_dir, metadata, args.datasets, args.methods)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Result directory: {result_dir}")
    print(f"Output directory: {output_dir}")

    rows = []
    summaries = {}
    for dataset in datasets:
        for method in methods:
            regret_path = result_dir / f"{dataset}_{method}_regret.npy"
            if not regret_path.exists():
                print(f"Warning: missing {regret_path.name}, skipped")
                continue

            regret_summary = summarize_array(np.load(regret_path))
            summaries[(dataset, method, "regret")] = regret_summary

            time_summary = None
            time_path = result_dir / f"{dataset}_{method}_time.npy"
            if time_path.exists():
                time_array = np.cumsum(np.asarray(np.load(time_path), dtype=float), axis=-1)
                time_summary = summarize_array(time_array)
                summaries[(dataset, method, "time")] = time_summary

            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "runs": regret_summary["runs"],
                    "T": regret_summary["T"],
                    "final_regret_mean": regret_summary["final_mean"],
                    "final_regret_std": regret_summary["final_std"],
                    "final_time_mean": None if time_summary is None else time_summary["final_mean"],
                }
            )

    print("\n=== Baseline Summary ===")
    print_table(rows)

    saved = []
    for dataset in datasets:
        path = plot_metric(dataset, methods, summaries, output_dir, "regret", "Cumulative Regret")
        if path is not None:
            saved.append(path)
        if args.plot_time:
            path = plot_metric(dataset, methods, summaries, output_dir, "time", "Cumulative Time (s)")
            if path is not None:
                saved.append(path)

    print("\n=== Saved Plots ===")
    if saved:
        for path in saved:
            print(path)
    else:
        print("No plots saved.")


if __name__ == "__main__":
    main()
