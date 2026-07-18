import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_configs import RESULTS_DIR


KNOWN_BASELINE_METHODS = [
    "EE-Net",
    "NeuralUCB",
    "NeuralTS",
    "NeuralGreedy",
    "Neural_epsilon",
    "LinUCB",
    "KernelUCB",
]


def load_curve(path, kind):
    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if kind == "online":
        if data.ndim != 3 or data.shape[2] < 2:
            raise ValueError(f"Expected online final_results shape (runs, T, >=2), got {data.shape}")
        regret = data[:, :, 1]
        step_time = data[:, :, 0]
    else:
        regret = data
        step_time = None
    return regret, step_time


def summarize(regret, step_time=None):
    regret = np.asarray(regret, dtype=float)
    if regret.ndim == 1:
        regret = regret.reshape(1, -1)
    summary = {
        "runs": regret.shape[0],
        "T": regret.shape[1],
        "regret_mean": np.mean(regret, axis=0),
        "regret_std": np.std(regret, axis=0),
        "final_regret_mean": float(np.mean(regret[:, -1])),
        "final_regret_std": float(np.std(regret[:, -1])),
    }
    if step_time is not None:
        step_time = np.asarray(step_time, dtype=float)
        if step_time.ndim == 1:
            step_time = step_time.reshape(1, -1)
        cumulative = np.cumsum(step_time, axis=1)
        summary["time_mean"] = np.mean(cumulative, axis=0)
        summary["final_time_mean"] = float(np.mean(cumulative[:, -1]))
    return summary


def latest_baseline_dir(root):
    root = Path(root)
    if (root / "metadata.json").exists():
        return root
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and (path / "metadata.json").exists())
    if not candidates:
        return root
    return candidates[-1]


def read_metadata(path):
    metadata_path = Path(path) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_dataset_method(stem, methods):
    for method in methods:
        suffix = f"_{method}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], method
    return None, None


def collect_baseline(result_dir, datasets, methods):
    result_dir = latest_baseline_dir(result_dir)
    metadata = read_metadata(result_dir)
    datasets = datasets or metadata.get("datasets") or []
    methods = methods or metadata.get("methods") or KNOWN_BASELINE_METHODS
    summaries = {}
    for regret_path in sorted(result_dir.glob("*_regret.npy")):
        stem = regret_path.name.removesuffix("_regret.npy")
        dataset, method = split_dataset_method(stem, methods)
        if not dataset or not method:
            continue
        if datasets and dataset not in datasets:
            continue
        time_path = result_dir / f"{dataset}_{method}_time.npy"
        regret, _ = load_curve(regret_path, "baseline")
        step_time = np.load(time_path) if time_path.exists() else None
        summaries[(dataset, method)] = summarize(regret, step_time)
    return result_dir, summaries


def collect_online(paths):
    summaries = {}
    for item in paths:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            result_path = path / "final_results.npy"
            label = path.name
        else:
            result_path = path
            label = path.parent.name
        regret, step_time = load_curve(result_path, "online")
        summaries[label] = summarize(regret, step_time)
    return summaries


def print_summary(rows):
    if not rows:
        print("No data found.")
        return
    header = ("Label", "Runs", "T", "FinalRegretMean", "FinalRegretStd", "FinalTimeMeanSec")
    table = [header]
    for row in rows:
        table.append(
            (
                row["label"],
                str(row["runs"]),
                str(row["T"]),
                f"{row['final_regret_mean']:.3f}",
                f"{row['final_regret_std']:.3f}",
                "NA" if row.get("final_time_mean") is None else f"{row['final_time_mean']:.3f}",
            )
        )
    widths = [max(len(cell) for cell in col) for col in zip(*table)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*header))
    print(fmt.format(*["-" * width for width in widths]))
    for row in table[1:]:
        print(fmt.format(*row))


def plot_regret(summary_items, output_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, summary in summary_items:
        x = np.arange(summary["T"])
        mean = summary["regret_mean"]
        std = summary["regret_std"]
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Regret")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified plotting for active result formats.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Plot baseline regret files.")
    baseline.add_argument("--result-dir", default=str(Path(RESULTS_DIR) / "baselines_new"))
    baseline.add_argument("--datasets", nargs="+", default=None)
    baseline.add_argument("--methods", nargs="+", default=None)
    baseline.add_argument("--output-dir", default=str(Path(RESULTS_DIR) / "plots"))

    online = subparsers.add_parser("online", help="Plot one or more online final_results.npy files.")
    online.add_argument("paths", nargs="+", help="Run directories or final_results.npy files")
    online.add_argument("--output-dir", default=str(Path(RESULTS_DIR) / "plots"))
    online.add_argument("--name", default="online_regret")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "baseline":
        result_dir, summaries = collect_baseline(args.result_dir, args.datasets, args.methods)
        rows = []
        for (dataset, method), summary in sorted(summaries.items()):
            label = f"{dataset}_{method}"
            rows.append({"label": label, **summary})
        print(f"Result directory: {result_dir}")
        print_summary(rows)
        for dataset in sorted({dataset for dataset, _method in summaries}):
            items = [
                (method, summaries[(dataset, method)])
                for method in (args.methods or KNOWN_BASELINE_METHODS)
                if (dataset, method) in summaries
            ]
            if not items:
                continue
            output_path = output_dir / f"{dataset}_baseline_regret.png"
            plot_regret(items, output_path, f"{dataset} Baseline Regret")
            print(f"Saved {output_path}")
    else:
        summaries = collect_online(args.paths)
        rows = [{"label": label, **summary} for label, summary in summaries.items()]
        print_summary(rows)
        output_path = output_dir / f"{args.name}.png"
        plot_regret(list(summaries.items()), output_path, "Online Regret")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
