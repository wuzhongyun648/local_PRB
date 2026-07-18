"""Plot the full LocPRB comparison against the archived PRB results.

The script discovers the newest complete 10-run directory for each new method
from its saved config, rather than relying on timestamped directory names.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_configs import RESULTS_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_SPECS = {
    "MovieLens": {"display": "MovieLens", "T": 10000, "eps": 8.33e-05},
    "Amazon_fashion": {"display": "AmazonFashion", "T": 5000, "eps": 1.25e-04},
    "Collab": {"display": "ogbl-Collab", "T": 5000, "eps": 4.24e-06},
    "Vessel": {"display": "ogbl-Vessel", "T": 5000, "eps": 2.86e-07},
}

NEW_METHOD_SPECS = {
    "LocPRB": {"method": "LocPRB"},
    "Dyn-LocPRB": {"method": "dyn_locPRB"},
}

OLD_PRB_REGRET_DIRS = {
    "MovieLens": REPO_ROOT
    / "results/final/data/table1/"
    "MovieLens_PRB_alpha0.85_powT50_T10000_lr10.0073_lr20.0004_202601290025",
    "Amazon_fashion": REPO_ROOT
    / "archive/results_nonfinal_20260709_070314/online_link_prediction/"
    "Amazon_fashion_PRB_alpha0.85_powT50_T5000_lr10.1_lr20.01_202601261332",
    "Collab": REPO_ROOT
    / "results/final/data/table1/"
    "Collab_PRB_alpha0.85_powT50_T5000_lr10.01_lr20.004_202601290026",
    "Vessel": REPO_ROOT
    / "results/final/data/table1/"
    "Vessel_PRB_alpha0.85_powT50_T5000_lr10.01_lr20.004_202601271451",
}

OLD_PRB_ACCURACY_DIRS = {
    **OLD_PRB_REGRET_DIRS,
    "Amazon_fashion": REPO_ROOT
    / "archive/results_nonfinal_20260709_070314/online_link_prediction/"
    "Amazon_fashion_PRB_alpha0.85_powT50_T5000_lr10.1_lr20.01_202601291452",
}

METHOD_ORDER = ["PRB", *NEW_METHOD_SPECS]
STYLES = {
    "PRB": {"color": "#0072B2", "linestyle": "--"},
    "LocPRB": {"color": "#009E73", "linestyle": "-"},
    "Dyn-LocPRB": {"color": "#E69F00", "linestyle": "-"},
}


def read_config(path):
    config = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            config[key.strip()] = value.strip()
    return config


def config_int(config, key):
    try:
        return int(config[key])
    except (KeyError, TypeError, ValueError):
        return None


def config_float(config, key):
    try:
        return float(config[key])
    except (KeyError, TypeError, ValueError):
        return None


def is_expected_config(config, dataset, method_spec):
    expected = DATASET_SPECS[dataset]
    alpha = config_float(config, "alpha")
    eps = config_float(config, "appr_eps")
    return (
        config.get("graph_name") == dataset
        and config.get("method") == method_spec["method"]
        and config.get("variant", "baseline") == "baseline"
        and config_int(config, "T") == expected["T"]
        and config_int(config, "runs") == 10
        and alpha is not None
        and np.isclose(alpha, 0.85, rtol=0.0, atol=1e-12)
        and eps is not None
        and np.isclose(eps, expected["eps"], rtol=1e-12, atol=0.0)
    )


def validate_final_results(run_dir, expected_t):
    result_path = run_dir / "final_results.npy"
    if not result_path.is_file():
        return False
    try:
        data = np.load(result_path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return data.ndim == 3 and data.shape[0] == 10 and data.shape[1] == expected_t and data.shape[2] >= 2


def discover_new_run(result_root, dataset, method_spec):
    expected_t = DATASET_SPECS[dataset]["T"]
    candidates = []
    for config_path in Path(result_root).rglob("config.txt"):
        try:
            config = read_config(config_path)
        except OSError:
            continue
        run_dir = config_path.parent
        if is_expected_config(config, dataset, method_spec) and validate_final_results(run_dir, expected_t):
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No complete 10-run result for {dataset}: "
            f"method={method_spec['method']} under {result_root}"
        )
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    if len(candidates) > 1:
        warnings.warn(
            f"Found {len(candidates)} matching runs for {dataset} "
            f"{method_spec['method']}; using newest: {candidates[-1]}",
            stacklevel=2,
        )
    return candidates[-1]


def load_regret(run_dir, expected_t):
    data = np.load(Path(run_dir) / "final_results.npy")
    if data.ndim != 3 or data.shape[0] != 10 or data.shape[1] != expected_t or data.shape[2] < 2:
        raise ValueError(f"Incomplete final_results.npy in {run_dir}: shape={data.shape}")
    regret = np.asarray(data[:, :, 1], dtype=float)
    if not np.all(np.isfinite(regret)):
        raise ValueError(f"Non-finite regret values in {run_dir}")
    return regret


def timeacc_sort_key(path):
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return path.name


def load_time_accuracy(run_dir):
    paths = sorted(Path(run_dir).glob("worker_*_TimeAcc.npy"), key=timeacc_sort_key)
    if len(paths) != 10:
        warnings.warn(
            f"Expected 10 TimeAcc files in {run_dir}, found {len(paths)}; skipping this accuracy curve.",
            stacklevel=2,
        )
        return None

    arrays = []
    expected_shape = None
    for path in paths:
        try:
            data = np.asarray(np.load(path), dtype=float)
        except (OSError, ValueError) as exc:
            warnings.warn(f"Cannot load {path}: {exc}; skipping this accuracy curve.", stacklevel=2)
            return None
        if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] == 0 or not np.all(np.isfinite(data[:, :2])):
            warnings.warn(f"Invalid TimeAcc data in {path}: shape={data.shape}; skipping this accuracy curve.", stacklevel=2)
            return None
        if expected_shape is None:
            expected_shape = data.shape
        elif data.shape != expected_shape:
            warnings.warn(
                f"Mismatched TimeAcc shapes in {run_dir}: {expected_shape} vs {data.shape}; "
                "skipping this accuracy curve.",
                stacklevel=2,
            )
            return None
        arrays.append(data[:, :2])
    return np.stack(arrays, axis=0)


def configure_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=20, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16, width=1.5, length=6)
    ax.grid(False)
    ax.legend(frameon=True, fontsize=14, loc="best")


def save_figure(fig, output_base):
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_regret(dataset, run_dirs, output_dir):
    spec = DATASET_SPECS[dataset]
    fig, ax = plt.subplots(figsize=(12, 9))
    for label in METHOD_ORDER:
        regret = load_regret(run_dirs[label], spec["T"])
        mean = regret.mean(axis=0)
        std = regret.std(axis=0)
        rounds = np.arange(1, spec["T"] + 1)
        style = STYLES[label]
        ax.plot(rounds, mean, label=label, linewidth=3.2, **style)
        ax.fill_between(rounds, mean - std, mean + std, color=style["color"], alpha=0.10, linewidth=0)
    configure_axes(ax, spec["display"], "Rounds", "Cumulative Regret")
    save_figure(fig, output_dir / f"{spec['display']}_Regret_vs_Rounds")


def plot_accuracy(dataset, run_dirs, output_dir):
    spec = DATASET_SPECS[dataset]
    fig, ax = plt.subplots(figsize=(12, 9))
    plotted = 0
    for label in METHOD_ORDER:
        time_accuracy = load_time_accuracy(run_dirs[label])
        if time_accuracy is None:
            continue
        mean_time = time_accuracy[:, :, 0].mean(axis=0)
        accuracy = time_accuracy[:, :, 1] * 100.0
        mean_accuracy = accuracy.mean(axis=0)
        std_accuracy = accuracy.std(axis=0)
        style = STYLES[label]
        ax.plot(mean_time, mean_accuracy, label=label, linewidth=3.2, **style)
        ax.fill_between(
            mean_time,
            mean_accuracy - std_accuracy,
            mean_accuracy + std_accuracy,
            color=style["color"],
            alpha=0.10,
            linewidth=0,
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        warnings.warn(f"No complete Accuracy-Time data for {dataset}; no accuracy figure written.", stacklevel=2)
        return
    configure_axes(ax, spec["display"], "Time (s)", "Accuracy (%)")
    save_figure(fig, output_dir / f"{spec['display']}_Accuracy_vs_Time")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot archived PRB and new LocPRB variants as Regret-Rounds and Accuracy-Time curves."
    )
    parser.add_argument(
        "--result-root",
        default=str(Path(RESULTS_DIR) / "online_link_prediction"),
        help="Root recursively searched for the two new-method run directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(RESULTS_DIR) / "plots" / "locprb_comparison"),
        help="Directory for PNG and PDF figures.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_SPECS),
        default=list(DATASET_SPECS),
        help="Datasets to plot (default: all four).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {result_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        old_regret_dir = OLD_PRB_REGRET_DIRS[dataset]
        old_accuracy_dir = OLD_PRB_ACCURACY_DIRS[dataset]
        if not validate_final_results(old_regret_dir, DATASET_SPECS[dataset]["T"]):
            raise FileNotFoundError(f"Archived PRB result is missing or incomplete: {old_regret_dir}")
        if not validate_final_results(old_accuracy_dir, DATASET_SPECS[dataset]["T"]):
            raise FileNotFoundError(f"Archived PRB accuracy result is missing or incomplete: {old_accuracy_dir}")
        regret_dirs = {"PRB": old_regret_dir}
        accuracy_dirs = {"PRB": old_accuracy_dir}
        for label, method_spec in NEW_METHOD_SPECS.items():
            run_dir = discover_new_run(result_root, dataset, method_spec)
            regret_dirs[label] = run_dir
            accuracy_dirs[label] = run_dir

        print(f"[{dataset}]")
        for label in METHOD_ORDER:
            if label == "PRB" and old_regret_dir != old_accuracy_dir:
                print(f"  PRB regret: {old_regret_dir}")
                print(f"  PRB accuracy: {old_accuracy_dir}")
            else:
                print(f"  {label}: {regret_dirs[label]}")
        plot_regret(dataset, regret_dirs, output_dir)
        plot_accuracy(dataset, accuracy_dirs, output_dir)

    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
