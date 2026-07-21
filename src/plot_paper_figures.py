import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_configs import RESULTS_DIR


FINAL_DATA_DIR = Path(RESULTS_DIR) / "final" / "data"
GENERATED_PLOT_DIR = Path(RESULTS_DIR) / "final" / "plots" / "generated"

RUN_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<method>(?:Loc|Fast)PRB|PRB)(?:_compareGraph)?_"
    r"alpha(?P<alpha>[^_]+)_(?P<param>eps[^_]+|powT[^_]+)_"
    r"T(?P<T>\d+)"
)

DATASET_DISPLAY = {
    "MovieLens": "MovieLens",
    "Collab": "ogbl-Collab",
    "PPA": "ogbl-PPA",
    "Vessel": "ogbl-Vessel",
}

COLORS = {
    "PRB": "#0072B2",
    "LocPRB": "#D55E00",
}


def normalize_dataset(name):
    return {
        "Amazon_fashion": "AmazonFashion",
        "Grqc": "GrQc",
    }.get(name, name)


def parse_run_dir(path):
    match = RUN_RE.search(Path(path).name)
    if not match:
        return None
    parsed = match.groupdict()
    parsed["dataset"] = normalize_dataset(parsed["dataset"])
    if parsed["method"] != "PRB":
        parsed["method"] = "LocPRB"
    parsed["path"] = Path(path)
    return parsed


def collect_final_runs(data_dir):
    runs = []
    for result_path in sorted(Path(data_dir).glob("*/*/final_results.npy")):
        parsed = parse_run_dir(result_path.parent.name)
        if parsed:
            parsed["result_path"] = result_path
            runs.append(parsed)
    return runs


def load_summary(result_path):
    raw = np.load(result_path)
    if raw.ndim != 3 or raw.shape[2] < 2:
        raise ValueError(f"Expected final_results.npy shape (runs, T, >=2), got {raw.shape}")
    regret = raw[:, :, 1]
    step_time = raw[:, :, 0]
    cumulative_time = np.cumsum(step_time, axis=1)
    return {
        "T": regret.shape[1],
        "regret_mean": regret.mean(axis=0),
        "regret_std": regret.std(axis=0),
        "final_regret_mean": float(regret[:, -1].mean()),
        "final_regret_std": float(regret[:, -1].std()),
        "final_time_mean": float(cumulative_time[:, -1].mean()),
        "final_time_std": float(cumulative_time[:, -1].std()),
    }


def choose_run(runs, dataset, method, alpha=None, param=None):
    matches = [
        run
        for run in runs
        if run["dataset"] == dataset
        and run["method"] == method
        and (alpha is None or run["alpha"] == alpha)
        and (param is None or run["param"] == param)
    ]
    if not matches:
        raise FileNotFoundError(f"No final run found for {dataset} {method} alpha={alpha} param={param}")
    return matches[0]


def plot_regret_curves(curves, output_base, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, summary, color, linestyle in curves:
        x = np.arange(1, summary["T"] + 1)
        mean = summary["regret_mean"]
        std = summary["regret_std"]
        ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=2.4)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Regret")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_main_regret(runs, output_dir):
    specs = [
        ("MovieLens", "MovieLens_Regret_vs_Rounds"),
        ("Collab", "ogbl-Collab_Regret_vs_Rounds"),
        ("PPA", "ogbl-PPA_Regret_vs_Rounds"),
        ("Vessel", "ogbl-Vessel_Regret_vs_Rounds"),
    ]
    written = []
    for dataset, filename in specs:
        prb = choose_run(runs, dataset, "PRB", alpha="0.85", param="powT50")
        loc = choose_run(runs, dataset, "LocPRB", alpha="0.85")
        curves = [
            ("PRB", load_summary(prb["result_path"]), COLORS["PRB"], (0, (3, 2))),
            ("LocPRB", load_summary(loc["result_path"]), COLORS["LocPRB"], "-"),
        ]
        output_base = output_dir / filename
        plot_regret_curves(curves, output_base, DATASET_DISPLAY[dataset])
        written.extend([output_base.with_suffix(".pdf"), output_base.with_suffix(".png")])
    return written


def plot_eps_ablation(runs, output_dir):
    specs = [
        ("10000/n", "0.0424"),
        ("1000/n", "0.00424"),
        ("100/n", "0.000424"),
        ("10/n", "4.24e-05"),
        ("1/n", "4.24e-06"),
        ("0.1/n", "4.24e-07"),
    ]
    palette = ["#7F3C8D", "#11A579", "#3969AC", "#F2B701", "#E73F74", "#80BA5A"]
    curves = []
    for (label, eps), color in zip(specs, palette):
        run = choose_run(runs, "Collab", "LocPRB", alpha="0.85", param=f"eps{eps}")
        curves.append((label, load_summary(run["result_path"]), color, "-"))
    output_base = output_dir / "ogbl-Collab_Eps_Ablation_Regret_vs_Rounds"
    plot_regret_curves(curves, output_base, "ogbl-Collab")
    return [output_base.with_suffix(".pdf"), output_base.with_suffix(".png")]


def plot_alpha_ablation(runs, output_dir):
    specs = ["0.6", "0.7", "0.8", "0.9"]
    palette = ["#3969AC", "#11A579", "#F2B701", "#E73F74"]
    written = []
    for method, label_prefix in [("PRB", "PRB"), ("LocPRB", "LocPRB")]:
        curves = []
        for alpha, color in zip(specs, palette):
            param = "powT50" if method == "PRB" else "eps4.24e-06"
            run = choose_run(runs, "Collab", method, alpha=alpha, param=param)
            curves.append((f"{label_prefix} alpha={alpha}", load_summary(run["result_path"]), color, "-"))
        output_base = output_dir / f"ogbl-Collab_{label_prefix}_Alpha_Ablation_Regret_vs_Rounds"
        plot_regret_curves(curves, output_base, "ogbl-Collab")
        written.extend([output_base.with_suffix(".pdf"), output_base.with_suffix(".png")])
    return written


def write_summary(written, output_dir):
    lines = [
        "# Generated Paper Figures",
        "",
        "Generated by `python -m src.plot_paper_figures`.",
        "",
        "Input data root: `results/final/data/`.",
        "",
        "## Files",
        "",
    ]
    for path in written:
        lines.append(f"- `{path.relative_to(Path(RESULTS_DIR).parent)}`")
    summary_path = output_dir / "GENERATED_FIGURES.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figures from results/final/data.")
    parser.add_argument("--data-dir", default=str(FINAL_DATA_DIR))
    parser.add_argument("--output-dir", default=str(GENERATED_PLOT_DIR))
    parser.add_argument("--skip-ablation", action="store_true", help="Only generate main regret figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_final_runs(args.data_dir)
    written = []
    written.extend(plot_main_regret(runs, output_dir))
    if not args.skip_ablation:
        written.extend(plot_eps_ablation(runs, output_dir))
        written.extend(plot_alpha_ablation(runs, output_dir))
    summary = write_summary(written, output_dir)
    print(f"Generated {len(written)} figure files in {output_dir}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
