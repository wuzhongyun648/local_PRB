import argparse
import ast
import csv
import os
import re
from pathlib import Path

import numpy as np

from src.experiment_configs import REPO_ROOT, RESULTS_DIR


ONLINE_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<method>FastPRB|PRB)(?:_compareGraph)?_"
    r"alpha(?P<alpha>[^_]+)_(?P<param>eps[^_]+|powT[^_]+)_"
    r"T(?P<T>\d+)"
)

SCRIPT_PATHS = [
    Path(REPO_ROOT) / "archive" / "old_plots" / "final_plot_online.py",
    Path(REPO_ROOT) / "archive" / "old_plots" / "accuracy_plot.py",
]


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalize_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(REPO_ROOT) / path


def extract_custom_paths(script_path):
    text = script_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(script_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CUSTOM_PATHS":
                    return ast.literal_eval(node.value)
    return {}


def parse_online_dir(path):
    result_path = Path(path)
    run_dir = result_path.parent if result_path.name == "final_results.npy" else result_path
    match = ONLINE_RE.search(run_dir.name)
    return match.groupdict() if match else {}


def array_shape(path):
    if not path.exists():
        return "", "", ""
    try:
        arr = np.load(path, mmap_mode="r")
        shape = "x".join(str(part) for part in arr.shape)
        runs = str(arr.shape[0]) if arr.ndim >= 1 else ""
        T = str(arr.shape[1]) if arr.ndim >= 2 else ""
        return shape, runs, T
    except Exception as exc:
        return f"error:{exc}", "", ""


def collect_data_candidates():
    rows = []
    seen = set()
    for script_path in SCRIPT_PATHS:
        custom_paths = extract_custom_paths(script_path)
        for dataset, methods in custom_paths.items():
            for method, path_text in methods.items():
                path = normalize_path(path_text)
                if not path.exists() and "/results/PPA_PRB_" in str(path):
                    fixed = Path(str(path).replace("/results/PPA_PRB_", "/results/online_link_prediction/PPA_PRB_"))
                    if fixed.exists():
                        path = fixed
                key = (script_path.name, dataset, method, rel(path))
                if key in seen:
                    continue
                seen.add(key)
                parsed = parse_online_dir(path)
                shape, runs, shape_T = array_shape(path)
                declared_T = parsed.get("T", "")
                notes = []
                if not path.exists():
                    notes.append("missing")
                if declared_T and shape_T and declared_T != shape_T:
                    notes.append(f"name_T={declared_T};shape_T={shape_T}")
                rows.append(
                    {
                        "source": script_path.name,
                        "candidate_type": "strong_data_candidate",
                        "dataset": dataset,
                        "method": method,
                        "path": rel(path),
                        "exists": str(path.exists()),
                        "shape": shape,
                        "runs": runs,
                        "T": shape_T or declared_T,
                        "alpha": parsed.get("alpha", ""),
                        "param": parsed.get("param", ""),
                        "notes": ";".join(notes),
                    }
                )
    return rows


def infer_plot_dataset(path):
    name = path.name
    stem = name.removesuffix(path.suffix)
    dataset = stem.split("_Regret")[0].split("_Accuracy")[0]
    return dataset


def collect_plot_candidates():
    rows = []
    plot_roots = [
        Path(RESULTS_DIR) / "online_link_prediction" / "plots",
        Path(RESULTS_DIR) / "plots",
    ]
    for root in plot_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.pdf")):
            metric = "accuracy" if "Accuracy" in path.name else "regret" if "Regret" in path.name else "unknown"
            axis = "time" if "_vs_Time" in path.name else "rounds" if "_vs_Rounds" in path.name else ""
            rows.append(
                {
                    "source": rel(root),
                    "candidate_type": "plot_candidate",
                    "dataset": infer_plot_dataset(path),
                    "method": "",
                    "path": rel(path),
                    "exists": "True",
                    "shape": "",
                    "runs": "",
                    "T": "",
                    "alpha": "",
                    "param": "",
                    "notes": f"{metric};{axis};{path.stat().st_size} bytes",
                }
            )
    return rows


def write_csv(rows, path):
    fields = ["source", "candidate_type", "dataset", "method", "path", "exists", "shape", "runs", "T", "alpha", "param", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(data_rows, plot_rows, output_path, csv_path):
    lines = [
        "# Final Results Candidates",
        "",
        "Generated by `python archive/maintenance_scripts/final_candidates.py`.",
        "",
        "No result files were moved while generating this candidate list.",
        "",
        f"CSV detail file: `{rel(csv_path)}`",
        "",
        "## Strong Data Candidates",
        "",
        "These are `final_results.npy` paths explicitly referenced by archived plotting scripts.",
        "",
        "| Source | Dataset | Method | Alpha | Param | Runs | T | Exists | Path | Notes |",
        "|---|---|---|---|---|---:|---:|---|---|---|",
    ]
    for row in data_rows:
        lines.append(
            "| {source} | {dataset} | {method} | {alpha} | {param} | {runs} | {T} | {exists} | `{path}` | {notes} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## PDF Plot Candidates",
            "",
            "PDFs are scanned from both `results/online_link_prediction/plots/` and `results/plots/`.",
            "",
            "| Source Root | Dataset/Figure | Path | Notes |",
            "|---|---|---|---|",
        ]
    )
    for row in plot_rows:
        lines.append("| {source} | {dataset} | `{path}` | {notes} |".format(**row))
    lines.extend(
        [
            "",
            "## Confirmation Needed",
            "",
            "- Choose one final data directory per paper dataset/method/metric when duplicates exist.",
            "- Confirm whether PDF figures should come from `results/online_link_prediction/plots/`, `results/plots/`, or both.",
            "- Confirm whether PNG companions should also be retained in `results/final/plots/`.",
            "- `archive/old_plots/accuracy_plot.py` references newer January 29 runs for accuracy plots; `final_plot_online.py` references older runs for regret plots.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate final result candidates without moving files.")
    parser.add_argument("--csv", default=str(Path(REPO_ROOT) / "final_candidates.csv"))
    parser.add_argument("--markdown", default=str(Path(REPO_ROOT) / "FINAL_CANDIDATES.md"))
    return parser.parse_args()


def main():
    args = parse_args()
    data_rows = collect_data_candidates()
    plot_rows = collect_plot_candidates()
    rows = data_rows + plot_rows
    csv_path = Path(args.csv).resolve()
    md_path = Path(args.markdown).resolve()
    write_csv(rows, csv_path)
    write_markdown(data_rows, plot_rows, md_path, csv_path)
    print(f"Wrote {len(data_rows)} strong data candidates and {len(plot_rows)} PDF plot candidates.")
    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote summary to {md_path}")


if __name__ == "__main__":
    main()
