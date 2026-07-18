import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

from src.experiment_configs import RESULTS_DIR, REPO_ROOT


ONLINE_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<method>FastPRB|PRB)(?:_compareGraph)?_"
    r"alpha(?P<alpha>[^_]+)_(?P<param>eps[^_]+|powT[^_]+)_"
    r"T(?P<T>\d+)"
)


def dir_size(path):
    total = 0
    count = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            count += 1
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return count, total


def rel(path):
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def read_metadata(path):
    metadata_path = Path(path) / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def classify_online(name, parsed):
    notes = []
    if "compareGraph" in name:
        notes.append("compareGraph")
        return "tuning", notes
    if parsed and int(parsed["T"]) <= 1:
        notes.append("T<=1")
        return "smoke_test", notes
    if parsed and parsed["alpha"] != "0.85":
        return "ablation_alpha", notes
    if parsed and parsed["method"] == "FastPRB":
        param = parsed["param"]
        if param not in {"eps8.33e-05", "eps4.24e-06", "eps1.74e-06", "eps2.86e-07", "eps0.000125", "eps0.000191", "eps0.000248"}:
            return "ablation_eps", notes
    if "_h" in name or "_ks" in name or "_initH" in name or "_k5_" in name or "_k20_" in name:
        return "tuning", notes
    return "paper_candidate", notes


def scan_online(root):
    rows = []
    online_root = root / "online_link_prediction"
    if not online_root.exists():
        return rows
    for path in sorted(child for child in online_root.iterdir() if child.is_dir()):
        parsed_match = ONLINE_RE.search(path.name)
        parsed = parsed_match.groupdict() if parsed_match else {}
        category, notes = classify_online(path.name, parsed)
        file_count, total_bytes = dir_size(path)
        rows.append(
            {
                "path": rel(path),
                "entry_type": "online_run_dir",
                "category": category,
                "dataset": parsed.get("dataset", ""),
                "method": parsed.get("method", ""),
                "T": parsed.get("T", ""),
                "alpha": parsed.get("alpha", ""),
                "param": parsed.get("param", ""),
                "created": path.name.rsplit("_", 1)[-1],
                "has_final_results": str((path / "final_results.npy").exists()),
                "file_count": str(file_count),
                "total_bytes": str(total_bytes),
                "notes": ";".join(notes),
            }
        )
    return rows


def infer_baseline_file(path):
    stem = path.name.removesuffix(".npy")
    metric = ""
    for suffix in ("_regret", "_time"):
        if stem.endswith(suffix):
            metric = suffix[1:]
            stem = stem[: -len(suffix)]
            break
    methods = ["Neural_epsilon", "NeuralGreedy", "NeuralUCB", "NeuralTS", "NeuralNoExplore", "KernelUCB", "LinUCB", "EE-Net"]
    for method in methods:
        suffix = f"_{method}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], method, metric
    return "", "", metric


def scan_baselines(root):
    rows = []
    for subdir_name in ("baselines", "baselines_new"):
        subdir = root / subdir_name
        if not subdir.exists():
            continue
        for path in sorted(subdir.iterdir()):
            if path.is_dir():
                metadata = read_metadata(path)
                args = metadata.get("args", {})
                category = "baseline"
                notes = []
                if args.get("T") == 1:
                    category = "smoke_test"
                    notes.append("metadata T=1")
                file_count, total_bytes = dir_size(path)
                rows.append(
                    {
                        "path": rel(path),
                        "entry_type": f"{subdir_name}_run_dir",
                        "category": category,
                        "dataset": ",".join(metadata.get("datasets", [])),
                        "method": ",".join(metadata.get("methods", [])),
                        "T": str(args.get("T", "")),
                        "alpha": "",
                        "param": "",
                        "created": path.name,
                        "has_final_results": "False",
                        "file_count": str(file_count),
                        "total_bytes": str(total_bytes),
                        "notes": ";".join(notes),
                    }
                )
            elif path.suffix == ".npy":
                dataset, method, metric = infer_baseline_file(path)
                size = path.stat().st_size
                category = "baseline"
                notes = [metric] if metric else []
                if size <= 1024:
                    category = "incomplete_or_tiny"
                    notes.append("small npy")
                rows.append(
                    {
                        "path": rel(path),
                        "entry_type": f"{subdir_name}_file",
                        "category": category,
                        "dataset": dataset,
                        "method": method,
                        "T": "",
                        "alpha": "",
                        "param": "",
                        "created": "",
                        "has_final_results": "False",
                        "file_count": "1",
                        "total_bytes": str(size),
                        "notes": ";".join(notes),
                    }
                )
    return rows


def scan_plots(root):
    rows = []
    for plot_root in (root / "plots", root / "online_link_prediction" / "plots", root / "baselines_new" / "plots"):
        if not plot_root.exists():
            continue
        for path in sorted(plot_root.rglob("*")):
            if not path.is_file():
                continue
            rows.append(
                {
                    "path": rel(path),
                    "entry_type": "plot_file",
                    "category": "plot",
                    "dataset": "",
                    "method": "",
                    "T": "",
                    "alpha": "",
                    "param": "",
                    "created": "",
                    "has_final_results": "False",
                    "file_count": "1",
                    "total_bytes": str(path.stat().st_size),
                    "notes": path.suffix.lstrip("."),
                }
            )
    return rows


def write_csv(rows, output_path):
    fields = [
        "path",
        "entry_type",
        "category",
        "dataset",
        "method",
        "T",
        "alpha",
        "param",
        "created",
        "has_final_results",
        "file_count",
        "total_bytes",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, output_path, csv_path):
    category_counts = Counter(row["category"] for row in rows)
    type_counts = Counter(row["entry_type"] for row in rows)
    smoke_rows = [row for row in rows if row["category"] == "smoke_test"]
    paper_rows = [row for row in rows if row["category"] == "paper_candidate"]

    lines = [
        "# Results Manifest",
        "",
        "Generated by `python archive/maintenance_scripts/results_manifest.py`.",
        "",
        "No result files were moved while generating this manifest.",
        "",
        f"CSV detail file: `{rel(csv_path)}`",
        "",
        "## Summary By Category",
        "",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Summary By Entry Type", ""])
    for entry_type, count in sorted(type_counts.items()):
        lines.append(f"- `{entry_type}`: {count}")
    lines.extend(["", "## Smoke Test Candidates", ""])
    if smoke_rows:
        for row in smoke_rows[:80]:
            lines.append(f"- `{row['path']}` ({row['entry_type']}; {row['notes']})")
        if len(smoke_rows) > 80:
            lines.append(f"- ... {len(smoke_rows) - 80} more entries in CSV")
    else:
        lines.append("- None detected.")
    lines.extend(["", "## Paper Candidate Online Runs", ""])
    if paper_rows:
        for row in paper_rows[:80]:
            label = " ".join(part for part in [row["dataset"], row["method"], row["param"], f"T{row['T']}"] if part)
            lines.append(f"- `{row['path']}` ({label})")
        if len(paper_rows) > 80:
            lines.append(f"- ... {len(paper_rows) - 80} more entries in CSV")
    else:
        lines.append("- None detected.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a manifest for result files without moving them.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--csv", default=str(Path(REPO_ROOT) / "results_manifest.csv"))
    parser.add_argument("--markdown", default=str(Path(REPO_ROOT) / "RESULTS_MANIFEST.md"))
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_dir).resolve()
    rows = []
    rows.extend(scan_online(root))
    rows.extend(scan_baselines(root))
    rows.extend(scan_plots(root))
    rows.sort(key=lambda row: row["path"])

    csv_path = Path(args.csv).resolve()
    md_path = Path(args.markdown).resolve()
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, csv_path)
    print(f"Wrote {len(rows)} entries to {csv_path}")
    print(f"Wrote summary to {md_path}")


if __name__ == "__main__":
    main()
