import argparse
import csv
import re
from pathlib import Path

import numpy as np

from archive.maintenance_scripts.confirm_paper_results import TARGETS
from src.experiment_configs import REPO_ROOT, RESULTS_DIR


ONLINE_DIR = Path(RESULTS_DIR) / "online_link_prediction"
ONLINE_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<method>FastPRB|PRB)(?:_compareGraph)?_"
    r"alpha(?P<alpha>[^_]+)_(?P<param>eps[^_]+|powT[^_]+)_"
    r"T(?P<T>\d+)"
)


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalize_dataset(name):
    return {
        "Amazon_fashion": "Amazon",
        "Grqc": "GrQc",
        "ogbl-Collab": "Collab",
        "ogbl-PPA": "PPA",
        "ogbl-Vessel": "Vessel",
    }.get(name, name)


def parse_run_dir(path):
    match = ONLINE_RE.search(Path(path).name)
    if not match:
        return {}
    parsed = match.groupdict()
    parsed["dataset"] = normalize_dataset(parsed["dataset"])
    return parsed


def result_stats(path):
    raw = np.load(path)
    if raw.ndim == 3:
        if raw.shape[2] < 2:
            raise ValueError(f"unexpected final_results shape {raw.shape}")
        final_regrets = raw[:, -1, 1]
        total_times = np.sum(raw[:, :, 0], axis=1)
        return {
            "shape": "x".join(str(part) for part in raw.shape),
            "runs": raw.shape[0],
            "T": raw.shape[1],
            "regret_mean": float(np.mean(final_regrets)),
            "regret_std": float(np.std(final_regrets)),
            "time_mean": float(np.mean(total_times)),
            "time_std": float(np.std(total_times)),
        }
    if raw.ndim == 2:
        final_regret = raw[-1, 1]
        total_time = np.sum(raw[:, 0])
        return {
            "shape": "x".join(str(part) for part in raw.shape),
            "runs": 1,
            "T": raw.shape[0],
            "regret_mean": float(final_regret),
            "regret_std": 0.0,
            "time_mean": float(total_time),
            "time_std": 0.0,
        }
    raise ValueError(f"unexpected final_results shape {raw.shape}")


def collect_all_results():
    rows = []
    for result_path in sorted(ONLINE_DIR.glob("*/final_results.npy")):
        run_dir = result_path.parent
        parsed = parse_run_dir(run_dir)
        if not parsed:
            continue
        try:
            stats = result_stats(result_path)
        except Exception as exc:
            rows.append(
                {
                    "dataset": parsed.get("dataset", ""),
                    "method": parsed.get("method", ""),
                    "alpha": parsed.get("alpha", ""),
                    "param": parsed.get("param", ""),
                    "T": parsed.get("T", ""),
                    "path": rel(run_dir),
                    "error": str(exc),
                }
            )
            continue
        rows.append(
            {
                "dataset": parsed["dataset"],
                "method": parsed["method"],
                "alpha": parsed["alpha"],
                "param": parsed["param"],
                "declared_T": parsed["T"],
                "path": rel(run_dir),
                "error": "",
                **stats,
            }
        )
    return rows


def score(row, target):
    _, _, _, _, _, paper_regret, paper_regret_std, paper_time, paper_time_std = target
    return (
        abs(row["regret_mean"] - paper_regret)
        + abs(row["regret_std"] - paper_regret_std)
        + abs(row["time_mean"] - paper_time) / 10.0
        + abs(row["time_std"] - paper_time_std) / 10.0
    )


def param_matches(row, alpha_hint, param_hint):
    if alpha_hint and f"alpha{row.get('alpha', '')}" != alpha_hint:
        return False
    if param_hint == "powT50":
        return row.get("param") == "powT50"
    if param_hint.startswith("eps") and "/n" not in param_hint:
        return row.get("param") == param_hint
    return True


def select_candidates(rows, top_n):
    selected = []
    usable_rows = [row for row in rows if not row.get("error")]
    for target_id, target in enumerate(TARGETS, start=1):
        table, dataset, method, alpha_hint, param_hint, paper_regret, paper_regret_std, paper_time, paper_time_std = target
        compatible = [
            row
            for row in usable_rows
            if row["dataset"] == dataset
            and row["method"] == method
            and param_matches(row, alpha_hint, param_hint)
        ]
        fallback = False
        if not compatible:
            compatible = [
                row
                for row in usable_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            fallback = True
        ranked = sorted(compatible, key=lambda row: score(row, target))[:top_n]
        for rank, row in enumerate(ranked, start=1):
            match_score = score(row, target)
            selected.append(
                {
                    "target_id": target_id,
                    "rank": rank,
                    "table": table,
                    "dataset": dataset,
                    "method": method,
                    "alpha_hint": alpha_hint,
                    "param_hint": param_hint,
                    "paper_regret_mean": paper_regret,
                    "paper_regret_std": paper_regret_std,
                    "paper_time_mean": paper_time,
                    "paper_time_std": paper_time_std,
                    "computed_regret_mean": f"{row['regret_mean']:.4f}",
                    "computed_regret_std": f"{row['regret_std']:.4f}",
                    "computed_time_mean": f"{row['time_mean']:.4f}",
                    "computed_time_std": f"{row['time_std']:.4f}",
                    "score": f"{match_score:.4f}",
                    "status": "confirmed" if rank == 1 and match_score < 1.5 and not fallback else "candidate",
                    "fallback_used": str(fallback),
                    "runs": row["runs"],
                    "T": row["T"],
                    "alpha": row["alpha"],
                    "param": row["param"],
                    "shape": row["shape"],
                    "path": row["path"],
                }
            )
    return selected


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(selected_rows, all_rows, output_path, selected_csv, all_csv):
    confirmed = [row for row in selected_rows if row["status"] == "confirmed"]
    target_count = len(TARGETS)
    lines = [
        "# Paper Result Selection",
        "",
        "Generated by `python archive/maintenance_scripts/select_paper_results.py`.",
        "",
        "This is a dry-run selection report. No result files were moved.",
        "",
        f"- Scanned raw result directories: {sum(1 for row in all_rows if not row.get('error'))}",
        f"- Paper table targets: {target_count}",
        f"- Confirmed rank-1 matches: {len(confirmed)}",
        f"- Selected candidates CSV: `{rel(selected_csv)}`",
        f"- All scanned raw stats CSV: `{rel(all_csv)}`",
        "",
        "## Confirmed Rank-1 Matches",
        "",
        "| Table | Dataset | Method | Hint | Paper Regret | Computed Regret | Paper Time | Computed Time | Path |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in confirmed:
        lines.append(
            "| {table} | {dataset} | {method} | {alpha_hint}, {param_hint} | {paper_regret_mean}±{paper_regret_std} | {computed_regret_mean}±{computed_regret_std} | {paper_time_mean}±{paper_time_std} | {computed_time_mean}±{computed_time_std} | `{path}` |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Top Candidates Per Target",
            "",
            "The top candidates are sorted by `abs(regret_mean diff) + abs(regret_std diff) + abs(time_mean diff)/10 + abs(time_std diff)/10`.",
            "",
            "| Target | Rank | Status | Dataset | Method | Paper Regret | Computed Regret | Paper Time | Computed Time | Score | Path |",
            "|---:|---:|---|---|---|---|---|---|---|---:|---|",
        ]
    )
    for row in selected_rows:
        lines.append(
            "| {target_id} | {rank} | {status} | {dataset} | {method} | {paper_regret_mean}±{paper_regret_std} | {computed_regret_mean}±{computed_regret_std} | {paper_time_mean}±{paper_time_std} | {computed_time_mean}±{computed_time_std} | {score} | `{path}` |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The statistics follow the old plotting scripts' raw-data convention: regret from column 1 and per-step time from column 0.",
            "- A row is `confirmed` only when it is rank 1, parameter-compatible, and its score is below 1.5.",
            "- This report intentionally does not select AmazonFashion/Facebook/GrQc appendix regret data because the PDF text does not contain their final numeric table values.",
        ]
    )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Select paper-result raw directories by matching paper table values.")
    parser.add_argument("--top", type=int, default=5, help="Number of top candidates to keep per paper target.")
    parser.add_argument("--selected-csv", default=str(Path(REPO_ROOT) / "paper_result_selection.csv"))
    parser.add_argument("--all-csv", default=str(Path(REPO_ROOT) / "all_online_result_stats.csv"))
    parser.add_argument("--markdown", default=str(Path(REPO_ROOT) / "PAPER_RESULT_SELECTION.md"))
    return parser.parse_args()


def main():
    args = parse_args()
    all_rows = collect_all_results()
    selected_rows = select_candidates(all_rows, args.top)
    selected_csv = Path(args.selected_csv).resolve()
    all_csv = Path(args.all_csv).resolve()
    markdown = Path(args.markdown).resolve()
    write_csv(all_rows, all_csv)
    write_csv(selected_rows, selected_csv)
    write_markdown(selected_rows, all_rows, markdown, selected_csv, all_csv)
    confirmed = sum(1 for row in selected_rows if row["status"] == "confirmed")
    print(f"Scanned {sum(1 for row in all_rows if not row.get('error'))} raw result directories.")
    print(f"Wrote {len(selected_rows)} selected candidate rows ({confirmed} confirmed).")
    print(f"Wrote all raw stats to {all_csv}")
    print(f"Wrote selected candidates to {selected_csv}")
    print(f"Wrote summary to {markdown}")


if __name__ == "__main__":
    main()
