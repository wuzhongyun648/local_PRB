"""Aggregate the six-dataset formal-horizon results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def main():
    root = Path(parse_args().result_dir)
    records = []
    for path in sorted(root.glob("*/seed_*_T*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "complete":
            continue
        records.append(
            {
                "dataset": item["dataset"],
                "seed": item["seed"],
                "rounds": item["rounds"],
                "final_regret": item["final_regret"],
                "duration_seconds": item["duration_seconds"],
            }
        )
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    grouped = defaultdict(list)
    for item in records:
        grouped[item["dataset"]].append(item)
    summaries = []
    for dataset, values in sorted(grouped.items()):
        regrets = np.asarray([item["final_regret"] for item in values], dtype=float)
        summaries.append(
            {
                "dataset": dataset,
                "runs": len(values),
                "rounds": values[0]["rounds"],
                "mean_final_regret": float(np.mean(regrets)),
                "std_final_regret": float(np.std(regrets)),
                "mean_duration_seconds": float(
                    np.mean([item["duration_seconds"] for item in values])
                ),
            }
        )
    payload = {"runs": records, "datasets": summaries}
    (report_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Six-Dataset g1-Current LocPRB",
        "",
        "| Dataset | T | Runs | Final regret | Mean duration (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['dataset']} | {item['rounds']} | {item['runs']} | "
            f"{item['mean_final_regret']:.2f} ± "
            f"{item['std_final_regret']:.2f} | "
            f"{item['mean_duration_seconds']:.2f} |"
        )
    (report_dir / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} runs across {len(summaries)} datasets")


if __name__ == "__main__":
    main()
