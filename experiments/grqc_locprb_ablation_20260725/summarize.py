"""Summarize GrQc ablation results into JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HISTORICAL_PRB_MEAN = 4386.9
HISTORICAL_PRB_STD = 325.34180487604107


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--rounds", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.result_dir)
    records = []
    for path in sorted(root.glob(f"group*/*/seed_*_T{args.rounds}/summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete":
            continue
        records.append(
            {
                "group": value["variant"]["group"],
                "variant": value["variant"]["name"],
                "seed": value["seed"],
                "rounds": value["rounds"],
                "final_regret": value["final_regret"],
                "direct_shadow": value["shadow_final_regret"]["direct"],
                "loc_shadow": value["shadow_final_regret"]["loc"],
                "power_shadow": value["shadow_final_regret"]["power"],
                "net_reranking_gain": value["net_reranking_gain"],
                "loc_power_disagreement": value["shadow_counts"][
                    "loc_power_disagreement"
                ],
                "inserted_edges": value["graph"]["inserted_edges"],
                "positive_visible_before_round": value["graph"][
                    "positive_visible_before_round"
                ],
                "endpoint_mismatches": value["graph"][
                    "update_endpoint_mismatches"
                ],
                "online_eval_identical": value["events"][
                    "online_evaluation_identical_events"
                ],
                "duration_seconds": value["duration_seconds"],
                "event_digest": value["events"]["online"]["event_digest"],
            }
        )

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    columns = list(records[0]) if records else []
    with (report_dir / f"screen_T{args.rounds}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    grouped = defaultdict(list)
    for record in records:
        grouped[(record["group"], record["variant"])].append(record)
    summaries = []
    for (group, variant), values in sorted(grouped.items()):
        regrets = np.asarray([v["final_regret"] for v in values], dtype=float)
        summaries.append(
            {
                "group": group,
                "variant": variant,
                "runs": len(values),
                "mean_final_regret": float(np.mean(regrets)),
                "std_final_regret": float(np.std(regrets)),
                "mean_reranking_gain": float(
                    np.mean([v["net_reranking_gain"] for v in values])
                ),
                "mean_loc_power_disagreement": float(
                    np.mean([v["loc_power_disagreement"] for v in values])
                ),
                "mean_inserted_edges": float(
                    np.mean([v["inserted_edges"] for v in values])
                ),
                "mean_endpoint_mismatches": float(
                    np.mean([v["endpoint_mismatches"] for v in values])
                ),
                "mean_online_eval_identical": float(
                    np.mean([v["online_eval_identical"] for v in values])
                ),
                "mean_duration_seconds": float(
                    np.mean([v["duration_seconds"] for v in values])
                ),
            }
        )
    (report_dir / f"screen_T{args.rounds}.json").write_text(
        json.dumps(
            {
                "historical_original_prb": {
                    "rounds": 10000,
                    "mean": HISTORICAL_PRB_MEAN,
                    "std": HISTORICAL_PRB_STD,
                    "paired_with_screen": False,
                },
                "variants": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        f"# GrQc Ablation Summary (T={args.rounds})",
        "",
        "Historical original PRB: "
        f"`{HISTORICAL_PRB_MEAN:.1f} +/- {HISTORICAL_PRB_STD:.1f}` at "
        "T=10000; audit anchor only, not paired with this screen.",
        "",
        "| Group | Variant | Runs | Final regret | Reranking gain | "
        "Loc/Power disagree | Inserted | Endpoint mismatch | Eval overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['group']} | {item['variant']} | {item['runs']} | "
            f"{item['mean_final_regret']:.2f} +/- "
            f"{item['std_final_regret']:.2f} | "
            f"{item['mean_reranking_gain']:.2f} | "
            f"{item['mean_loc_power_disagreement']:.2f} | "
            f"{item['mean_inserted_edges']:.2f} | "
            f"{item['mean_endpoint_mismatches']:.2f} | "
            f"{item['mean_online_eval_identical']:.2f} |"
        )
    (report_dir / f"screen_T{args.rounds}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} run records and {len(summaries)} variants")


if __name__ == "__main__":
    main()

