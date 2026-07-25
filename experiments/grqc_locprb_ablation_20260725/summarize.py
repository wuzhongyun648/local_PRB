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
T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}

# Positive delta means the candidate has higher (worse) regret.
PAIRED_COMPARISONS = [
    ("g1_current_loc", "g1_original_loc"),
    ("g1_current_power", "g1_current_loc"),
    ("s1_selected_edge_update", "s0_legacy"),
    ("s2_disjoint_undirected_split", "s0_legacy"),
    ("s3_shared_serving", "s0_legacy"),
    ("s4_no_positive_replacement", "s0_legacy"),
    ("s5_independent_eval", "s0_legacy"),
    ("sf_all_fixed", "s0_legacy"),
    ("r1_locprb", "r0_eenet_direct"),
    ("r2_power", "r0_eenet_direct"),
    ("r2_power", "r1_locprb"),
    ("p1_l1", "p0_raw"),
    ("p2_clip_l1", "p0_raw"),
    ("p3_softmax", "p0_raw"),
    ("p4_exploit_only", "p0_raw"),
    ("p5_exploit_explore", "p0_raw"),
]

EQUIVALENT_REPRESENTATIVE = {
    "s0_legacy": "g1_current_loc",
    "r1_locprb": "sf_all_fixed",
    "p0_raw": "sf_all_fixed",
    "p5_exploit_explore": "sf_all_fixed",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--rounds", type=int, default=1000)
    return parser.parse_args()


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None, None
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    critical = T_CRITICAL_975.get(len(values) - 1, 1.96)
    half_width = critical * standard_error
    return mean, mean - half_width, mean + half_width


def main():
    args = parse_args()
    root = Path(args.result_dir)
    records = []
    for path in sorted(root.glob(f"group*/*/seed_*_T{args.rounds}/summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete":
            continue
        curve_path = path.with_name("curves.npz")
        with np.load(curve_path) as curves:
            regret_curve = curves["regret"]
            curve_fields = {
                f"regret_at_{round_number}": float(
                    regret_curve[round_number - 1]
                )
                for round_number in (100, 500, 1000, 10000)
                if round_number <= len(regret_curve)
            }
            integrate = getattr(np, "trapezoid", np.trapz)
            curve_fields["cumulative_regret_auc"] = float(
                integrate(regret_curve, dx=1.0)
            )
        record = {
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
        record.update(curve_fields)
        records.append(record)

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
        regret_mean, regret_ci_low, regret_ci_high = mean_ci95(regrets)
        summaries.append(
            {
                "group": group,
                "variant": variant,
                "runs": len(values),
                "mean_final_regret": regret_mean,
                "std_final_regret": float(np.std(regrets)),
                "ci95_final_regret_low": regret_ci_low,
                "ci95_final_regret_high": regret_ci_high,
                "mean_cumulative_regret_auc": float(
                    np.mean([v["cumulative_regret_auc"] for v in values])
                ),
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
    by_variant_seed = {
        (record["variant"], record["seed"]): record for record in records
    }
    available_variants = {record["variant"] for record in records}

    def representative(name):
        if name in available_variants:
            return name
        alias = EQUIVALENT_REPRESENTATIVE.get(name)
        return alias if alias in available_variants else None

    comparisons = []
    for candidate_label, reference_label in PAIRED_COMPARISONS:
        candidate = representative(candidate_label)
        reference = representative(reference_label)
        if candidate is None or reference is None or candidate == reference:
            continue
        common_seeds = sorted(
            {
                seed
                for variant, seed in by_variant_seed
                if variant == candidate
            }
            & {
                seed
                for variant, seed in by_variant_seed
                if variant == reference
            }
        )
        if not common_seeds:
            continue
        deltas = [
            by_variant_seed[(candidate, seed)]["final_regret"]
            - by_variant_seed[(reference, seed)]["final_regret"]
            for seed in common_seeds
        ]
        mean_delta, ci_low, ci_high = mean_ci95(deltas)
        comparisons.append(
            {
                "candidate": candidate_label,
                "reference": reference_label,
                "candidate_representative": candidate,
                "reference_representative": reference,
                "seeds": common_seeds,
                "mean_paired_delta": mean_delta,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "candidate_wins": sum(delta < 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "reference_wins": sum(delta > 0 for delta in deltas),
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
                "paired_comparisons": comparisons,
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
    lines.extend(
        [
            "",
            "## Paired final-regret deltas",
            "",
            "Delta is candidate minus reference; negative is better.",
            "",
            "| Candidate | Reference | Seeds | Mean delta (95% CI) | W/T/L |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in comparisons:
        ci_text = (
            "n/a"
            if item["ci95_low"] is None
            else f"[{item['ci95_low']:.2f}, {item['ci95_high']:.2f}]"
        )
        lines.append(
            f"| {item['candidate']} | {item['reference']} | "
            f"{len(item['seeds'])} | {item['mean_paired_delta']:.2f} "
            f"{ci_text} | {item['candidate_wins']}/"
            f"{item['ties']}/{item['reference_wins']} |"
        )
    (report_dir / f"screen_T{args.rounds}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} run records and {len(summaries)} variants")


if __name__ == "__main__":
    main()
