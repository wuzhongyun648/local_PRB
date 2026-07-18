import csv
import hashlib
import re
import zlib
from pathlib import Path

import numpy as np

from src.experiment_configs import REPO_ROOT, RESULTS_DIR


PDF_PATH = Path(REPO_ROOT) / "Local_Bandits_on_VLGs (3).pdf"
ONLINE_DIR = Path(RESULTS_DIR) / "online_link_prediction"


TARGETS = [
    # Table 1: main online link prediction results.
    ("table1", "MovieLens", "PRB", "alpha0.85", "powT50", 2054, 56.9, 7636, 103.4),
    ("table1", "MovieLens", "FastPRB", "alpha0.85", "eps8.33e-05", 2054, 56.9, 4247, 73.9),
    ("table1", "Collab", "PRB", "alpha0.85", "powT50", 2199, 114.5, 29952, 483.3),
    ("table1", "Collab", "FastPRB", "alpha0.85", "eps4.24e-06", 2194, 129.5, 1676, 115.5),
    ("table1", "PPA", "PRB", "alpha0.85", "powT50", 3276, 137.3, 56877, 6710.3),
    ("table1", "PPA", "FastPRB", "alpha0.85", "eps1.74e-06", 3264, 155.5, 3869, 374.3),
    ("table1", "Vessel", "PRB", "alpha0.85", "powT50", 4455, 28.5, 15111, 714.2),
    ("table1", "Vessel", "FastPRB", "alpha0.85", "eps2.86e-07", 4404, 37.4, 1276, 87.4),
    # Table 3: APPR tolerance ablation on ogbl-Collab.
    ("table3_eps", "Collab", "FastPRB", "alpha0.85", "eps10000/n", 4568, 34.9, 2781, 11.9),
    ("table3_eps", "Collab", "FastPRB", "alpha0.85", "eps1000/n", 2504, 152.8, 2956, 15.3),
    ("table3_eps", "Collab", "FastPRB", "alpha0.85", "eps100/n", 2271, 83.2, 2961, 22.9),
    ("table3_eps", "Collab", "FastPRB", "alpha0.85", "eps10/n", 2195, 105.3, 2983, 22.5),
    ("table3_eps", "Collab", "FastPRB", "alpha0.85", "eps1/n", 2195, 129.5, 1676, 115.5),
    ("table3_eps", "Collab", "FastPRB", "alpha0.85", "eps0.1/n", 2195, 109.4, 2832, 183.7),
    # Table 4: PageRank damping factor ablation on ogbl-Collab.
    ("table4_alpha", "Collab", "PRB", "alpha0.6", "powT50", 1826, 95.5, 30731, 221.5),
    ("table4_alpha", "Collab", "FastPRB", "alpha0.6", "eps4.24e-06", 1815, 89.2, 1145, 19.9),
    ("table4_alpha", "Collab", "PRB", "alpha0.7", "powT50", 1917, 99.0, 30751, 171.9),
    ("table4_alpha", "Collab", "FastPRB", "alpha0.7", "eps4.24e-06", 1920, 97.3, 1161, 18.2),
    ("table4_alpha", "Collab", "PRB", "alpha0.8", "powT50", 2068, 144.0, 30739, 222.7),
    ("table4_alpha", "Collab", "FastPRB", "alpha0.8", "eps4.24e-06", 2067, 148.7, 1580, 7.63),
    ("table4_alpha", "Collab", "PRB", "alpha0.9", "powT50", 2375, 119.7, 30278, 167.7),
    ("table4_alpha", "Collab", "FastPRB", "alpha0.9", "eps4.24e-06", 2355, 105.6, 1620, 12.8),
]


ONLINE_RE = re.compile(r"^(?P<dataset>.+?)_(?P<method>FastPRB|PRB).*")
EMBEDDED_FIGURE_RE = re.compile(rb"/PTEX.FileName \(([^)]*?(?:Regret_vs_Rounds|Accuracy_vs_Time)\.pdf)\)")


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


def result_stats(path):
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[2] < 2:
        raise ValueError(f"unexpected shape {arr.shape}")
    regret = arr[:, -1, 1]
    total_time = arr[:, :, 0].sum(axis=1)
    return {
        "runs": arr.shape[0],
        "T": arr.shape[1],
        "regret_mean": float(regret.mean()),
        "regret_std": float(regret.std()),
        "time_mean": float(total_time.mean()),
        "time_std": float(total_time.std()),
    }


def collect_online_results():
    rows = []
    for path in sorted(ONLINE_DIR.glob("*/final_results.npy")):
        match = ONLINE_RE.match(path.parent.name)
        if not match:
            continue
        parsed = match.groupdict()
        dataset = normalize_dataset(parsed["dataset"])
        method = parsed["method"]
        try:
            stats = result_stats(path)
        except Exception:
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "path": rel(path.parent),
                **stats,
            }
        )
    return rows


def score(row, target):
    _, _, _, _, _, r_mean, r_std, t_mean, t_std = target
    return (
        abs(row["regret_mean"] - r_mean)
        + abs(row["regret_std"] - r_std)
        + abs(row["time_mean"] - t_mean) / 10.0
        + abs(row["time_std"] - t_std) / 10.0
    )


def is_param_compatible(path, alpha_hint, param_hint):
    name = Path(path).name
    if alpha_hint and alpha_hint not in name:
        return False
    if param_hint.startswith("eps") and "/n" not in param_hint:
        return param_hint in name
    if param_hint == "powT50":
        return "powT50" in name
    return True


def best_matches(online_rows):
    rows = []
    for target in TARGETS:
        table, dataset, method, alpha_hint, param_hint, r_mean, r_std, t_mean, t_std = target
        candidates = [
            row
            for row in online_rows
            if row["dataset"] == dataset
            and row["method"] == method
            and is_param_compatible(row["path"], alpha_hint, param_hint)
        ]
        if not candidates:
            candidates = [
                row
                for row in online_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
        best = min(candidates, key=lambda row: score(row, target)) if candidates else None
        if best is None:
            rows.append(
                {
                    "table": table,
                    "dataset": dataset,
                    "method": method,
                    "paper_regret": f"{r_mean}±{r_std}",
                    "paper_time": f"{t_mean}±{t_std}",
                    "match_status": "missing",
                    "path": "",
                }
            )
            continue
        exact = score(best, target) < 1.5
        rows.append(
            {
                "table": table,
                "dataset": dataset,
                "method": method,
                "paper_regret": f"{r_mean}±{r_std}",
                "paper_time": f"{t_mean}±{t_std}",
                "computed_regret": f"{best['regret_mean']:.1f}±{best['regret_std']:.1f}",
                "computed_time": f"{best['time_mean']:.1f}±{best['time_std']:.1f}",
                "runs": best["runs"],
                "T": best["T"],
                "score": f"{score(best, target):.2f}",
                "match_status": "confirmed" if exact else "nearest",
                "path": best["path"],
            }
        )
    return rows


def embedded_figures():
    data = PDF_PATH.read_bytes()
    names = []
    for match in EMBEDDED_FIGURE_RE.finditer(data):
        names.append(match.group(1).decode("latin1").split("/")[-1])
    return sorted(set(names))


def embedded_figure_hashes():
    data = PDF_PATH.read_bytes()
    hashes = {}
    pattern = re.compile(
        rb"/PTEX\.FileName \([^)]*/([^/)]+\.pdf)\).*?stream\r?\n(.*?)\r?\nendstream",
        re.S,
    )
    for match in pattern.finditer(data):
        name = match.group(1).decode("latin1")
        stream = match.group(2).strip(b"\r\n")
        try:
            decoded = zlib.decompress(stream)
        except Exception:
            continue
        hashes[name] = hashlib.sha256(decoded).hexdigest()
    return hashes


def pdf_stream_hashes(path):
    data = Path(path).read_bytes()
    hashes = set()
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        stream = match.group(1).strip(b"\r\n")
        try:
            decoded = zlib.decompress(stream)
        except Exception:
            continue
        hashes.add(hashlib.sha256(decoded).hexdigest())
    return hashes


def local_figure_candidates(names):
    rows = []
    embedded_hashes = embedded_figure_hashes()
    roots = [
        Path(RESULTS_DIR) / "online_link_prediction" / "plots",
        Path(RESULTS_DIR) / "plots",
    ]
    for name in names:
        matches = []
        for root in roots:
            if root.exists():
                matches.extend(sorted(root.rglob(name)))
        exact_matches = []
        embedded_hash = embedded_hashes.get(name)
        if embedded_hash:
            for path in matches:
                if embedded_hash in pdf_stream_hashes(path):
                    exact_matches.append(path)
        rows.append(
            {
                "figure": name,
                "local_count": len(matches),
                "local_paths": "; ".join(rel(path) for path in matches[:8]),
                "exact_stream_count": len(exact_matches),
                "exact_stream_paths": "; ".join(rel(path) for path in exact_matches[:8]),
                "truncated": "yes" if len(matches) > 8 else "no",
            }
        )
    return rows


def write_csv(rows, path):
    fields = [
        "table",
        "dataset",
        "method",
        "paper_regret",
        "paper_time",
        "computed_regret",
        "computed_time",
        "runs",
        "T",
        "score",
        "match_status",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(match_rows, figure_rows, output_path, csv_path):
    confirmed = sum(1 for row in match_rows if row["match_status"] == "confirmed")
    nearest = sum(1 for row in match_rows if row["match_status"] == "nearest")
    lines = [
        "# Paper Results Confirmation",
        "",
        "Generated by `python archive/maintenance_scripts/confirm_paper_results.py`.",
        "",
        "No result files were moved while generating this report.",
        "",
        f"CSV detail file: `{rel(csv_path)}`",
        "",
        "## Summary",
        "",
        f"- Confirmed table rows: {confirmed}",
        f"- Nearest-only table rows: {nearest}",
        "- Regret is computed from `final_results.npy[:, -1, 1]`.",
        "- Total time is computed from `final_results.npy[:, :, 0].sum(axis=1)`.",
        "",
        "## Table Value Matches",
        "",
        "| Table | Dataset | Method | Paper Regret | Computed Regret | Paper Time | Computed Time | Runs | T | Status | Path |",
        "|---|---|---|---|---|---|---|---:|---:|---|---|",
    ]
    for row in match_rows:
        lines.append(
            "| {table} | {dataset} | {method} | {paper_regret} | {computed_regret} | {paper_time} | {computed_time} | {runs} | {T} | {match_status} | `{path}` |".format(
                **{key: row.get(key, "") for key in [
                    "table",
                    "dataset",
                    "method",
                    "paper_regret",
                    "computed_regret",
                    "paper_time",
                    "computed_time",
                    "runs",
                    "T",
                    "match_status",
                    "path",
                ]}
            )
        )
    lines.extend(
        [
            "",
            "## Embedded Paper Figures",
            "",
            "These figure names were extracted from the PDF's embedded `PTEX.FileName` objects.",
            "",
            "| Figure | Local PDF Count | Exact Stream Matches | Exact Paths | Candidate Paths | Truncated |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for row in figure_rows:
        lines.append(
            f"| {row['figure']} | {row['local_count']} | {row['exact_stream_count']} | `{row['exact_stream_paths']}` | `{row['local_paths']}` | {row['truncated']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The final paper table values match result directories from later January runs, not all paths hardcoded in `archive/old_plots/final_plot_online.py`.",
            "- `archive/old_plots/accuracy_plot.py` has one Collab PRB path with `alpha0.9`; Table 1 uses `alpha0.85`, while Table 4 intentionally includes `alpha0.9`.",
            "- The paper embeds 7 regret figures and 5 accuracy-time figures. No PPA/Vessel accuracy-time figure is embedded in the PDF.",
            "- Accuracy-time PDFs all have exact stream matches in `results/plots/custom_plots_acc/`.",
            "- AmazonFashion/Facebook/GrQc regret PDFs have exact stream matches in January 29 batch output directories.",
            "- MovieLens/ogbl-Collab/ogbl-PPA/ogbl-Vessel regret PDFs have local same-name candidates but no exact stream match; keep these candidates until the final figure source is selected or the figures are regenerated.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    online_rows = collect_online_results()
    match_rows = best_matches(online_rows)
    figure_rows = local_figure_candidates(embedded_figures())
    csv_path = Path(REPO_ROOT) / "paper_results_confirmation.csv"
    md_path = Path(REPO_ROOT) / "PAPER_RESULTS_CONFIRMATION.md"
    write_csv(match_rows, csv_path)
    write_markdown(match_rows, figure_rows, md_path, csv_path)
    confirmed = sum(1 for row in match_rows if row["match_status"] == "confirmed")
    print(f"Wrote {len(match_rows)} table checks ({confirmed} confirmed).")
    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote summary to {md_path}")


if __name__ == "__main__":
    main()
