import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path

from archive.maintenance_scripts.confirm_paper_results import embedded_figures, local_figure_candidates
from src.experiment_configs import REPO_ROOT, RESULTS_DIR


RESULTS_ROOT = Path(RESULTS_DIR)
FINAL_ROOT = RESULTS_ROOT / "final"
FINAL_DATA = FINAL_ROOT / "data"
FINAL_PLOTS = FINAL_ROOT / "plots"
FINAL_METADATA = FINAL_ROOT / "metadata"
ARCHIVE_ROOT = Path(REPO_ROOT) / "archive"


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def unique_dest(parent, name):
    dest = parent / name
    if not dest.exists():
        return dest
    index = 2
    while True:
        candidate = parent / f"{name}__dup{index}"
        if not candidate.exists():
            return candidate
        index += 1


def move_path(src, dest):
    src = Path(src)
    dest = Path(dest)
    if not src.exists():
        return False, f"missing: {rel(src)}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest = unique_dest(dest.parent, dest.name)
    shutil.move(str(src), str(dest))
    return True, f"{rel(src)} -> {rel(dest)}"


def copy_path(src, dest):
    src = Path(src)
    dest = Path(dest)
    if not src.exists():
        return False, f"missing: {rel(src)}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest = unique_dest(dest.parent, dest.name)
    shutil.copy2(src, dest)
    return True, f"{rel(src)} -> {rel(dest)}"


def confirmed_data_dirs():
    selection_csv = Path(REPO_ROOT) / "paper_result_selection.csv"
    dirs = []
    seen = set()
    with selection_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] != "confirmed":
                continue
            src = Path(REPO_ROOT) / row["path"]
            if src in seen:
                continue
            seen.add(src)
            dirs.append((src, row))
    return dirs


def final_plot_paths():
    rows = local_figure_candidates(embedded_figures())
    exact_paths = []
    candidate_paths = []
    for row in rows:
        exact = [Path(REPO_ROOT) / path for path in row["exact_stream_paths"].split("; ") if path]
        candidates = []
        if not exact:
            figure_name = row["figure"]
            for root in [
                Path(RESULTS_DIR) / "online_link_prediction" / "plots",
                Path(RESULTS_DIR) / "plots",
            ]:
                if root.exists():
                    candidates.extend(sorted(root.rglob(figure_name)))
        if exact:
            exact_paths.extend(exact)
        else:
            candidate_paths.extend(candidates)
    exact_set = set(exact_paths)
    candidates = [path for path in candidate_paths if path not in exact_set]
    return sorted(exact_set), sorted(set(candidates))


def copy_metadata():
    logs = []
    for name in [
        "PAPER_RESULT_SELECTION.md",
        "paper_result_selection.csv",
        "all_online_result_stats.csv",
        "PAPER_RESULTS_CONFIRMATION.md",
        "paper_results_confirmation.csv",
        "FINAL_CANDIDATES.md",
        "final_candidates.csv",
    ]:
        ok, message = copy_path(Path(REPO_ROOT) / name, FINAL_METADATA / name)
        logs.append(("copied_metadata" if ok else "missing_metadata", message))
    return logs


def write_manifest(logs, archive_dest):
    manifest = FINAL_ROOT / "FINAL_ARCHIVE_MANIFEST.md"
    lines = [
        "# Final Archive Manifest",
        "",
        f"Generated at UTC `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`.",
        "",
        f"Archived non-final results to `{rel(archive_dest)}`.",
        "",
        "## Operations",
        "",
        "| Action | Path |",
        "|---|---|",
    ]
    for action, message in logs:
        lines.append(f"| {action} | `{message}` |")
    manifest.write_text("\n".join(lines), encoding="utf-8")


def archive_remaining_results(timestamp):
    archive_dest = ARCHIVE_ROOT / f"results_nonfinal_{timestamp}"
    archive_dest.mkdir(parents=True, exist_ok=False)
    logs = []
    for child in sorted(RESULTS_ROOT.iterdir()):
        if child.name == "final":
            continue
        ok, message = move_path(child, archive_dest / child.name)
        logs.append(("archived_nonfinal" if ok else "archive_skip", message))
    return archive_dest, logs


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    FINAL_DATA.mkdir(parents=True, exist_ok=True)
    FINAL_PLOTS.mkdir(parents=True, exist_ok=True)
    FINAL_METADATA.mkdir(parents=True, exist_ok=True)

    logs = []
    for src, row in confirmed_data_dirs():
        table_dir = FINAL_DATA / row["table"]
        dest = table_dir / src.name
        ok, message = move_path(src, dest)
        logs.append(("moved_final_data" if ok else "missing_final_data", message))

    exact_plots, candidate_plots = final_plot_paths()
    for src in exact_plots:
        dest = FINAL_PLOTS / "exact" / src.parent.name / src.name
        ok, message = move_path(src, dest)
        logs.append(("moved_exact_plot" if ok else "missing_exact_plot", message))
    for src in candidate_plots:
        dest = FINAL_PLOTS / "candidates" / src.parent.name / src.name
        ok, message = move_path(src, dest)
        logs.append(("moved_candidate_plot" if ok else "missing_candidate_plot", message))

    logs.extend(copy_metadata())
    archive_dest, archive_logs = archive_remaining_results(timestamp)
    logs.extend(archive_logs)
    write_manifest(logs, archive_dest)

    print(f"Moved {sum(1 for action, _ in logs if action == 'moved_final_data')} final data directories.")
    print(f"Moved {sum(1 for action, _ in logs if action == 'moved_exact_plot')} exact paper PDFs.")
    print(f"Moved {sum(1 for action, _ in logs if action == 'moved_candidate_plot')} candidate paper PDFs.")
    print(f"Archived remaining results to {archive_dest}")
    print(f"Wrote manifest to {FINAL_ROOT / 'FINAL_ARCHIVE_MANIFEST.md'}")


if __name__ == "__main__":
    main()
