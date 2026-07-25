"""Run one ablation group sequentially, with restart-safe task status."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from variants import GROUPS, VARIANTS


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument(
        "--variants",
        choices=sorted(VARIANTS),
        nargs="+",
        help="Optional explicit queue; defaults to every variant in --group.",
    )
    parser.add_argument(
        "--queue-name",
        help="Status/log namespace; defaults to --group.",
    )
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation-samples", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_dir)
    queue_dir = root / "queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_name = args.queue_name or args.group
    variants = args.variants or GROUPS[args.group]
    queue_status = queue_dir / f"{queue_name}_T{args.rounds}.json"
    tasks = [
        {"variant": variant, "seed": seed}
        for variant in variants
        for seed in args.seeds
    ]
    completed = 0
    failed = []
    started = time.time()
    for index, task in enumerate(tasks):
        variant = task["variant"]
        seed = task["seed"]
        log_dir = root / "task_logs" / queue_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{variant}_seed{seed:03d}_T{args.rounds}.log"
        command = [
            sys.executable,
            "-u",
            str(RUNNER),
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--rounds",
            str(args.rounds),
            "--data-dir",
            args.data_dir,
            "--output-dir",
            args.output_dir,
            "--evaluation-samples",
            str(args.evaluation_samples),
            "--progress-every",
            str(args.progress_every),
        ]
        state = {
            "status": "running",
            "group": args.group,
            "queue_name": queue_name,
            "rounds": args.rounds,
            "task_index": index,
            "task_count": len(tasks),
            "current": task,
            "completed": completed,
            "failed": failed,
            "started_unix": started,
            "updated_unix": time.time(),
        }
        queue_status.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(
            f"QUEUE {queue_name}: {index + 1}/{len(tasks)} "
            f"{variant} seed={seed}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0:
            completed += 1
        else:
            failed.append({**task, "returncode": result.returncode})
            if not args.continue_on_error:
                break

    final = {
        "status": "complete" if not failed and completed == len(tasks) else "failed",
        "group": args.group,
        "queue_name": queue_name,
        "rounds": args.rounds,
        "task_count": len(tasks),
        "completed": completed,
        "failed": failed,
        "started_unix": started,
        "finished_unix": time.time(),
    }
    queue_status.write_text(json.dumps(final, indent=2), encoding="utf-8")
    if final["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
