"""Run one restart-safe GPU queue sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from configs import build_queues


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=int, choices=range(4), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    queues, loads = build_queues()
    tasks = queues[args.queue]
    root = Path(args.output_dir)
    queue_dir = root / "queues"
    log_dir = root / "task_logs" / f"gpu{args.queue}"
    queue_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = queue_dir / f"gpu{args.queue}.json"
    completed = 0
    failed = []
    started = time.time()
    for index, (dataset, seed) in enumerate(tasks):
        state = {
            "status": "running",
            "queue": args.queue,
            "estimated_seconds": loads[args.queue],
            "task_index": index,
            "task_count": len(tasks),
            "current": {"dataset": dataset, "seed": seed},
            "completed": completed,
            "failed": failed,
            "started_unix": started,
            "updated_unix": time.time(),
        }
        atomic_json(status_path, state)
        log_path = log_dir / f"{dataset}_seed{seed:03d}.log"
        command = [
            sys.executable,
            "-u",
            str(RUNNER),
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--output-dir",
            args.output_dir,
        ]
        print(
            f"QUEUE gpu{args.queue}: {index + 1}/{len(tasks)} "
            f"{dataset} seed={seed}",
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
            failed.append(
                {"dataset": dataset, "seed": seed, "returncode": result.returncode}
            )
            if not args.continue_on_error:
                break
    final = {
        "status": "complete" if not failed and completed == len(tasks) else "failed",
        "queue": args.queue,
        "task_count": len(tasks),
        "completed": completed,
        "failed": failed,
        "started_unix": started,
        "finished_unix": time.time(),
    }
    atomic_json(status_path, final)
    if final["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
