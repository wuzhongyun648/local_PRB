"""Run one restart-safe GPU queue with fixed one-CPU worker lanes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from configs import build_queues


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=int, choices=range(4), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    queues, loads = build_queues()
    tasks = queues[args.queue]
    root = Path(args.output_dir)
    queue_dir = root / "queues"
    log_dir = root / "task_logs" / f"gpu{args.queue}"
    queue_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = queue_dir / f"gpu{args.queue}.json"
    completed = []
    failed = []
    started = time.time()
    active = {}
    lock = threading.Lock()
    allowed_cpus = sorted(os.sched_getaffinity(0))
    required_cpus = 4 * args.workers
    if len(allowed_cpus) < required_cpus:
        raise RuntimeError(
            f"need {required_cpus} allowed CPUs for four queues, "
            f"but only {len(allowed_cpus)} are available"
        )
    cpu_ids = allowed_cpus[
        args.queue * args.workers : (args.queue + 1) * args.workers
    ]

    def write_state():
        atomic_json(
            status_path,
            {
                "status": "running",
                "queue": args.queue,
                "workers": args.workers,
                "cpu_ids": cpu_ids,
                "estimated_seconds": loads[args.queue],
                "task_count": len(tasks),
                "active": list(active.values()),
                "completed": len(completed),
                "completed_tasks": completed,
                "failed": failed,
                "started_unix": started,
                "updated_unix": time.time(),
            },
        )

    def run_lane(lane: int, lane_tasks):
        cpu_id = cpu_ids[lane]
        for index, (dataset, seed) in lane_tasks:
            task = {
                "task_index": index,
                "dataset": dataset,
                "seed": seed,
                "lane": lane,
                "cpu_id": cpu_id,
            }
            with lock:
                active[lane] = task
                write_state()
            log_path = log_dir / f"{dataset}_seed{seed:03d}.log"
            command = [
                "taskset",
                "--cpu-list",
                str(cpu_id),
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
                f"QUEUE gpu{args.queue} lane={lane} cpu={cpu_id}: "
                f"{index + 1}/{len(tasks)} {dataset} seed={seed}",
                flush=True,
            )
            worker_env = os.environ.copy()
            worker_env.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "TORCH_NUM_THREADS": "1",
                }
            )
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env=worker_env,
                )
            with lock:
                active.pop(lane, None)
                if result.returncode == 0:
                    completed.append(task)
                else:
                    failed.append({**task, "returncode": result.returncode})
                write_state()
            if result.returncode != 0 and not args.continue_on_error:
                break

    lanes = [[] for _ in range(args.workers)]
    for index, task in enumerate(tasks):
        lanes[index % args.workers].append((index, task))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(run_lane, lane, lane_tasks)
            for lane, lane_tasks in enumerate(lanes)
            if lane_tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    final = {
        "status": (
            "complete"
            if not failed and len(completed) == len(tasks)
            else "failed"
        ),
        "queue": args.queue,
        "workers": args.workers,
        "cpu_ids": cpu_ids,
        "task_count": len(tasks),
        "completed": len(completed),
        "completed_tasks": completed,
        "failed": failed,
        "started_unix": started,
        "finished_unix": time.time(),
    }
    atomic_json(status_path, final)
    if final["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
