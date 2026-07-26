"""Run one formal-horizon g1-current-style LocPRB trial."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import CONFIGS  # noqa: E402
from src.main import run_experiment  # noqa: E402


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int)
    return parser.parse_args()


def main():
    cli = parse_args()
    config = CONFIGS[cli.dataset]
    rounds = cli.rounds or config.rounds
    output = (
        Path(cli.output_dir)
        / cli.dataset
        / f"seed_{cli.seed:03d}_T{rounds}"
    )
    status_path = output / "status.json"
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"SKIP complete {cli.dataset} seed={cli.seed}", flush=True)
            return
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        status_path,
        {
            "status": "running",
            "dataset": cli.dataset,
            "seed": cli.seed,
            "rounds": rounds,
            "started_unix": time.time(),
        },
    )

    formal_args = Namespace(
        graph_name=config.graph_name,
        method="numba-locPRB",
        alpha=0.85,
        appr_eps=config.eps,
        power_T=None,
        T=rounds,
        lr1=config.lr1,
        lr2=config.lr2,
        if_save=False,
        workers=1,
        runs=1,
        seed=cli.seed,
        n_neg=9,
        init_hops=0,
        init_topk=0,
        init_edges=None,
        ppr_diagnostics=False,
        ppr_diagnostic_every=50,
        evaluation_every=0,
        hidden=100,
        kernel_size=config.kernel_size,
    )
    started = time.time()
    try:
        curve, timing = run_experiment(0, formal_args, str(output))
        final_regret = float(curve[-1, 1])
        np.savez_compressed(
            output / "curves.npz",
            online_metrics=curve,
        )
        summary = {
            "status": "complete",
            "dataset": cli.dataset,
            "seed": cli.seed,
            "rounds": rounds,
            "final_regret": final_regret,
            "duration_seconds": time.time() - started,
            "config": {
                **config.to_dict(),
                "actual_rounds": rounds,
                "alpha": 0.85,
                "method": "numba-locPRB",
                "evaluation_every": 0,
                "positive_sampling": "legacy_with_replacement",
            },
            "timing": timing,
            "finished_unix": time.time(),
        }
        atomic_json(output / "summary.json", summary)
        atomic_json(status_path, summary)
        print(
            f"COMPLETE {cli.dataset} seed={cli.seed} "
            f"T={rounds} regret={final_regret:.0f}",
            flush=True,
        )
    except Exception as exc:
        atomic_json(
            status_path,
            {
                "status": "failed",
                "dataset": cli.dataset,
                "seed": cli.seed,
                "rounds": rounds,
                "error": repr(exc),
                "finished_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
