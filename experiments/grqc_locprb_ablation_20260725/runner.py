"""Standalone GrQc LocPRB ablation runner.

This module imports the production neural model and propagation kernels but
does not modify the production CLI or loaders.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.EENet import EE_Net  # noqa: E402
from src import ppr_solver  # noqa: E402
from src.experiment_configs import (  # noqa: E402
    TRAIN_EVERY_AFTER_2000,
    TRAIN_EVERY_BEFORE_2000,
)

from events import (  # noqa: E402
    NUM_NODES,
    build_event_tape,
    canonical_edge,
    identical_event_count,
    load_grqc_data,
    positive_edge_overlap_count,
    validate_tape,
)
from variants import VARIANTS  # noqa: E402


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


class GraphState:
    def __init__(self, warm_edges):
        rows = []
        cols = []
        for u, v in sorted(warm_edges):
            if u == v:
                continue
            rows.extend((u, v))
            cols.extend((v, u))
        values = np.ones(len(rows), dtype=np.float64)
        self.adjacency = sp.csr_matrix(
            (values, (rows, cols)), shape=(NUM_NODES, NUM_NODES)
        )
        self.adjacency.data[:] = 1.0
        self._refresh()

    def _refresh(self):
        self.adjacency = self.adjacency.tocsr()
        self.adjacency.sum_duplicates()
        self.adjacency.data[:] = 1.0
        self.degree_raw = np.asarray(
            self.adjacency.sum(axis=1)
        ).reshape(-1).astype(np.int64)
        self.degree = self.degree_raw.copy()
        self.degree[self.degree == 0] = 1
        inverse = np.zeros(NUM_NODES, dtype=np.float64)
        nonzero = self.degree_raw > 0
        inverse[nonzero] = 1.0 / self.degree_raw[nonzero]
        self.transition = (
            self.adjacency.tocsc() @ sp.diags(inverse, format="csc")
        ).tocsr()

    def contains(self, u: int, v: int) -> bool:
        return bool(self.adjacency[int(u), int(v)] != 0)

    def insert(self, u: int, v: int) -> bool:
        u, v = int(u), int(v)
        if u == v or self.contains(u, v):
            return False
        mutable = self.adjacency.tolil()
        mutable[u, v] = 1.0
        mutable[v, u] = 1.0
        self.adjacency = mutable.tocsr()
        self._refresh()
        return True


def transform_source(raw, exploit, mode: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    exploit = np.asarray(exploit, dtype=np.float64).reshape(-1)
    if mode in ("raw", "exploit_explore"):
        return raw.copy()
    if mode == "exploit_only":
        return exploit.copy()
    if mode == "l1":
        norm = float(np.sum(np.abs(raw)))
        return raw / norm if norm > 0 else np.full_like(raw, 1.0 / len(raw))
    if mode == "clip_l1":
        clipped = np.maximum(raw, 0.0)
        norm = float(np.sum(clipped))
        return (
            clipped / norm
            if norm > 0
            else np.full_like(raw, 1.0 / len(raw))
        )
    if mode == "softmax":
        shifted = raw - float(np.max(raw))
        values = np.exp(shifted)
        return values / float(np.sum(values))
    raise ValueError(f"unknown source mode: {mode}")


def edge_contexts(features: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.concatenate((features[int(u)], features[int(v)]))
            for u, v in candidates
        ],
        dtype=np.float32,
    )


def solve_three(
    graph: GraphState,
    candidate_edges: np.ndarray,
    candidate_source: np.ndarray,
    alpha: float,
    eps: float,
    power_steps: int,
    loc_kernel,
):
    targets = candidate_edges[:, 1].astype(np.int64)
    dense = np.zeros(NUM_NODES, dtype=np.float64)
    for target, value in zip(targets, candidate_source):
        dense[int(target)] = float(value)
    support = np.unique(targets)

    loc = loc_kernel(
        NUM_NODES,
        graph.transition.indptr,
        graph.transition.indices,
        graph.degree,
        dense,
        alpha,
        eps,
        support,
    )[0]
    power = ppr_solver.power_iteration(
        graph.transition, alpha, dense, power_steps
    )
    return {
        "direct": int(np.argmax(candidate_source)),
        "loc": int(np.argmax(loc[targets])),
        "power": int(np.argmax(power[targets])),
    }


def should_train(t: int) -> bool:
    if t < 2000:
        return t % TRAIN_EVERY_BEFORE_2000 == 0
    return t % TRAIN_EVERY_AFTER_2000 == 0


def evaluate(
    model,
    graph,
    features,
    tape,
    variant,
    loc_kernel,
    alpha,
    online_round,
):
    hits = {"direct": 0, "loc": 0, "power": 0}
    for t in range(len(tape)):
        candidates = tape.candidates[t]
        context = edge_contexts(features, candidates)
        _, raw = model.predict(context, online_round)
        source = transform_source(raw, model.exploit_scores, variant.source_mode)
        choices = solve_three(
            graph, candidates, source, alpha, variant.eps,
            variant.power_steps, loc_kernel,
        )
        positive = int(tape.positive_arm[t])
        for method, arm in choices.items():
            hits[method] += int(arm == positive)
    return {
        method: value / len(tape)
        for method, value in hits.items()
    }


def run(args) -> dict:
    variant = VARIANTS[args.variant]
    output = Path(args.output_dir) / variant.group / variant.name / (
        f"seed_{args.seed:03d}_T{args.rounds}"
    )
    status_path = output / "status.json"
    if status_path.exists() and not args.force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            print(f"SKIP complete {output}", flush=True)
            return status
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        status_path,
        {
            "status": "running",
            "variant": args.variant,
            "seed": args.seed,
            "rounds": args.rounds,
            "pid": os.getpid(),
            "started_unix": time.time(),
        },
    )

    started = time.perf_counter()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = load_grqc_data(args.data_dir)
    online = build_event_tape(data, variant, args.rounds, args.seed)
    online_contract = validate_tape(online, variant, data)
    eval_seed = (
        args.seed + 1_000_003 if variant.independent_eval else args.seed
    )
    excluded_eval_edges = (
        {
            canonical_edge(u, v)
            for u, v in online.positive_edge
        }
        if variant.independent_eval
        else None
    )
    evaluation_tape = build_event_tape(
        data,
        variant,
        args.evaluation_samples,
        eval_seed,
        excluded_positive_edges=excluded_eval_edges,
    )
    eval_contract = validate_tape(evaluation_tape, variant, data)
    online_eval_identical = identical_event_count(online, evaluation_tape)
    online_eval_positive_overlap = positive_edge_overlap_count(
        online, evaluation_tape
    )
    if variant.independent_eval and (
        online_eval_identical or online_eval_positive_overlap
    ):
        raise AssertionError(
            "independent evaluation overlap: "
            f"{online_eval_identical} events, "
            f"{online_eval_positive_overlap} positive edges"
        )

    graph = GraphState(data.warm_undirected)
    model = EE_Net(
        dim=20,
        n_arm=10,
        pool_step_size=50,
        lr_1=variant.lr1,
        lr_2=variant.lr2,
        hidden=100,
        neural_decision_maker=False,
        kernel_size=40,
    )
    loc_kernel = ppr_solver.get_timing_kernel("numba")
    # Compile before measured online work.
    zero = np.zeros(NUM_NODES, dtype=np.float64)
    loc_kernel(
        NUM_NODES,
        graph.transition.indptr,
        graph.transition.indices,
        graph.degree,
        zero,
        args.alpha,
        variant.eps,
        np.empty(0, dtype=np.int64),
    )

    regret = 0.0
    regrets = np.empty(args.rounds, dtype=np.float64)
    shadow_regret = {
        name: np.empty(args.rounds, dtype=np.float64)
        for name in ("direct", "loc", "power")
    }
    shadow_total = {"direct": 0.0, "loc": 0.0, "power": 0.0}
    shadow_counts = {
        "direct_wrong_loc_correct": 0,
        "direct_correct_loc_wrong": 0,
        "loc_power_disagreement": 0,
        "first_loc_power_disagreement": None,
        "first_direct_loc_disagreement": None,
    }
    inserted = 0
    selected_positive = 0
    selected_positive_already_visible = 0
    positive_visible_before_round = 0
    update_endpoint_mismatches = 0
    source_negative_entries = 0
    source_l1_sum = 0.0
    round_seconds = np.empty(args.rounds, dtype=np.float64)

    for t in range(args.rounds):
        round_start = time.perf_counter()
        candidates = online.candidates[t]
        positive_arm = int(online.positive_arm[t])
        positive_u, positive_v = map(int, online.positive_edge[t])
        positive_visible_before_round += int(
            graph.contains(positive_u, positive_v)
        )

        context = edge_contexts(data.features, candidates)
        _, raw = model.predict(context, t)
        source = transform_source(raw, model.exploit_scores, variant.source_mode)
        source_negative_entries += int(np.count_nonzero(source < 0))
        source_l1_sum += float(np.sum(np.abs(source)))
        choices = solve_three(
            graph, candidates, source, args.alpha, variant.eps,
            variant.power_steps, loc_kernel,
        )

        direct_correct = choices["direct"] == positive_arm
        loc_correct = choices["loc"] == positive_arm
        shadow_counts["direct_wrong_loc_correct"] += int(
            not direct_correct and loc_correct
        )
        shadow_counts["direct_correct_loc_wrong"] += int(
            direct_correct and not loc_correct
        )
        if choices["loc"] != choices["power"]:
            shadow_counts["loc_power_disagreement"] += 1
            if shadow_counts["first_loc_power_disagreement"] is None:
                shadow_counts["first_loc_power_disagreement"] = t
        if choices["direct"] != choices["loc"]:
            if shadow_counts["first_direct_loc_disagreement"] is None:
                shadow_counts["first_direct_loc_disagreement"] = t

        for method, arm in choices.items():
            shadow_total[method] += float(arm != positive_arm)
            shadow_regret[method][t] = shadow_total[method]

        primary_arm = choices[variant.solver]
        reward = float(primary_arm == positive_arm)
        regret += 1.0 - reward
        regrets[t] = regret

        if reward == 1.0:
            selected_positive += 1
            selected_target = int(candidates[primary_arm, 1])
            if variant.corrected_update:
                update_u, update_v = map(int, candidates[primary_arm])
            else:
                update_u = int(online.legacy_user[t])
                update_v = selected_target
            update_endpoint_mismatches += int(
                canonical_edge(update_u, update_v)
                != canonical_edge(positive_u, positive_v)
            )
            was_visible = graph.contains(update_u, update_v)
            selected_positive_already_visible += int(was_visible)
            inserted += int(graph.insert(update_u, update_v))

        model.arm_select = primary_arm
        model.update(context, reward, t)
        if should_train(t) and not args.skip_training:
            model.train(t)
        round_seconds[t] = time.perf_counter() - round_start

        if (t + 1) % args.progress_every == 0 or t + 1 == args.rounds:
            print(
                f"{variant.name} seed={args.seed} "
                f"round={t + 1}/{args.rounds} regret={regret:.0f}",
                flush=True,
            )

    evaluation_accuracy = evaluate(
        model, graph, data.features, evaluation_tape, variant, loc_kernel,
        args.alpha, args.rounds,
    )
    duration = time.perf_counter() - started
    np.savez_compressed(
        output / "curves.npz",
        regret=regrets,
        direct_shadow_regret=shadow_regret["direct"],
        loc_shadow_regret=shadow_regret["loc"],
        power_shadow_regret=shadow_regret["power"],
        round_seconds=round_seconds,
    )
    summary = {
        "status": "complete",
        "variant": variant.to_dict(),
        "seed": args.seed,
        "rounds": args.rounds,
        "alpha": args.alpha,
        "final_regret": regret,
        "shadow_final_regret": {
            key: float(value[-1]) for key, value in shadow_regret.items()
        },
        "shadow_counts": shadow_counts,
        "net_reranking_gain": (
            shadow_counts["direct_wrong_loc_correct"]
            - shadow_counts["direct_correct_loc_wrong"]
        ),
        "graph": {
            "initial_undirected_edges": len(data.warm_undirected),
            "inserted_edges": inserted,
            "selected_positive": selected_positive,
            "selected_positive_already_visible": (
                selected_positive_already_visible
            ),
            "positive_visible_before_round": positive_visible_before_round,
            "update_endpoint_mismatches": update_endpoint_mismatches,
            "final_undirected_edges": int(graph.adjacency.nnz // 2),
        },
        "source": {
            "negative_entries": source_negative_entries,
            "mean_l1": source_l1_sum / args.rounds,
        },
        "events": {
            "online": online_contract,
            "evaluation": eval_contract,
            "online_evaluation_identical_events": online_eval_identical,
            "online_evaluation_positive_edge_overlap": (
                online_eval_positive_overlap
            ),
            "evaluation_seed": eval_seed,
        },
        "evaluation_accuracy": evaluation_accuracy,
        "duration_seconds": duration,
        "device": str(next(model.f_1.func.parameters()).device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "finished_unix": time.time(),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(status_path, summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data" / "GrQc"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "grqc_locprb_ablation_20260725"),
    )
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--evaluation-samples", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        variant = VARIANTS[args.variant]
        output = Path(args.output_dir) / variant.group / variant.name / (
            f"seed_{args.seed:03d}_T{args.rounds}"
        )
        atomic_json(
            output / "status.json",
            {
                "status": "failed",
                "variant": args.variant,
                "seed": args.seed,
                "rounds": args.rounds,
                "error": repr(exc),
                "finished_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
