#!/usr/bin/env python3
"""Compare scratch, Python-DYN, and Numba-DYN APPR on one event stream."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dynamic_appr import DynamicAPPR, appr_push
from src.ppr_solver import appr_with_stats


def regular_graph(num_nodes, degree):
    if degree < 2 or degree % 2:
        raise ValueError("degree must be an even integer >= 2")
    nodes = np.arange(num_nodes, dtype=np.int64)
    rows = []
    cols = []
    for offset in range(1, degree // 2 + 1):
        neighbor = (nodes + offset) % num_nodes
        rows.extend((nodes, neighbor))
        cols.extend((neighbor, nodes))
    adjacency = sp.csr_matrix(
        (
            np.ones(sum(len(part) for part in rows), dtype=np.float64),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(num_nodes, num_nodes),
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return adjacency


def source_stream(num_nodes, rounds, support_size, seed):
    rng = np.random.default_rng(seed)
    stream = []
    candidates = []
    for _ in range(rounds):
        support = rng.choice(num_nodes, size=support_size, replace=False)
        source = np.zeros(num_nodes, dtype=np.float64)
        # Raw EE-Net outputs are not L1-normalized and may be signed.
        source[support] = rng.normal(loc=0.5, scale=0.75, size=support_size)
        stream.append(source)
        candidates.append(support)
    return stream, candidates


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "total_ms": float(values.sum() * 1000.0),
        "mean_ms": float(values.mean() * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
    }


def run_case(num_nodes, degree, rounds, support_size, alpha, eps, seed):
    adjacency = regular_graph(num_nodes, degree)
    degree_array = np.diff(adjacency.indptr).astype(np.float64)
    sources, candidates = source_stream(num_nodes, rounds, support_size, seed)

    if not hasattr(appr_push, "py_func"):
        raise RuntimeError("Numba is required for the Python-vs-JIT diagnosis")

    # Compile both kernels before timing.
    appr_with_stats(
        num_nodes,
        adjacency.indptr,
        adjacency.indices,
        degree_array,
        sources[0],
        alpha,
        eps,
    )
    warm = DynamicAPPR()
    warm.solve(
        num_nodes,
        adjacency.indptr,
        adjacency.indices,
        degree_array,
        sources[0],
        alpha,
        eps,
    )

    dynamic_jit = DynamicAPPR(push_impl=appr_push, diagnostics=True)
    dynamic_python = DynamicAPPR(
        push_impl=appr_push.py_func, diagnostics=True
    )
    timings = {"scratch": [], "dynamic_numba": [], "dynamic_python": []}
    scratch_pushes = []
    numba_pushes = []
    l1_errors = []
    decision_disagreements = 0
    python_numba_max_abs = 0.0

    for source, candidate in zip(sources, candidates):
        start = time.perf_counter()
        scratch, pushes = appr_with_stats(
            num_nodes,
            adjacency.indptr,
            adjacency.indices,
            degree_array,
            source,
            alpha,
            eps,
        )
        timings["scratch"].append(time.perf_counter() - start)

        start = time.perf_counter()
        dyn_jit = dynamic_jit.solve(
            num_nodes,
            adjacency.indptr,
            adjacency.indices,
            degree_array,
            source,
            alpha,
            eps,
        )
        timings["dynamic_numba"].append(time.perf_counter() - start)

        start = time.perf_counter()
        dyn_python = dynamic_python.solve(
            num_nodes,
            adjacency.indptr,
            adjacency.indices,
            degree_array,
            source,
            alpha,
            eps,
        )
        timings["dynamic_python"].append(time.perf_counter() - start)

        scratch_pushes.append(int(pushes))
        numba_pushes.append(dynamic_jit.last_stats["pushes"])
        l1_errors.append(float(np.sum(np.abs(dyn_jit - scratch))))
        python_numba_max_abs = max(
            python_numba_max_abs,
            float(np.max(np.abs(dyn_jit - dyn_python))),
        )
        decision_disagreements += int(
            np.argmax(dyn_jit[candidate]) != np.argmax(scratch[candidate])
        )

    return {
        "num_nodes": num_nodes,
        "degree": degree,
        "rounds": rounds,
        "eps": eps,
        "timing": {name: summarize(value) for name, value in timings.items()},
        "speedup_numba_over_python_dyn": float(
            np.sum(timings["dynamic_python"])
            / np.sum(timings["dynamic_numba"])
        ),
        "speedup_dyn_numba_over_scratch": float(
            np.sum(timings["scratch"])
            / np.sum(timings["dynamic_numba"])
        ),
        "pushes": {
            "scratch_mean": float(np.mean(scratch_pushes)),
            "dynamic_mean": float(np.mean(numba_pushes)),
            "dynamic_over_scratch": float(
                np.sum(numba_pushes) / np.sum(scratch_pushes)
            ),
        },
        "equivalence": {
            "python_numba_max_abs": python_numba_max_abs,
            "dynamic_scratch_l1_mean": float(np.mean(l1_errors)),
            "dynamic_scratch_l1_max": float(np.max(l1_errors)),
            "candidate_decision_disagreements": decision_disagreements,
        },
        "dynamic_stats": dynamic_jit.stats,
        "final_state": {
            "p_l1": dynamic_jit.last_stats["p_l1"],
            "residual_l1": dynamic_jit.last_stats["residual_l1"],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-nodes", type=int, default=10_000)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--support-size", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_case(
        args.num_nodes,
        args.degree,
        args.rounds,
        args.support_size,
        args.alpha,
        args.eps,
        args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
