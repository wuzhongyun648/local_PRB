#!/usr/bin/env python3
"""Non-training microbenchmark for graph updates and LocPRB solvers."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dynamic_appr import DynamicAPPR
from src.ppr_solver import appr_with_stats
from src import utils


def build_regular_graph(num_nodes, degree):
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
        (np.ones(sum(len(part) for part in rows)), (np.concatenate(rows), np.concatenate(cols))),
        shape=(num_nodes, num_nodes),
    )
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    return adjacency


def make_manager(adjacency):
    manager = utils.graph("synthetic")
    manager.A = adjacency.tolil()
    manager.P = utils._normalize_columns(manager.A, format="csc")
    manager.degree = np.asarray(manager.A.sum(axis=1)).reshape(-1)
    manager.num_nodes = adjacency.shape[0]
    manager.num_edges = manager.A.nnz // 2
    return manager


def insertion_edges(adjacency, rounds, rng):
    selected = []
    seen = set()
    num_nodes = adjacency.shape[0]
    while len(selected) < rounds:
        u = int(rng.integers(num_nodes))
        v = int(rng.integers(num_nodes))
        edge = (min(u, v), max(u, v))
        if u == v or edge in seen or adjacency[u, v] != 0:
            continue
        seen.add(edge)
        selected.append((u, v))
    return selected


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean() * 1000.0),
        "p50_ms": float(np.percentile(values, 50) * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
    }


def benchmark_size(num_nodes, degree=8, rounds=20, alpha=0.85, eps=1e-4, seed=0):
    rng = np.random.default_rng(seed + num_nodes)
    adjacency = build_regular_graph(num_nodes, degree)
    manager = make_manager(adjacency)
    edges = insertion_edges(adjacency, rounds, rng)

    source_nodes = rng.choice(num_nodes, size=min(10, num_nodes), replace=False)
    source = np.zeros(num_nodes, dtype=np.float64)
    source[source_nodes] = 1.0 / len(source_nodes)
    initial_csr = manager.P.tocsr()
    # Warm JIT compilation outside the timed section.
    appr_with_stats(
        num_nodes,
        initial_csr.indptr,
        initial_csr.indices,
        manager.degree,
        source,
        alpha,
        eps,
    )
    dynamic = DynamicAPPR()
    dynamic.solve(
        num_nodes,
        initial_csr.indptr,
        initial_csr.indices,
        manager.degree,
        source,
        alpha,
        eps,
    )

    measurements = {
        "graph_update": [],
        "csr_rebuild": [],
        "scratch_appr": [],
        "dynamic_appr": [],
    }
    scratch_pushes = []
    dynamic_pushes = []

    for round_index, (u, v) in enumerate(edges):
        start = time.perf_counter()
        added = manager._add_undirected_edge(u, v)
        measurements["graph_update"].append(time.perf_counter() - start)
        if not added:
            raise RuntimeError("microbenchmark generated an existing edge")

        start = time.perf_counter()
        current_csr = manager.P.tocsr()
        measurements["csr_rebuild"].append(time.perf_counter() - start)

        source.fill(0.0)
        source[(source_nodes + round_index) % num_nodes] = 1.0 / len(source_nodes)
        degree_array = np.asarray(manager.degree, dtype=np.float64)

        start = time.perf_counter()
        _, pushes = appr_with_stats(
            num_nodes,
            current_csr.indptr,
            current_csr.indices,
            degree_array,
            source,
            alpha,
            eps,
        )
        measurements["scratch_appr"].append(time.perf_counter() - start)
        scratch_pushes.append(int(pushes))

        start = time.perf_counter()
        dynamic.solve(
            num_nodes,
            current_csr.indptr,
            current_csr.indices,
            degree_array,
            source,
            alpha,
            eps,
        )
        measurements["dynamic_appr"].append(time.perf_counter() - start)
        dynamic_pushes.append(int(dynamic.last_stats["pushes"]))

    return {
        "num_nodes": int(num_nodes),
        "initial_edges": int(adjacency.nnz // 2),
        "degree": int(degree),
        "rounds": int(rounds),
        "timing": {name: summarize(values) for name, values in measurements.items()},
        "pushes": {
            "scratch_mean": float(np.mean(scratch_pushes)),
            "scratch_p95": float(np.percentile(scratch_pushes, 95)),
            "dynamic_mean": float(np.mean(dynamic_pushes)),
            "dynamic_p95": float(np.percentile(dynamic_pushes, 95)),
        },
        "dynamic_cumulative": dynamic.stats,
    }


def run_benchmark(sizes, degree=8, rounds=20, alpha=0.85, eps=1e-4, seed=0):
    return {
        "protocol": "graph-ppr-only-v1",
        "alpha": float(alpha),
        "eps": float(eps),
        "seed": int(seed),
        "results": [
            benchmark_size(size, degree, rounds, alpha, eps, seed)
            for size in sizes
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_benchmark(
        args.sizes, args.degree, args.rounds, args.alpha, args.eps, args.seed
    )
    output = args.output or os.path.join(
        REPO_ROOT, "results", "benchmarks", "graph_ppr_latest.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
