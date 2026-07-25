"""Deterministic GrQc event generation and protocol contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NUM_NODES = 5242
N_ARMS = 10


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    u, v = int(u), int(v)
    return (u, v) if u <= v else (v, u)


def positive_edges(array: np.ndarray) -> np.ndarray:
    return np.asarray(array[array[:, 2] == 1, :2], dtype=np.int32)


def negative_edges(array: np.ndarray) -> np.ndarray:
    return np.asarray(array[array[:, 2] != 1, :2], dtype=np.int32)


@dataclass
class GrQcData:
    features: np.ndarray
    root_entry: np.ndarray
    insert_entry: np.ndarray
    legacy_warm: np.ndarray
    full_undirected: set[tuple[int, int]]
    warm_undirected: set[tuple[int, int]]
    online_disjoint: list[tuple[int, int]]
    neighbors: list[set[int]]


def load_grqc_data(data_dir) -> GrQcData:
    root = Path(data_dir)
    features = np.load(root / "GrQc_ALLusers_features.npy")
    root_entry = np.load(root / "GrQc_ALLusers_entry.npy")
    insert_entry = np.load(root / "Insert/GrQc_ALLusers_entry.npy")
    legacy_warm = np.load(root / "Insert/GrQc_ALLusers_noedge.npy")

    full = {
        canonical_edge(u, v)
        for u, v in positive_edges(root_entry)
        if int(u) != int(v)
    }
    warm = {
        canonical_edge(u, v)
        for u, v in positive_edges(legacy_warm)
        if int(u) != int(v)
    }
    online = sorted(full - warm)
    neighbors = [set() for _ in range(NUM_NODES)]
    for u, v in full:
        neighbors[u].add(v)
        neighbors[v].add(u)
    return GrQcData(
        features=np.asarray(features, dtype=np.float32),
        root_entry=root_entry,
        insert_entry=insert_entry,
        legacy_warm=legacy_warm,
        full_undirected=full,
        warm_undirected=warm,
        online_disjoint=online,
        neighbors=neighbors,
    )


@dataclass
class EventTape:
    candidates: np.ndarray
    positive_arm: np.ndarray
    positive_edge: np.ndarray
    legacy_user: np.ndarray
    serving_node: np.ndarray
    digest: str

    def __len__(self):
        return int(self.candidates.shape[0])


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _orient_edges(
    edges: list[tuple[int, int]] | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.asarray(edges, dtype=np.int32).copy()
    flips = rng.integers(0, 2, size=len(result), dtype=np.int8).astype(bool)
    result[flips] = result[flips, ::-1]
    return result


def _sample_shared_negatives(
    serving: int,
    positive_target: int,
    data: GrQcData,
    rng: np.random.Generator,
) -> np.ndarray:
    selected: set[int] = {int(serving), int(positive_target)}
    result = []
    while len(result) < N_ARMS - 1:
        target = int(rng.integers(0, NUM_NODES))
        if target in selected or target in data.neighbors[serving]:
            continue
        selected.add(target)
        result.append((serving, target))
    return np.asarray(result, dtype=np.int32)


def build_event_tape(
    data: GrQcData,
    variant,
    rounds: int,
    seed: int,
    excluded_positive_edges=None,
) -> EventTape:
    """Build one immutable online/evaluation event tape."""
    rng = np.random.default_rng(seed)
    entry = data.root_entry if variant.entry_style == "root" else data.insert_entry
    entry_pos = positive_edges(entry)
    entry_neg = negative_edges(entry)

    excluded = set(excluded_positive_edges or ())
    if variant.disjoint_split:
        base_edges = [
            edge for edge in data.online_disjoint if edge not in excluded
        ]
        if rounds > len(base_edges) and variant.positive_without_replacement:
            raise ValueError(
                f"rounds={rounds} exceeds {len(base_edges)} disjoint positives"
            )
        if variant.positive_without_replacement:
            indices = rng.choice(len(base_edges), size=rounds, replace=False)
        else:
            indices = rng.choice(len(base_edges), size=rounds, replace=True)
        positives = _orient_edges([base_edges[int(i)] for i in indices], rng)
    else:
        if variant.positive_without_replacement:
            base_edges = sorted(
                {
                    canonical_edge(u, v)
                    for u, v in entry_pos
                    if int(u) != int(v)
                    and canonical_edge(u, v) not in excluded
                }
            )
            if rounds > len(base_edges):
                raise ValueError(
                    f"rounds={rounds} exceeds {len(base_edges)} entry positives"
                )
            indices = rng.choice(len(base_edges), size=rounds, replace=False)
            positives = _orient_edges(
                [base_edges[int(i)] for i in indices], rng
            )
        else:
            if excluded:
                entry_pos = np.asarray(
                    [
                        (u, v)
                        for u, v in entry_pos
                        if canonical_edge(u, v) not in excluded
                    ],
                    dtype=np.int32,
                )
            if len(entry_pos) == 0:
                raise ValueError("no positive edges remain after exclusions")
            indices = rng.choice(len(entry_pos), size=rounds, replace=True)
            positives = np.asarray(entry_pos[indices], dtype=np.int32)

    candidates = np.empty((rounds, N_ARMS, 2), dtype=np.int32)
    positive_arm = rng.integers(0, N_ARMS, size=rounds, dtype=np.int16)
    legacy_user = np.empty(rounds, dtype=np.int32)
    serving_node = positives[:, 0].astype(np.int32, copy=True)

    for t in range(rounds):
        pos = positives[t]
        arm = int(positive_arm[t])
        if variant.shared_serving:
            neg = _sample_shared_negatives(
                int(pos[0]), int(pos[1]), data, rng
            )
        else:
            neg_indices = rng.choice(
                len(entry_neg), size=N_ARMS - 1, replace=False
            )
            neg = np.asarray(entry_neg[neg_indices], dtype=np.int32)
        candidates[t, :arm] = neg[:arm]
        candidates[t, arm] = pos
        candidates[t, arm + 1:] = neg[arm:]
        endpoint_index = arm if arm == N_ARMS - 1 else arm + 1
        legacy_user[t] = candidates[t, endpoint_index, 0]

    digest = _digest_arrays(
        candidates, positive_arm, positives, legacy_user, serving_node
    )
    return EventTape(
        candidates=candidates,
        positive_arm=positive_arm,
        positive_edge=positives,
        legacy_user=legacy_user,
        serving_node=serving_node,
        digest=digest,
    )


def validate_tape(
    tape: EventTape,
    variant,
    data: GrQcData,
) -> dict:
    rounds = len(tape)
    rows = np.arange(rounds)
    selected_positive = tape.candidates[rows, tape.positive_arm]
    if not np.array_equal(selected_positive, tape.positive_edge):
        raise AssertionError("positive arm does not identify positive edge")

    shared_violations = int(
        np.count_nonzero(
            tape.candidates[:, :, 0]
            != tape.serving_node.reshape(-1, 1)
        )
    )
    if variant.shared_serving and shared_violations:
        raise AssertionError(
            f"shared-serving contract has {shared_violations} violations"
        )

    warm_online_overlap = 0
    if variant.disjoint_split:
        online = {
            canonical_edge(u, v)
            for u, v in tape.positive_edge
            if int(u) != int(v)
        }
        warm_online_overlap = len(online & data.warm_undirected)
        if warm_online_overlap:
            raise AssertionError(
                f"warm/online overlap contains {warm_online_overlap} edges"
            )

    positive_unique = len(
        {
            canonical_edge(u, v)
            for u, v in tape.positive_edge
        }
    )
    if variant.positive_without_replacement and positive_unique != rounds:
        raise AssertionError("positive-without-replacement contract violated")

    return {
        "rounds": rounds,
        "shared_serving_violations": shared_violations,
        "warm_online_overlap": warm_online_overlap,
        "unique_positive_edges": positive_unique,
        "repeated_positive_events": rounds - positive_unique,
        "event_digest": tape.digest,
    }


def identical_event_count(left: EventTape, right: EventTape) -> int:
    limit = min(len(left), len(right))
    same_candidates = np.all(
        left.candidates[:limit] == right.candidates[:limit], axis=(1, 2)
    )
    same_arms = left.positive_arm[:limit] == right.positive_arm[:limit]
    return int(np.count_nonzero(same_candidates & same_arms))


def positive_edge_overlap_count(left: EventTape, right: EventTape) -> int:
    left_edges = {
        canonical_edge(u, v) for u, v in left.positive_edge
    }
    right_edges = {
        canonical_edge(u, v) for u, v in right.positive_edge
    }
    return len(left_edges & right_edges)
