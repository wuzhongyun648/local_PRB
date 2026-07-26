"""Dataset configurations and deterministic four-GPU task assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    graph_name: str
    rounds: int
    lr1: float
    lr2: float
    eps: float
    kernel_size: int
    estimated_seconds: int

    def to_dict(self):
        return asdict(self)


CONFIGS = {
    "MovieLens": DatasetConfig(
        "MovieLens", "MovieLens", 10_000, 0.0073, 0.0004,
        8.33e-5, 40, 2124,
    ),
    "AmazonFashion": DatasetConfig(
        "AmazonFashion", "Amazon_fashion", 5_000, 0.1, 0.01,
        1.25e-4, 40, 1081,
    ),
    "Facebook": DatasetConfig(
        "Facebook", "Facebook", 10_000, 0.06, 0.02,
        2.48e-4, 40, 1812,
    ),
    "Collab": DatasetConfig(
        "Collab", "Collab", 5_000, 0.01, 0.004,
        4.24e-6, 40, 1010,
    ),
    "PPA": DatasetConfig(
        "PPA", "PPA", 5_000, 0.01, 0.004,
        1.74e-6, 40, 4709,
    ),
    "Vessel": DatasetConfig(
        "Vessel", "Vessel", 5_000, 0.01, 0.004,
        2.86e-7, 5, 803,
    ),
}

SEEDS = tuple(range(200, 210))
INITIAL_DATASETS = ("MovieLens", "AmazonFashion", "Facebook", "Collab")


def build_queues():
    """Greedy duration balancing with four distinct datasets at launch."""
    queues = [[] for _ in range(4)]
    loads = [0] * 4
    assigned = set()
    for gpu, dataset in enumerate(INITIAL_DATASETS):
        task = (dataset, SEEDS[0])
        queues[gpu].append(task)
        loads[gpu] += CONFIGS[dataset].estimated_seconds
        assigned.add(task)

    remaining = [
        (dataset, seed)
        for dataset in CONFIGS
        for seed in SEEDS
        if (dataset, seed) not in assigned
    ]
    remaining.sort(
        key=lambda item: (
            -CONFIGS[item[0]].estimated_seconds,
            item[0],
            item[1],
        )
    )
    for task in remaining:
        gpu = min(range(4), key=lambda index: (loads[index], index))
        queues[gpu].append(task)
        loads[gpu] += CONFIGS[task[0]].estimated_seconds
    return queues, loads
