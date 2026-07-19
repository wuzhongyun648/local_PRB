"""Compact, reproducible online-bandit event streams shared by all methods."""

from dataclasses import dataclass
import hashlib
import json
import os
import tempfile

import numpy as np


STREAM_VERSION = 1


def canonical_dataset_name(dataset):
    aliases = {
        "Amazon": "AmazonFashion",
        "Amazon_fashion": "AmazonFashion",
        "Grqc": "GrQc",
    }
    return aliases.get(str(dataset), str(dataset))


@dataclass(frozen=True)
class EventStream:
    candidate_edges: np.ndarray
    rewards: np.ndarray
    positive_arms: np.ndarray
    metadata: dict

    def __post_init__(self):
        edges = np.asarray(self.candidate_edges, dtype=np.int64)
        rewards = np.asarray(self.rewards, dtype=np.float64)
        arms = np.asarray(self.positive_arms, dtype=np.int64)
        if edges.ndim != 3 or edges.shape[2] != 2:
            raise ValueError("candidate_edges must have shape [rounds, arms, 2]")
        if rewards.shape != edges.shape[:2]:
            raise ValueError("rewards must have shape [rounds, arms]")
        if arms.shape != (len(edges),):
            raise ValueError("positive_arms must have shape [rounds]")
        if not np.all(rewards[np.arange(len(edges)), arms] == 1.0):
            raise ValueError("each positive_arm must point to reward 1")
        object.__setattr__(self, "candidate_edges", edges)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "positive_arms", arms)

    def __len__(self):
        return len(self.candidate_edges)

    @property
    def event_hash(self):
        digest = hashlib.sha256()
        digest.update(self.candidate_edges.tobytes(order="C"))
        digest.update(self.rewards.tobytes(order="C"))
        digest.update(self.positive_arms.tobytes(order="C"))
        digest.update(
            json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        return digest.hexdigest()

    def materialize(self, loader, round_index):
        edges = self.candidate_edges[round_index]
        rewards = self.rewards[round_index]
        arm = int(self.positive_arms[round_index])
        user, item = edges[arm]
        return (
            loader.context_from_edges(edges),
            edges,
            rewards,
            arm,
            int(user),
            int(item),
        )

    def save(self, path):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(prefix=".event_stream_", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(
                    handle,
                    candidate_edges=self.candidate_edges,
                    rewards=self.rewards,
                    positive_arms=self.positive_arms,
                    metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
                    event_hash=np.asarray(self.event_hash),
                )
            os.replace(temporary_path, path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            stream = cls(
                data["candidate_edges"],
                data["rewards"],
                data["positive_arms"],
                json.loads(str(data["metadata"])),
            )
            stored_hash = str(data["event_hash"])
        if stored_hash != stream.event_hash:
            raise ValueError(f"event stream hash mismatch: {path}")
        return stream


def generate_event_stream(loader, rounds, dataset, seed):
    dataset = canonical_dataset_name(dataset)
    candidate_edges = []
    rewards = []
    positive_arms = []
    for _ in range(int(rounds)):
        _, edges, reward, arm, _, _ = loader.step()
        candidate_edges.append(np.asarray(edges, dtype=np.int64))
        rewards.append(np.asarray(reward, dtype=np.float64))
        positive_arms.append(int(arm))
    return EventStream(
        np.asarray(candidate_edges),
        np.asarray(rewards),
        np.asarray(positive_arms),
        {
            "version": STREAM_VERSION,
            "dataset": str(dataset),
            "split": str(loader.split),
            "seed": int(seed),
            "rounds": int(rounds),
            "n_arm": int(loader.n_arm),
        },
    )


def load_or_generate_event_stream(loader, path, rounds, dataset, seed):
    dataset = canonical_dataset_name(dataset)
    if os.path.exists(path):
        stream = EventStream.load(path)
        expected = {
            "dataset": str(dataset),
            "split": str(loader.split),
            "seed": int(seed),
            "rounds": int(rounds),
            "n_arm": int(loader.n_arm),
        }
        if any(stream.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"cached event stream metadata mismatch: {path}")
        return stream
    stream = generate_event_stream(loader, rounds, dataset, seed)
    stream.save(path)
    return stream


def canonical_stream_path(root, dataset, split, seed, split_seed, n_arm, rounds):
    safe_dataset = canonical_dataset_name(dataset).replace(os.sep, "_")
    filename = (
        f"{split}_seed{int(seed)}_split{int(split_seed)}_"
        f"k{int(n_arm)}_T{int(rounds)}.npz"
    )
    return os.path.join(root, safe_dataset, filename)
