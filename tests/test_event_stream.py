import os
import tempfile
import unittest

import numpy as np

from src.event_stream import EventStream, generate_event_stream
from src.load_data import load_movielen


def make_loader(seed):
    loader = load_movielen.__new__(load_movielen)
    loader.n_neg = 2
    loader.n_arm = 3
    loader.dim = 4
    loader.U = np.arange(400, dtype=np.float64).reshape(200, 2)
    loader.I = np.arange(400, 800, dtype=np.float64).reshape(200, 2)
    positives = np.array([[i, i + 1] for i in range(30)], dtype=np.int64)
    negatives = np.array([[i, i + 100] for i in range(30)], dtype=np.int64)
    loader._configure_splits(
        positives, negatives, "online", seed, split_seed=1729, undirected=False
    )
    return loader


class EventStreamContractTests(unittest.TestCase):
    def test_same_seed_produces_identical_stream_hash(self):
        first = generate_event_stream(make_loader(5), 20, "synthetic", 5)
        second = generate_event_stream(make_loader(5), 20, "synthetic", 5)
        self.assertEqual(first.event_hash, second.event_hash)
        np.testing.assert_array_equal(first.candidate_edges, second.candidate_edges)

    def test_round_trip_preserves_hash_and_materialization(self):
        loader = make_loader(11)
        stream = generate_event_stream(loader, 10, "synthetic", 11)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "events.npz")
            stream.save(path)
            restored = EventStream.load(path)
        self.assertEqual(stream.event_hash, restored.event_hash)
        expected = stream.materialize(loader, 3)
        actual = restored.materialize(loader, 3)
        for expected_value, actual_value in zip(expected, actual):
            np.testing.assert_array_equal(expected_value, actual_value)

    def test_stream_stores_edges_not_dense_contexts(self):
        stream = generate_event_stream(make_loader(3), 5, "synthetic", 3)
        self.assertEqual(stream.candidate_edges.shape, (5, 3, 2))
        self.assertFalse(hasattr(stream, "contexts"))


if __name__ == "__main__":
    unittest.main()
