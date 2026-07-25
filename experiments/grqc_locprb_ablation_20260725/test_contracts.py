import unittest
from pathlib import Path

import numpy as np

from events import (
    build_event_tape,
    canonical_edge,
    identical_event_count,
    load_grqc_data,
    positive_edge_overlap_count,
    validate_tape,
)
from runner import transform_source
from variants import VARIANTS


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "GrQc"


class EventContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_grqc_data(DATA_DIR)

    def test_current_files_have_undirected_warm_online_overlap(self):
        insert_positive = {
            canonical_edge(u, v)
            for u, v, label in self.data.insert_entry
            if label == 1 and u != v
        }
        self.assertEqual(len(insert_positive & self.data.warm_undirected), 2586)

    def test_all_fixed_contracts(self):
        variant = VARIANTS["sf_all_fixed"]
        tape = build_event_tape(self.data, variant, rounds=100, seed=7)
        contract = validate_tape(tape, variant, self.data)
        self.assertEqual(contract["shared_serving_violations"], 0)
        self.assertEqual(contract["warm_online_overlap"], 0)
        self.assertEqual(contract["repeated_positive_events"], 0)

    def test_legacy_eval_replays_online_prefix(self):
        variant = VARIANTS["s0_legacy"]
        online = build_event_tape(self.data, variant, rounds=100, seed=9)
        evaluation = build_event_tape(self.data, variant, rounds=100, seed=9)
        self.assertEqual(identical_event_count(online, evaluation), 100)

    def test_independent_eval_does_not_replay_online_prefix(self):
        variant = VARIANTS["sf_all_fixed"]
        online = build_event_tape(self.data, variant, rounds=100, seed=9)
        evaluation = build_event_tape(
            self.data,
            variant,
            rounds=100,
            seed=1_000_012,
            excluded_positive_edges={
                canonical_edge(u, v) for u, v in online.positive_edge
            },
        )
        self.assertEqual(identical_event_count(online, evaluation), 0)
        self.assertEqual(
            positive_edge_overlap_count(online, evaluation), 0
        )

    def test_without_replacement_is_undirected(self):
        variant = VARIANTS["s4_no_positive_replacement"]
        tape = build_event_tape(self.data, variant, rounds=1000, seed=11)
        contract = validate_tape(tape, variant, self.data)
        self.assertEqual(contract["repeated_positive_events"], 0)

    def test_raw_and_explicit_control_match(self):
        raw = np.array([-2.0, 1.0, 3.0])
        exploit = np.array([0.0, 0.0, 0.0])
        self.assertTrue(
            np.array_equal(
                transform_source(raw, exploit, "raw"),
                transform_source(raw, exploit, "exploit_explore"),
            )
        )

    def test_nonnegative_source_transforms_are_probabilities(self):
        raw = np.array([-2.0, 1.0, 3.0])
        exploit = np.array([0.0, 0.0, 0.0])
        for mode in ("clip_l1", "softmax"):
            value = transform_source(raw, exploit, mode)
            self.assertTrue(np.all(value >= 0))
            self.assertAlmostEqual(float(np.sum(value)), 1.0)


if __name__ == "__main__":
    unittest.main()
