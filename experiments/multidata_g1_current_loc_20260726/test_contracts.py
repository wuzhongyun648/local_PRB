"""Static contracts for the six-dataset task matrix."""

from __future__ import annotations

import unittest

from configs import CONFIGS, INITIAL_DATASETS, SEEDS, build_queues


class QueueContracts(unittest.TestCase):
    def test_all_dataset_seed_pairs_appear_once(self):
        queues, _ = build_queues()
        tasks = [task for queue in queues for task in queue]
        expected = {
            (dataset, seed) for dataset in CONFIGS for seed in SEEDS
        }
        self.assertEqual(len(tasks), 60)
        self.assertEqual(len(set(tasks)), 60)
        self.assertEqual(set(tasks), expected)

    def test_first_wave_has_four_datasets(self):
        queues, _ = build_queues()
        self.assertEqual(
            tuple(queue[0][0] for queue in queues),
            INITIAL_DATASETS,
        )

    def test_estimated_load_is_balanced(self):
        _, loads = build_queues()
        self.assertLess(max(loads) - min(loads), 900)

    def test_formal_horizons(self):
        self.assertEqual(CONFIGS["MovieLens"].rounds, 10_000)
        self.assertEqual(CONFIGS["Facebook"].rounds, 10_000)
        for dataset in ("AmazonFashion", "Collab", "PPA", "Vessel"):
            self.assertEqual(CONFIGS[dataset].rounds, 5_000)


if __name__ == "__main__":
    unittest.main()
