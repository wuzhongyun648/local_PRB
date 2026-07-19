import unittest

from benchmarks.benchmark_graph_ppr import run_benchmark


class GraphPPRMicrobenchmarkTests(unittest.TestCase):
    def test_tiny_protocol_runs_without_training(self):
        report = run_benchmark([30], degree=4, rounds=2, eps=1e-3, seed=1)
        result = report["results"][0]
        self.assertEqual(result["num_nodes"], 30)
        self.assertEqual(result["rounds"], 2)
        self.assertGreater(result["pushes"]["scratch_mean"], 0)
        self.assertIn("csr_rebuild", result["timing"])


if __name__ == "__main__":
    unittest.main()
