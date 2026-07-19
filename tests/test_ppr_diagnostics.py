import unittest

import numpy as np
import scipy.sparse as sp

from src.dynamic_appr import DynamicAPPR
from src.ppr_solver import appr, appr_with_stats


def path_graph():
    adjacency = sp.csr_matrix(
        np.array(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64
        )
    )
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return adjacency, degree


class PPRDiagnosticsTests(unittest.TestCase):
    def test_static_appr_reports_push_count_without_changing_result(self):
        graph, degree = path_graph()
        source = np.array([1.0, 0.0, 0.0])
        expected = appr(3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4)
        actual, pushes = appr_with_stats(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(pushes, 0)

    def test_dynamic_solver_exposes_cumulative_and_last_solve_stats(self):
        graph, degree = path_graph()
        source = np.array([1.0, 0.0, 0.0])
        solver = DynamicAPPR()
        solver.solve(3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4)
        self.assertTrue(solver.last_stats["cold_start"])
        self.assertGreater(solver.last_stats["pushes"], 0)
        solver.solve(3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4)
        self.assertEqual(solver.stats["solves"], 2)
        self.assertEqual(solver.stats["initializations"], 1)
        self.assertGreaterEqual(solver.stats["pushes"], solver.last_stats["pushes"])


if __name__ == "__main__":
    unittest.main()
