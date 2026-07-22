import unittest

import numpy as np
import scipy.sparse as sp

from src.dynamic_appr import DynamicAPPR, appr_push
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
    def test_static_stats_do_not_change_result(self):
        graph, degree = path_graph()
        source = np.array([1.0, 0.0, 0.0])
        expected = appr(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4
        )
        actual, pushes = appr_with_stats(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(pushes, 0)

    def test_numba_and_python_push_are_identical(self):
        if not hasattr(appr_push, "py_func"):
            self.skipTest("Numba is not installed")
        graph, degree = path_graph()
        p_numba = np.zeros(3)
        p_python = np.zeros(3)
        r_numba = np.array([1.0, -0.25, 0.0])
        r_python = r_numba.copy()
        stats_numba = appr_push(
            graph.indptr,
            graph.indices,
            degree,
            p_numba,
            r_numba,
            0.85,
            1e-4,
        )
        stats_python = appr_push.py_func(
            graph.indptr,
            graph.indices,
            degree,
            p_python,
            r_python,
            0.85,
            1e-4,
        )
        np.testing.assert_array_equal(p_numba, p_python)
        np.testing.assert_array_equal(r_numba, r_python)
        self.assertEqual(stats_numba, stats_python)

    def test_dynamic_solver_exposes_cumulative_stats(self):
        graph, degree = path_graph()
        source = np.array([1.0, 0.0, 0.0])
        solver = DynamicAPPR(diagnostics=True)
        solver.solve(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4
        )
        self.assertTrue(solver.last_stats["cold_start"])
        self.assertGreater(solver.last_stats["pushes"], 0)
        self.assertIn("residual_l1", solver.last_stats)
        solver.solve(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4
        )
        self.assertEqual(solver.stats["solves"], 2)
        self.assertEqual(solver.stats["initializations"], 1)

    def test_numba_and_python_dynamic_sequences_match(self):
        if not hasattr(appr_push, "py_func"):
            self.skipTest("Numba is not installed")
        graph, degree = path_graph()
        compiled = DynamicAPPR(push_impl=appr_push)
        python = DynamicAPPR(push_impl=appr_push.py_func)
        sources = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.75, -0.25]),
            np.array([0.0, 0.0, 1.0]),
        )
        for source in sources:
            actual = compiled.solve(
                3,
                graph.indptr,
                graph.indices,
                degree,
                source,
                0.85,
                1e-4,
            )
            expected = python.solve(
                3,
                graph.indptr,
                graph.indices,
                degree,
                source,
                0.85,
                1e-4,
            )
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(compiled.r, python.r)
            self.assertEqual(compiled.last_stats, python.last_stats)


if __name__ == "__main__":
    unittest.main()
