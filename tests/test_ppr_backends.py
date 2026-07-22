import unittest

import numpy as np
import scipy.sparse as sp

from src.dynamic_appr import DynamicAPPR, get_push_impl
from src.ppr_solver import NUMBA_AVAILABLE, appr_with_diagnostics, get_appr_kernel


def path_graph():
    adjacency = sp.csr_matrix(
        np.array(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64
        )
    )
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return adjacency, degree


class PPRBackendContractTests(unittest.TestCase):
    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba is not installed")
    def test_scratch_python_and_numba_are_identical(self):
        graph, degree = path_graph()
        source = np.array([1.0, -0.25, 0.5])
        compiled = appr_with_diagnostics(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
            backend="numba",
        )
        python = appr_with_diagnostics(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
            backend="python",
        )
        np.testing.assert_array_equal(compiled[0], python[0])
        self.assertEqual(compiled[1:], python[1:])

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba is not installed")
    def test_dynamic_python_and_numba_are_identical(self):
        graph, degree = path_graph()
        compiled = DynamicAPPR(push_impl=get_push_impl("numba"))
        python = DynamicAPPR(push_impl=get_push_impl("python"))
        for source in (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.75, -0.25]),
            np.array([0.0, 0.0, 1.0]),
        ):
            actual = compiled.solve(
                3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4
            )
            expected = python.solve(
                3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4
            )
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(compiled.r, python.r)
            self.assertEqual(compiled.last_stats, python.last_stats)
        self.assertEqual(compiled.stats, python.stats)

    def test_backend_resolvers_reject_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "Unknown PPR backend"):
            get_appr_kernel("invalid")
        with self.assertRaisesRegex(ValueError, "Unknown PPR backend"):
            get_push_impl("invalid")


if __name__ == "__main__":
    unittest.main()
