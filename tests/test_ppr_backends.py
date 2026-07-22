import unittest

import numpy as np
import scipy.sparse as sp

from src.dynamic_appr import DynamicAPPR, get_push_impl
from src.ppr_solver import (
    NUMBA_AVAILABLE,
    appr_with_diagnostics,
    get_appr_kernel,
    get_reuse_queue_kernel,
    get_scratch_into_kernel,
    get_timing_kernel,
)


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

    def test_sparse_scratch_seed_matches_full_scan(self):
        graph, degree = path_graph()
        source = np.array([1.0, 0.0, -0.25])
        full = appr_with_diagnostics(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
            backend="python",
        )
        sparse = appr_with_diagnostics(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
            backend="python", seed_nodes=np.array([0, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(full[0], sparse[0])
        self.assertEqual(full[1:], sparse[1:])

    def test_uninstrumented_kernels_match_diagnostic_output(self):
        graph, degree = path_graph()
        source = np.array([1.0, 0.0, -0.25])
        support = np.array([0, 2], dtype=np.int64)
        expected = appr_with_diagnostics(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
            backend="python", seed_nodes=support,
        )[0]
        timed = get_timing_kernel("python")(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
            support,
        )
        reused = get_reuse_queue_kernel("python")(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
            support, np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.bool_),
        )
        np.testing.assert_array_equal(timed[0], expected)
        np.testing.assert_array_equal(reused[0], expected)
        self.assertEqual(timed[1:4], (0, 0, 0))
        self.assertEqual(reused[1:4], (0, 0, 0))

    def test_sparse_dynamic_hints_match_legacy_scans(self):
        graph, degree = path_graph()
        legacy = DynamicAPPR(push_impl=get_push_impl("python"))
        sparse = DynamicAPPR(push_impl=get_push_impl("python"))
        for source, support in (
            (np.array([1.0, 0.0, 0.0]), np.array([0], dtype=np.int64)),
            (np.array([0.0, 0.75, -0.25]), np.array([1, 2], dtype=np.int64)),
            (np.array([0.0, 0.0, 1.0]), np.array([2], dtype=np.int64)),
        ):
            expected = legacy.solve(
                3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4
            )
            actual = sparse.solve(
                3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
                source_indices=support,
                changed_nodes_hint=np.empty(0, dtype=np.int64),
            )
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(sparse.r, legacy.r)

        triangle = graph.copy().tolil()
        triangle[0, 2] = 1.0
        triangle[2, 0] = 1.0
        triangle = triangle.tocsr()
        triangle_degree = np.asarray(triangle.sum(axis=1)).reshape(-1)
        source = np.array([0.0, 1.0, 0.0])
        expected = legacy.solve(
            3, triangle.indptr, triangle.indices, triangle_degree,
            source, 0.85, 1e-4,
        )
        actual = sparse.solve(
            3, triangle.indptr, triangle.indices, triangle_degree,
            source, 0.85, 1e-4,
            source_indices=np.array([1], dtype=np.int64),
            changed_nodes_hint=np.array([0, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(sparse.r, legacy.r)

    def test_adaptive_reset_matches_fresh_scratch_after_insert(self):
        graph, degree = path_graph()
        solver = DynamicAPPR(
            push_impl=get_push_impl("python"),
            scratch_impl=get_scratch_into_kernel("python"),
        )
        solver.solve(
            3, graph.indptr, graph.indices, degree,
            np.array([1.0, 0.0, 0.0]), 0.85, 1e-4,
            source_indices=np.array([0], dtype=np.int64),
            changed_nodes_hint=np.empty(0, dtype=np.int64),
        )

        triangle = graph.copy().tolil()
        triangle[0, 2] = 1.0
        triangle[2, 0] = 1.0
        triangle = triangle.tocsr()
        triangle_degree = np.asarray(triangle.sum(axis=1)).reshape(-1)
        source = np.array([0.0, 0.0, 1.0])
        actual = solver.solve(
            3, triangle.indptr, triangle.indices, triangle_degree,
            source, 0.85, 1e-4,
            source_indices=np.array([2], dtype=np.int64),
            changed_nodes_hint=np.array([0, 2], dtype=np.int64),
        ).copy()
        expected = appr_with_diagnostics(
            3, triangle.indptr, triangle.indices, triangle_degree,
            source, 0.85, 1e-4, backend="python",
            seed_nodes=np.array([2], dtype=np.int64),
        )[0]
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(solver.last_stats["adaptive_reset"])
        self.assertEqual(solver.stats["adaptive_resets"], 1)
        self.assertEqual(solver.stats["insert_updates"], 0)
        self.assertFalse(np.any(solver.queued))


if __name__ == "__main__":
    unittest.main()
