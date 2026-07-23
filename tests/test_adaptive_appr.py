import unittest

import numpy as np
import scipy.sparse as sp

from src.adaptive_appr import (
    DYNAMIC,
    SCRATCH,
    AdaptiveAPPR,
    NUMBA_AVAILABLE,
    predict_adaptive_branch,
)
from src.ppr_solver import appr_with_diagnostics


def path_graph():
    graph = sp.csr_matrix(
        np.array(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64
        )
    )
    degree = np.diff(graph.indptr).astype(np.int64)
    return graph, degree


class AdaptiveAPPRTests(unittest.TestCase):
    def assert_linear_invariant(
        self, solver, graph, degree, source, alpha=0.85
    ):
        transition = graph.T @ sp.diags(1.0 / degree)
        error = (
            solver.p
            - alpha * (transition @ solver.p)
            + (1.0 - alpha) * solver.r
            - (1.0 - alpha) * source
        )
        self.assertLess(float(np.max(np.abs(error))), 1e-12)

    def test_forced_scratch_matches_reference_and_reuses_workspace(self):
        graph, degree = path_graph()
        solver = AdaptiveAPPR("python")
        source = np.array([1.0, 0.0, -0.25])
        support = np.array([0, 2], dtype=np.int64)
        actual = solver.solve(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
            support, np.empty(0, dtype=np.int64), force_scratch=True,
        ).copy()
        expected = appr_with_diagnostics(
            3, graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
            backend="python", seed_nodes=support,
        )[0]
        np.testing.assert_array_equal(actual, expected)
        first_p = solver.p
        solver.solve(
            3, graph.indptr, graph.indices, degree,
            np.array([0.0, 1.0, 0.0]), 0.85, 1e-4,
            np.array([1], dtype=np.int64), np.empty(0, dtype=np.int64),
            force_scratch=True,
        )
        self.assertIs(solver.p, first_p)
        self.assertFalse(np.any(solver.queued))
        self.assertEqual(solver.stats["scratch_resets"], 2)

    def test_prediction_is_read_only_and_uses_exact_candidate_residual(self):
        degree = np.array([2, 3, 1], dtype=np.int64)
        p = np.zeros(3)
        residual = np.array([0.01, -0.02, 0.0])
        previous_source = np.array([1.0, 0.0, 0.0])
        source = np.array([0.0, 0.5, 0.0])
        before = tuple(
            value.copy()
            for value in (degree, p, residual, previous_source, source)
        )
        predictor = getattr(
            predict_adaptive_branch, "py_func", predict_adaptive_branch
        )
        result = predictor(
            degree, p, residual, previous_source, source,
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.empty(0, dtype=np.int64),
            0.85, 0.1, True, 0,
        )
        # Candidate residuals are exactly -0.99 at node 0 and 0.48 at node 1.
        expected_dynamic = 3 * 2 + (4.0 + 2.0) + (4.0 + 3.0)
        expected_scratch = 2 * 3 + 3 * 1 + 1 + (4.0 + 3.0)
        self.assertAlmostEqual(result[1], expected_dynamic)
        self.assertAlmostEqual(result[2], expected_scratch)
        self.assertEqual(result[0], SCRATCH)
        for actual, expected in zip(
            (degree, p, residual, previous_source, source), before
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_diagnostics_do_not_mutate_live_state(self):
        graph, degree = path_graph()
        solver = AdaptiveAPPR("python")
        solver.solve(
            3, graph.indptr, graph.indices, degree,
            np.array([1.0, 0.0, 0.0]), 0.85, 1e-4,
            np.array([0], dtype=np.int64), np.empty(0, dtype=np.int64),
        )
        source = np.array([0.0, 1.0, 0.0])
        prediction = solver.predict(
            3, graph.indptr, degree, source, 0.85, 1e-4,
            np.array([1], dtype=np.int64), np.empty(0, dtype=np.int64),
        )
        before = (
            solver.p.copy(), solver.r.copy(), solver.previous_source.copy(),
            solver.previous_support.copy(),
        )
        outputs = solver.diagnose_branches(
            graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
            np.array([1], dtype=np.int64), prediction,
        )
        self.assertEqual(set(outputs), {"dynamic", "scratch"})
        for actual, expected in zip(
            (solver.p, solver.r, solver.previous_source,
             solver.previous_support),
            before,
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_edge_insertion_sequence_remains_reversible(self):
        graph, degree = path_graph()
        solver = AdaptiveAPPR("python")
        solver.solve(
            3, graph.indptr, graph.indices, degree,
            np.array([1.0, 0.0, 0.0]), 0.85, 1e-4,
            np.array([0], dtype=np.int64), np.empty(0, dtype=np.int64),
        )
        triangle = graph.copy().tolil()
        triangle[0, 2] = 1.0
        triangle[2, 0] = 1.0
        triangle = triangle.tocsr()
        triangle_degree = np.diff(triangle.indptr).astype(np.int64)
        source = np.array([0.0, 0.0, 1.0])
        prediction = solver.predict(
            3, triangle.indptr, triangle_degree, source, 0.85, 1e-4,
            np.array([2], dtype=np.int64), np.array([0, 2], dtype=np.int64),
        )
        # Exercise the exact DYN transition regardless of the cost choice.
        prediction["mode"] = DYNAMIC
        actual = solver.execute(
            triangle.indptr, triangle.indices, triangle_degree, source,
            0.85, 1e-4, np.array([2], dtype=np.int64), prediction,
        ).copy()
        expected = appr_with_diagnostics(
            3, triangle.indptr, triangle.indices, triangle_degree, source,
            0.85, 1e-4, backend="python",
            seed_nodes=np.array([2], dtype=np.int64),
        )[0]
        self.assertLess(float(np.sum(np.abs(actual - expected))), 0.01)
        self.assertFalse(np.any(solver.queued))
        reset_prediction = solver.scratch_prediction(3, triangle.indptr)
        solver.execute(
            triangle.indptr, triangle.indices, triangle_degree, source,
            0.85, 1e-4, np.array([2], dtype=np.int64), reset_prediction,
        )
        np.testing.assert_array_equal(solver.p, expected)

    def test_directed_edge_insertion_is_supported(self):
        graph, degree = path_graph()
        solver = AdaptiveAPPR("python")
        solver.solve(
            3, graph.indptr, graph.indices, degree,
            np.array([1.0, 0.0, 0.0]), 0.85, 1e-4,
            np.array([0], dtype=np.int64), np.empty(0, dtype=np.int64),
        )
        directed = graph.copy().tolil()
        directed[0, 2] = 1.0
        directed = directed.tocsr()
        directed_degree = np.diff(directed.indptr).astype(np.int64)
        source = np.array([0.0, 0.0, 1.0])
        prediction = solver.predict(
            3, directed.indptr, directed_degree, source, 0.85, 1e-4,
            np.array([2], dtype=np.int64), np.array([0, 2], dtype=np.int64),
        )
        self.assertEqual(prediction["insertion_kind"], 1)
        np.testing.assert_array_equal(
            prediction["changed_nodes"], np.array([0, 2], dtype=np.int64)
        )
        prediction["mode"] = DYNAMIC
        actual = solver.execute(
            directed.indptr, directed.indices, directed_degree, source,
            0.85, 1e-4, np.array([2], dtype=np.int64), prediction,
        ).copy()
        expected = appr_with_diagnostics(
            3, directed.indptr, directed.indices, directed_degree, source,
            0.85, 1e-4, backend="python",
            seed_nodes=np.array([2], dtype=np.int64),
        )[0]
        self.assertLess(float(np.sum(np.abs(actual - expected))), 0.01)
        self.assert_linear_invariant(
            solver, directed, directed_degree, source
        )

    def test_scratch_dynamic_scratch_dynamic_sequence(self):
        graph, degree = path_graph()
        solver = AdaptiveAPPR("python")
        sources = (
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        supports = (
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([2], dtype=np.int64),
        )
        modes = []
        for source, support in zip(sources, supports):
            prediction = solver.predict(
                3, graph.indptr, degree, source, 0.85, 1e-4, support,
                np.empty(0, dtype=np.int64),
            )
            modes.append(prediction["mode"])
            solver.execute(
                graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
                support, prediction,
            )
            self.assert_linear_invariant(solver, graph, degree, source)
            self.assertFalse(np.any(solver.queued))
        self.assertEqual(modes, [SCRATCH, DYNAMIC, SCRATCH, DYNAMIC])

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba is not installed")
    def test_python_and_numba_sequences_are_identical(self):
        graph, degree = path_graph()
        compiled = AdaptiveAPPR("numba")
        python = AdaptiveAPPR("python")
        for source, support in (
            (np.array([1.0, 0.0, 0.0]), np.array([0], dtype=np.int64)),
            (np.array([0.0, 0.75, -0.25]), np.array([1, 2], dtype=np.int64)),
            (np.array([0.0, 0.0, 1.0]), np.array([2], dtype=np.int64)),
        ):
            pred_numba = compiled.predict(
                3, graph.indptr, degree, source, 0.85, 1e-4, support,
                np.empty(0, dtype=np.int64),
            )
            pred_python = python.predict(
                3, graph.indptr, degree, source, 0.85, 1e-4, support,
                np.empty(0, dtype=np.int64),
            )
            for key in pred_numba:
                if isinstance(pred_numba[key], np.ndarray):
                    np.testing.assert_array_equal(
                        pred_numba[key], pred_python[key]
                    )
                else:
                    self.assertEqual(pred_numba[key], pred_python[key])
            actual = compiled.execute(
                graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
                support, pred_numba,
            )
            expected = python.execute(
                graph.indptr, graph.indices, degree, source, 0.85, 1e-4,
                support, pred_python,
            )
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(compiled.r, python.r)


if __name__ == "__main__":
    unittest.main()
