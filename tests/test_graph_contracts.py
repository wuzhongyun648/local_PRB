import unittest
from unittest import mock
import os
import tempfile

import numpy as np
import scipy.sparse as sp

from src import utils


GRAPH_CLASSES = (
    utils.MovieLens,
    utils.Amazon_fashion,
    utils.Facebook,
    utils.Grqc,
    utils.PPA,
    utils.Collab,
    utils.Vessel,
)


def make_empty_graph(graph_class):
    manager = graph_class("synthetic")
    manager.A = sp.lil_matrix((5, 5), dtype=np.float64)
    manager.P = sp.csc_matrix((5, 5), dtype=np.float64)
    manager.degree = np.zeros(5, dtype=np.float64)
    manager.num_nodes = 5
    manager.num_edges = 0
    if graph_class in (utils.MovieLens, utils.Amazon_fashion):
        manager.num_users = 2
        return manager, (1, 0), (0, 3)
    return manager, (0, 3), (0, 3)


def assert_graph_state_contract(testcase, manager, edge):
    u, v = edge
    testcase.assertEqual(float(manager.A[u, v]), 1.0)
    testcase.assertEqual(float(manager.A[v, u]), 1.0)

    actual_degree = np.asarray(manager.A.sum(axis=1)).reshape(-1)
    stored_degree = np.asarray(manager.degree).reshape(-1)
    np.testing.assert_array_equal(stored_degree, actual_degree)
    testcase.assertEqual(int(manager.num_edges), int(manager.A.nnz // 2))

    column_sums = np.asarray(manager.P.sum(axis=0)).reshape(-1)
    non_isolated = actual_degree > 0
    np.testing.assert_allclose(column_sums[non_isolated], 1.0)
    testcase.assertTrue(sp.isspmatrix_csc(manager.P))


class GraphStateContractTests(unittest.TestCase):
    def test_movielens_initial_graph_uses_only_positive_labels(self):
        entries = np.array(
            [
                [0, 0, 1],
                [0, 1, -1],
                [1, 0, 0],
            ],
            dtype=np.int64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "movielens.npy")
            np.save(path, entries)
            manager = utils.MovieLens(path)
            manager.load(num_users=2, num_items=2)

        self.assertEqual(float(manager.A[0, 2]), 1.0)
        self.assertEqual(float(manager.A[2, 0]), 1.0)
        self.assertEqual(float(manager.A[0, 3]), 0.0)
        self.assertEqual(float(manager.A[3, 0]), 0.0)
        self.assertEqual(float(manager.A[1, 2]), 0.0)
        self.assertEqual(manager.num_edges, 1)

    def assert_update_contract(self, graph_class):
        manager, update_args, canonical_edge = make_empty_graph(graph_class)
        manager.update(*update_args)
        assert_graph_state_contract(self, manager, canonical_edge)

        adjacency_before = manager.A.toarray().copy()
        degree_before = np.asarray(manager.degree).copy()
        num_edges_before = manager.num_edges
        manager.update(*update_args)
        np.testing.assert_array_equal(manager.A.toarray(), adjacency_before)
        np.testing.assert_array_equal(np.asarray(manager.degree), degree_before)
        self.assertEqual(manager.num_edges, num_edges_before)

    def test_movielens_graph_state_contract(self):
        self.assert_update_contract(utils.MovieLens)

    def test_amazon_graph_state_contract(self):
        self.assert_update_contract(utils.Amazon_fashion)

    def test_facebook_graph_state_contract(self):
        self.assert_update_contract(utils.Facebook)

    def test_grqc_graph_state_contract(self):
        self.assert_update_contract(utils.Grqc)

    def test_ppa_graph_state_contract(self):
        self.assert_update_contract(utils.PPA)

    def test_collab_graph_state_contract(self):
        self.assert_update_contract(utils.Collab)

    def test_vessel_graph_state_contract(self):
        self.assert_update_contract(utils.Vessel)

    def test_updates_report_whether_an_edge_was_added(self):
        for graph_class in GRAPH_CLASSES:
            with self.subTest(graph=graph_class.__name__):
                manager, update_args, _ = make_empty_graph(graph_class)
                self.assertIs(manager.update(*update_args), True)
                self.assertIs(manager.update(*update_args), False)

    def test_new_edges_change_only_two_endpoint_degrees(self):
        for graph_class in GRAPH_CLASSES:
            with self.subTest(graph=graph_class.__name__):
                manager, update_args, canonical_edge = make_empty_graph(graph_class)
                degree_before = np.asarray(manager.degree).reshape(-1).copy()
                nnz_before = manager.A.nnz
                self.assertTrue(manager.update(*update_args))

                degree_delta = np.asarray(manager.degree).reshape(-1) - degree_before
                changed = set(np.flatnonzero(degree_delta))
                self.assertSetEqual(changed, set(canonical_edge))
                np.testing.assert_array_equal(
                    degree_delta[list(canonical_edge)],
                    np.ones(2),
                )
                self.assertEqual(manager.A.nnz - nnz_before, 2)

    def test_main_skips_csr_rebuild_for_existing_edges(self):
        from src.main import apply_graph_update

        manager = mock.Mock()
        manager.update.return_value = False
        edge_added, updated_csr = apply_graph_update(manager, 1, 2)
        self.assertFalse(edge_added)
        self.assertIsNone(updated_csr)
        manager.P.tocsr.assert_not_called()

    def test_main_rebuilds_csr_for_new_edges(self):
        from src.main import apply_graph_update

        manager = mock.Mock()
        manager.update.return_value = True
        manager.P.tocsr.return_value = "updated-csr"
        edge_added, updated_csr = apply_graph_update(manager, 1, 2)
        self.assertTrue(edge_added)
        self.assertEqual(updated_csr, "updated-csr")
        manager.P.tocsr.assert_called_once_with()


class ActiveRegistryContractTests(unittest.TestCase):
    def test_selected_graph_edge_comes_from_selected_candidate(self):
        from src.main import selected_graph_edge

        candidates = np.array([[9, 2], [4, 7], [3, 1]], dtype=np.int64)
        self.assertTupleEqual(selected_graph_edge(candidates, 1), (4, 7))
        self.assertTupleEqual(
            selected_graph_edge(candidates, 1, target_offset=100),
            (4, 107),
        )

    def test_main_registers_exactly_the_active_graphs(self):
        from src import main

        expected = {
            "MovieLens",
            "Amazon_fashion",
            "Facebook",
            "Grqc",
            "PPA",
            "Collab",
            "Vessel",
        }
        self.assertSetEqual(set(main.LOADER_BY_DATASET), expected)
        self.assertSetEqual(set(main.GRAPH_BY_DATASET), expected)

    def test_baseline_runner_registers_exactly_the_active_datasets(self):
        import online_baselines_run as baselines

        expected = {
            "MovieLens",
            "AmazonFashion",
            "Facebook",
            "GrQc",
            "Collab",
            "PPA",
            "Vessel",
        }
        self.assertSetEqual(set(baselines.SUPPORTED_DATASETS), expected)
        normalized_registry = {
            baselines.normalize_dataset_name(name)
            for name in baselines.DATASET_LOADERS
        }
        self.assertSetEqual(normalized_registry, expected)


if __name__ == "__main__":
    unittest.main()
