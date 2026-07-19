import unittest

import numpy as np
import scipy.sparse as sp

from src import load_data


SMALL_LOADER_CLASSES = (
    load_data.load_movielen,
    load_data.load_amazon_fashion,
    load_data.load_facebook,
    load_data.load_grqc,
)


def make_small_loader(loader_class):
    loader = loader_class.__new__(loader_class)
    loader.n_neg = 2
    loader.n_arm = 3
    loader.dim = 4
    loader.pos_index = np.array([[0, 1]], dtype=np.int64)
    loader.neg_index = np.array([[1, 2], [2, 3], [3, 0]], dtype=np.int64)
    loader.p_d = len(loader.pos_index)
    loader.n_d = len(loader.neg_index)
    loader.U = np.arange(10, dtype=np.float64).reshape(5, 2)
    if loader_class in (load_data.load_movielen, load_data.load_amazon_fashion):
        loader.I = np.arange(20, 30, dtype=np.float64).reshape(5, 2)
    loader.rng = np.random.default_rng(0)
    return loader


def make_split_small_loader(loader_class, seed=7):
    loader = loader_class.__new__(loader_class)
    loader.n_neg = 2
    loader.n_arm = 3
    loader.dim = 4
    loader.U = np.arange(400, dtype=np.float64).reshape(200, 2)
    if loader_class in (load_data.load_movielen, load_data.load_amazon_fashion):
        loader.I = np.arange(400, 800, dtype=np.float64).reshape(200, 2)
    positive_edges = np.array([[i, i + 1] for i in range(30)], dtype=np.int64)
    negative_edges = np.array([[i, i + 100] for i in range(30)], dtype=np.int64)
    loader._configure_splits(
        positive_edges,
        negative_edges,
        split="online",
        seed=seed,
        split_seed=1729,
        undirected=False,
    )
    return loader


class DeterministicRNG:
    def __init__(self, positive_arm, integer_values=()):
        self.positive_arm = positive_arm
        self.scalar_calls = 0
        self.integer_values = iter(integer_values)

    def choice(self, values, size=None, replace=True):
        del replace
        if size is not None:
            return np.arange(size, dtype=np.int64)
        self.scalar_calls += 1
        return self.positive_arm if self.scalar_calls == 1 else 0

    def integers(self, _high):
        return next(self.integer_values)


def assert_step_contract(testcase, loader, result):
    context, context_ind, reward, arm, user, item = result
    testcase.assertEqual(len(context), loader.n_arm)
    testcase.assertEqual(len(context_ind), loader.n_arm)
    testcase.assertEqual(len(reward), loader.n_arm)
    testcase.assertGreaterEqual(int(arm), 0)
    testcase.assertLess(int(arm), loader.n_arm)
    testcase.assertEqual(int(np.count_nonzero(reward)), 1)
    testcase.assertEqual(float(reward[arm]), 1.0)
    testcase.assertTupleEqual(
        tuple(np.asarray(context_ind[arm], dtype=np.int64)),
        (int(user), int(item)),
    )

    for row, (u, v) in zip(context, context_ind):
        right_features = loader.I[v] if hasattr(loader, "I") else loader.U[v]
        expected = np.concatenate((loader.U[u], right_features))
        np.testing.assert_array_equal(row, expected)


class SmallLoaderContractTests(unittest.TestCase):
    def assert_loader_all_arm_positions(self, loader_class):
        for arm in range(3):
            with self.subTest(loader=loader_class.__name__, arm=arm):
                loader = make_small_loader(loader_class)
                loader.rng = DeterministicRNG(arm)
                assert_step_contract(self, loader, loader.step())

    def test_movielens_step_contract(self):
        self.assert_loader_all_arm_positions(load_data.load_movielen)

    def test_amazon_step_contract(self):
        self.assert_loader_all_arm_positions(load_data.load_amazon_fashion)

    def test_facebook_step_contract(self):
        self.assert_loader_all_arm_positions(load_data.load_facebook)

    def test_grqc_step_contract(self):
        self.assert_loader_all_arm_positions(load_data.load_grqc)


def make_ogb_loader(loader_class):
    loader = loader_class.__new__(loader_class)
    loader.n_pos = 1
    loader.n_neg = 2
    loader.n_arm = 3
    loader.num_nodes = 5
    loader.pos_index = np.array([[0, 1]], dtype=np.int64)
    loader.p_d = len(loader.pos_index)
    loader.U = np.arange(10, dtype=np.float64).reshape(5, 2)
    loader.I = loader.U.copy()
    loader.node_feat = loader.U
    loader.adj_gt = sp.csr_matrix((5, 5), dtype=np.bool_)
    return loader


class FakeKDTree:
    def query(self, _features, k, return_distance=False):
        del k, return_distance
        return np.array([[0, 1, 2, 3, 4]], dtype=np.int64)


class OGBLoaderContractTests(unittest.TestCase):
    def test_collab_and_ppa_step_contract(self):
        for loader_class in (load_data.load_ogb_collab, load_data.load_ogb_ppa):
            for arm in range(3):
                with self.subTest(loader=loader_class.__name__, arm=arm):
                    loader = make_ogb_loader(loader_class)
                    loader.rng = DeterministicRNG(arm, (2, 3, 3, 4))
                    assert_step_contract(self, loader, loader.step())

    def test_vessel_step_contract(self):
        for arm in range(3):
            with self.subTest(arm=arm):
                loader = make_ogb_loader(load_data.load_ogb_vessel)
                loader.tree = FakeKDTree()
                loader.rng = DeterministicRNG(arm)
                assert_step_contract(self, loader, loader.step())


class LoaderStreamContractTests(unittest.TestCase):
    @staticmethod
    def edge_keys(results):
        return [
            tuple(np.asarray(result[1], dtype=np.int64).reshape(-1))
            for result in results
        ]

    @staticmethod
    def positive_edge_keys(results):
        return {(int(result[4]), int(result[5])) for result in results}

    def test_fixed_test_events_do_not_replay_online_prefix(self):
        from src.main import build_fixed_test_set

        loader = make_split_small_loader(load_data.load_movielen)
        fixed_test_set = build_fixed_test_set(loader, run_seed=7)

        online_events = [loader.step() for _ in range(100)]
        self.assertNotEqual(
            self.edge_keys(fixed_test_set),
            self.edge_keys(online_events),
        )

    def test_fixed_test_edges_are_disjoint_from_online_edges(self):
        from src.main import build_fixed_test_set

        loader = make_split_small_loader(load_data.load_movielen)
        fixed_test_set = build_fixed_test_set(loader, run_seed=7)

        online_events = [loader.step() for _ in range(100)]
        test_keys = self.positive_edge_keys(fixed_test_set)
        online_keys = self.positive_edge_keys(online_events)
        self.assertTrue(test_keys.isdisjoint(online_keys))

    def test_all_small_loader_edge_pools_are_pairwise_disjoint(self):
        for loader_class in SMALL_LOADER_CLASSES:
            with self.subTest(loader=loader_class.__name__):
                loader = make_split_small_loader(loader_class)
                for pool_name in ("_positive_splits", "_negative_splits"):
                    pools = getattr(loader, pool_name)
                    keys = {
                        name: {tuple(edge) for edge in edges}
                        for name, edges in pools.items()
                    }
                    self.assertTrue(keys["online"].isdisjoint(keys["validation"]))
                    self.assertTrue(keys["online"].isdisjoint(keys["test"]))
                    self.assertTrue(keys["validation"].isdisjoint(keys["test"]))

    def test_undirected_reverse_edges_stay_in_the_same_split(self):
        edges = np.array(
            [[0, 1], [1, 0], [2, 3], [4, 5], [6, 7]], dtype=np.int64
        )
        splits = load_data._partition_edges(edges, seed=3, undirected=True)
        memberships = {
            tuple(edge): name for name, values in splits.items() for edge in values
        }
        self.assertEqual(memberships[(0, 1)], memberships[(1, 0)])

    def test_model_numpy_randomness_does_not_change_loader_events(self):
        first = make_small_loader(load_data.load_movielen)
        second = make_small_loader(load_data.load_movielen)
        first.rng = np.random.default_rng(19)
        second.rng = np.random.default_rng(19)

        first_events = []
        second_events = []
        for _ in range(20):
            first_events.append(first.step())
            np.random.shuffle(np.arange(1000))
            np.random.random(1000)
            second_events.append(second.step())

        self.assertListEqual(
            self.edge_keys(first_events),
            self.edge_keys(second_events),
        )


if __name__ == "__main__":
    unittest.main()
