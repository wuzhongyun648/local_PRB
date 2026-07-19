import unittest
from unittest import mock

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
    return loader


def deterministic_choice(positive_arm):
    scalar_calls = 0

    def choice(_values, size=None, replace=True):
        nonlocal scalar_calls
        if size is not None:
            return np.arange(size, dtype=np.int64)
        scalar_calls += 1
        return positive_arm if scalar_calls == 1 else 0

    return choice


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
                with mock.patch.object(
                    load_data.np.random,
                    "choice",
                    side_effect=deterministic_choice(arm),
                ):
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
                    randint_values = iter((2, 3, 3, 4))
                    with mock.patch.object(
                        load_data.np.random,
                        "choice",
                        side_effect=deterministic_choice(arm),
                    ), mock.patch.object(
                        load_data.np.random,
                        "randint",
                        side_effect=lambda *_args, **_kwargs: next(randint_values),
                    ):
                        assert_step_contract(self, loader, loader.step())

    def test_vessel_step_contract(self):
        for arm in range(3):
            with self.subTest(arm=arm):
                loader = make_ogb_loader(load_data.load_ogb_vessel)
                loader.tree = FakeKDTree()
                with mock.patch.object(
                    load_data.np.random,
                    "choice",
                    side_effect=deterministic_choice(arm),
                ):
                    assert_step_contract(self, loader, loader.step())


class LoaderStreamContractTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_fixed_test_events_do_not_overlap_online_events(self):
        """Known P0: restoring the pre-test RNG reproduces the test events."""
        loader = make_small_loader(load_data.load_movielen)
        np.random.seed(7)
        pre_test_state = np.random.get_state()
        fixed_test_set = loader.testing_dataset()

        np.random.set_state(pre_test_state)
        online_events = [loader.step() for _ in range(100)]

        def edge_key(result):
            return tuple(np.asarray(result[1], dtype=np.int64).reshape(-1))

        test_keys = {edge_key(result) for result in fixed_test_set}
        online_keys = {edge_key(result) for result in online_events}
        self.assertTrue(test_keys.isdisjoint(online_keys))


if __name__ == "__main__":
    unittest.main()
