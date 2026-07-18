"""Run one controlled A0-A10 ablation without modifying the active runners."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


TESTA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTA_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _extract_ablation(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ablation", choices=[f"A{i}" for i in range(11)]
    )
    known, remaining = parser.parse_known_args(argv[1:])
    argv[:] = [argv[0], *remaining]
    ablation = known.ablation or os.environ.get("TESTA_ABLATION")
    if ablation not in {f"A{i}" for i in range(11)}:
        parser.error("--ablation A0...A10 is required")
    os.environ["TESTA_ABLATION"] = ablation
    return ablation


ABLATION = _extract_ablation(sys.argv)

from src import EENetClass as net_classes
from src import main as base_main
from src import ppr_solver
from src.EENet import EE_Net


RESULTS_ROOT = os.path.join(TESTA_DIR, "results", ABLATION)
base_main.RESULTS_DIR = RESULTS_ROOT


def _isolate_loader_randomness():
    """Keep the sampled candidate stream independent of model-side NumPy use."""
    classes = set(base_main.LOADER_BY_DATASET.values())
    for loader_class in classes:
        if getattr(loader_class, "_testa_rng_wrapped", False):
            continue
        original_init = loader_class.__init__
        original_step = loader_class.step

        def wrapped_init(self, *args, _init=original_init, **kwargs):
            _init(self, *args, **kwargs)
            self._testa_rng_state = np.random.get_state()

        def wrapped_step(self, _step=original_step):
            caller_state = np.random.get_state()
            np.random.set_state(self._testa_rng_state)
            try:
                result = _step(self)
                self._testa_rng_state = np.random.get_state()
                return result
            finally:
                np.random.set_state(caller_state)

        loader_class.__init__ = wrapped_init
        loader_class.step = wrapped_step
        loader_class._testa_rng_wrapped = True


class PaperExplorationNetwork(nn.Module):
    """A1: two-layer fully-connected ReLU exploration network."""

    def __init__(self, input_dim, _kernel_size=100, _stride=50, _channels=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(1)
        return self.fc2(self.activate(self.fc1(x)))


def _install_a1():
    net_classes.Network_exploration = PaperExplorationNetwork


def _install_a2():
    original_init = EE_Net.__init__

    def no_pool_init(self, dim, n_arm, pool_step_size, *args, **kwargs):
        return original_init(self, dim, n_arm, 1, *args, **kwargs)

    EE_Net.__init__ = no_pool_init


def _paper_initialize(module):
    weighted_layers = [
        layer for layer in module.modules() if isinstance(layer, (nn.Linear, nn.Conv1d))
    ]
    for index, layer in enumerate(weighted_layers):
        std = (1.0 / 100.0) ** 0.5 if index == len(weighted_layers) - 1 else (2.0 / 100.0) ** 0.5
        nn.init.normal_(layer.weight, mean=0.0, std=std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def _install_a3():
    original_init = EE_Net.__init__

    def gaussian_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _paper_initialize(self.f_1.func)
        _paper_initialize(self.f_2.func)

    EE_Net.__init__ = gaussian_init


def _predict_scores(self, context, t, *, decay=True, shifted=True):
    self.exploit_scores, self.grad_list = self.f_1.output_and_gradient(context)
    self.explore_scores = self.f_2.output(self.grad_list)
    self.ee_scores = np.concatenate(
        (self.exploit_scores, self.explore_scores), axis=1
    )
    weight = 1.0
    if decay and t > 500:
        weight = 1.0 / np.sqrt(t)
    exploration = self.explore_scores - 1.0 if shifted else self.explore_scores
    scores = self.exploit_scores + weight * exploration
    self.arm_select = int(np.argmax(scores))
    self.h = scores
    return self.arm_select, self.h


def _install_a4():
    def no_decay_predict(self, context, t):
        return _predict_scores(self, context, t, decay=False, shifted=True)

    EE_Net.predict = no_decay_predict


def _update_scores(self, context, reward, t, *, shifted=True, auxiliary=True):
    self.f_1.update(context[self.arm_select], reward)
    self.contexts.append(context[self.arm_select])
    self.rewards.append(reward)
    prediction = self.exploit_scores[self.arm_select][0]
    target = reward - prediction + (1.0 if shifted else 0.0)
    self.f_2.update(self.grad_list[self.arm_select], target)
    if auxiliary and t < 1000 and reward == 0:
        for index, gradient in enumerate(self.grad_list):
            if index != self.arm_select:
                self.f_2.update(gradient, 1.2)


def _install_a5():
    def no_shift_predict(self, context, t):
        return _predict_scores(self, context, t, decay=True, shifted=False)

    def no_shift_update(self, context, reward, t):
        return _update_scores(
            self, context, reward, t, shifted=False, auxiliary=True
        )

    EE_Net.predict = no_shift_predict
    EE_Net.update = no_shift_update


def _install_a6():
    def no_auxiliary_update(self, context, reward, t):
        return _update_scores(
            self, context, reward, t, shifted=True, auxiliary=False
        )

    EE_Net.update = no_auxiliary_update


def _normalize_context(context):
    context = np.asarray(context)
    norms = np.linalg.norm(context, axis=1, keepdims=True)
    return context / np.maximum(norms, 1e-12)


def _install_a7():
    original_predict = EE_Net.predict
    original_update = EE_Net.update

    def normalized_predict(self, context, t):
        return original_predict(self, _normalize_context(context), t)

    def normalized_update(self, context, reward, t):
        return original_update(self, _normalize_context(context), reward, t)

    EE_Net.predict = normalized_predict
    EE_Net.update = normalized_update


def _normalize_source(source):
    norm = np.sum(np.abs(source))
    return source if norm == 0.0 else source / norm


def _install_a8():
    original_appr = ppr_solver.appr
    original_power = ppr_solver.power_iteration

    def normalized_appr(num_nodes, indptr, indices, degree, source, alpha, eps):
        return original_appr(
            num_nodes,
            indptr,
            indices,
            degree,
            _normalize_source(source),
            alpha,
            eps,
        )

    def normalized_power(P, alpha, source, iterations):
        return original_power(P, alpha, _normalize_source(source), iterations)

    ppr_solver.appr = normalized_appr
    ppr_solver.power_iteration = normalized_power


def _latest_exploitation_step(self):
    optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
    context = self.context_list[-1].to(net_classes.device)
    reward = self.reward[-1]
    optimizer.zero_grad()
    loss = (self.func(context) - reward) ** 2
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _latest_exploration_step(self):
    optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
    index = getattr(self, "_testa_latest_selected_index", len(self.reward) - 1)
    context = self.context_list[index].to(net_classes.device)
    reward = self.reward[index]
    optimizer.zero_grad()
    loss = (self.func(context) - reward) ** 2
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _install_a9():
    original_update = EE_Net.update

    def track_selected_sample(self, context, reward, t):
        selected_index = len(self.f_2.reward)
        original_update(self, context, reward, t)
        self.f_2._testa_latest_selected_index = selected_index

    base_main.TRAIN_EVERY_BEFORE_2000 = 1
    base_main.TRAIN_EVERY_AFTER_2000 = 1
    EE_Net.update = track_selected_sample
    net_classes.Exploitation.train = _latest_exploitation_step
    net_classes.Exploration.train = _latest_exploration_step


INSTALLERS = {
    "A1": _install_a1,
    "A2": _install_a2,
    "A3": _install_a3,
    "A4": _install_a4,
    "A5": _install_a5,
    "A6": _install_a6,
    "A7": _install_a7,
    "A8": _install_a8,
    "A9": _install_a9,
}


_isolate_loader_randomness()
if ABLATION in INSTALLERS:
    INSTALLERS[ABLATION]()


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    if ABLATION == "A10":
        from src import main_dyn

        main_dyn.main()
    else:
        base_main.main()


if __name__ == "__main__":
    main()
