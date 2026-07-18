import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import online_baselines_prb_style as common

DEVICE = common.DEVICE
COMMON_RUN_SINGLE_TASK = common.run_single_task
COMMON_BUILD_MODEL = common.build_model


class OriginalScalarNetwork(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class OriginalNeuralUCBDiag:
    """NeuralUCB learner_diag.py aligned implementation without CUDA-only assumptions."""

    def __init__(self, dim, lamdba=1, nu=1, hidden=100):
        self.func = OriginalScalarNetwork(dim, hidden_size=hidden).to(DEVICE)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,), device=DEVICE)
        self.nu = nu

    def select(self, context):
        tensor = common.to_tensor(context)
        mu = self.func(tensor)
        g_list = []
        sampled = []
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sampled.append(fx.item() + sigma.item())
        arm = int(np.argmax(sampled))
        self.U += g_list[arm] * g_list[arm]
        return arm

    def update(self, context, reward):
        self.context_list.append(torch.as_tensor(context.reshape(1, -1), dtype=torch.float32))
        self.reward.append(float(reward))

    def train(self, t=None):
        if not self.reward:
            return 0
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        index = np.arange(len(self.reward))
        np.random.shuffle(index)
        cnt = 0
        total_loss = 0.0
        while True:
            batch_loss = 0.0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                loss = (self.func(c.to(DEVICE)) - r) ** 2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                total_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return total_loss / 1000
            if batch_loss / len(self.reward) <= 1e-3:
                return batch_loss / len(self.reward)


class OriginalNeuralTSDiag:
    """NeuralTS learner_diag.py aligned implementation using per-arm gradients instead of backpack."""

    def __init__(self, dim, lamdba=1, nu=1, hidden=100, style="ts"):
        self.func = OriginalScalarNetwork(dim, hidden_size=hidden).to(DEVICE)
        self.context_list = None
        self.len = 0
        self.reward = None
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,), device=DEVICE)
        self.nu = nu
        self.style = style
        self.loss_func = nn.MSELoss()

    def _grad_for_output(self, output):
        self.func.zero_grad()
        output.backward(retain_graph=True)
        return torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])

    def select(self, context):
        tensor = common.to_tensor(context)
        mu = self.func(tensor).view(-1)
        g_list = []
        sigma_values = []
        for fx in mu:
            g = self._grad_for_output(fx)
            g_list.append(g)
            sigma_values.append(torch.sqrt(torch.sum(self.lamdba * self.nu * g * g / self.U)))
        sigma = torch.stack(sigma_values)
        if self.style == "ts":
            sample_r = torch.normal(mu, torch.clamp(sigma, min=1e-12))
        elif self.style == "ucb":
            sample_r = mu + sigma
        else:
            raise ValueError(f"Unknown NeuralTS style: {self.style}")
        arm = int(torch.argmax(sample_r).item())
        self.U += g_list[arm] * g_list[arm]
        return arm

    def update(self, context, reward):
        self.len += 1
        context_tensor = torch.as_tensor(context.reshape(1, -1), dtype=torch.float32, device=DEVICE)
        reward_tensor = torch.tensor([float(reward)], dtype=torch.float32, device=DEVICE)
        if self.context_list is None:
            self.context_list = context_tensor
            self.reward = reward_tensor
        else:
            self.context_list = torch.cat((self.context_list, context_tensor), dim=0)
            self.reward = torch.cat((self.reward, reward_tensor), dim=0)

    def train(self, t=None):
        if self.reward is None:
            return 0
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba / self.len)
        final_loss = 0.0
        for _ in range(100):
            self.func.zero_grad()
            optimizer.zero_grad()
            pred = self.func(self.context_list).view(-1)
            loss = self.loss_func(pred, self.reward)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        return final_loss


def build_model(method_name, bandit, args, dataset_name):
    if method_name == "NeuralUCB":
        return OriginalNeuralUCBDiag(bandit.dim, lamdba=args.lamdba, nu=args.nu, hidden=args.hidden)
    if method_name == "NeuralTS":
        return OriginalNeuralTSDiag(
            bandit.dim,
            lamdba=args.lamdba,
            nu=args.nu,
            hidden=args.hidden,
            style=args.ts_style,
        )
    return COMMON_BUILD_MODEL(method_name, bandit, args, dataset_name)


def run_single_task(dataset_name, method_name, run_id, args):
    original_builder = common.build_model
    common.build_model = build_model
    try:
        return COMMON_RUN_SINGLE_TASK(dataset_name, method_name, run_id, args)
    finally:
        common.build_model = original_builder


def parse_args():
    parser = argparse.ArgumentParser(description="Run original-code-aligned online baselines for LocPRB")
    parser.add_argument("--datasets", nargs="+", default=common.SUPPORTED_DATASETS)
    parser.add_argument("--methods", nargs="+", default=common.DEFAULT_METHODS)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--n_neg", default=common.DEFAULT_N_NEG, type=int)
    parser.add_argument("--lamdba", default=0.1, type=float)
    parser.add_argument("--nu", default=0.001, type=float)
    parser.add_argument("--epsilon", default=0.01, type=float)
    parser.add_argument("--hidden", default=100, type=int)
    parser.add_argument("--lr1", default=0.1, type=float)
    parser.add_argument("--lr2", default=0.01, type=float)
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--ts_style", default="ts", choices=["ts", "ucb"])
    return parser.parse_args()


def main():
    args = parse_args()
    common.run_single_task = run_single_task
    common.run_runner(args, runner_style="original_style", result_subdir="baselines_original_style")


if __name__ == "__main__":
    main()
