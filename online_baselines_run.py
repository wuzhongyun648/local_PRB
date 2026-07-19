import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.baselines_new.EENet import EE_Net
from src.experiment_configs import RESULTS_DIR
from src.load_data import (
    load_amazon_fashion,
    load_facebook,
    load_grqc,
    load_movielen,
    load_ogb_collab,
    load_ogb_ppa,
    load_ogb_vessel,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SUPPORTED_DATASETS = ["MovieLens", "AmazonFashion", "Facebook", "GrQc", "Collab", "PPA", "Vessel"]
DEFAULT_METHODS = ["EE-Net", "NeuralUCB", "NeuralTS", "NeuralGreedy", "LinUCB", "KernelUCB"]
TRAIN_EVERY_BEFORE_2000 = 50
TRAIN_EVERY_AFTER_2000 = 100
DEFAULT_N_NEG = 9

DATASET_LOADERS = {
    "MovieLens": load_movielen,
    "AmazonFashion": load_amazon_fashion,
    "Amazon": load_amazon_fashion,
    "Facebook": load_facebook,
    "GrQc": load_grqc,
    "Grqc": load_grqc,
    "Collab": load_ogb_collab,
    "PPA": load_ogb_ppa,
    "Vessel": load_ogb_vessel,
}


def to_tensor(array):
    return torch.as_tensor(array, dtype=torch.float32, device=DEVICE)


class ScalarNetwork(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class PRBNeuralUCBDiag:
    """PRB/EE-Net released NeuralUCB baseline with device/import compatibility."""

    def __init__(self, dim, lamdba=1, nu=1, hidden=100):
        self.func = ScalarNetwork(dim, hidden_size=hidden).to(DEVICE)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,), device=DEVICE)
        self.nu = nu
        self.lr = 0.01

    def select(self, context):
        tensor = to_tensor(context)
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
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
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
                if cnt >= 2000:
                    return total_loss / 2000
            if batch_loss / len(self.reward) <= 1e-3:
                return batch_loss / len(self.reward)


def flatten_grad(tensor_tuple):
    flat = torch.tensor([], device=DEVICE)
    for element in tensor_tuple:
        flat = torch.cat([flat, element.to(DEVICE).flatten()])
    return flat


class PRBNeuralTS:
    """PRB/EE-Net released NeuralTS baseline with device/import compatibility."""

    def __init__(self, dim, n_arm, m=100, sigma=1, nu=0.15):
        self.K = n_arm
        self.nu = nu
        self.sigma = sigma
        self.m = m
        self.d = dim
        self.estimator = ScalarNetwork(self.d, hidden_size=m).to(DEVICE)
        self.optimizer = optim.SGD(self.estimator.parameters(), lr=0.01)
        self.current_loss = 0
        self.t = 1
        self.total_param = sum(p.numel() for p in self.estimator.parameters() if p.requires_grad)
        self.Design = torch.ones((self.total_param,), device=DEVICE)
        self.rewards = []
        self.context_list = []

    def select(self, context):
        features = to_tensor(context)
        estimated_rewards = []
        for k in range(self.K):
            f = self.estimator(features[k])
            g = torch.autograd.grad(outputs=f, inputs=self.estimator.parameters())
            g = flatten_grad(g).detach()
            sigma2 = g * g / self.Design
            sigma = torch.sqrt(torch.sum(sigma2)) * self.nu
            sample = torch.normal(
                mean=torch.tensor(float(f.item()), device=DEVICE),
                std=torch.clamp(sigma, min=1e-12),
            )
            estimated_rewards.append(sample.item())
        return int(np.argmax(estimated_rewards))

    def update(self, context, reward):
        self.context_list.append(torch.as_tensor(context.reshape(1, -1), dtype=torch.float32))
        new_context = torch.as_tensor(context.reshape(1, -1), dtype=torch.float32, device=DEVICE)
        self.rewards.append(float(reward))
        f_t = self.estimator(new_context)
        g = torch.autograd.grad(outputs=f_t, inputs=self.estimator.parameters())
        g = flatten_grad(g)
        g = g / np.sqrt(self.m)
        self.Design += g * g
        self.t += 1

    def train(self, t=None):
        if not self.rewards:
            return 0
        index = np.arange(len(self.rewards))
        np.random.shuffle(index)
        cnt = 0
        total_loss = 0.0
        while True:
            batch_loss = 0.0
            for idx in index:
                c = self.context_list[idx].to(DEVICE)
                r = self.rewards[idx]
                self.current_loss = (self.estimator(c) - r) ** 2
                self.optimizer.zero_grad()
                self.current_loss.backward()
                self.optimizer.step()
                batch_loss += self.current_loss.item()
                total_loss += self.current_loss.item()
                cnt += 1
                if cnt >= 2000:
                    return total_loss / 2000
            if batch_loss / len(self.rewards) <= 1e-3:
                return batch_loss / len(self.rewards)


class PRBNeuralNoExplore:
    def __init__(self, dim, hidden=100):
        self.func = ScalarNetwork(dim, hidden_size=hidden).to(DEVICE)
        self.context_list = []
        self.reward = []
        self.lr = 0.01

    def select(self, context):
        with torch.no_grad():
            mu = self.func(to_tensor(context)).view(-1)
        return int(torch.argmax(mu).item())

    def update(self, context, reward):
        self.context_list.append(torch.as_tensor(context.reshape(1, -1), dtype=torch.float32))
        self.reward.append(float(reward))

    def train(self, t=None):
        if not self.reward:
            return 0
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
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
                if cnt >= 2000:
                    return total_loss / 1000
            if batch_loss / len(self.reward) <= 1e-3:
                return batch_loss / len(self.reward)


class PRBNeuralEpsilon(PRBNeuralNoExplore):
    def __init__(self, dim, p=0.01, hidden=100):
        super().__init__(dim, hidden=hidden)
        self.p = p

    def select(self, context):
        if np.random.binomial(1, self.p):
            return int(np.random.choice(len(context), 1)[0])
        return super().select(context)


class Linearucb:
    def __init__(self, dim, lamdba=0.001, nu=1, style="ts"):
        self.dim = dim
        self.U = lamdba * np.eye(dim)
        self.Uinv = 1 / lamdba * np.eye(dim)
        self.nu = nu
        self.jr = np.zeros((dim,))
        self.mu = np.zeros((dim,))
        self.lamdba = lamdba
        self.style = style

    def select(self, context):
        sig = np.diag(np.matmul(np.matmul(context, self.Uinv), context.T))
        r = np.dot(context, self.mu) + np.sqrt(self.lamdba * self.nu) * sig
        return int(np.argmax(r))

    def train(self, context, reward):
        self.jr += reward * context
        self.U += np.matmul(context.reshape((-1, 1)), context.reshape((1, -1)))
        zz, _ = sp.linalg.lapack.dpotrf(self.U, False, False)
        linv, _ = sp.linalg.lapack.dpotri(zz)
        self.Uinv = np.triu(linv) + np.triu(linv, k=1).T
        self.mu = np.dot(self.Uinv, self.jr)
        return 0


class KernelUCB:
    def __init__(self, dim, lamdba=1, nu=1):
        self.dim = dim
        self.lamdba = lamdba
        self.nu = nu
        self.x_t = None
        self.r_t = None
        self.history_len = 0
        self.scale = self.lamdba * self.nu
        self.U_t = None
        self.K_t = None

    def select(self, context):
        a, _ = context.shape
        if self.history_len == 0:
            mu_t = torch.zeros((a,), device=DEVICE)
            sigma_t = self.scale * torch.ones((a,), device=DEVICE)
        else:
            c_t = to_tensor(context)
            delta_t = c_t.reshape((a, 1, -1)) - self.x_t.reshape((1, self.history_len, -1))
            k_t = torch.exp(-delta_t.norm(dim=2))
            mu_t = k_t.matmul(self.U_t.matmul(self.r_t))
            sigma_t = self.scale * (
                torch.ones((a,), device=DEVICE) - torch.diag(k_t.matmul(self.U_t.matmul(k_t.T)))
            )
            sigma_t = torch.clamp(sigma_t, min=1e-12)
        return int(torch.argmax(mu_t + torch.sqrt(sigma_t)).item())

    def train(self, context, reward):
        if self.history_len < 1000:
            if self.x_t is None:
                self.x_t = to_tensor(context).reshape((1, -1))
                self.r_t = torch.tensor(reward, device=DEVICE, dtype=torch.float32).reshape((-1,))
                self.K_t = torch.ones((1, 1), device=DEVICE, dtype=torch.float32)
            else:
                c_t = to_tensor(context).reshape((1, -1))
                r_t = torch.tensor(reward, device=DEVICE, dtype=torch.float32).reshape((-1,))
                delta_t = c_t.reshape((1, 1, -1)) - self.x_t.reshape((1, self.history_len, -1))
                self.x_t = torch.cat((self.x_t, c_t), dim=0)
                self.r_t = torch.cat((self.r_t, r_t), dim=0)
                k_t = torch.exp(-delta_t.norm(dim=2)).reshape((-1, 1))
                a = torch.cat((k_t.T, torch.ones((1, 1), dtype=torch.float32, device=DEVICE)), dim=1)
                b = torch.cat((self.K_t, k_t), dim=1)
                self.K_t = torch.cat((b, a), dim=0)
            self.history_len += 1
            self.U_t = torch.inverse(self.K_t + self.lamdba * torch.eye(self.history_len, device=DEVICE))
        return 0


def normalize_dataset_name(name):
    if name == "Amazon":
        return "AmazonFashion"
    if name == "Grqc":
        return "GrQc"
    return name


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for_run(base_seed, run_id):
    return base_seed + run_id * 100 + 43


def load_bandit(dataset_name, n_neg, seed, split_seed):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_LOADERS[dataset_name](
        n_neg=n_neg,
        seed=seed,
        split="online",
        split_seed=split_seed,
    )


def should_train(t):
    if t < 2000:
        return t % TRAIN_EVERY_BEFORE_2000 == 0
    return t % TRAIN_EVERY_AFTER_2000 == 0


def build_model(method_name, bandit, args, dataset_name):
    if method_name == "KernelUCB":
        return KernelUCB(bandit.dim, args.lamdba, args.nu)
    if method_name == "LinUCB":
        return Linearucb(bandit.dim, args.lamdba, args.nu)
    if method_name == "NeuralGreedy":
        return PRBNeuralNoExplore(bandit.dim, hidden=args.hidden)
    if method_name == "Neural_epsilon":
        return PRBNeuralEpsilon(bandit.dim, p=args.epsilon, hidden=args.hidden)
    if method_name == "NeuralTS":
        return PRBNeuralTS(bandit.dim, bandit.n_arm, m=args.hidden, sigma=args.lamdba, nu=args.nu)
    if method_name == "NeuralUCB":
        return PRBNeuralUCBDiag(bandit.dim, lamdba=args.lamdba, nu=args.nu, hidden=args.hidden)
    if method_name == "NeuralNoExplore":
        return PRBNeuralNoExplore(bandit.dim, hidden=args.hidden)
    if method_name == "EE-Net":
        kernel_size = 5 if normalize_dataset_name(dataset_name) == "Vessel" else 40
        return EE_Net(
            bandit.dim,
            bandit.n_arm,
            pool_step_size=50,
            lr_1=args.lr1,
            lr_2=args.lr2,
            lr_3=0.01,
            hidden=args.hidden,
            neural_decision_maker=False,
            kernel_size=kernel_size,
        )
    raise ValueError(f"Unknown method: {method_name}")


def unpack_step(step_result):
    if len(step_result) == 6:
        context, _context_ind, rwd, *_ = step_result
        return context, rwd
    if len(step_result) == 2:
        return step_result
    raise ValueError(f"Unsupported b.step() return length: {len(step_result)}")


def update_and_train(method_name, model, context, arm_select, reward, t):
    if method_name in ["LinUCB", "KernelUCB"]:
        return model.train(context[arm_select], reward)
    if method_name == "EE-Net":
        model.update(context, reward, t)
        return model.train(t) if should_train(t) else 0
    model.update(context[arm_select], reward)
    return model.train(t) if should_train(t) else 0


def run_single_task(dataset_name, method_name, run_id, args):
    try:
        dataset_name = normalize_dataset_name(dataset_name)
        seed = seed_for_run(args.seed, run_id)
        seed_everything(seed)
        bandit = load_bandit(dataset_name, args.n_neg, seed, args.split_seed)
        if bandit.n_arm != args.n_neg + 1:
            raise ValueError(f"{dataset_name} produced n_arm={bandit.n_arm}, expected {args.n_neg + 1}")
        model = build_model(method_name, bandit, args, dataset_name)

        regrets = []
        time_records = []
        sum_regret = 0.0
        start_time_total = time.time()

        for t in range(args.T):
            step_start = time.time()
            context, rwd = unpack_step(bandit.step())
            if len(rwd) != args.n_neg + 1:
                raise ValueError(f"{dataset_name} step produced {len(rwd)} arms, expected {args.n_neg + 1}")

            if method_name == "EE-Net":
                arm_select = model.predict(context, t)
            else:
                arm_select = model.select(context)
            arm_select = int(arm_select)
            reward = float(rwd[arm_select])
            update_and_train(method_name, model, context, arm_select, reward, t)

            sum_regret += float(np.max(rwd) - reward)
            regrets.append(sum_regret)
            time_records.append(time.time() - step_start)

            if t % 100 == 0:
                elapsed = time.time() - start_time_total
                print(
                    f"[{dataset_name} | {method_name} | Run {run_id}] "
                    f"Step {t} | Regret: {sum_regret:.0f} | Time: {elapsed:.2f}s",
                    flush=True,
                )

        print(f"[Done] {dataset_name} | {method_name} | Run {run_id} | Final: {sum_regret:.0f}", flush=True)
        return {
            "dataset": dataset_name,
            "method": method_name,
            "run_id": run_id,
            "seed": seed,
            "regrets": regrets,
            "time_records": time_records,
            "error": None,
        }
    except Exception as exc:
        print(f"[Error] {dataset_name} | {method_name} | Run {run_id}: {exc}", flush=True)
        traceback.print_exc()
        return {
            "dataset": normalize_dataset_name(dataset_name),
            "method": method_name,
            "run_id": run_id,
            "seed": seed_for_run(args.seed, run_id),
            "regrets": None,
            "time_records": None,
            "error": repr(exc),
        }


def save_results(results, args, save_dir, elapsed_seconds, run_name, runner_style):
    os.makedirs(save_dir, exist_ok=True)
    data_store = defaultdict(lambda: defaultdict(list))
    time_store = defaultdict(lambda: defaultdict(list))
    failures = []
    successes = []

    for result in results:
        if result is None or result.get("error"):
            failures.append(result)
            continue
        ds = result["dataset"]
        method = result["method"]
        data_store[ds][method].append(result["regrets"])
        time_store[ds][method].append(result["time_records"])
        successes.append({"dataset": ds, "method": method, "run_id": result["run_id"], "seed": result["seed"]})

    saved_files = []
    for ds in data_store:
        for method in data_store[ds]:
            regret_path = os.path.join(save_dir, f"{ds}_{method}_regret.npy")
            time_path = os.path.join(save_dir, f"{ds}_{method}_time.npy")
            np.save(regret_path, np.array(data_store[ds][method]))
            np.save(time_path, np.array(time_store[ds][method]))
            saved_files.extend([regret_path, time_path])
            print(f"Saved: {os.path.basename(regret_path)} & {os.path.basename(time_path)}", flush=True)

    metadata = {
        "runner_style": runner_style,
        "run_name": run_name,
        "save_dir": save_dir,
        "args": vars(args),
        "datasets": [normalize_dataset_name(ds) for ds in args.datasets],
        "methods": args.methods,
        "runs": args.runs,
        "workers": args.workers,
        "n_neg": args.n_neg,
        "n_arm": args.n_neg + 1,
        "train_schedule": {"before_2000": "every 50 rounds", "from_2000": "every 100 rounds"},
        "seed_rule": "seed = base_seed + run_id * 100 + 43",
        "successes": successes,
        "failures": failures,
        "saved_files": saved_files,
        "elapsed_seconds": elapsed_seconds,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {os.path.basename(metadata_path)}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRB-style online baselines for LocPRB")
    parser.add_argument("--datasets", nargs="+", default=SUPPORTED_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--n_neg", default=DEFAULT_N_NEG, type=int)
    parser.add_argument("--lamdba", default=0.1, type=float)
    parser.add_argument("--nu", default=0.001, type=float)
    parser.add_argument("--epsilon", default=0.01, type=float)
    parser.add_argument("--hidden", default=100, type=int)
    parser.add_argument("--lr1", default=0.1, type=float)
    parser.add_argument("--lr2", default=0.01, type=float)
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--split_seed", default=1729, type=int)
    return parser.parse_args()


def run_runner(args, runner_style, result_subdir):
    if args.n_neg != DEFAULT_N_NEG:
        print(f"[Warning] n_neg={args.n_neg}; default protocol is 9 negatives + 1 positive.", flush=True)
    os.chdir(REPO_ROOT)
    tasks = [
        (normalize_dataset_name(ds), method, run_id, args)
        for ds in args.datasets
        for method in args.methods
        for run_id in range(args.runs)
    ]
    print(f"=== Starting {runner_style} online baselines ===", flush=True)
    print(f"Datasets: {[normalize_dataset_name(ds) for ds in args.datasets]}", flush=True)
    print(f"Methods:  {args.methods}", flush=True)
    print(f"T: {args.T}, runs: {args.runs}, workers: {args.workers}, arms: {args.n_neg + 1}", flush=True)

    start_time = time.time()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    with mp.Pool(processes=args.workers) as pool:
        results = pool.starmap(run_single_task, tasks)
    elapsed_seconds = time.time() - start_time

    run_name = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(RESULTS_DIR, result_subdir, run_name)
    save_results(results, args, save_dir, elapsed_seconds, run_name, runner_style)


def main():
    args = parse_args()
    run_runner(args, runner_style="prb_style", result_subdir="baselines_prb_style")


if __name__ == "__main__":
    main()
