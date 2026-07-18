import argparse
import json
import multiprocessing as mp
import os
import random
import time
import traceback
import warnings
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
warnings.filterwarnings("ignore", message=".*The use of `x.T` on tensors.*")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)


def _install_optional_torch_geometric_stub():
    """Allow importing load_data when PyG-only loaders are unused."""
    try:
        import torch_geometric  # noqa: F401
    except ModuleNotFoundError:
        import sys
        import types

        tg = types.ModuleType("torch_geometric")
        loader = types.ModuleType("torch_geometric.loader")
        datasets = types.ModuleType("torch_geometric.datasets")
        transforms = types.ModuleType("torch_geometric.transforms")

        class _MissingPyG:
            def __init__(self, *args, **kwargs):
                raise ModuleNotFoundError("torch_geometric is required for Planetoid loaders")

        loader.DataLoader = _MissingPyG
        datasets.Planetoid = _MissingPyG
        transforms.NormalizeFeatures = _MissingPyG
        tg.loader = loader
        tg.datasets = datasets
        tg.transforms = transforms
        sys.modules.setdefault("torch_geometric", tg)
        sys.modules.setdefault("torch_geometric.loader", loader)
        sys.modules.setdefault("torch_geometric.datasets", datasets)
        sys.modules.setdefault("torch_geometric.transforms", transforms)


_install_optional_torch_geometric_stub()

from src.baselines_new.EENet import EE_Net
from src.baselines_new.KernelUCB import KernelUCB
from src.baselines_new.LinUCB import Linearucb
from src.baselines_new.NeuralGreedy import NeuralGreedy
from src.baselines_new.NeuralNoExplore import NeuralNoExplore
from src.baselines_new.NeuralTS import NeuralTS
from src.baselines_new.NeuralUCB import NeuralUCBDiag
from src.experiment_configs import (
    DEFAULT_BASELINE_DATASETS,
    DEFAULT_BASELINE_METHODS,
    DEFAULT_RUNS,
    DEFAULT_WORKERS,
    RESULTS_DIR,
    TRAIN_EVERY_AFTER_2000,
    TRAIN_EVERY_BEFORE_2000,
)
from src.load_data import (
    load_amazon_fashion,
    load_facebook,
    load_grqc,
    load_movielen,
    load_ogb_collab,
    load_ogb_ppa,
    load_ogb_vessel,
)


DATASET_LOADERS = {
    "MovieLens": load_movielen,
    "Amazon": load_amazon_fashion,
    "Facebook": load_facebook,
    "GrQc": load_grqc,
    "Collab": load_ogb_collab,
    "PPA": load_ogb_ppa,
    "Vessel": load_ogb_vessel,
}

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for_run(run_id):
    return run_id * 100 + 43


def load_bandit(dataset_name):
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_LOADERS[dataset_name]()


def build_model(method_name, bandit, args, dataset_name):
    if method_name == "KernelUCB":
        return KernelUCB(bandit.dim, args.lamdba, args.nu)
    if method_name == "LinUCB":
        return Linearucb(bandit.dim, args.lamdba, args.nu)
    if method_name == "NeuralGreedy":
        return NeuralGreedy(bandit.dim, epsilon=0.0, hidden=100)
    if method_name == "Neural_epsilon":
        return NeuralGreedy(bandit.dim, epsilon=0.01, hidden=100)
    if method_name == "NeuralTS":
        return NeuralTS(bandit.dim, bandit.n_arm, m=100, lamdba=args.lamdba, nu=args.nu)
    if method_name == "NeuralUCB":
        return NeuralUCBDiag(bandit.dim, lamdba=args.lamdba, nu=args.nu, hidden=100)
    if method_name == "NeuralNoExplore":
        return NeuralNoExplore(bandit.dim)
    if method_name == "EE-Net":
        kernel_size = 5 if dataset_name == "Vessel" else 40
        return EE_Net(
            bandit.dim,
            bandit.n_arm,
            pool_step_size=50,
            lr_1=args.lr1,
            lr_2=args.lr2,
            lr_3=0.01,
            hidden=100,
            neural_decision_maker=False,
            kernel_size=kernel_size,
        )
    raise ValueError(f"Unknown method: {method_name}")


def should_train(t):
    if t < 2000:
        return t % TRAIN_EVERY_BEFORE_2000 == 0
    return t % TRAIN_EVERY_AFTER_2000 == 0


def maybe_train(method_name, model, context, arm_select, reward, t):
    if method_name in ["LinUCB", "KernelUCB"]:
        return model.train(context[arm_select], reward)
    if method_name == "EE-Net":
        model.update(context, reward, t)
        if should_train(t):
            return model.train(t)
        return 0

    model.update(context[arm_select], reward)
    if should_train(t):
        return model.train(t)
    return 0


def unpack_step(step_result):
    if len(step_result) == 6:
        context, _context_ind, rwd, *_ = step_result
        return context, rwd
    if len(step_result) == 2:
        return step_result
    raise ValueError(f"Unsupported b.step() return length: {len(step_result)}")


def run_single_task(dataset_name, method_name, run_id, args):
    try:
        seed = seed_for_run(run_id)
        seed_everything(seed)
        bandit = load_bandit(dataset_name)
        model = build_model(method_name, bandit, args, dataset_name)

        regrets = []
        time_records = []
        sum_regret = 0.0
        start_time_total = time.time()

        for t in range(args.T):
            step_start = time.time()
            context, rwd = unpack_step(bandit.step())

            if method_name == "EE-Net":
                arm_select = model.predict(context, t)
            else:
                arm_select = model.select(context)

            reward = float(rwd[arm_select])
            maybe_train(method_name, model, context, arm_select, reward, t)

            regret = float(np.max(rwd) - reward)
            sum_regret += regret
            regrets.append(sum_regret)

            step_duration = time.time() - step_start
            time_records.append(step_duration)
            if t % 100 == 0:
                total_elapsed = time.time() - start_time_total
                print(
                    f"[{dataset_name} | {method_name} | Run {run_id}] "
                    f"Step {t} | Regret: {sum_regret:.0f} | Time: {total_elapsed:.2f}s",
                    flush=True,
                )

        print(
            f"[Done] {dataset_name} | {method_name} | Run {run_id} | Final: {sum_regret:.0f}",
            flush=True,
        )
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
            "dataset": dataset_name,
            "method": method_name,
            "run_id": run_id,
            "seed": seed_for_run(run_id),
            "regrets": None,
            "time_records": None,
            "error": repr(exc),
        }


def save_results(results, args, save_dir, elapsed_seconds, run_name):
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
        successes.append(
            {
                "dataset": ds,
                "method": method,
                "run_id": result["run_id"],
                "seed": result["seed"],
            }
        )

    print("\n=== Saving Results ===", flush=True)
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
        "run_name": run_name,
        "save_dir": save_dir,
        "args": vars(args),
        "datasets": args.datasets,
        "methods": args.methods,
        "runs": args.runs,
        "workers": args.workers,
        "train_schedule": {
            "before_2000": "every 50 rounds",
            "from_2000": "every 100 rounds",
        },
        "seed_rule": "seed = run_id * 100 + 43",
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
    parser = argparse.ArgumentParser(description="Run self-contained online baselines in parallel")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_BASELINE_DATASETS, help="List of datasets to run")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_BASELINE_METHODS, help="List of methods")
    parser.add_argument("--T", default=10000, type=int, help="Total number of rounds")
    parser.add_argument("--lamdba", default=0.1, type=float)
    parser.add_argument("--nu", default=0.001, type=float)
    parser.add_argument("--lr1", default=0.1, type=float, help="EE-Net lr1")
    parser.add_argument("--lr2", default=0.01, type=float, help="EE-Net lr2")
    parser.add_argument("--runs", default=DEFAULT_RUNS, type=int, help="Number of independent runs per method")
    parser.add_argument("--workers", default=DEFAULT_WORKERS, type=int, help="Number of parallel workers")
    return parser.parse_args()


def main():
    os.chdir(REPO_ROOT)
    args = parse_args()

    tasks = [(ds, method, run, args) for ds in args.datasets for method in args.methods for run in range(args.runs)]

    print("=== Starting Parallel Baselines (new) ===", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Methods:  {args.methods}", flush=True)
    print(f"Total Tasks: {len(tasks)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print("=" * 50, flush=True)

    start_time = time.time()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    with mp.Pool(processes=args.workers) as pool:
        results = pool.starmap(run_single_task, tasks)

    elapsed_seconds = time.time() - start_time
    run_name = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(RESULTS_DIR, "baselines_new", run_name)
    save_results(results, args, save_dir, elapsed_seconds, run_name)
    print(f"Results directory: {save_dir}", flush=True)
    print(f"\nAll tasks finished in {elapsed_seconds / 60:.2f} minutes.", flush=True)


if __name__ == "__main__":
    main()
