
import argparse
import numpy as np
import scipy.sparse as sp
import os
import sys
import time
os.environ['TZ'] = 'Asia/Shanghai'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
import datetime
import multiprocessing as mp
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_FOR_IMPORTS = os.path.dirname(SRC_DIR)
sys.path.append(REPO_ROOT_FOR_IMPORTS)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
warnings.filterwarnings("ignore", message=".*The use of `x.T` on tensors.*")
warnings.filterwarnings("ignore", message=".*SparseEfficiencyWarning.*")


from src.EENet import EE_Net
from src import ppr_solver
from src import utils
from src.dynamic_appr import DynamicAPPR
from src.experiment_configs import (
    DEFAULT_EE_NET_POOL_STEP,
    DEFAULT_HIDDEN,
    DEFAULT_MAIN_T,
    DEFAULT_N_NEG,
    DEFAULT_RUNS,
    DEFAULT_WORKERS,
    MAIN_DATASET_CONFIGS,
    RESULTS_DIR,
    TRAIN_EVERY_AFTER_2000,
    TRAIN_EVERY_BEFORE_2000,
)
from src.load_data import load_movielen, load_facebook, load_amazon_fashion, load_grqc, load_ogb_collab, load_ogb_ppa, load_ogb_vessel

LOADER_BY_DATASET = {
    "MovieLens": load_movielen,
    "Amazon_fashion": load_amazon_fashion,
    "Facebook": load_facebook,
    "Grqc": load_grqc,
    "PPA": load_ogb_ppa,
    "Collab": load_ogb_collab,
    "Vessel": load_ogb_vessel,
}

GRAPH_BY_DATASET = {
    "MovieLens": utils.MovieLens,
    "Amazon_fashion": utils.Amazon_fashion,
    "Facebook": utils.Facebook,
    "Grqc": utils.Grqc,
    "PPA": utils.PPA,
    "Collab": utils.Collab,
    "Vessel": utils.Vessel,
}

METHOD_ALIASES = {
    "PRB": "PRB",
    "LocPRB": "LocPRB",
    "FastPRB": "LocPRB",
    "dyn_locPRB": "dyn_locPRB",
    "dyn-LocPRB": "dyn_locPRB",
    "DynLocPRB": "dyn_locPRB",
}
METHOD_LABEL_SUFFIX = ""
TEST_SEED_OFFSET = 1_000_003


def parse_method(value):
    try:
        return METHOD_ALIASES[value]
    except KeyError as exc:
        choices = ", ".join(METHOD_ALIASES)
        raise argparse.ArgumentTypeError(
            f"unknown method {value!r}; choose one of: {choices}"
        ) from exc


def prepare_ppr_source(source, method):
    """L1-normalize the personalization vector used by LocPRB solvers."""
    if method in ("LocPRB", "dyn_locPRB"):
        norm = np.sum(np.abs(source))
        if norm != 0.0:
            return source / norm
    return source


def build_ee_net(dim, n_arm, args, kernel_size):
    """Build the EE-Net shared by PRB and both LocPRB solvers."""
    return EE_Net(
        dim,
        n_arm,
        pool_step_size=DEFAULT_EE_NET_POOL_STEP,
        lr_1=args.lr1,
        lr_2=args.lr2,
        hidden=args.hidden,
        neural_decision_maker=False,
        kernel_size=kernel_size,
    )


def build_fixed_test_set(loader, run_seed):
    """Build a deterministic test set without consuming the online RNG stream."""
    test_seed = (int(run_seed) + TEST_SEED_OFFSET) % (2**32)
    online_rng = loader.rng
    try:
        loader.rng = np.random.default_rng(test_seed)
        return loader.testing_dataset()
    finally:
        loader.rng = online_rng


def apply_graph_update(graph_manager, *edge_args):
    """Apply one graph update and rebuild CSR only for a newly inserted edge."""
    edge_added = graph_manager.update(*edge_args)
    if not edge_added:
        return False, None
    return True, graph_manager.P.tocsr()


def selected_graph_edge(context_ind, arm, target_offset=0):
    """Return canonical graph-node endpoints for the selected candidate."""
    raw_u, raw_v = context_ind[int(arm)]
    return int(raw_u), int(raw_v) + int(target_offset)

def get_configurations(graph_name):
    """
    Retrieves the dataset-specific configurations based on the input graph name.
    
    This function acts as a factory, returning the appropriate bandit data loader class,
    graph structure manager class, file path, and node metadata (user/item counts)
    required to initialize the experiment environment.
    """
    if graph_name not in MAIN_DATASET_CONFIGS:
        raise ValueError(f"Unknown graph_name: {graph_name}")
    config = MAIN_DATASET_CONFIGS[graph_name]
    return (
        LOADER_BY_DATASET[graph_name],
        GRAPH_BY_DATASET[graph_name],
        config["path"],
        config["n_users"],
        config["n_items"],
    )

def parse_arguments():
    """
    Parses and validates command-line arguments for the experiment.

    This function defines the available hyperparameters, dataset options, and 
    algorithm choices (FastPRB vs. PRB). It also enforces constraints, such as 
    mutual exclusivity between the approximation error (epsilon) and power 
    iteration steps.
    """
    parser = argparse.ArgumentParser(description="Run PRB/LocPRB Experiments")
    parser.add_argument('--graph_name', type=str, required=True, choices=sorted(MAIN_DATASET_CONFIGS))
    parser.add_argument('--method', type=parse_method, required=True)
    parser.add_argument('--alpha', type=float, required=True, help='PPR alpha (Damping factor)')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--appr_eps', type=float, help='Epsilon for FastPRB (APPR)')
    group.add_argument('--power_T', type=int, help='Iterations for PRB (Power Iteration)')
    parser.add_argument('--T', type=int, default=DEFAULT_MAIN_T, help='Total rounds')
    parser.add_argument('--lr1', type=float, required=True, help='Learning rate for exploitation')
    parser.add_argument('--lr2', type=float, required=True, help='Learning rate for exploration')
    parser.add_argument('--if_save', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False, help='Save checkpoint every 1000 steps?')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help='Number of multiprocessing workers')
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS, help='Total number of independent runs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_neg', type=int, default=DEFAULT_N_NEG, help='Number of negative candidates per round (k = n_neg + 1)')
    parser.add_argument('--init_hops', type=int, default=0, help='H-hop warm-start radius for the initial graph; 0 keeps the default graph')
    parser.add_argument('--init_topk', type=int, default=0, help='Use top-k highest-degree seed nodes for warm-start; 0 disables warm-start')
    
    parser.add_argument('--init_edges', type=int, default=None, help='Limit the number of initial edges for PPA/Vessel')
    parser.add_argument('--hidden', type=int, default=DEFAULT_HIDDEN, help='EE-Net exploitation hidden width (Network_exploitation)')
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=None,
        help='EE-Net exploration Conv1d kernel length; default 5 for Vessel, 40 otherwise',
    )
    args = parser.parse_args()
    if args.method in ('LocPRB', 'dyn_locPRB') and args.appr_eps is None:
        parser.error("Methods LocPRB and dyn_locPRB require --appr_eps")
    if args.method == 'PRB' and args.power_T is None:
        parser.error("Method PRB requires --power_T")
    if args.n_neg < 1:
        parser.error("--n_neg must be >= 1")
    if args.init_hops < 0:
        parser.error("--init_hops must be >= 0")
    if args.init_topk < 0:
        parser.error("--init_topk must be >= 0")
        
    return args


def run_experiment(run_id,args, save_dir):
    """
    Executes a single independent experimental trial.

    This function initializes the environment with a specific random seed derived 
    from the run ID. It performs the sequential link prediction loop for T rounds, 
    handling context observation, PPR computation (approximate or exact), 
    arm selection, dynamic graph updates, and neural network training.
    """
    seed = args.seed + run_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f">>> [Worker {os.getpid()}] Starting Run {run_id}...", flush=True)
    LoaderClass, GraphClass, data_path, n_users, n_items = get_configurations(args.graph_name)
    bandit_loader = LoaderClass(n_neg=args.n_neg, seed=seed)
    print(f"-> Loading Graph from {data_path} ...")
    graph_manager = GraphClass(data_path)
    if args.graph_name in ['MovieLens', 'Amazon_fashion']:
        graph_manager.load(n_users, n_items, init_hops=args.init_hops, init_topk=args.init_topk)
    elif args.graph_name in ['Facebook', 'Grqc']:
        graph_manager.load(n_users, init_hops=args.init_hops, init_topk=args.init_topk)
    else:
        graph_manager.load(init_hops=args.init_hops, init_topk=args.init_topk)
    if args.graph_name in ['PPA', 'Vessel'] and args.init_edges is not None:
        print(f"-> [Main] Overwriting GraphManager P with subsampled graph ({args.init_edges} edges)")
        graph_manager.P = bandit_loader.adj_init
        if hasattr(graph_manager, 'A'):
            graph_manager.A = bandit_loader.adj_init.tolil()
        graph_manager.degree = np.diff(graph_manager.P.indptr)    
    graph_data = graph_manager.get()
    
    num_nodes = graph_data['num_nodes']
    current_kernel_size = utils.resolve_ee_net_kernel_size(
        args.graph_name, args.kernel_size
    )
    ee_net = build_ee_net(
        bandit_loader.dim, bandit_loader.n_arm, args, current_kernel_size
    )
    dynamic_solver = DynamicAPPR() if args.method == 'dyn_locPRB' else None
    results_list = [] # [time, regret, loss1, loss2, ppr_norm]

    start_time_trial = time.time()
    sum_regret = 0.0
    if sp.isspmatrix_csr(graph_manager.P):
        P_current_csr = graph_manager.P
    else:
        P_current_csr = graph_manager.P.tocsr()
        
        
    print(f">>> [Worker {run_id}] Generating fixed testing dataset (100 samples)...", flush=True)
    time_acc_results = []
    latest_test_acc = 0.0
    # 从报告的 step_duration / Time / Total / TimeAcc 横轴中累计扣除：测试集构造 + 周期评测
    total_excluded_seconds = 0.0
    timing_breakdown = {
        'ppr_time': 0.0,
        'train_time': 0.0,
        'other_time': 0.0,
    }
    _t_build0 = time.time()
    fixed_test_set = build_fixed_test_set(bandit_loader, seed)
    total_excluded_seconds += time.time() - _t_build0

    for t in range(args.T):
        step_start = time.time()
        eval_dt = 0.0
        if t % 50 == 0:
            eval_t0 = time.time()
            test_hits = 0
            test_user_offset = 0
            if args.graph_name in ['MovieLens', 'Amazon_fashion']:
                test_user_offset = n_users

            for sample in fixed_test_set:
                if len(sample) == 6:
                    t_ctx, t_ctx_ind, t_rwd, _, _, _ = sample
                else:
                    t_ctx, t_ctx_ind, t_rwd, _, _, _ = sample
                
                _, t_h_observe = ee_net.predict(t_ctx, t)
                
                t_h_dense = np.zeros(num_nodes, dtype=np.float64)
                for i in range(len(t_ctx_ind)):
                    raw_id = t_ctx_ind[i][1]
                    real_id = raw_id + test_user_offset
                    val = t_h_observe[i]
                    if isinstance(val, (list, np.ndarray)): val = val[0]
                    t_h_dense[real_id] = val
                
                t_h_dense = prepare_ppr_source(t_h_dense, args.method)
                if args.method in ('LocPRB', 'dyn_locPRB'):
                    t_degree = np.array(graph_manager.degree).flatten().astype(np.int64)
                    t_degree[t_degree == 0] = 1
                    t_p = ppr_solver.appr(
                        num_nodes, P_current_csr.indptr, P_current_csr.indices, 
                        t_degree, t_h_dense, args.alpha, args.appr_eps
                    )
                elif args.method == 'PRB':
                    t_p = ppr_solver.power_iteration(
                        P_current_csr, args.alpha, t_h_dense, args.power_T
                    )
                
                t_cand_ids = [pair[1] + test_user_offset for pair in t_ctx_ind]
                t_scores = t_p[t_cand_ids]
                t_final = int(np.argmax(t_scores))
                
                if t_rwd[t_final] == 1.0:
                    test_hits += 1
            
            test_acc = test_hits / 100.0
            eval_dt = time.time() - eval_t0
            total_excluded_seconds += eval_dt
            # 不含测试集构造与评测的累计墙钟时间（本轮评测结束后）
            current_eval_time = time.time() - start_time_trial - total_excluded_seconds
            time_acc_results.append([current_eval_time, test_acc])
            latest_test_acc = test_acc
        # --- A. Bandit Step (Context) ---
        online_step_t0 = time.perf_counter()
        
        step_result = bandit_loader.step()
        context, context_ind, rwd, _, _, _ = step_result
            
        # --- B. Neural Net Predict ---
        _, h_observe = ee_net.predict(context, t)
        
        # --- C. Construct h Vector ---
        h_dense = np.zeros(num_nodes, dtype=np.float64)
        
        
        current_user_offset = 0
        
        if args.graph_name in ['MovieLens', 'Amazon_fashion']:
            current_user_offset = n_users
            
        for i in range(len(context_ind)):
            raw_item_id = context_ind[i][1]
            real_node_id = raw_item_id + current_user_offset
            
            val = h_observe[i]
            if isinstance(val, (list, np.ndarray)): val = val[0]
            
            h_dense[real_node_id] = val
            
        #--- D. Solver Calculation ---

        current_p = None
        ppr_dt = 0.0
        
        h_dense = prepare_ppr_source(h_dense, args.method)
        if args.method in ('LocPRB', 'dyn_locPRB'):
            ppr_t0 = time.perf_counter()
            degree = np.array(graph_manager.degree).flatten().astype(np.int64)
            degree[degree == 0] = 1
            if args.method == 'dyn_locPRB':
                current_p = dynamic_solver.solve(
                    num_nodes,
                    P_current_csr.indptr,
                    P_current_csr.indices,
                    degree,
                    h_dense,
                    args.alpha,
                    args.appr_eps,
                )
            else:
                current_p = ppr_solver.appr(
                    num_nodes,
                    P_current_csr.indptr,
                    P_current_csr.indices,
                    degree,
                    h_dense,
                    args.alpha,
                    args.appr_eps,
                )
            ppr_dt = time.perf_counter() - ppr_t0
            
        elif args.method == 'PRB':
            ppr_t0 = time.perf_counter()
            current_p = ppr_solver.power_iteration(
                P_current_csr, 
                args.alpha, 
                h_dense, 
                args.power_T
            )
            ppr_dt = time.perf_counter() - ppr_t0
        
        
        # --- E. Decision & Reward ---
        
        cand_graph_ids = []
        for pair in context_ind:
            cand_graph_ids.append(pair[1] + current_user_offset)
            
        p_scores = current_p[cand_graph_ids]
        
        final_arm = int(np.argmax(p_scores))
        
        if rwd[final_arm] == 1.0:
            reward = 1.0
            connected_u, connected_v = selected_graph_edge(
                context_ind,
                final_arm,
                target_offset=current_user_offset,
            )
        else:
            reward = 0.0
            connected_u = None
        sum_regret += (1.0 - reward)
        
        # --- F. Graph Update ---
        if reward == 1.0 and connected_u is not None:
            if args.graph_name in ['MovieLens', 'Amazon_fashion']:
                raw_item_id = connected_v - current_user_offset
                edge_added, updated_csr = apply_graph_update(
                    graph_manager, raw_item_id, connected_u
                )
            else:
                edge_added, updated_csr = apply_graph_update(
                    graph_manager, connected_u, connected_v
                )
            if edge_added:
                P_current_csr = updated_csr
        # --- G. Net Update & Train ---
        ee_net.update(context, reward, t)
        
        loss1, loss2 = 0.0, 0.0
        train_dt = 0.0
        if  t < 2000:
            if t % TRAIN_EVERY_BEFORE_2000 == 0:
                train_t0 = time.perf_counter()
                loss1, loss2 = ee_net.train(t)
                train_dt = time.perf_counter() - train_t0
        else:
            if t % TRAIN_EVERY_AFTER_2000 == 0:
                train_t0 = time.perf_counter()
                loss1, loss2 = ee_net.train(t)
                train_dt = time.perf_counter() - train_t0
            
        # --- H. Recording ---
        step_end = time.time()
        step_duration = step_end - step_start - eval_dt
        ppr_norm = np.sum(np.abs(current_p))
        current_total_time_excl_overhead = step_end - start_time_trial - total_excluded_seconds
        
        results_list.append([step_duration, sum_regret, loss1, loss2, ppr_norm])
        
        if t % 500 == 0:
            print(f"Round {t} | Regret: {sum_regret:.0f} | Loss1: {loss1:.4f} | Loss2: {loss2:.4f} | TestAcc: {latest_test_acc:.2%} | Time: {current_total_time_excl_overhead:.4f}s (excl. testset&eval) | Norm: {ppr_norm:.2f}", flush=True)
        
        if args.if_save and t % 1000 == 0 and t > 0:
            
            utils.save_results(save_dir, results_list, is_final=False)

        online_step_elapsed = time.perf_counter() - online_step_t0
        other_dt = max(0.0, online_step_elapsed - ppr_dt - train_dt)
        timing_breakdown['ppr_time'] += ppr_dt
        timing_breakdown['train_time'] += train_dt
        timing_breakdown['other_time'] += other_dt

    # --- Final Save ---
    total_time = time.time() - start_time_trial - total_excluded_seconds
    
    print(
        f">>> [Worker {run_id}] Finished. Total Time: {total_time:.2f}s "
        f"(excl. testset build & eval; deducted {total_excluded_seconds:.2f}s) | Total Regret: {sum_regret:.0f}",
        flush=True,
    )
    time_acc_save_path = os.path.join(save_dir, f"worker_{run_id}_TimeAcc.npy")
    np.save(time_acc_save_path, np.array(time_acc_results))
    print(f"-> Saved Time-Accuracy results to {time_acc_save_path}")
    return np.array(results_list), timing_breakdown

def main():
    mp.set_start_method('spawn', force=True)
    args = parse_arguments()
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    if args.method in ('LocPRB', 'dyn_locPRB'):
        param_str = f"eps{args.appr_eps}"
    else:
        param_str = f"powT{args.power_T}"
    ks_resolved = utils.resolve_ee_net_kernel_size(args.graph_name, args.kernel_size)
    method_label = f"{args.method}{METHOD_LABEL_SUFFIX}"
    folder_name = (
        f"{args.graph_name}_{method_label}_alpha{args.alpha}_{param_str}_"
        f"T{args.T}_k{args.n_neg + 1}_initH{args.init_hops}_initK{args.init_topk}_"
        f"lr1{args.lr1}_lr2{args.lr2}_h{args.hidden}_ks{ks_resolved}_{current_time}"
    )
    
    
    base_dir = os.path.join(RESULTS_DIR, "online_link_prediction")
    save_dir = os.path.join(base_dir, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"=== Experiment Start ===")
    print(f"Output Dir: {save_dir}")
    print(f"Workers: {args.workers}, Runs: {args.runs}")
    
    task_args = [(i, args, save_dir) for i in range(args.runs)]
    
    with mp.Pool(processes=args.workers) as pool:
        raw_results = pool.starmap(run_experiment, task_args)
    
    valid_results = [r for r in raw_results if r is not None]
    final_data = np.stack([result for result, _ in valid_results])
    total_train_time = sum(stats['train_time'] for _, stats in valid_results)
    total_ppr_time = sum(stats['ppr_time'] for _, stats in valid_results)
    total_other_time = sum(stats['other_time'] for _, stats in valid_results)
    print(f"=== All Done. Aggregated Shape: {final_data.shape} ===")
    print(
        "=== Online Time Breakdown Across All Runs "
        f"(excl. testset build & eval) | "
        f"Train: {total_train_time:.2f}s | "
        f"PPR: {total_ppr_time:.2f}s | "
        f"Other: {total_other_time:.2f}s ==="
    )
    utils.save_results(save_dir, final_data, is_final=True, args=args)


if __name__ == "__main__":
    main()
