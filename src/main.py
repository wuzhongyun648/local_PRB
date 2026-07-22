
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import numpy as np
import scipy
import scipy.sparse as sp
import sys
import time
os.environ['TZ'] = 'Asia/Shanghai'
import torch
import datetime
import multiprocessing as mp
import platform
import subprocess
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
from src.dynamic_appr import DynamicAPPR, get_push_impl
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
    "dyn_locPRB": "dyn_locPRB",
    "dyn-LocPRB": "dyn_locPRB",
    "DynLocPRB": "dyn_locPRB",
}
METHOD_LABEL_SUFFIX = ""


def parse_method(value):
    try:
        return METHOD_ALIASES[value]
    except KeyError as exc:
        choices = ", ".join(METHOD_ALIASES)
        raise argparse.ArgumentTypeError(
            f"unknown method {value!r}; choose one of: {choices}"
        ) from exc


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
    algorithm choices (LocPRB vs. PRB). It also enforces constraints, such as
    mutual exclusivity between the approximation error (epsilon) and power 
    iteration steps.
    """
    parser = argparse.ArgumentParser(description="Run PRB/LocPRB Experiments")
    parser.add_argument('--graph_name', type=str, required=True, choices=sorted(MAIN_DATASET_CONFIGS))
    parser.add_argument('--method', type=parse_method, required=True)
    parser.add_argument('--alpha', type=float, required=True, help='PPR alpha (Damping factor)')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--appr_eps', type=float, help='Epsilon for LocPRB (APPR)')
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
    parser.add_argument(
        '--ppr_backend',
        choices=('numba', 'python'),
        default='numba',
        help='Execution backend for LocPRB and dyn_locPRB; PRB always uses SciPy',
    )
    
    parser.add_argument('--init_edges', type=int, default=None, help='Limit the number of initial edges for PPA/Vessel')
    parser.add_argument(
        '--ppr_diagnostics',
        action='store_true',
        help='Compare DYN-APPR with scratch APPR periodically (diagnostic runs only)',
    )
    parser.add_argument(
        '--ppr_diagnostic_every',
        type=int,
        default=50,
        help='Rounds between scratch-vs-DYN diagnostic checks',
    )
    parser.add_argument(
        '--evaluation_every',
        type=int,
        default=50,
        help='Periodic fixed-test interval; 0 disables excluded evaluation for timing runs',
    )
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
    if args.ppr_diagnostic_every < 1:
        parser.error("--ppr_diagnostic_every must be >= 1")
    if args.evaluation_every < 0:
        parser.error("--evaluation_every must be >= 0")
        
    return args


def run_experiment(run_id,args, save_dir):
    """
    Executes a single independent experimental trial.

    This function initializes the environment with a specific random seed derived 
    from the run ID. It performs the sequential link prediction loop for T rounds, 
    handling context observation, PPR computation (approximate or exact), 
    arm selection, dynamic graph updates, and neural network training.
    """
    end_to_end_t0 = time.perf_counter()
    seed = args.seed + run_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f">>> [Worker {os.getpid()}] Starting Run {run_id}...", flush=True)
    LoaderClass, GraphClass, data_path, n_users, n_items = get_configurations(args.graph_name)
    if args.graph_name in ['PPA', 'Vessel'] and args.init_edges is not None:
        bandit_loader = LoaderClass(max_init_edges=args.init_edges)
    else:
        bandit_loader = LoaderClass(n_neg=args.n_neg)
    loader_rng_state = np.random.get_state()

    def loader_step():
        nonlocal loader_rng_state
        caller_state = np.random.get_state()
        np.random.set_state(loader_rng_state)
        try:
            result = bandit_loader.step()
            loader_rng_state = np.random.get_state()
            return result
        finally:
            np.random.set_state(caller_state)
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
    resolved_backend = 'scipy' if args.method == 'PRB' else args.ppr_backend
    scratch_kernel = (
        ppr_solver.get_appr_kernel(args.ppr_backend)
        if args.method in ('LocPRB', 'dyn_locPRB')
        else None
    )
    push_impl = (
        get_push_impl(args.ppr_backend)
        if args.method == 'dyn_locPRB'
        else None
    )
    dynamic_solver = (
        DynamicAPPR(push_impl=push_impl)
        if args.method == 'dyn_locPRB'
        else None
    )
    results_list = [] # [time, regret, loss1, loss2, ppr_norm]
    if sp.isspmatrix_csr(graph_manager.P):
        P_current_csr = graph_manager.P
    else:
        P_current_csr = graph_manager.P.tocsr()

    # Compile Numba against the real graph dtypes before any reported timer.
    warmup_seconds = 0.0
    if resolved_backend == 'numba':
        warmup_t0 = time.perf_counter()
        warm_degree = np.asarray(graph_manager.degree).reshape(-1).astype(np.int64)
        warm_degree[warm_degree == 0] = 1
        warm_source = np.zeros(num_nodes, dtype=np.float64)
        if args.method == 'LocPRB':
            scratch_kernel(
                num_nodes,
                P_current_csr.indptr,
                P_current_csr.indices,
                warm_degree,
                warm_source,
                args.alpha,
                args.appr_eps,
            )
        elif args.method == 'dyn_locPRB':
            push_impl(
                P_current_csr.indptr,
                P_current_csr.indices,
                warm_degree.astype(np.float64),
                np.zeros(num_nodes, dtype=np.float64),
                warm_source.copy(),
                args.alpha,
                args.appr_eps,
            )
        warmup_seconds = time.perf_counter() - warmup_t0

    setup_seconds = time.perf_counter() - end_to_end_t0 - warmup_seconds
    start_time_trial = time.perf_counter()
    sum_regret = 0.0
        
        
    print(f">>> [Worker {run_id}] Generating fixed testing dataset (100 samples)...", flush=True)
    time_acc_results = []
    latest_test_acc = 0.0
    # 从报告的 step_duration / Time / Total / TimeAcc 横轴中累计扣除：测试集构造 + 周期评测
    total_excluded_seconds = 0.0
    timing_breakdown = {
        'ppr_time': 0.0,
        'train_time': 0.0,
        'other_time': 0.0,
        'loader_time': 0.0,
        'predict_source_time': 0.0,
        'decision_time': 0.0,
        'graph_update_time': 0.0,
        'evaluation_time': 0.0,
        'testset_build_time': 0.0,
        'scratch_solves': 0,
        'scratch_pushes': 0,
        'scratch_edge_visits': 0,
        'scratch_initial_active_nodes': 0,
        'graph_update_attempts': 0,
        'diagnostic_time': 0.0,
        'diagnostic_checks': 0,
        'diagnostic_l1_sum': 0.0,
        'diagnostic_l1_max': 0.0,
        'diagnostic_decision_disagreements': 0,
        'diagnostic_scratch_pushes': 0,
    }
    _t_build0 = time.perf_counter()
    fixed_test_set = (
        bandit_loader.testing_dataset()
        if args.evaluation_every > 0
        else []
    )
    testset_build_dt = time.perf_counter() - _t_build0
    timing_breakdown['testset_build_time'] = testset_build_dt
    total_excluded_seconds += testset_build_dt

    for t in range(args.T):
        step_start = time.perf_counter()
        eval_dt = 0.0
        if args.evaluation_every > 0 and t % args.evaluation_every == 0:
            eval_t0 = time.perf_counter()
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
                
                if args.method in ('LocPRB', 'dyn_locPRB'):
                    t_degree = np.array(graph_manager.degree).flatten().astype(np.int64)
                    t_degree[t_degree == 0] = 1
                    t_p = scratch_kernel(
                        num_nodes, P_current_csr.indptr, P_current_csr.indices, 
                        t_degree, t_h_dense, args.alpha, args.appr_eps
                    )[0]
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
            eval_dt = time.perf_counter() - eval_t0
            timing_breakdown['evaluation_time'] += eval_dt
            total_excluded_seconds += eval_dt
            # 不含测试集构造与评测的累计墙钟时间（本轮评测结束后）
            current_eval_time = time.perf_counter() - start_time_trial - total_excluded_seconds
            time_acc_results.append([current_eval_time, test_acc])
            latest_test_acc = test_acc
        # --- A. Bandit Step (Context) ---
        online_step_t0 = time.perf_counter()
        
        loader_t0 = time.perf_counter()
        step_result = loader_step()
        loader_dt = time.perf_counter() - loader_t0
        if len(step_result) == 6:
            context, context_ind, rwd, _, user_id, _ = step_result
        else:
            # OGB 
            context, context_ind, rwd, _, user_id, _ = step_result
            
        # --- B. Neural Net Predict ---
        predict_t0 = time.perf_counter()
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
        predict_source_dt = time.perf_counter() - predict_t0
            
        #--- D. Solver Calculation ---

        current_p = None
        ppr_dt = 0.0
        diagnostic_dt = 0.0
        scratch_diagnostic_p = None
        
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
                current_p, scratch_pushes, scratch_edge_visits, scratch_active = scratch_kernel(
                    num_nodes,
                    P_current_csr.indptr,
                    P_current_csr.indices,
                    degree,
                    h_dense,
                    args.alpha,
                    args.appr_eps,
                )
                timing_breakdown['scratch_solves'] += 1
                timing_breakdown['scratch_pushes'] += int(scratch_pushes)
                timing_breakdown['scratch_edge_visits'] += int(scratch_edge_visits)
                timing_breakdown['scratch_initial_active_nodes'] += int(scratch_active)
            ppr_dt = time.perf_counter() - ppr_t0
            if (
                args.method == 'dyn_locPRB'
                and args.ppr_diagnostics
                and t % args.ppr_diagnostic_every == 0
            ):
                diagnostic_t0 = time.perf_counter()
                scratch_diagnostic_p, scratch_pushes, _, _ = (
                    scratch_kernel(
                        num_nodes,
                        P_current_csr.indptr,
                        P_current_csr.indices,
                        degree,
                        h_dense,
                        args.alpha,
                        args.appr_eps,
                    )
                )
                l1_error = float(
                    np.sum(np.abs(current_p - scratch_diagnostic_p))
                )
                timing_breakdown['diagnostic_checks'] += 1
                timing_breakdown['diagnostic_l1_sum'] += l1_error
                timing_breakdown['diagnostic_l1_max'] = max(
                    timing_breakdown['diagnostic_l1_max'], l1_error
                )
                timing_breakdown['diagnostic_scratch_pushes'] += int(
                    scratch_pushes
                )
                diagnostic_candidate_ids = [
                    pair[1] + current_user_offset for pair in context_ind
                ]
                timing_breakdown['diagnostic_decision_disagreements'] += int(
                    np.argmax(current_p[diagnostic_candidate_ids])
                    != np.argmax(scratch_diagnostic_p[diagnostic_candidate_ids])
                )
                diagnostic_dt = time.perf_counter() - diagnostic_t0
                total_excluded_seconds += diagnostic_dt
            
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
        decision_t0 = time.perf_counter()
        cand_graph_ids = []
        for pair in context_ind:
            cand_graph_ids.append(pair[1] + current_user_offset)
            
        p_scores = current_p[cand_graph_ids]
        
        final_arm = int(np.argmax(p_scores))
        
        if rwd[final_arm] == 1.0:
            reward = 1.0
            connected_u = user_id
            connected_v = cand_graph_ids[final_arm]
        else:
            reward = 0.0
            connected_u = None
        sum_regret += (1.0 - reward)
        decision_dt = time.perf_counter() - decision_t0
        
        # --- F. Graph Update ---
        graph_update_t0 = time.perf_counter()
        if reward == 1.0 and connected_u is not None:
            timing_breakdown['graph_update_attempts'] += 1
            if args.graph_name in ['MovieLens', 'Amazon_fashion']:
                raw_item_id = connected_v - current_user_offset
                graph_manager.update(raw_item_id, connected_u)
            else:
                graph_manager.update(connected_u, connected_v)
            P_current_csr = graph_manager.P.tocsr()  
            degree = np.array(graph_manager.degree).flatten().astype(np.int64)
            degree[degree == 0] = 1  
        graph_update_dt = time.perf_counter() - graph_update_t0
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
        step_end = time.perf_counter()
        step_duration = step_end - step_start - eval_dt - diagnostic_dt
        ppr_norm = np.sum(np.abs(current_p))
        current_total_time_excl_overhead = step_end - start_time_trial - total_excluded_seconds
        
        results_list.append([step_duration, sum_regret, loss1, loss2, ppr_norm])
        
        if t % 500 == 0:
            print(f"Round {t} | Regret: {sum_regret:.0f} | Loss1: {loss1:.4f} | Loss2: {loss2:.4f} | TestAcc: {latest_test_acc:.2%} | Time: {current_total_time_excl_overhead:.4f}s (excl. testset&eval) | Norm: {ppr_norm:.2f}", flush=True)
        
        if args.if_save and t % 1000 == 0 and t > 0:
            np.save(
                os.path.join(
                    save_dir,
                    f"worker_{run_id}_checkpoint_step_{len(results_list)}.npy",
                ),
                np.asarray(results_list),
            )

        online_step_elapsed = time.perf_counter() - online_step_t0
        other_dt = max(
            0.0,
            online_step_elapsed
            - ppr_dt
            - train_dt
            - diagnostic_dt
            - graph_update_dt
            - loader_dt
            - predict_source_dt
            - decision_dt,
        )
        timing_breakdown['ppr_time'] += ppr_dt
        timing_breakdown['train_time'] += train_dt
        timing_breakdown['other_time'] += other_dt
        timing_breakdown['loader_time'] += loader_dt
        timing_breakdown['predict_source_time'] += predict_source_dt
        timing_breakdown['decision_time'] += decision_dt
        timing_breakdown['graph_update_time'] += graph_update_dt
        timing_breakdown['diagnostic_time'] += diagnostic_dt

    if dynamic_solver is not None:
        for key, value in dynamic_solver.stats.items():
            timing_breakdown[f'dynamic_{key}'] = value

    # --- Final Save ---
    total_time = time.perf_counter() - start_time_trial - total_excluded_seconds
    end_to_end_seconds = time.perf_counter() - end_to_end_t0
    timing_breakdown['online_total_time'] = total_time
    timing_breakdown['setup_time'] = setup_seconds
    timing_breakdown['warmup_time'] = warmup_seconds
    timing_breakdown['end_to_end_time'] = end_to_end_seconds
    
    print(
        f">>> [Worker {run_id}] Finished. Total Time: {total_time:.2f}s "
        f"(excl. testset build & eval; deducted {total_excluded_seconds:.2f}s) | Total Regret: {sum_regret:.0f}",
        flush=True,
    )
    time_acc_save_path = os.path.join(save_dir, f"worker_{run_id}_TimeAcc.npy")
    np.save(time_acc_save_path, np.array(time_acc_results))
    print(f"-> Saved Time-Accuracy results to {time_acc_save_path}")
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=REPO_ROOT_FOR_IMPORTS,
            text=True,
        ).strip()
    except Exception:
        git_commit = 'unknown'
    try:
        git_dirty = bool(
            subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=REPO_ROOT_FOR_IMPORTS,
                text=True,
            ).strip()
        )
    except Exception:
        git_dirty = None
    try:
        import numba
        numba_version = numba.__version__
    except ModuleNotFoundError:
        numba_version = None
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    run_metrics = {
        'schema_version': 1,
        'run_id': run_id,
        'seed': seed,
        'dataset': args.graph_name,
        'method': args.method,
        'requested_backend': args.ppr_backend,
        'resolved_backend': resolved_backend,
        'rounds': args.T,
        'final_regret': sum_regret,
        'timing': timing_breakdown,
        'environment': {
            'git_commit': git_commit,
            'git_dirty': git_dirty,
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'numba': numba_version,
            'torch': torch.__version__,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_available': cuda_available,
            'gpu_name': gpu_name,
            'thread_environment': {
                name: os.environ.get(name) for name in (
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'
                )
            },
        },
        'config': vars(args),
    }
    utils.save_json(
        os.path.join(save_dir, f'worker_{run_id}_metrics.json'),
        run_metrics,
    )
    return np.array(results_list), timing_breakdown

def main():
    mp.set_start_method('spawn', force=True)
    args = parse_arguments()
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if args.method in ('LocPRB', 'dyn_locPRB'):
        param_str = f"eps{args.appr_eps}"
    else:
        param_str = f"powT{args.power_T}"
    ks_resolved = utils.resolve_ee_net_kernel_size(args.graph_name, args.kernel_size)
    method_label = f"{args.method}{METHOD_LABEL_SUFFIX}"
    backend_label = (
        f"_backend{args.ppr_backend}"
        if args.method in ('LocPRB', 'dyn_locPRB')
        else ''
    )
    folder_name = (
        f"{args.graph_name}_{method_label}{backend_label}_alpha{args.alpha}_{param_str}_"
        f"T{args.T}_k{args.n_neg + 1}_initH{args.init_hops}_initK{args.init_topk}_"
        f"lr1{args.lr1}_lr2{args.lr2}_h{args.hidden}_ks{ks_resolved}_{current_time}"
    )
    
    
    base_dir = os.path.join(RESULTS_DIR, "online_link_prediction")
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=False)
    
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
    diagnostic_checks = sum(
        stats.get('diagnostic_checks', 0) for _, stats in valid_results
    )
    if diagnostic_checks:
        diagnostic_l1_sum = sum(
            stats.get('diagnostic_l1_sum', 0.0)
            for _, stats in valid_results
        )
        diagnostic_l1_max = max(
            stats.get('diagnostic_l1_max', 0.0)
            for _, stats in valid_results
        )
        disagreements = sum(
            stats.get('diagnostic_decision_disagreements', 0)
            for _, stats in valid_results
        )
        dynamic_pushes = sum(
            stats.get('dynamic_pushes', 0) for _, stats in valid_results
        )
        scratch_pushes = sum(
            stats.get('diagnostic_scratch_pushes', 0)
            for _, stats in valid_results
        )
        dynamic_edge_visits = sum(
            stats.get('dynamic_edge_visits', 0)
            for _, stats in valid_results
        )
        fallback_resets = sum(
            stats.get('dynamic_fallback_resets', 0)
            for _, stats in valid_results
        )
        insert_updates = sum(
            stats.get('dynamic_insert_updates', 0)
            for _, stats in valid_results
        )
        print(
            "=== DYN Diagnostics | "
            f"Checks: {diagnostic_checks} | "
            f"L1 Mean: {diagnostic_l1_sum / diagnostic_checks:.6f} | "
            f"L1 Max: {diagnostic_l1_max:.6f} | "
            f"Decision Disagreements: {disagreements} | "
            f"Pushes: {dynamic_pushes} | "
            f"Scratch Pushes ({diagnostic_checks} checks): {scratch_pushes} | "
            f"Edge Visits: {dynamic_edge_visits} | "
            f"Insert Updates: {insert_updates} | "
            f"Fallback Resets: {fallback_resets} ==="
        )
    aggregate_metrics = {
        'schema_version': 1,
        'dataset': args.graph_name,
        'method': args.method,
        'requested_backend': args.ppr_backend,
        'resolved_backend': 'scipy' if args.method == 'PRB' else args.ppr_backend,
        'runs': args.runs,
        'rounds': args.T,
        'final_regret_mean': float(np.mean(final_data[:, -1, 1])),
        'final_regret_std': float(np.std(final_data[:, -1, 1])),
        'timing_totals': {
            key: sum(stats.get(key, 0) for _, stats in valid_results)
            for key in (
                'ppr_time', 'train_time', 'graph_update_time', 'other_time',
                'loader_time', 'predict_source_time', 'decision_time',
                'evaluation_time', 'testset_build_time', 'diagnostic_time',
                'online_total_time', 'setup_time', 'warmup_time', 'end_to_end_time'
            )
        },
        'solver_totals': {
            key: sum(stats.get(key, 0) for _, stats in valid_results)
            for key in set().union(*(stats.keys() for _, stats in valid_results))
            if key.startswith('scratch_') or key.startswith('dynamic_')
        },
        'config': vars(args),
    }
    utils.save_json(os.path.join(save_dir, 'metrics_summary.json'), aggregate_metrics)
    utils.save_results(save_dir, final_data, is_final=True, args=args)


if __name__ == "__main__":
    main()
