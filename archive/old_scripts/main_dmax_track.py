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
import traceback
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
warnings.filterwarnings("ignore", message=".*The use of `x.T` on tensors.*")
warnings.filterwarnings("ignore", message=".*SparseEfficiencyWarning.*")


from EENet import EE_Net
import ppr_solver
import utils

from load_data import load_movielen, load_facebook, load_amazon_fashion, load_grqc, load_ogb_collab, load_ogb_ppa, load_ogb_vessel

PROJECT_ROOT = "/mnt/data/xinyu/bandits_pj/PRB"

# Paper Table: node counts |V|, for d_max / n plots
PAPER_NUM_NODES = {
    "MovieLens": 12_000,
    "Amazon_fashion": 8_000,
    "Facebook": 4_039,
    "Grqc": 5_242,
    "Collab": 235_868,
    "PPA": 576_289,
    "Vessel": 3_538_495,
}


def paper_num_nodes(graph_name):
    """Return paper #Nodes for the dataset name used in argparse."""
    if graph_name not in PAPER_NUM_NODES:
        raise ValueError(
            f"No paper #Nodes entry for graph_name={graph_name!r}. "
            f"Add it to PAPER_NUM_NODES in main_dmax_track.py."
        )
    return PAPER_NUM_NODES[graph_name]


def get_configurations(graph_name):
    """
    Retrieves the dataset-specific configurations based on the input graph name.
    
    This function acts as a factory, returning the appropriate bandit data loader class,
    graph structure manager class, file path, and node metadata (user/item counts)
    required to initialize the experiment environment.
    """
    if graph_name == 'MovieLens':
        path = os.path.join(PROJECT_ROOT, "online_link_prediction/data/MovieLens/movie_2000users_10000items_noedge.npy")
        # Bandit Loader, Graph Manager Class, Path, Num_Users, Num_Items
        return load_movielen, utils.MovieLens, path, 2000, 10000
    
    elif graph_name == 'Amazon_fashion':
        path = os.path.join(PROJECT_ROOT, "online_link_prediction/data/Amazon_fashion/new/Insert/Amazon_fashion_4000users_noedge.npy")
        return load_amazon_fashion, utils.Amazon_fashion, path, 4000, 4000
        
    elif graph_name == 'Facebook':
        path = os.path.join(PROJECT_ROOT, "online_link_prediction/data/Facebook/Insert/facebook_combined_ALLusers_noedge.npy")
        return load_facebook, utils.Facebook, path, 4039, 0 
        
    elif graph_name == 'Grqc':
        path = os.path.join(PROJECT_ROOT, "online_link_prediction/data/GrQc/Insert/GrQc_ALLusers_noedge.npy")
        return load_grqc, utils.Grqc, path, 5242, 0 
    elif graph_name == 'PPA':
        path = os.path.join(PROJECT_ROOT, "dataset")
        return load_ogb_ppa, utils.PPA, path, 0, 0
        
    elif graph_name == 'Collab':
        path = os.path.join(PROJECT_ROOT, "dataset")
        return load_ogb_collab, utils.Collab, path, 0, 0

    elif graph_name == 'Vessel':
        path = os.path.join(PROJECT_ROOT, "dataset")
        return load_ogb_vessel, utils.Vessel, path, 0, 0
        
    
    else:
        raise ValueError(f"Unknown graph_name: {graph_name}")

def parse_arguments():
    """
    Parses and validates command-line arguments for the experiment.

    This function defines the available hyperparameters, dataset options, and 
    algorithm choices (FastPRB vs. PRB). It also enforces constraints, such as 
    mutual exclusivity between the approximation error (epsilon) and power 
    iteration steps.
    """
    parser = argparse.ArgumentParser(description="Run PRB/FastPRB Experiments")
    parser.add_argument('--graph_name', type=str, required=True, choices=['MovieLens', 'Amazon_fashion', 'Facebook', 'Grqc', 'PPA', 'Collab', 'Vessel', 'DDI' ,'Citation2'])
    parser.add_argument('--method', type=str, required=True, choices=['FastPRB', 'PRB'])
    parser.add_argument('--alpha', type=float, required=True, help='PPR alpha (Damping factor)')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--appr_eps', type=float, help='Epsilon for FastPRB (APPR)')
    group.add_argument('--power_T', type=int, help='Iterations for PRB (Power Iteration)')
    parser.add_argument('--T', type=int, default=10000, help='Total rounds')
    parser.add_argument('--lr1', type=float, required=True, help='Learning rate for exploitation')
    parser.add_argument('--lr2', type=float, required=True, help='Learning rate for exploration')
    parser.add_argument('--if_save', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False, help='Save checkpoint every 1000 steps?')
    parser.add_argument('--workers', type=int, default=10, help='Number of multiprocessing workers')
    parser.add_argument('--runs', type=int, default=10, help='Total number of independent runs')
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--init_edges', type=int, default=None, help='Limit the number of initial edges for PPA/Vessel')
    parser.add_argument('--hidden', type=int, default=100, help='EE-Net exploitation hidden width (Network_exploitation)')
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=None,
        help='EE-Net exploration Conv1d kernel length; default 5 for Vessel, 40 otherwise',
    )
    args = parser.parse_args()
    if args.method == 'FastPRB' and args.appr_eps is None:
        parser.error("Method FastPRB requires --appr_eps")
    if args.method == 'PRB' and args.power_T is None:
        parser.error("Method PRB requires --power_T")
    if args.graph_name in ('DDI', 'Citation2'):
        parser.error(
            "main_dmax_track: DDI/Citation2 are not supported here (no PAPER_NUM_NODES / get_configurations)."
        )
        
    return args


def _max_degree_pre_update(graph_manager):
    """Maximum degree on the current graph (before this round's edge insertion)."""
    deg = np.asarray(graph_manager.degree).flatten()
    if deg.size == 0:
        return 0
    return int(np.max(deg))


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
    
    device = torch.device('cuda:0')
    
    print(f">>> [Worker {os.getpid()}] Starting Run {run_id}...", flush=True)
    LoaderClass, GraphClass, data_path, n_users, n_items = get_configurations(args.graph_name)
    if args.graph_name in ['PPA', 'Vessel'] and args.init_edges is not None:
        bandit_loader = LoaderClass(max_init_edges=args.init_edges)
    else:
        bandit_loader = LoaderClass()
    print(f"-> Loading Graph from {data_path} ...")
    graph_manager = GraphClass(data_path)
    if args.graph_name in ['MovieLens', 'Amazon_fashion']:
        graph_manager.load(n_users, n_items)
    elif args.graph_name in ['Facebook', 'Grqc']:
        graph_manager.load(n_users)
    else:
        graph_manager.load()
    if args.graph_name in ['PPA', 'Vessel'] and args.init_edges is not None:
        print(f"-> [Main] Overwriting GraphManager P with subsampled graph ({args.init_edges} edges)")
        graph_manager.P = bandit_loader.adj_init
        if hasattr(graph_manager, 'A'):
            graph_manager.A = bandit_loader.adj_init.tolil()
        graph_manager.degree = np.diff(graph_manager.P.indptr)    
    graph_data = graph_manager.get()
    
    A_lil = graph_manager.A 
    P_matrix = graph_data['P'] 
    num_nodes = graph_data['num_nodes']
    current_pool_step = 50
    current_kernel_size = utils.resolve_ee_net_kernel_size(
        args.graph_name, args.kernel_size
    )
    ee_net = EE_Net(
        bandit_loader.dim, 
        bandit_loader.n_arm, 
        pool_step_size=current_pool_step, 
        lr_1=args.lr1, 
        lr_2=args.lr2, 
        hidden=args.hidden, 
        neural_decision_maker=False, 
        kernel_size=current_kernel_size
        
    )
    results_list = [] # [time, regret, loss1, loss2, ppr_norm]
    dmax_per_round = []

    start_time_trial = time.time()
    sum_regret = 0.0
    if sp.isspmatrix_csr(graph_manager.P):
        P_current_csr = graph_manager.P
    else:
        P_current_csr = graph_manager.P.tocsr()
        
        
    print(f">>> [Worker {run_id}] Generating fixed testing dataset (100 samples)...", flush=True)
    fixed_test_set = bandit_loader.testing_dataset() 
    time_acc_results = []   
    latest_test_acc = 0.0    
        
        
        
    for t in range(args.T):
        step_start = time.time()
        if t % 50 == 0:
            test_hits = 0
            current_eval_time = time.time() - start_time_trial
            
            test_user_offset = 0
            if args.graph_name in ['MovieLens', 'Amazon_fashion']:
                test_user_offset = n_users

            for sample in fixed_test_set:
                if len(sample) == 6:
                    t_ctx, t_ctx_ind, t_rwd, t_arm, _, _ = sample
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
                
                if args.method == 'FastPRB':
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
            time_acc_results.append([current_eval_time, test_acc])
            latest_test_acc = test_acc
        # --- A. Bandit Step (Context) ---
        
        step_result = bandit_loader.step()
        if len(step_result) == 6:
            context, context_ind, rwd, arm, user_id, item_id_repr = step_result
            correct_arm_indices = [arm]
        else:
            # OGB 
            context, context_ind, rwd, correct_arm_indices, user_id, item_id_repr = step_result
            
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
        
        if args.method == 'FastPRB':
            
            degree = np.array(graph_manager.degree).flatten().astype(np.int64)
            degree[degree == 0] = 1
            current_p = ppr_solver.appr(
                num_nodes, 
                P_current_csr.indptr, 
                P_current_csr.indices, 
                degree, 
                h_dense, 
                args.alpha, 
                args.appr_eps
            )
            
        elif args.method == 'PRB':
            
            current_p = ppr_solver.power_iteration(
                P_current_csr, 
                args.alpha, 
                h_dense, 
                args.power_T
            )
        
        
        # --- E. Decision & Reward ---
        
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
        
        # d_max: graph state *before* this round's update (same graph used for PPR above)
        dmax_per_round.append(_max_degree_pre_update(graph_manager))
        
        # --- F. Graph Update ---
        if reward == 1.0 and connected_u is not None:
            if args.graph_name in ['MovieLens', 'Amazon_fashion']:
                raw_item_id = connected_v - current_user_offset
                graph_manager.update(raw_item_id, connected_u)
            else:
                graph_manager.update(connected_u, connected_v)
            P_current_csr = graph_manager.P.tocsr()  
            degree = np.array(graph_manager.degree).flatten().astype(np.int64)
            degree[degree == 0] = 1  
        # --- G. Net Update & Train ---
        ee_net.update(context, reward, t)
        
        loss1, loss2 = 0.0, 0.0
        if  t < 2000:
            if t % 50 == 0: loss1, loss2 = ee_net.train(t)
        else:
            if t % 100 == 0: loss1, loss2 = ee_net.train(t)
            
        # --- H. Recording ---
        step_end = time.time()
        step_duration = step_end - step_start
        ppr_norm = np.sum(np.abs(current_p))
        current_total_time = step_end - start_time_trial
        
        results_list.append([step_duration, sum_regret, loss1, loss2, ppr_norm])
        
        if t % 200 == 0:
            print(f"Round {t} | Regret: {sum_regret:.0f} | Loss1: {loss1:.4f} | Loss2: {loss2:.4f} | TestAcc: {latest_test_acc:.2%} | Time: {current_total_time:.4f}s | Norm: {ppr_norm:.2f}", flush=True)
        
        if args.if_save and t % 1000 == 0 and t > 0:
            
            utils.save_results(save_dir, results_list, is_final=False)

    # --- Final Save ---
    total_time = time.time() - start_time_trial
    
    dmax_arr = np.array(dmax_per_round, dtype=np.int64)
    dmax_global = int(dmax_arr.max()) if dmax_arr.size else 0
    print(
        f">>> [Worker {run_id}] Finished. Total Time: {total_time:.2f}s | Total Regret: {sum_regret:.0f} | "
        f"d_max max-over-rounds: {dmax_global} (last round d_max: {int(dmax_arr[-1]) if dmax_arr.size else 0})",
        flush=True,
    )
    time_acc_save_path = os.path.join(save_dir, f"worker_{run_id}_TimeAcc.npy")
    np.save(time_acc_save_path, np.array(time_acc_results))
    print(f"-> Saved Time-Accuracy results to {time_acc_save_path}")
    dmax_save_path = os.path.join(save_dir, f"worker_{run_id}_dmax_per_round.npy")
    np.save(dmax_save_path, dmax_arr)
    print(f"-> Saved d_max trajectory to {dmax_save_path}")
    return np.array(results_list)


def save_dmax_plots(save_dir, graph_name, n_paper, runs_expected):
    """
    Aggregate worker_*_dmax_per_round.npy and save mean ± sqrt(sample variance) across runs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    for r in range(runs_expected):
        path = os.path.join(save_dir, f"worker_{r}_dmax_per_round.npy")
        if not os.path.isfile(path):
            print(f"[dmax plots] Missing {path}, skip run {r}")
            continue
        series.append(np.load(path))
    if not series:
        print("[dmax plots] No dmax files found, skip plotting.")
        return
    T_lens = {s.shape[0] for s in series}
    if len(T_lens) != 1:
        print(f"[dmax plots] Inconsistent lengths {T_lens}, skip plotting.")
        return
    mat = np.stack(series, axis=0).astype(np.float64)
    n_run, T = mat.shape
    mean = mat.mean(axis=0)
    # sample variance (ddof=1); band uses ± sqrt(var) i.e. sample std when n_run>1
    if n_run > 1:
        var = mat.var(axis=0, ddof=1)
        std = np.sqrt(np.maximum(var, 0.0))
    else:
        var = np.zeros(T)
        std = np.zeros(T)

    rounds = np.arange(T)
    label_band = f"mean ± √(sample var) over {n_run} runs"

    def _style_ax(ax, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # --- d_max vs round ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, mean, color="C0", label="mean $d_{\\max}$")
    if n_run > 1:
        ax.fill_between(rounds, mean - std, mean + std, color="C0", alpha=0.25, label=label_band)
    _style_ax(ax, f"$d_{{\\max}}$ vs round ({graph_name})", "$d_{\\max}$")
    p1 = os.path.join(save_dir, "dmax_vs_round.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "dmax_vs_round.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[dmax plots] Saved {p1}")

    # --- d_max / n vs round ---
    ratio = mat / float(n_paper)
    mean_r = ratio.mean(axis=0)
    if n_run > 1:
        std_r = np.sqrt(np.maximum(ratio.var(axis=0, ddof=1), 0.0))
    else:
        std_r = np.zeros(T)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(rounds, mean_r, color="C1", label=f"mean $d_{{\\max}}/n$ ($n={n_paper}$)")
    if n_run > 1:
        ax2.fill_between(rounds, mean_r - std_r, mean_r + std_r, color="C1", alpha=0.25, label=label_band)
    _style_ax(ax2, f"$d_{{\\max}}/n$ vs round ({graph_name}, $n={n_paper}$)", "$d_{\\max} / n$")
    p2 = os.path.join(save_dir, "dmax_over_n_vs_round.png")
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    fig2.savefig(os.path.join(save_dir, "dmax_over_n_vs_round.pdf"), bbox_inches="tight")
    plt.close(fig2)
    print(f"[dmax plots] Saved {p2}")

    meta_path = os.path.join(save_dir, "dmax_plot_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"graph_name={graph_name}\n")
        f.write(f"paper_n={n_paper}\n")
        f.write(f"runs_used={n_run}\n")
        f.write(f"T={T}\n")
        f.write("band=mean ± sqrt(sample variance) per round across runs\n")
    print(f"[dmax plots] Wrote {meta_path}")


if __name__ == "__main__":
    
    mp.set_start_method('spawn', force=True)
    args = parse_arguments()
    n_paper = paper_num_nodes(args.graph_name)
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    if args.method == 'FastPRB':
        param_str = f"eps{args.appr_eps}"
    else:
        param_str = f"powT{args.power_T}"
    ks_resolved = utils.resolve_ee_net_kernel_size(args.graph_name, args.kernel_size)
    folder_name = (
        f"{args.graph_name}_{args.method}_alpha{args.alpha}_{param_str}_"
        f"T{args.T}_lr1{args.lr1}_lr2{args.lr2}_h{args.hidden}_ks{ks_resolved}_{current_time}"
    )
    
    
    base_dir = os.path.join(os.getcwd(), "results", "online_link_prediction")
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
    final_data = np.stack(valid_results)
    print(f"=== All Done. Aggregated Shape: {final_data.shape} ===")
    utils.save_results(save_dir, final_data, is_final=True, args=args)

    per_run_dmax_peak = []
    for r in range(args.runs):
        fp = os.path.join(save_dir, f"worker_{r}_dmax_per_round.npy")
        if os.path.isfile(fp):
            per_run_dmax_peak.append(int(np.max(np.load(fp))))
    if per_run_dmax_peak:
        peaks = np.array(per_run_dmax_peak, dtype=np.float64)
        std_str = f"{peaks.std(ddof=1):.2f}" if len(peaks) > 1 else "nan"
        print(
            f"=== d_max summary (max degree over rounds, per run): "
            f"mean={peaks.mean():.2f}, std={std_str}, "
            f"min={peaks.min():.0f}, max={peaks.max():.0f} (n_runs={len(peaks)}) ==="
        )

    save_dmax_plots(save_dir, args.graph_name, n_paper, args.runs)
