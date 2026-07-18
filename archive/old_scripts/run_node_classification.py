import argparse
import numpy as np
import scipy.sparse as sp
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import time , datetime
import torch
import multiprocessing as mp
import traceback
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from EENet import EE_Net
import ppr_solver
import utils
from load_data import (
    load_cora_train, load_cora_test,
    load_citeseer_train, load_citeseer_test,
    load_pubmed_train, load_pubmed_test
)

os.environ['TZ'] = 'Asia/Shanghai'
try:
    time.tzset()
except:
    pass

PROJECT_ROOT = "/mnt/data/xinyu/bandits_pj/PRB"

def construct_initial_matrices(num_users, num_classes):
    """
    """
    num_nodes = num_users + num_classes
    A = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    return A

def build_graph_dict(edge_index):
    """
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
        
    graph_dict = {}
    src, dst = edge_index[0], edge_index[1]
    
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        if u not in graph_dict: graph_dict[u] = []
        if v not in graph_dict: graph_dict[v] = []
        
        graph_dict[u].append(v)
        graph_dict[v].append(u)
        
    return graph_dict

def connect_node_to_neighbors(A, graph_dict, current_nodes_set, new_node_id):
    """
    """
    if new_node_id in graph_dict:
        neighbors = graph_dict[new_node_id]
        for neighbor in neighbors:
            if neighbor in current_nodes_set:
                A[new_node_id, neighbor] = 1.0
                A[neighbor, new_node_id] = 1.0
                
    return A

def get_dataset_config(name):
    if name == 'Cora':
        return load_cora_train, load_cora_test, 2708
    elif name == 'Citeseer':
        return load_citeseer_train, load_citeseer_test, 3327
    elif name == 'Pubmed':
        return load_pubmed_train, load_pubmed_test, 19717
    else:
        raise ValueError(f"Unknown dataset: {name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Node Classification Experiments")
    
    parser.add_argument('--dataset', type=str, required=True, choices=['Cora', 'Citeseer', 'Pubmed'])
    parser.add_argument('--method', type=str, required=True, choices=['FastPRB', 'PRB'])
    
    parser.add_argument('--alpha', type=float, default=0.85)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--appr_eps', type=float, help='Epsilon for FastPRB')
    group.add_argument('--power_T', type=int, help='Iterations for PRB')
    
    parser.add_argument('--T', type=int, default=5000, help='Total rounds (usually smaller for NC)')
    parser.add_argument('--lr1', type=float, default=0.01)
    parser.add_argument('--lr2', type=float, default=0.001)
    
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    
    return parser.parse_args()

def run_single_trial(run_id, args):
    try:
        seed = args.seed + run_id
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print(f">>> [Run {run_id}] Start {args.dataset} | {args.method}...", flush=True)
        
        LoadTrain, LoadTest, num_users = get_dataset_config(args.dataset)
        
        b = LoadTrain()
        num_classes = b.n_arm 
        total_nodes = num_users + num_classes
        
        hidden_dim = 500 if args.dataset == 'Pubmed' else 100
        
        ee_net = EE_Net(
            b.dim, 
            b.n_arm, 
            pool_step_size=50, 
            lr_1=args.lr1, 
            lr_2=args.lr2, 
            hidden=hidden_dim,  
            neural_decision_maker=False, 
            kernel_size=40
        )
        graph_dict = build_graph_dict(b.edge_index_all)
        
        A_lil = construct_initial_matrices(num_users, num_classes)
        P_csr = sp.csr_matrix((total_nodes, total_nodes), dtype=np.float32)
        degree = np.zeros(total_nodes, dtype=np.int64)
        
        current_nodes_set = set()
        
        regrets = []
        time_records = []
        sum_regret = 0.0
        
        #max_rounds = min(args.T, len(b.X_all))
        
        for t in range(args.T):
            step_start = time.time()
            
            context, context_ind, rwd, correct_arm, user_id, correct_class_node_id = b.step()
            A_lil = connect_node_to_neighbors(A_lil, graph_dict, current_nodes_set, user_id)
            current_nodes_set.add(user_id)
            A_csr = A_lil.tocsr()
            degree = np.array(A_csr.sum(axis=1)).flatten().astype(np.int64)
            P_current = utils.to_prmatrix(A_csr)
            
            _, h_observe = ee_net.predict(context, t)
            h_dense = np.zeros(total_nodes, dtype=np.float64)
            
            for i in range(len(context_ind)):
                class_node_id = context_ind[i][1]
                val = h_observe[i]
                if isinstance(val, (list, np.ndarray)): val = val[0]
                
                h_dense[class_node_id] = val
            current_p = None
            if args.method == 'FastPRB':
                degree_safe = degree.copy()
                degree_safe[degree_safe == 0] = 1
                
                current_p = ppr_solver.appr(
                    total_nodes, 
                    P_current.indptr, 
                    P_current.indices, 
                    degree_safe, 
                    h_dense,   
                    args.alpha, 
                    args.appr_eps
                )
            elif args.method == 'PRB':
                current_p = ppr_solver.power_iteration(
                    P_current, 
                    args.alpha, 
                    h_dense, 
                    args.power_T
                )
            
            class_node_indices = [pair[1] for pair in context_ind]
            p_scores = current_p[class_node_indices]
            final_arm = np.argmax(p_scores)
            
            if rwd[final_arm] == 1.0:
                reward = 1.0
                pred_class_node = class_node_indices[final_arm]
            else:
                reward = 0.0
                pred_class_node = None
                
            sum_regret += (1.0 - reward)
            regrets.append(sum_regret)
            
            if reward == 1.0 and pred_class_node is not None:
                u, v = int(user_id), int(pred_class_node)
                if A_lil[u, v] == 0:
                    A_lil[u, v] = 1.0
                    A_lil[v, u] = 1.0
                    
            ee_net.update(context, reward, t)
            loss1, loss2 = 0.0, 0.0
            if t < 2000:
                if t % 50 == 0: loss1, loss2 = ee_net.train(t)
            else:
                if t % 100 == 0: loss1, loss2 = ee_net.train(t)
                
            step_duration = time.time() - step_start
            time_records.append(step_duration)
            
            if t % 100 == 0:
                print(f"[Run {run_id}] Step {t} | Regret: {sum_regret:.0f}", flush=True)
                
        result_data = np.stack([time_records, regrets], axis=1)
        return result_data

    except Exception as e:
        print(f"!!! Error in Run {run_id}: {e}", flush=True)
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try: mp.set_start_method('spawn', force=True)
    except: pass
    
    args = parse_args()
    
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    if args.method == 'FastPRB':
        param_str = f"eps{args.appr_eps}"
    else:
        param_str = f"powT{args.power_T}"
        
    folder_name = f"{args.dataset}_{args.method}_alpha{args.alpha}_{param_str}_T{args.T}_{current_time}"
    
    base_dir = os.path.join(os.getcwd(), "results", "node_classification")
    save_dir = os.path.join(base_dir, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"=== Node Classification Experiment ===")
    print(f"Dataset: {args.dataset}, Method: {args.method}")
    print(f"Output: {save_dir}")
    
    task_args = [(i, args) for i in range(args.runs)]
    
    with mp.Pool(processes=args.workers) as pool:
        raw_results = pool.starmap(run_single_trial, task_args)
        
    valid_results = [r for r in raw_results if r is not None]
    if valid_results:
        final_data = np.stack(valid_results)
        
        T = final_data.shape[1]
        padding = np.zeros((len(valid_results), T, 3))
        final_data_padded = np.concatenate([final_data, padding], axis=2)
        
        np.save(os.path.join(save_dir, "final_results.npy"), final_data_padded)
        with open(os.path.join(save_dir, "config.txt"), "w") as f:
            f.write(str(args))
            
        print("Done.")
    else:
        print("All runs failed.")