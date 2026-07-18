import argparse
import numpy as np
import scipy.sparse as sp
import os
import sys
import time

import multiprocessing as mp
import traceback
from datetime import datetime

# 硬件限制
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 路径 Hack
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import utils,ppr_solver

import torch
from tqdm import tqdm
import argparse


# 引入你的模块
from load_data import (
    load_collab_train, load_collab_test,
    load_ppa_train, load_ppa_test,
    load_ddi_train, load_ddi_test,
    load_cora_train, load_cora_test,
    load_citeseer_train, load_citeseer_test,
    load_pubmed_train, load_pubmed_test
)
# 假设你的 EENet 在 EENet.py 中
from EENet import EE_Net 

# ================= Configuration =================
# 基于 PRB 论文 Table 3 的指标设置
METRICS = {
    'collab': 50,    # Hits@50
    'ppa': 100,      # Hits@100
    'ddi': 20,       # Hits@20
    'Cora': 100,     # Hits@100
    'Citeseer': 100, # Hits@100
    'PubMed': 100    # Hits@100
}

DATASET_MAP = {
    'collab': (load_collab_train, load_collab_test),
    'ppa': (load_ppa_train, load_ppa_test),
    'ddi': (load_ddi_train, load_ddi_test),
    'Cora': (load_cora_train, load_cora_test),
    'Citeseer': (load_citeseer_train, load_citeseer_test),
    'PubMed': (load_pubmed_train, load_pubmed_test),
}

def to_prmatrix(A):
    A = A.tocsr()
    d = np.array(A.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        d_inv = 1.0 / d
    d_inv[np.isinf(d_inv)] = 0.0 
    D_inv = sp.diags(d_inv)
    P = D_inv.dot(A)
    return P

# ================= Main Run =================
def run(args):
    dataset_name = args.dataset
    print(f"[{dataset_name}] Running PRB Offline Link Prediction (Correct PPR Scheme)...")
    
    # 1. Initialize Loaders
    TrainLoaderCls, TestLoaderCls = DATASET_MAP[dataset_name]
    train_loader = TrainLoaderCls()
    test_loader = TestLoaderCls()
    
    num_nodes = train_loader.num_nodes
    feat_dim = train_loader.dim
    hit_k = METRICS[dataset_name]

    # 2. Initialize Model
    # 建议使用较低的学习率以保持数值稳定
    ee_net = EE_Net(
        dim=feat_dim, 
        n_arm=train_loader.n_arm, 
        pool_step_size=train_loader.n_arm,
        lr_1=args.lr1, 
        lr_2=args.lr2, 
        hidden=args.hidden
    )

    # 3. Initialize Graph Storage (使用列表以高效添加)
    edge_rows = []
    edge_cols = []
    
    # ==========================================
    # Phase 1: Sequential Training
    # ==========================================
    print("\n>>> Phase 1: Sequential Training...")
    
    total_samples = train_loader.num_samples
    # 截断训练步数 (如果设置了 max_train_steps)
    train_limit = args.max_train_steps if args.max_train_steps else total_samples
    train_limit = min(train_limit, total_samples)
    
    print(f"Training on first {train_limit} steps (Total: {total_samples}).")
    pbar = tqdm(range(train_limit), desc="Training NN")
    
    for t in pbar:
        # Step A: 获取数据
        _, X_ind, rwd, correct_idx, u, v = train_loader.step()
        
        # Step B: 构建特征
        u_feat = train_loader.node_feat[X_ind[:, 0]]
        v_feat = train_loader.node_feat[X_ind[:, 1]]
        context = np.concatenate((u_feat, v_feat), axis=1)
        
        # Input Check
        if np.isnan(context).any():
            print(f"Error: Context contains NaN at step {t}")
            break

        # Step C: Bandit Update
        arm_select, _ = ee_net.predict(context, t)
        reward = rwd[arm_select]
        ee_net.update(context, reward, t)
        
        # Step D: Periodic Training
        loss_1, loss_2 = 0.0, 0.0
        if t < 1000:
            if t % 50 == 0: loss_1, loss_2 = ee_net.train(t)
        else:
            if t % 100 == 0: loss_1, loss_2 = ee_net.train(t)
            
        # Step E: 记录边 (用于构建图)
        edge_rows.extend([u, v])
        edge_cols.extend([v, u])
        
        if t % 1000 == 0:
            pbar.set_postfix({'L1': f"{loss_1:.4f}"})
    
    # --- 快速补全剩余边 (如果使用了截断) ---
    # 必须保证图结构的完整性，否则 PageRank 无法跑
    if train_limit < total_samples:
        print(f"\n>>> Fast-forwarding: Adding remaining edges to Graph...")
        remaining_edges = train_loader.pos_edges[train_limit:]
        
        if hasattr(remaining_edges, 'source_node'): # OGB dict list
             rem_u = [e['source_node'] for e in remaining_edges]
             rem_v = [e['target_node'] for e in remaining_edges]
        elif isinstance(remaining_edges, np.ndarray):
            rem_u = remaining_edges[:, 0]
            rem_v = remaining_edges[:, 1]
        else:
            rem_u = [e[0] for e in remaining_edges]
            rem_v = [e[1] for e in remaining_edges]
            
        edge_rows.extend(rem_u)
        edge_cols.extend(rem_v)
        edge_rows.extend(rem_v) # 反向
        edge_cols.extend(rem_u)

    print("Training finished.")

    # ==========================================
    # Phase 2: Finalizing Graph & Masking
    # ==========================================
    print("\n>>> Phase 2: Finalizing Graph...")
    
    # 1. 注入验证集边 (Validation Edges)
    val_u, val_v = [], []
    if hasattr(train_loader, 'dataset'):
        split_edge = train_loader.dataset.get_edge_split()
        if 'valid' in split_edge:
            val_edges = split_edge['valid']['edge']
            if 'source_node' in val_edges:
                val_u = val_edges['source_node']
                val_v = val_edges['target_node']
            else:
                val_u = val_edges[:, 0]
                val_v = val_edges[:, 1]
    elif hasattr(train_loader, 'all_val_edges'): 
         val_u = train_loader.all_val_edges[:, 0]
         val_v = train_loader.all_val_edges[:, 1]

    if len(val_u) > 0:
        print(f"Injecting {len(val_u)} validation edges...")
        edge_rows.extend(val_u)
        edge_cols.extend(val_v)
        edge_rows.extend(val_v)
        edge_cols.extend(val_u)
    
    # 2. 构建初始矩阵 (COO -> CSR)
    print("Constructing initial CSR matrix...")
    data = np.ones(len(edge_rows), dtype=np.float32)
    A = sp.coo_matrix((data, (edge_rows, edge_cols)), shape=(num_nodes, num_nodes)).tocsr()
    A.data = np.ones_like(A.data) # 强制二值化
    
    # 3. 【强力屏蔽】Masking Test Edges (防止数据泄露)
    print(">>> [Security] Masking Test Edges to prevent leakage...")
    
    raw_test_edges = test_loader.pos_edges
    mask_u, mask_v = None, None
    
    # 稳健的格式提取
    if isinstance(raw_test_edges, dict):
        mask_u = raw_test_edges['source_node']
        mask_v = raw_test_edges['target_node']
    elif torch.is_tensor(raw_test_edges):
        raw_test_edges = raw_test_edges.numpy()
        mask_u = raw_test_edges[:, 0]
        mask_v = raw_test_edges[:, 1]
    elif isinstance(raw_test_edges, np.ndarray):
        mask_u = raw_test_edges[:, 0]
        mask_v = raw_test_edges[:, 1]
    elif isinstance(raw_test_edges, list):
        arr = np.array(raw_test_edges)
        mask_u = arr[:, 0]
        mask_v = arr[:, 1]
        
    if torch.is_tensor(mask_u): mask_u = mask_u.numpy()
    if torch.is_tensor(mask_v): mask_v = mask_v.numpy()

    # Masking Execution
    A = A.tolil() # 转换为 LIL 以便修改
    removed_count = 0
    for i in tqdm(range(len(mask_u)), desc="Masking"):
        u, v = mask_u[i], mask_v[i]
        if A[u, v] != 0: removed_count += 1
        A[u, v] = 0.0
        A[v, u] = 0.0 
        
    print(f"Masked {removed_count} edges that existed in the graph.")
    A = A.tocsr() # 转回 CSR
    A.eliminate_zeros() # 物理删除

    print("Calculating Transition Matrix P...")
    P = to_prmatrix(A)
    print("Graph finalized.")

    # ==========================================
    # Phase 3: Testing (方案 2: PPR Seeded at u)
    # ==========================================
    print(f"\n>>> Phase 3: Testing (Metric: Hits@{hit_k})...")
    
    hits = 0
    total_samples = 0
    
    pbar = tqdm(range(test_loader.num_samples), desc="Testing")
    
    for _ in pbar:
        # Step A: 获取数据
        _, X_ind, _, correct_idx, u, v_target = test_loader.step()
        
        # Step B: 特征
        u_feat = test_loader.node_feat[X_ind[:, 0]]
        v_feat = test_loader.node_feat[X_ind[:, 1]]
        context = np.concatenate((u_feat, v_feat), axis=1)
        
        # Step C: Model Prediction (Bandit Score)
        # 模型根据特征判断 u 和 v 是否匹配
        _, h_scores = ee_net.predict(context, 1e9) # 1e9 关闭探索
        h_scores = h_scores.flatten()
        
        if np.isnan(h_scores).any():
            total_samples += 1
            continue 

        # 对模型分数做 Softmax 归一化，保证它是正数且有概率意义
        # 这样才能和 PPR 分数相乘
        max_val = np.max(h_scores)
        exp_scores = np.exp(h_scores - max_val)
        model_prob = exp_scores / (exp_scores.sum() + 1e-9)

        # Step D: Topological Prediction (PPR Score)
        # 【关键修改】从 User u 开始跑 PageRank
        # 这计算的是：v 离 u 有多近？
        h_vec = np.zeros(num_nodes, dtype=np.float32)
        h_vec[u] = 1.0  # <--- 能量源头是 u
        
        # 运行 PPR
        # 注意：这步比较耗时，如果太慢可以减小 iter 或只跑一部分测试集
        v_ppr = ppr_solver.power_iteration(P, args.alpha, h_vec, t=20)
        
        # 获取候选人的 PPR 分数
        candidate_indices = X_ind[:, 1]
        ppr_scores = v_ppr[candidate_indices]
        
        # Step E: Combination (Final Score)
        # 最终分数 = 拓扑亲密度 (PPR) * 内容匹配度 (Model)
        # 这种组合既利用了图结构，又利用了特征
        final_scores = ppr_scores * model_prob
        
        # 如果你想纯测 Bandit 能力，就用: final_scores = model_prob
        # 如果你想纯测 PPR 基线，就用:    final_scores = ppr_scores
        
        # Step F: Pessimistic Ranking
        pos_score = final_scores[correct_idx]
        
        # 悲观排名：分数 >= 正样本的都算
        rank = (final_scores >= pos_score).sum()
        
        if rank <= hit_k:
            hits += 1
        
        total_samples += 1
        pbar.set_postfix({f'Hits@{hit_k}': f"{hits / total_samples:.4f}"})

    final_acc = hits / total_samples
    print(f"\n{'='*30}")
    print(f"Dataset: {dataset_name}")
    print(f"Result Hits@{hit_k}: {final_acc:.4f}")
    print(f"{'='*30}\n")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='collab', 
                        choices=['collab', 'ppa', 'ddi', 'Cora', 'Citeseer', 'PubMed'])
    # 默认参数参考 PRB 论文 Appendix A.1
    parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate for Exploitation')
    parser.add_argument('--lr2', type=float, default=0.0001, help='Learning rate for Exploration')
    parser.add_argument('--hidden', type=int, default=500, help='Hidden dimension')
    parser.add_argument('--alpha', type=float, default=0.85, help='PageRank damping factor')
    parser.add_argument('--max_train_steps', type=int, default=None, help="Truncate training steps for speed")
    args = parser.parse_args()
    run(args)