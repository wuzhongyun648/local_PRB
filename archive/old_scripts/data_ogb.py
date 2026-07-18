import os
import numpy as np
import scipy.sparse as sp
import torch
from ogb.linkproppred import LinkPropPredDataset
from tqdm import tqdm

# ================= 配置 =================
# 项目根目录 (根据你的服务器路径调整)
PROJECT_ROOT = "/mnt/data/xinyu/Fast_bandit"
# 输出数据的根目录 (对应代码里的 ./data)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
# OGB 原始数据下载目录
OGB_ROOT = os.path.join(PROJECT_ROOT, "dataset")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_negatives(pos_edge_index, num_nodes, num_neg):
    """
    简单的随机负采样
    """
    print(f"  -> Generating {num_neg} negative edges...")
    
    # 使用 Set 加速查找
    # 注意：对于超大图(PPA)，转 Set 可能会慢，这里做个简单处理
    # 为了速度，在大图上我们假设随机采样的边大概率是负的（碰撞率极低）
    
    neg_u = np.random.randint(0, num_nodes, num_neg)
    neg_v = np.random.randint(0, num_nodes, num_neg)
    
    # 简单的自环剔除，暂不进行严格的冲突检测（在大图上效率优先）
    mask = neg_u != neg_v
    neg_u = neg_u[mask]
    neg_v = neg_v[mask]
    
    return np.stack([neg_u, neg_v], axis=1)

def save_npy_format(dataset_name, output_subdir, train_pos, test_pos, node_feat, num_nodes):
    save_path = os.path.join(OUTPUT_DIR, output_subdir)
    ensure_dir(save_path)
    
    print(f"Processing {dataset_name} -> {save_path}")
    
    # 1. 保存特征 (Features)
    # 你的 Loader 需要两个特征文件 U 和 I
    # 对于同构图，U = I
    feat_u_path = os.path.join(save_path, f"{output_subdir}_users_items_features.npy")
    feat_i_path = os.path.join(save_path, f"{output_subdir}_items_users_features.npy")
    
    if node_feat is None:
        print("  -> No features found. Generating Random Embeddings (dim=64)...")
        # DDI 没有特征，生成随机特征
        node_feat = np.random.randn(num_nodes, 64).astype(np.float32)
    else:
        # Collab/PPA 有特征，强制转 float32
        node_feat = node_feat.astype(np.float32)
        
    np.save(feat_u_path, node_feat)
    np.save(feat_i_path, node_feat) # 复制一份，满足 loader 需求
    print("  -> Features saved.")
    
    # 2. 保存训练集 (Train Entry)
    # 格式: [u, v, 1] (正) 和 [u, v, 0] (负)
    # 为了满足 loader 的 self.neg_index，我们需要生成等量的负样本
    num_train = train_pos.shape[0]
    
    # 构造正样本: [u, v, 1]
    train_pos_with_w = np.column_stack([train_pos, np.ones(num_train)])
    
    # 构造负样本: [u, v, 0]
    train_neg = generate_negatives(train_pos, num_nodes, num_train) # 1:1 采样
    train_neg_with_w = np.column_stack([train_neg, np.zeros(len(train_neg))])
    
    # 合并
    train_all = np.vstack([train_pos_with_w, train_neg_with_w])
    # 打乱
    np.random.shuffle(train_all)
    
    train_path = os.path.join(save_path, f"{output_subdir}_users_entry_train.npy")
    np.save(train_path, train_all)
    print(f"  -> Train entries saved: {train_all.shape}")
    
    # 3. 保存测试集 (Test Entry)
    # 同样逻辑
    num_test = test_pos.shape[0]
    test_pos_with_w = np.column_stack([test_pos, np.ones(num_test)])
    test_neg = generate_negatives(test_pos, num_nodes, num_test)
    test_neg_with_w = np.column_stack([test_neg, np.zeros(len(test_neg))])
    
    test_all = np.vstack([test_pos_with_w, test_neg_with_w])
    np.random.shuffle(test_all)
    
    test_path = os.path.join(save_path, f"{output_subdir}_users_entry_test.npy")
    np.save(test_path, test_all)
    print(f"  -> Test entries saved: {test_all.shape}")
    print("------------------------------------------------")

def process_collab():
    dataset = LinkPropPredDataset(name='ogbl-collab', root=OGB_ROOT)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]
    
    # Train: 原始 edge index
    train_pos = split_edge['train']['edge']
    # Test: 原始 edge index (pos)
    test_pos = split_edge['test']['edge']
    
    save_npy_format(
        dataset_name='ogbl-collab',
        output_subdir='collab', # 对应 ./data/collab
        train_pos=train_pos,
        test_pos=test_pos,
        node_feat=graph['node_feat'],
        num_nodes=graph['num_nodes']
    )

def process_ppa():
    dataset = LinkPropPredDataset(name='ogbl-ppa', root=OGB_ROOT)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]
    
    train_pos = split_edge['train']['edge']
    test_pos = split_edge['test']['edge']
    
    # PPA 特征是 One-Hot (58维)，ogb loader 会直接返回
    save_npy_format(
        dataset_name='ogbl-ppa',
        output_subdir='ppa',
        train_pos=train_pos,
        test_pos=test_pos,
        node_feat=graph['node_feat'],
        num_nodes=graph['num_nodes']
    )

def process_ddi():
    dataset = LinkPropPredDataset(name='ogbl-ddi', root=OGB_ROOT)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]
    
    train_pos = split_edge['train']['edge']
    test_pos = split_edge['test']['edge']
    
    # DDI 没有 node_feat，传 None，save_npy_format 会自动生成随机特征
    save_npy_format(
        dataset_name='ogbl-ddi',
        output_subdir='ddi',
        train_pos=train_pos,
        test_pos=test_pos,
        node_feat=None, 
        num_nodes=graph['num_nodes']
    )

if __name__ == "__main__":
    print(f"Preprocessing OGB datasets for Offline Link Prediction...")
    print(f"Source: {OGB_ROOT}")
    print(f"Target: {OUTPUT_DIR}")
    
    process_collab()
    process_ppa()
    process_ddi()
    
    print("\n✅ All datasets processed! You can now run the training script.")