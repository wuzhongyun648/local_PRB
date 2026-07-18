from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd 
import scipy.sparse as sp
from sklearn.neighbors import KDTree
import torch, os
_original_torch_load = torch.load
def _safe_load_global(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_load_global
import torchvision
from torchvision import datasets, transforms
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import pdb
import gc
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, 'data')
OGB_ROOT = os.path.join(BASE_PATH, 'dataset')

class _PlanetoidBaseLoader:
    def __init__(self, name, mode, n_neg_train=1, max_test_neg=1000, seed=42):
        """
        Args:
            name: 'Cora', 'Citeseer', 'PubMed'
            mode: 'train' or 'test' (val is skipped for simplicity here)
            n_neg_train: 训练时负采样数 (默认 1)
            max_test_neg: 测试时负采样数 (默认 1000，Ranking Metric)
        """
        print(f"Loading Planetoid {name} [{mode}] dataset with Random Edge Split...")
        # 1. 加载数据
        # 使用 public split 初始化，但我们会覆盖它做 Link Prediction Split
        dataset = Planetoid(root=f'./data/{name}', name=name)
        data = dataset[0]
        
        self.node_feat = data.x.numpy()
        self.dim = self.node_feat.shape[1] * 2
        self.num_nodes = data.num_nodes
        
        # 2. 随机划分边 (Random Edge Split: 70/10/20) 
        # 注意：Planetoid 原始 edge_index 是无向图 (包含 u->v 和 v->u)
        # 我们先转为无向 (只取 u < v) 进行划分，避免数据泄露
        edge_index = data.edge_index
        
        # 仅保留 u < v 的边进行划分
        row, col = edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        num_edges = row.size(0)
        
        # 打乱
        np.random.seed(seed)
        perm = np.random.permutation(num_edges)
        
        # 划分索引
        n_train = int(num_edges * 0.70)
        n_val = int(num_edges * 0.10)
        n_test = num_edges - n_train - n_val
        
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train+n_val]
        test_idx = perm[n_train+n_val:]
        
        # 提取对应的边
        edges_u = row.numpy()
        edges_v = col.numpy()
        
        if mode == 'train':
            self.pos_edges = np.stack([edges_u[train_idx], edges_v[train_idx]], axis=1)
        elif mode == 'test':
            self.pos_edges = np.stack([edges_u[test_idx], edges_v[test_idx]], axis=1)
        else:
            self.pos_edges = np.stack([edges_u[val_idx], edges_v[val_idx]], axis=1)
            
        

        # 3. 设置参数
        self.mode = mode
        self.n_neg_train = n_neg_train
        self.max_test_neg = max_test_neg
        self.n_arm = 1 + n_neg_train
        self.current_idx = 0
        self.num_samples = len(self.pos_edges)
        
        # 对于 Planetoid，负采样是动态的 (因为不像 OGB 那样提供固定负样本)
        # 但为了测试的公平性，我们可以在这里固定随机种子，或者在 step 里动态采
        
        print(f"Loaded {self.num_samples} positive edges for {mode}.")

    def step(self):
        if self.current_idx >= self.num_samples:
            self.current_idx = 0
            
        # A. 获取正样本
        u, v = self.pos_edges[self.current_idx]
        
        # B. 获取负样本
        if self.mode == 'train':
            n_neg = self.n_neg_train
        else:
            # Test 模式: 使用较多的负样本进行 Ranking
            n_neg = self.max_test_neg if self.max_test_neg is not None else 1000
            
        # 简单随机负采样 (不检查 collision，大规模图通常可忽略)
        # 严格来说应该检查 neg != u 和 neg not in neighbors
        # 但为了效率，这里采用简化版。如果需要严格复现，需维护 adj_set 进行过滤。
        negs = np.random.randint(0, self.num_nodes, n_neg)
        
        # C. 组装 (正样本在 idx=0)
        candidate_items = np.concatenate(([v], negs))
        users = np.full(len(candidate_items), u)
        
        X_ind = np.stack([users, candidate_items], axis=1)
        
        # Reward
        rwd = np.zeros(len(candidate_items))
        rwd[0] = 1.0
        correct_idx = 0
        
        # X 占位符
        X = np.zeros((1, 1))
        
        self.current_idx += 1
        return X, X_ind, rwd, correct_idx, u, v

# ==========================================
# 具体数据集 Wrapper
# ==========================================

# --- Cora ---
class load_cora_train(_PlanetoidBaseLoader):
    def __init__(self): super().__init__('Cora', 'train', n_neg_train=1)

class load_cora_test(_PlanetoidBaseLoader):
    def __init__(self): super().__init__('Cora', 'test', max_test_neg=1000) # Hits@100 需要足够的负样本

# --- Citeseer ---
class load_citeseer_train(_PlanetoidBaseLoader):
    def __init__(self): super().__init__('Citeseer', 'train', n_neg_train=1)

class load_citeseer_test(_PlanetoidBaseLoader):
    def __init__(self): super().__init__('Citeseer', 'test', max_test_neg=1000)

# --- PubMed ---
class load_pubmed_train(_PlanetoidBaseLoader):
    def __init__(self): super().__init__('PubMed', 'train', n_neg_train=1)

class load_pubmed_test(_PlanetoidBaseLoader):
    def __init__(self): super().__init__('PubMed', 'test', max_test_neg=1000)
    
class load_cora:
    def __init__(self):
        batch_size = 1
        dataset = Planetoid(root='./data/Cora_', name='Cora')
        self.data = dataset[0]
        
        self.X_all,self.edge_index_all, self.Y_all = self.data.x, self.data.edge_index, self.data.y
        self.n_arm = 7
        self.node_size = 2708
        self.dim = 10031
        self.data_index = 0

    def step(self):  
        x_idx = self.data_index
        x, y = self.X_all[x_idx, :], self.Y_all[x_idx]
        d = x.numpy()
        target = int(y.item())
        # print(target)
        X_n = []
        X_ind = [[x_idx, self.node_size + i] for i in range(self.n_arm)]
        for i in range(7):
            front = np.zeros((1433 * i))
            back = np.zeros((1433 * (6 - i)))
            new_d = np.concatenate((front, d, back), axis=0)
            X_n.append(new_d)

        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        
        self.data_index += 1
        if self.data_index == 2708:
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())
        # pdb.set_trace()
        return X_n, X_ind, rwd, target,node1_idx, node2_idx

class load_citeseer:
    def __init__(self):
        
        dataset = Planetoid(root='./data', name='CiteSeer')
        self.data = dataset[0]
        
        self.X_all,self.edge_index_all, self.Y_all = self.data.x, self.data.edge_index, self.data.y
        self.n_arm = 6
        self.node_size = 3327
        self.dim = 6 * 3703
        self.data_index = 0

    def step(self):  
        x_idx = self.data_index
        x, y = self.X_all[x_idx, :], self.Y_all[x_idx]
        d = x.numpy()
        target = int(y.item())
        X_n = []
        X_ind = [[x_idx, self.node_size + i] for i in range(self.n_arm)]
        for i in range(self.n_arm):
            front = np.zeros((3703 * i))
            back = np.zeros((3703 * (5 - i)))
            new_d = np.concatenate((front, d, back), axis=0)
            X_n.append(new_d)

        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        
        self.data_index += 1
        if self.data_index == 3327:
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())
        # pdb.set_trace()
        return X_n, X_ind, rwd, target,node1_idx, node2_idx
    
class load_pubmed:
    def __init__(self):
        
        dataset = Planetoid(root='./data', name='PubMed')
        self.data = dataset[0]
        
        self.X_all,self.edge_index_all, self.Y_all = self.data.x, self.data.edge_index, self.data.y
        self.n_arm = 3
        self.node_size = 19717
        self.dim = 3 * 500
        self.data_index = 0

    def step(self):  
        x_idx = self.data_index
        x, y = self.X_all[x_idx, :], self.Y_all[x_idx]
        d = x.numpy()
        target = int(y.item())
        X_n = []
        X_ind = [[x_idx, self.node_size + i] for i in range(self.n_arm)]
        for i in range(self.n_arm):
            front = np.zeros((500 * i))
            back = np.zeros((500 * (2 - i)))
            new_d = np.concatenate((front, d, back), axis=0)
            X_n.append(new_d)

        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        
        self.data_index += 1
        if self.data_index == 19717:
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())
        # pdb.set_trace()
        return X_n, X_ind, rwd, target,node1_idx, node2_idx
    
    
    
class load_movielen:
    def __init__(self):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "MovieLens/movie_2000users_10000items_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "MovieLens/movie_2000users_10000items_features.npy"))
        self.I = np.load(os.path.join(DATA_DIR, "MovieLens/movie_10000items_2000users_features.npy"))
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else: # i[2] == -1
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        # print('self.p_d and self.n_d is:',self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)

    def testing_dataset(self):
        test_data = []
        for _ in range(100):
            step_result = self.step()
            test_data.append(step_result)
        return test_data
    
    def step(self):        
        arm = np.random.choice(range(10))
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)] 
        neg = self.neg_index[np.random.choice(range(self.n_d), 9, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        
        X = []
        for i, ind in enumerate(X_ind):
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 
            if i == arm: 
                user = ind[0]
                item = ind[1]
                
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1 
        
        return np.array(X), X_ind, rwd, arm, user, item
    
class load_facebook:
    def __init__(self):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "Facebook/facebook_combined_ALLusers_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "Facebook/facebook_combined_ALLusers_features.npy"))
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else: # i[2] == -1
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        # print('self.p_d and self.n_d is:',self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)

    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 9, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.U[ind[1]]))) 
            if arm == 9 and i == arm:
                user = ind[0]
                item = ind[1]
            elif i == arm + 1 : 
                user = ind[0]
                item = ind[1]
        # print("X is \n",X)
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X),X_ind, rwd, arm, user, item  # arm is the one that randomly picked up and settled to 1
    
    def testing_dataset(self):
        test_data = []
        for _ in range(100):
            step_result = self.step()
            test_data.append(step_result)
        return test_data


class load_grqc:    
    def __init__(self):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "GrQc/Insert/GrQc_ALLusers_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "GrQc/GrQc_ALLusers_features.npy"))
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else: # i[2] == -1
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        # print('self.p_d and self.n_d is:',self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)
        
    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 9, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.U[ind[1]]))) 
            if arm == 9 and i == arm:
                user = ind[0]
                item = ind[1]
            elif i == arm + 1 : 
                user = ind[0]
                item = ind[1]
        # print("X is \n",X)
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X),X_ind, rwd, arm, user, item  # arm is the one that randomly picked up and settled to 1
    def testing_dataset(self):
        test_data = []
        for _ in range(100):
            step_result = self.step()
            test_data.append(step_result)
        return test_data


class load_amazon_fashion:
    def __init__(self):
        # Fetch data
        self.m = np.load("./data/Amazon_fashion/new/amazon_fashion_4000users_entry.npy")
        self.U = np.load("./data/Amazon_fashion/new/amazon_fashion_4000users_4000items_features.npy")
        self.I = np.load("./data/Amazon_fashion/new/amazon_fashion_4000items_4000users_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else:
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        print(self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 9, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 
            if arm == 9 and i == arm:
                user = ind[0]
                item = ind[1]
            elif i == arm + 1 : 
                user = ind[0]
                item = ind[1]
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X),X_ind, rwd, arm, user, item  # arm is the one that randomly picked up and settled to 1
    def testing_dataset(self):
        test_data = []
        for _ in range(100):
            step_result = self.step()
            test_data.append(step_result)
        return test_data
    


class _OfflineBaseLoader:
    def __init__(self, name, mode, n_neg_train=1, max_test_neg=None):
        """
        Args:
            name: 数据集名称 (e.g., 'collab', 'ppa', 'ddi')
            mode: 'train', 'valid', 或 'test'
            n_neg_train: 训练时的负采样数量 (通常为 1 或 5)
            max_test_neg: 测试时最大负样本数量 (用于防止 OOM，None 表示使用全部 OGB 负样本)
        """
        print(f"Loading ogbl-{name} [{mode}] dataset...")
        dataset = LinkPropPredDataset(name=f'ogbl-{name}')
        graph = dataset[0]
        split_edge = dataset.get_edge_split()
        
        # 1. 处理特征
        def to_numpy(x): return x.numpy() if hasattr(x, 'numpy') else x
        self.num_nodes = int(graph['num_nodes'])
        raw_feat = to_numpy(graph['node_feat'])
        
        # Collab 特征通常需要归一化，其他数据集视情况而定
        if raw_feat is not None:
            feat = raw_feat.astype(np.float32)
            if name == 'collab':
                # 标准化: (x - mean) / std
                mean = np.mean(feat, axis=0)
                std = np.std(feat, axis=0)
                print(f"Min std: {np.min(std)}")
                self.node_feat = (feat - mean) / (std + 1e-6)
            else:
                self.node_feat = feat
            self.dim = self.node_feat.shape[1] * 2
        else:
            # 如果没有特征 (如 DDI)，初始化随机特征或 One-hot (显存允许的话)
            # 这里使用随机特征作为示例，维度设为 64 或 128
            print("No node features found. Initializing random features.")
            np.random.seed(42)
            self.node_feat = np.random.randn(self.num_nodes, 64).astype(np.float32)
            self.dim = 64 * 2
            
        # 2. 获取边分割
        self.mode = mode
        self.pos_edges = split_edge[mode]['edge']  # Shape: (N_edges, 2)
        
        # 3. 获取负样本 (仅 Test/Valid 模式有固定负样本)
        if mode != 'train':
            self.neg_edges = split_edge[mode]['edge_neg'] # Shape: (N_edges, K_neg)
        
        # 4. 设置参数
        self.n_neg_train = n_neg_train
        self.max_test_neg = max_test_neg
        self.n_arm = 1 + n_neg_train
        self.current_idx = 0
        self.num_samples = len(self.pos_edges)
        
        print(f"Loaded {self.num_samples} positive edges for {mode}.")

    def step(self):
        # 循环遍历数据集
        if self.current_idx >= self.num_samples: 
            self.current_idx = 0
        
        # --- A. 获取正样本 (Source u, Target v) ---
        pos = self.pos_edges[self.current_idx]
        if hasattr(pos, 'source_node'): # 处理不同格式
            u, v = pos['source_node'], pos['target_node']
        else:
            u, v = pos[0], pos[1]
            
        # --- B. 获取候选集 (Candidates) ---
        # 约定：Candidates 列表的第 0 个位置永远是正样本 v
        
        if self.mode == 'train':
            # === Train 模式: 随机负采样 ===
            # 随机采样 n_neg_train 个负样本
            # 简单起见，这里没有做严格的 "碰撞检测" (即采样的负样本可能是正样本)
            # 在大规模稀疏图中，碰撞概率极低，这也是通用做法
            negs = np.random.randint(0, self.num_nodes, self.n_neg_train)
            
        else:
            # === Test/Valid 模式: 使用 OGB 固定负样本 ===
            negs = self.neg_edges[self.current_idx]
            
            # 截断逻辑 (防止 OOM 或为了快速评估)
            if self.max_test_neg is not None and len(negs) > self.max_test_neg:
                negs = negs[:self.max_test_neg]
        
        # --- C. 组装数据 ---
        # 候选目标节点: [v, neg1, neg2, ...]
        candidate_items = np.concatenate(([v], negs))
        
        # 对应的源节点: [u, u, u, ...]
        users = np.full(len(candidate_items), u)
        
        # 构造 X_ind: Shape (K, 2)
        # 每一行是 [u, candidate_v]
        X_ind = np.stack([users, candidate_items], axis=1)
        
        # 构造 Reward (仅用于 Train, Test 时忽略)
        # 正样本在第 0 位
        rwd = np.zeros(len(candidate_items))
        rwd[0] = 1.0
        correct_idx = 0
        
        # 构造 X (占位符)
        # 因为你的 main.py 会使用 b.node_feat[X_ind] 来构建特征
        # 所以这里返回一个空占位符即可，节省 IO
        X = np.zeros((1, 1)) 
        
        self.current_idx += 1
        
        # 返回:
        # X: 占位符
        # X_ind: 索引矩阵，用于去 node_feat 查表
        # rwd: 奖励向量
        # correct_idx: 正样本在列表中的索引 (始终为0)
        # u, v: 原始正样本边
        return X, X_ind, rwd, correct_idx, u, v

# ==========================================
# 具体数据集的 Wrapper 类
# ==========================================

# --- Collab ---
# Train: 1 pos vs 1 neg (PRB/BUDDY 默认配置)
# Test: 1 pos vs 500 negs (你可以改为 None 使用全量 100k，但速度会慢)
class load_collab_train(_OfflineBaseLoader):
    def __init__(self): 
        super().__init__('collab', 'train', n_neg_train=1)

class load_collab_test(_OfflineBaseLoader):
    def __init__(self): 
        super().__init__('collab', 'test', max_test_neg=10000) 


# --- PPA ---
# PPA 图非常大，建议 Test 截断，否则单步推理太慢
class load_ppa_train(_OfflineBaseLoader):
    def __init__(self): 
        super().__init__('ppa', 'train', n_neg_train=1)

class load_ppa_test(_OfflineBaseLoader):
    def __init__(self): 
        super().__init__('ppa', 'test', max_test_neg=1000)


# --- DDI ---
# DDI 只有结构没有特征，BaseLoader 会自动初始化随机特征
class load_ddi_train(_OfflineBaseLoader):
    def __init__(self): 
        super().__init__('ddi', 'train', n_neg_train=1)

class load_ddi_test(_OfflineBaseLoader):
    def __init__(self): 
        super().__init__('ddi', 'test', max_test_neg=500)

    
# OGB Data Loaders 
_original_load = torch.load
def _safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load

try:
    from ogb.linkproppred import LinkPropPredDataset
    HAS_OGB = True
except ImportError:
    HAS_OGB = False

class _OGBBaseLoader:
    def __init__(self, dataset_name, n_pos=1, n_neg=9, is_directed=False, need_norm=True):
        if not HAS_OGB:
            raise ImportError("Please install ogb: pip install ogb")
            
        self.dataset_name = dataset_name
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.n_arm = n_pos + n_neg
        self.is_directed = is_directed
        
        dataset = LinkPropPredDataset(name=dataset_name, root='./dataset')
        graph = dataset[0]
        split_edge = dataset.get_edge_split()
        
        def to_numpy(x):
            if hasattr(x, 'cpu'): x = x.cpu()
            if hasattr(x, 'detach'): x = x.detach()
            if hasattr(x, 'numpy'): return x.numpy()
            return np.array(x) if not isinstance(x, np.ndarray) else x
            
        self.num_nodes = int(graph['num_nodes'])
        
        if graph['node_feat'] is None:
            self.node_feat = np.random.randn(self.num_nodes, 64).astype(np.float32)
            self.dim = 128
        else:
            raw_feat = to_numpy(graph['node_feat'])
            feat = raw_feat.astype(np.float32)
            if need_norm:
                mean = np.mean(feat, axis=0)
                std = np.std(feat, axis=0)
                self.node_feat = (feat - mean) / (std + 1e-6)
            else:
                self.node_feat = feat
            self.dim = self.node_feat.shape[1] * 2
            del raw_feat, feat
            gc.collect()
        rows, cols = [], []
        for split in ['train', 'valid', 'test']:
            if split not in split_edge: continue
            edge = split_edge[split]
            if 'edge' in edge: e = to_numpy(edge['edge'])
            elif 'source_node' in edge: 
                src = to_numpy(edge['source_node'])
                dst = to_numpy(edge['target_node'])
                e = np.stack([src, dst], axis=1)
            elif 'edge_index' in edge: e = to_numpy(edge['edge_index']).T
            else: continue
            rows.append(e[:, 0])
            cols.append(e[:, 1])
            
        all_rows = np.concatenate(rows).astype(np.int32)
        all_cols = np.concatenate(cols).astype(np.int32)
        data = np.ones(len(all_rows), dtype=np.bool_) 
        self.adj_gt = sp.csr_matrix((data, (all_rows, all_cols)), shape=(self.num_nodes, self.num_nodes))
        

        train_edge = split_edge['train']
        if 'edge' in train_edge: self.pos_index = to_numpy(train_edge['edge'])
        elif 'source_node' in train_edge: 
            src = to_numpy(train_edge['source_node'])
            dst = to_numpy(train_edge['target_node'])
            self.pos_index = np.stack([src, dst], axis=1)
        elif 'edge_index' in train_edge: self.pos_index = to_numpy(train_edge['edge_index']).T
        
        self.p_d = len(self.pos_index)
        
        self.U = self.node_feat
        self.I = self.node_feat
        
        del split_edge, dataset, graph
        gc.collect()

    def step(self):
        arm = np.random.choice(range(self.n_arm))
        idx = np.random.choice(self.p_d)
        pos = self.pos_index[idx] # shape (2,)
        neg_list = []
        while len(neg_list) < self.n_neg:
            u = np.random.randint(self.num_nodes)
            v = np.random.randint(self.num_nodes)
            if self.adj_gt[u, v] == 0:
                neg_list.append([u, v])
        neg = np.array(neg_list) # shape (9, 2)
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        
        X = []
        for i, ind in enumerate(X_ind):
            u, v = ind[0], ind[1]
            X.append(np.concatenate((self.U[u], self.I[v]))) 
            
            if i == arm:
                user = u
                item = v
                
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1.0
        
        return np.array(X), X_ind, rwd, arm, user, item
    def testing_dataset(self):
        test_data = []
        for _ in range(100):
            step_result = self.step()
            test_data.append(step_result)
        return test_data

class load_ogb_collab(_OGBBaseLoader):
    def __init__(self):
        super().__init__('ogbl-collab', n_pos=1, n_neg=9, is_directed=False, need_norm=False)

class load_ogb_ppa(_OGBBaseLoader):
    def __init__(self):
        super().__init__('ogbl-ppa', n_pos=1, n_neg=9, is_directed=False, need_norm=True)

class load_ogb_vessel(_OGBBaseLoader):
    def __init__(self):
        # 1. 初始化：正常加载 Vessel 数据 (会加载原始的 3D 坐标特征)
        super().__init__('ogbl-vessel', n_pos=1, n_neg=9, is_directed=False, need_norm=True)
        
        # 2. 【修复】确保 self.U 和 self.I 指向原始特征
        # 之前方案一里我们生成了随机特征，现在要改回来使用 super() 加载的 node_feat
        self.U = self.node_feat
        self.I = self.node_feat
        
        # 3. 【方案二核心】构建 KDTree 索引
        # 这允许我们在 O(log N) 时间内找到空间上最近的邻居
        print(f">>> [Hard Negative] Building KDTree for {self.num_nodes} nodes (Spatial Hard Mining)...")
        self.tree = KDTree(self.node_feat) 
        
    def step(self):
        # 1. 随机选 Arm
        arm = np.random.choice(range(self.n_arm))
        
        # 2. 采样正样本 (u, v)
        idx = np.random.choice(self.p_d)
        pos = self.pos_index[idx] # [u, v]
        u, v = pos[0], pos[1]
        
        # 3. 采样硬负样本 (Spatial Hard Negatives)
        # 目标：找到离 u 很近，但不是 v，也不是其他真实邻居的点
        
        neg_list = []
        
        # 查询 u 附近的 k 个邻居 (k=50 是一个经验值，保证能筛出 9 个非邻居)
        # return_distance=False, 只返回索引
        # query 输入需要是 (1, dim)
        ind = self.tree.query(self.node_feat[u].reshape(1, -1), k=500,return_distance=False)
        candidates = ind[0] # [nearest_1, nearest_2, ...]
        
        for cand in candidates:
            # 跳过自己
            if cand == u: continue
            
            # 跳过正样本 v (虽然概率低，但以防万一)
            if cand == v: continue
            
            # 【关键】检查是否是真实邻居
            # 只有当 cand 和 u 没有边连接时，才算“硬负样本”
            if self.adj_gt[u, cand] == 0:
                neg_list.append([u, cand])
            
            # 凑够 9 个就停止
            if len(neg_list) == self.n_neg:
                break
        
        # 4. 兜底逻辑 (Fallback)
        # 如果该节点非常孤立，KDTree 找出来的全是邻居(不太可能)或者点太少，
        # 就用随机负样本填充剩余空位
        while len(neg_list) < self.n_neg:
            rnd = np.random.randint(self.num_nodes)
            if rnd != u and self.adj_gt[u, rnd] == 0:
                neg_list.append([u, rnd])
        
        neg = np.array(neg_list)
        
        # 5. 组装 Batch (和以前一样)
        pos_entry = np.array([[u, v]])
        X_ind = np.concatenate((neg[:arm], pos_entry, neg[arm:]), axis=0)
        
        X = []
        for i, idx_pair in enumerate(X_ind):
            curr_u, curr_v = idx_pair[0], idx_pair[1]
            X.append(np.concatenate((self.U[curr_u], self.I[curr_v])))
            
            if i == arm:
                target_user = curr_u
                target_item = curr_v
        
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1.0
        
        return np.array(X), X_ind, rwd, arm, target_user, target_item

class load_ogb_ddi(_OGBBaseLoader):
    def __init__(self):
        super().__init__('ogbl-ddi', n_pos=1, n_neg=9, is_directed=False, need_norm=True)

class OGBOfflineTestLoader:
    def __init__(self, dataset_name):
        try:
            from ogb.linkproppred import LinkPropPredDataset
        except ImportError:
            raise ImportError("Please install ogb: pip install ogb")
            
        print(f"-> [Loader] Loading OGB Test Data: {dataset_name} ...")
        self.dataset_name = dataset_name
        self.dataset = LinkPropPredDataset(name=dataset_name, root='./dataset')
        
        # 1. 获取官方划分 (Train/Valid/Test)
        split_edge = self.dataset.get_edge_split()
        self.test_edges = split_edge['test']['edge']     # [num_test, 2]
        self.test_neg = split_edge['test']['edge_neg']   # [num_test, num_neg]
        
        # 2. 处理节点特征
        self.graph = self.dataset[0]
        self.num_nodes = int(self.graph['num_nodes'])
        
        raw_feat = self.graph['node_feat']
        if raw_feat is not None:
            # 简单的归一化处理，与 ELPH/BUDDY 保持一致
            feat = np.array(raw_feat, dtype=np.float32)
            mean = np.mean(feat, axis=0)
            std = np.std(feat, axis=0)
            self.node_feat = (feat - mean) / (std + 1e-6)
            self.dim = self.node_feat.shape[1] * 2
        else:
            # 如果数据集没有特征 (如 DDI)，使用随机特征或 One-hot (取决于具体实现，这里给个默认)
            # 只有 ogbl-ddi 没有特征
            print("   [Warning] No node features found. Using random embedding.")
            np.random.seed(42)
            self.node_feat = np.random.randn(self.num_nodes, 64).astype(np.float32)
            self.dim = 128
            
        self.num_test = len(self.test_edges)
        print(f"   [Loader] Loaded {self.num_test} test samples.")

    def get_batch(self, idx):
        """
        获取第 idx 个测试样本及其对应的所有负样本
        Returns:
            X: (num_candidates, dim) - 特征矩阵
            u: int - 用户ID (Source Node)
            candidates: np.array - [Pos_Item, Neg_Item_1, Neg_Item_2, ...]
            pos_idx: int - 正样本在 candidates 中的索引 (通常是 0)
        """
        # 获取正样本 (u, v)
        edge = self.test_edges[idx]
        u, v = int(edge[0]), int(edge[1])
        
        # 获取对应的负样本列表
        neg_nodes = self.test_neg[idx]
        
        # 构建候选集: 第一个是正样本，后面是负样本
        candidates = np.concatenate(([v], neg_nodes)).astype(int)
        
        # 构建 Context Features: [User_Feat || Candidate_Feat]
        u_feat = self.node_feat[u]
        cand_feats = self.node_feat[candidates]
        
        # 广播 user feature 并拼接
        # Shape: [num_candidates, feat_dim]
        u_feat_repeated = np.tile(u_feat, (len(candidates), 1))
        X = np.concatenate([u_feat_repeated, cand_feats], axis=1)
        
        return X, u, candidates, 0 # 0 是正样本的位置        