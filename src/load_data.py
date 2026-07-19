import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KDTree
import os
import torch
from src.experiment_configs import DATA_DIR, OGB_ROOT
_original_torch_load = torch.load
def _safe_load_global(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_load_global
try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
except ModuleNotFoundError:
    DataLoader = None
    Planetoid = None
    NormalizeFeatures = None
import gc

class _PlanetoidBaseLoader:
    def __init__(self, name, mode, n_neg_train=1, max_test_neg=1000, seed=42):
        """
        Args:
            name: 'Cora', 'Citeseer', 'PubMed'
            mode: 'train' or 'test' (val is skipped for simplicity here)
        """
        print(f"Loading Planetoid {name} [{mode}] dataset with Random Edge Split...")
        dataset = Planetoid(root=os.path.join(DATA_DIR, name), name=name)
        data = dataset[0]
        
        self.node_feat = data.x.numpy()
        self.dim = self.node_feat.shape[1] * 2
        self.num_nodes = data.num_nodes
        
        edge_index = data.edge_index
        
        row, col = edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        num_edges = row.size(0)
        
        np.random.seed(seed)
        perm = np.random.permutation(num_edges)
        
        n_train = int(num_edges * 0.70)
        n_val = int(num_edges * 0.10)
        n_test = num_edges - n_train - n_val
        
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train+n_val]
        test_idx = perm[n_train+n_val:]
        
        edges_u = row.numpy()
        edges_v = col.numpy()
        
        if mode == 'train':
            self.pos_edges = np.stack([edges_u[train_idx], edges_v[train_idx]], axis=1)
        elif mode == 'test':
            self.pos_edges = np.stack([edges_u[test_idx], edges_v[test_idx]], axis=1)
        else:
            self.pos_edges = np.stack([edges_u[val_idx], edges_v[val_idx]], axis=1)
            
        

        self.mode = mode
        self.n_neg_train = n_neg_train
        self.max_test_neg = max_test_neg
        self.n_arm = 1 + n_neg_train
        self.current_idx = 0
        self.num_samples = len(self.pos_edges)
        
        
        print(f"Loaded {self.num_samples} positive edges for {mode}.")

    def step(self):
        if self.current_idx >= self.num_samples:
            self.current_idx = 0
            
        u, v = self.pos_edges[self.current_idx]
        
        if self.mode == 'train':
            n_neg = self.n_neg_train
        else:
            n_neg = self.max_test_neg if self.max_test_neg is not None else 1000
            
        negs = np.random.randint(0, self.num_nodes, n_neg)
        
        candidate_items = np.concatenate(([v], negs))
        users = np.full(len(candidate_items), u)
        
        X_ind = np.stack([users, candidate_items], axis=1)
        
        rwd = np.zeros(len(candidate_items))
        rwd[0] = 1.0
        correct_idx = 0
        
        X = np.zeros((1, 1))
        
        self.current_idx += 1
        return X, X_ind, rwd, correct_idx, u, v


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
        dataset = Planetoid(root=os.path.join(DATA_DIR, 'Cora_'), name='Cora')
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
        
        dataset = Planetoid(root=DATA_DIR, name='CiteSeer')
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
        
        dataset = Planetoid(root=DATA_DIR, name='PubMed')
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
    def __init__(self, n_neg=9):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "MovieLens/movie_2000users_10000items_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "MovieLens/movie_2000users_10000items_features.npy"))
        self.I = np.load(os.path.join(DATA_DIR, "MovieLens/movie_10000items_2000users_features.npy"))
        self.n_neg = n_neg
        self.n_arm = self.n_neg + 1
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
        arm = np.random.choice(range(self.n_arm))
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)] 
        neg = self.neg_index[np.random.choice(range(self.n_d), self.n_neg, replace=False)]
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
    def __init__(self, n_neg=9):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "Facebook/facebook_combined_ALLusers_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "Facebook/facebook_combined_ALLusers_features.npy"))
        self.n_neg = n_neg
        self.n_arm = self.n_neg + 1
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
        arm = np.random.choice(range(self.n_arm))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), self.n_neg, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.U[ind[1]]))) 
            if arm == self.n_arm - 1 and i == arm:
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
    def __init__(self, n_neg=9):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "GrQc/Insert/GrQc_ALLusers_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "GrQc/GrQc_ALLusers_features.npy"))
        self.n_neg = n_neg
        self.n_arm = self.n_neg + 1
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
        arm = np.random.choice(range(self.n_arm))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), self.n_neg, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.U[ind[1]]))) 
            if arm == self.n_arm - 1 and i == arm:
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
    def __init__(self, n_neg=9):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "Amazon_fashion/new/amazon_fashion_4000users_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "Amazon_fashion/new/amazon_fashion_4000users_4000items_features.npy"))
        self.I = np.load(os.path.join(DATA_DIR, "Amazon_fashion/new/amazon_fashion_4000items_4000users_features.npy"))
        self.n_neg = n_neg
        self.n_arm = self.n_neg + 1
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
        arm = np.random.choice(range(self.n_arm))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), self.n_neg, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 
            if arm == self.n_arm - 1 and i == arm:
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
        print(f"Loading ogbl-{name} [{mode}] dataset...")
        dataset = LinkPropPredDataset(name=f'ogbl-{name}')
        graph = dataset[0]
        split_edge = dataset.get_edge_split()
        
        def to_numpy(x): return x.numpy() if hasattr(x, 'numpy') else x
        self.num_nodes = int(graph['num_nodes'])
        raw_feat = to_numpy(graph['node_feat'])
        
        if raw_feat is not None:
            feat = raw_feat.astype(np.float32)
            if name == 'collab':
                mean = np.mean(feat, axis=0)
                std = np.std(feat, axis=0)
                print(f"Min std: {np.min(std)}")
                self.node_feat = (feat - mean) / (std + 1e-6)
            else:
                self.node_feat = feat
            self.dim = self.node_feat.shape[1] * 2
        else:
            print("No node features found. Initializing random features.")
            np.random.seed(42)
            self.node_feat = np.random.randn(self.num_nodes, 64).astype(np.float32)
            self.dim = 64 * 2
            
        self.mode = mode
        self.pos_edges = split_edge[mode]['edge']  # Shape: (N_edges, 2)
        
        if mode != 'train':
            self.neg_edges = split_edge[mode]['edge_neg'] # Shape: (N_edges, K_neg)
        
        self.n_neg_train = n_neg_train
        self.max_test_neg = max_test_neg
        self.n_arm = 1 + n_neg_train
        self.current_idx = 0
        self.num_samples = len(self.pos_edges)
        
        print(f"Loaded {self.num_samples} positive edges for {mode}.")

    def step(self):
        if self.current_idx >= self.num_samples: 
            self.current_idx = 0
        
        pos = self.pos_edges[self.current_idx]
        if hasattr(pos, 'source_node'): 
            u, v = pos['source_node'], pos['target_node']
        else:
            u, v = pos[0], pos[1]
            
        
        if self.mode == 'train':
            negs = np.random.randint(0, self.num_nodes, self.n_neg_train)
            
        else:
            negs = self.neg_edges[self.current_idx]
            
            if self.max_test_neg is not None and len(negs) > self.max_test_neg:
                negs = negs[:self.max_test_neg]
        
        candidate_items = np.concatenate(([v], negs))
        
        users = np.full(len(candidate_items), u)
        
        X_ind = np.stack([users, candidate_items], axis=1)
        
        rwd = np.zeros(len(candidate_items))
        rwd[0] = 1.0
        correct_idx = 0
        
        X = np.zeros((1, 1)) 
        
        self.current_idx += 1
        
        return X, X_ind, rwd, correct_idx, u, v


class load_collab_train(_OfflineBaseLoader):
    def __init__(self): 
        super().__init__('collab', 'train', n_neg_train=1)

class load_collab_test(_OfflineBaseLoader):
    def __init__(self): 
        super().__init__('collab', 'test', max_test_neg=10000) 


class load_ppa_train(_OfflineBaseLoader):
    def __init__(self): 
        super().__init__('ppa', 'train', n_neg_train=1)

class load_ppa_test(_OfflineBaseLoader):
    def __init__(self): 
        super().__init__('ppa', 'test', max_test_neg=1000)


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
        
        dataset = LinkPropPredDataset(name=dataset_name, root=OGB_ROOT)
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
    def __init__(self, n_neg=9):
        super().__init__('ogbl-collab', n_pos=1, n_neg=n_neg, is_directed=False, need_norm=False)

class load_ogb_ppa(_OGBBaseLoader):
    def __init__(self, n_neg=9):
        super().__init__('ogbl-ppa', n_pos=1, n_neg=n_neg, is_directed=False, need_norm=True)

class load_ogb_vessel(_OGBBaseLoader):
    def __init__(self, n_neg=9):
        super().__init__('ogbl-vessel', n_pos=1, n_neg=n_neg, is_directed=False, need_norm=True)
        
        self.U = self.node_feat
        self.I = self.node_feat
        
        print(f">>> [Hard Negative] Building KDTree for {self.num_nodes} nodes (Spatial Hard Mining)...")
        self.tree = KDTree(self.node_feat) 
        
    def step(self):
        arm = np.random.choice(range(self.n_arm))
        
        idx = np.random.choice(self.p_d)
        pos = self.pos_index[idx] # [u, v]
        u, v = pos[0], pos[1]
        
        
        neg_list = []
        
        ind = self.tree.query(self.node_feat[u].reshape(1, -1), k=500,return_distance=False)
        candidates = ind[0] # [nearest_1, nearest_2, ...]
        
        for cand in candidates:
            if cand == u: continue
            
            if cand == v: continue
            
            if self.adj_gt[u, cand] == 0:
                neg_list.append([u, cand])
            
            if len(neg_list) == self.n_neg:
                break
        
        while len(neg_list) < self.n_neg:
            rnd = np.random.randint(self.num_nodes)
            if rnd != u and self.adj_gt[u, rnd] == 0:
                neg_list.append([u, rnd])
        
        neg = np.array(neg_list)
        
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
    def __init__(self, n_neg=9):
        super().__init__('ogbl-ddi', n_pos=1, n_neg=n_neg, is_directed=False, need_norm=True)

class OGBOfflineTestLoader:
    def __init__(self, dataset_name):
        try:
            from ogb.linkproppred import LinkPropPredDataset
        except ImportError:
            raise ImportError("Please install ogb: pip install ogb")
            
        print(f"-> [Loader] Loading OGB Test Data: {dataset_name} ...")
        self.dataset_name = dataset_name
        self.dataset = LinkPropPredDataset(name=dataset_name, root=OGB_ROOT)
        
        split_edge = self.dataset.get_edge_split()
        self.test_edges = split_edge['test']['edge']     # [num_test, 2]
        self.test_neg = split_edge['test']['edge_neg']   # [num_test, num_neg]
        
        self.graph = self.dataset[0]
        self.num_nodes = int(self.graph['num_nodes'])
        
        raw_feat = self.graph['node_feat']
        if raw_feat is not None:
            feat = np.array(raw_feat, dtype=np.float32)
            mean = np.mean(feat, axis=0)
            std = np.std(feat, axis=0)
            self.node_feat = (feat - mean) / (std + 1e-6)
            self.dim = self.node_feat.shape[1] * 2
        else:
            print("   [Warning] No node features found. Using random embedding.")
            np.random.seed(42)
            self.node_feat = np.random.randn(self.num_nodes, 64).astype(np.float32)
            self.dim = 128
            
        self.num_test = len(self.test_edges)
        print(f"   [Loader] Loaded {self.num_test} test samples.")

    def get_batch(self, idx):
        edge = self.test_edges[idx]
        u, v = int(edge[0]), int(edge[1])
        
        neg_nodes = self.test_neg[idx]
        
        candidates = np.concatenate(([v], neg_nodes)).astype(int)
        
        u_feat = self.node_feat[u]
        cand_feats = self.node_feat[candidates]
        
        # Shape: [num_candidates, feat_dim]
        u_feat_repeated = np.tile(u_feat, (len(candidates), 1))
        X = np.concatenate([u_feat_repeated, cand_feats], axis=1)
        
        return X, u, candidates, 0         
