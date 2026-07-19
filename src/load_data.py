import gc
import os

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import KDTree

from src.experiment_configs import DATA_DIR, OGB_ROOT

_original_torch_load = torch.load


def _safe_load_global(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _safe_load_global


class load_movielen:
    def __init__(self, n_neg=9, seed=0):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "MovieLens/movie_2000users_10000items_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "MovieLens/movie_2000users_10000items_features.npy"))
        self.I = np.load(os.path.join(DATA_DIR, "MovieLens/movie_10000items_2000users_features.npy"))
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)
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
        arm = self.rng.choice(self.n_arm)
        pos = self.pos_index[self.rng.choice(self.p_d)]
        neg = self.neg_index[self.rng.choice(self.n_d, self.n_neg, replace=False)]
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
    def __init__(self, n_neg=9, seed=0):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "Facebook/facebook_combined_ALLusers_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "Facebook/facebook_combined_ALLusers_features.npy"))
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)
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
        arm = self.rng.choice(self.n_arm)
        #print(pos_index.shape)
        pos = self.pos_index[self.rng.choice(self.p_d)]
        user, item = int(pos[0]), int(pos[1])
        neg = self.neg_index[self.rng.choice(self.n_d, self.n_neg, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.U[ind[1]]))) 
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
    def __init__(self, n_neg=9, seed=0):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "GrQc/Insert/GrQc_ALLusers_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "GrQc/GrQc_ALLusers_features.npy"))
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)
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
        arm = self.rng.choice(self.n_arm)
        #print(pos_index.shape)
        pos = self.pos_index[self.rng.choice(self.p_d)]
        user, item = int(pos[0]), int(pos[1])
        neg = self.neg_index[self.rng.choice(self.n_d, self.n_neg, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.U[ind[1]]))) 
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
    def __init__(self, n_neg=9, seed=0):
        # Fetch data
        self.m = np.load(os.path.join(DATA_DIR, "Amazon_fashion/new/amazon_fashion_4000users_entry.npy"))
        self.U = np.load(os.path.join(DATA_DIR, "Amazon_fashion/new/amazon_fashion_4000users_4000items_features.npy"))
        self.I = np.load(os.path.join(DATA_DIR, "Amazon_fashion/new/amazon_fashion_4000items_4000users_features.npy"))
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)
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
        arm = self.rng.choice(self.n_arm)
        #print(pos_index.shape)
        pos = self.pos_index[self.rng.choice(self.p_d)]
        user, item = int(pos[0]), int(pos[1])
        neg = self.neg_index[self.rng.choice(self.n_d, self.n_neg, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X),X_ind, rwd, arm, user, item  # arm is the one that randomly picked up and settled to 1
    def testing_dataset(self):
        test_data = []
        for _ in range(100):
            step_result = self.step()
            test_data.append(step_result)
        return test_data
    


# OGB Data Loaders 

try:
    from ogb.linkproppred import LinkPropPredDataset
    HAS_OGB = True
except ImportError:
    HAS_OGB = False

class _OGBBaseLoader:
    def __init__(self, dataset_name, n_pos=1, n_neg=9, is_directed=False, need_norm=True, seed=0):
        if not HAS_OGB:
            raise ImportError("Please install ogb: pip install ogb")
            
        self.dataset_name = dataset_name
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)
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
            self.node_feat = self.rng.standard_normal((self.num_nodes, 64)).astype(np.float32)
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
        arm = self.rng.choice(self.n_arm)
        idx = self.rng.choice(self.p_d)
        pos = self.pos_index[idx] # shape (2,)
        neg_list = []
        while len(neg_list) < self.n_neg:
            u = self.rng.integers(self.num_nodes)
            v = self.rng.integers(self.num_nodes)
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
    def __init__(self, n_neg=9, seed=0):
        super().__init__('ogbl-collab', n_pos=1, n_neg=n_neg, is_directed=False, need_norm=False, seed=seed)

class load_ogb_ppa(_OGBBaseLoader):
    def __init__(self, n_neg=9, seed=0):
        super().__init__('ogbl-ppa', n_pos=1, n_neg=n_neg, is_directed=False, need_norm=True, seed=seed)

class load_ogb_vessel(_OGBBaseLoader):
    def __init__(self, n_neg=9, seed=0):
        super().__init__('ogbl-vessel', n_pos=1, n_neg=n_neg, is_directed=False, need_norm=True, seed=seed)
        
        self.U = self.node_feat
        self.I = self.node_feat
        
        print(f">>> [Hard Negative] Building KDTree for {self.num_nodes} nodes (Spatial Hard Mining)...")
        self.tree = KDTree(self.node_feat) 
        
    def step(self):
        arm = self.rng.choice(self.n_arm)
        
        idx = self.rng.choice(self.p_d)
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
            rnd = self.rng.integers(self.num_nodes)
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
