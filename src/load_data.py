import gc
import copy
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


SPLIT_NAMES = ("online", "validation", "test")
DEFAULT_SPLIT_SEED = 1729


def _partition_edges(edges, seed=DEFAULT_SPLIT_SEED, undirected=False):
    """Partition edge groups deterministically without cross-split overlap."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) < 3:
        raise ValueError("at least three edges are required for three data splits")

    keys = np.sort(edges, axis=1) if undirected else edges
    _, group_ids = np.unique(keys, axis=0, return_inverse=True)
    groups = np.unique(group_ids)
    if len(groups) < 3:
        raise ValueError("at least three unique edge groups are required")

    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    validation_size = max(1, int(0.1 * len(groups)))
    test_size = max(1, int(0.1 * len(groups)))
    online_size = len(groups) - validation_size - test_size
    if online_size < 1:
        online_size, validation_size, test_size = 1, 1, len(groups) - 2
    train_end = online_size
    validation_end = online_size + validation_size
    group_splits = {
        "online": groups[:train_end],
        "validation": groups[train_end:validation_end],
        "test": groups[validation_end:],
    }
    return {
        name: edges[np.isin(group_ids, selected_groups)]
        for name, selected_groups in group_splits.items()
    }


class _SplitLoaderMixin:
    def _configure_splits(
        self,
        positive_edges,
        negative_edges,
        split,
        seed,
        split_seed,
        undirected=False,
    ):
        self._positive_splits = _partition_edges(
            positive_edges, seed=split_seed, undirected=undirected
        )
        self._negative_splits = _partition_edges(
            negative_edges, seed=split_seed + 1, undirected=undirected
        )
        self._activate_split(split, seed)

    def _activate_split(self, split, seed):
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.pos_index = self._positive_splits[split]
        self.neg_index = self._negative_splits[split]
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)

    def for_split(self, split, seed):
        clone = copy.copy(self)
        clone._activate_split(split, seed)
        return clone


class load_movielen(_SplitLoaderMixin):
    def __init__(self, n_neg=9, seed=0, split="online", split_seed=DEFAULT_SPLIT_SEED):
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
            
        self._configure_splits(
            self.pos_index,
            self.neg_index,
            split,
            seed,
            split_seed,
            undirected=False,
        )

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
    
class load_facebook(_SplitLoaderMixin):
    def __init__(self, n_neg=9, seed=0, split="online", split_seed=DEFAULT_SPLIT_SEED):
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
            
        self._configure_splits(
            self.pos_index,
            self.neg_index,
            split,
            seed,
            split_seed,
            undirected=True,
        )

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


class load_grqc(_SplitLoaderMixin):
    def __init__(self, n_neg=9, seed=0, split="online", split_seed=DEFAULT_SPLIT_SEED):
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
            
        self._configure_splits(
            self.pos_index,
            self.neg_index,
            split,
            seed,
            split_seed,
            undirected=True,
        )
        
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


class load_amazon_fashion(_SplitLoaderMixin):
    def __init__(self, n_neg=9, seed=0, split="online", split_seed=DEFAULT_SPLIT_SEED):
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
            
        self._configure_splits(
            self.pos_index,
            self.neg_index,
            split,
            seed,
            split_seed,
            undirected=False,
        )


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

def _ogb_positive_edges(split_entry):
    def to_numpy(value):
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'numpy'):
            return value.numpy()
        return np.asarray(value)

    if 'edge' in split_entry:
        return to_numpy(split_entry['edge']).astype(np.int64)
    if 'source_node' in split_entry:
        return np.stack(
            [
                to_numpy(split_entry['source_node']),
                to_numpy(split_entry['target_node']),
            ],
            axis=1,
        ).astype(np.int64)
    if 'edge_index' in split_entry:
        return to_numpy(split_entry['edge_index']).T.astype(np.int64)
    raise ValueError("unsupported OGB positive-edge format")


def _edge_adjacency(num_nodes, edges, undirected=True):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    rows, cols = edges[:, 0], edges[:, 1]
    if undirected:
        rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
    data = np.ones(len(rows), dtype=np.bool_)
    return sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


class _OGBBaseLoader(_SplitLoaderMixin):
    def __init__(
        self,
        dataset_name,
        n_pos=1,
        n_neg=9,
        is_directed=False,
        need_norm=True,
        seed=0,
        split="online",
        split_seed=DEFAULT_SPLIT_SEED,
    ):
        del split_seed
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
        self._positive_splits = {
            "online": _ogb_positive_edges(split_edge['train']),
            "validation": _ogb_positive_edges(split_edge['valid']),
            "test": _ogb_positive_edges(split_edge['test']),
        }
        self._visible_edge_splits = {
            "online": (self._positive_splits["online"],),
            "validation": (
                self._positive_splits["online"],
                self._positive_splits["validation"],
            ),
            "test": (
                self._positive_splits["online"],
                self._positive_splits["validation"],
                self._positive_splits["test"],
            ),
        }
        # OGB graphs can be very large.  Build only the active split's
        # false-negative filter instead of retaining three CSR matrices.
        self._adjacency_cache = {}
        self._activate_split(split, seed)
        
        self.U = self.node_feat
        self.I = self.node_feat
        
        del split_edge, dataset, graph
        gc.collect()

    def _activate_split(self, split, seed):
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.pos_index = self._positive_splits[split]
        self.p_d = len(self.pos_index)
        if split not in self._adjacency_cache:
            visible_edges = np.concatenate(self._visible_edge_splits[split], axis=0)
            self._adjacency_cache[split] = _edge_adjacency(
                self.num_nodes, visible_edges, undirected=not self.is_directed
            )
        self.adj_gt = self._adjacency_cache[split]

    def for_split(self, split, seed):
        clone = copy.copy(self)
        # Do not let a short-lived test clone retain or mutate the online
        # loader's potentially huge adjacency cache.
        clone._adjacency_cache = {}
        clone._activate_split(split, seed)
        return clone

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
    def __init__(self, n_neg=9, seed=0, split="online", split_seed=DEFAULT_SPLIT_SEED):
        super().__init__('ogbl-collab', n_pos=1, n_neg=n_neg, is_directed=False, need_norm=False, seed=seed, split=split, split_seed=split_seed)

class load_ogb_ppa(_OGBBaseLoader):
    def __init__(self, n_neg=9, seed=0, split="online", split_seed=DEFAULT_SPLIT_SEED):
        super().__init__('ogbl-ppa', n_pos=1, n_neg=n_neg, is_directed=False, need_norm=True, seed=seed, split=split, split_seed=split_seed)

class load_ogb_vessel(_OGBBaseLoader):
    def __init__(self, n_neg=9, seed=0, split="online", split_seed=DEFAULT_SPLIT_SEED):
        super().__init__('ogbl-vessel', n_pos=1, n_neg=n_neg, is_directed=False, need_norm=True, seed=seed, split=split, split_seed=split_seed)
        
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
