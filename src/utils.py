import numpy as np
import scipy.sparse as sp
import torch
from ogb.linkproppred import LinkPropPredDataset
import os
import json
from src.experiment_configs import DATA_DIR

_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, payload):
    """Persist structured experiment metadata in a deterministic format."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def resolve_ee_net_kernel_size(graph_name, kernel_size_arg):
    """
    When kernel_size_arg is None: Vessel uses 5 (exploration Conv1d), other graphs use 40.
    If kernel_size_arg is set, use that value for all datasets.
    """
    if kernel_size_arg is not None:
        return kernel_size_arg
    return 5 if graph_name == "Vessel" else 40


def save_results(save_dir, results_list, is_final=False, args=None):
    """
    Persists experimental results and configuration metadata to storage.
    
    This function handles the serialization of runtime metrics (e.g., regret, time, loss) 
    into .npy files. It supports both intermediate checkpoints and final result saving, 
    and optionally records the experimental hyperparameters into a text file upon completion.
    """
    
    data_array = np.array(results_list)# [[time, regret, loss1, loss2, ppr_norm], ...]
    if is_final:
        file_name = "final_results.npy"
        if args is not None:
            config_path = os.path.join(save_dir, "config.txt")
            try:
                with open(config_path, "w") as f:
                    if hasattr(args, '__dict__'):
                        for k, v in vars(args).items():
                            f.write(f"{k}: {v}\n")
                    else:
                        f.write(str(args))
            except Exception as e:
                print(f"Warning: Failed to save config.txt: {e}")
    else:
        file_name = f"checkpoint_step_{len(data_array)}.npy"
        
    file_path = os.path.join(save_dir, file_name)
    np.save(file_path, data_array)
    
    if is_final:
        print(f"-> [Save] Final results saved to: {file_path}")


def _to_numpy(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _positive_entry_edges(entry_path):
    entry = np.load(entry_path)
    return entry[entry[:, 2] == 1][:, :2].astype(np.int64)


def _topk_hop_edges(num_nodes, edges, init_hops, init_topk):
    if init_hops <= 0 or init_topk <= 0:
        return edges, np.array([], dtype=np.int64)

    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return edges.reshape(0, 2), np.array([], dtype=np.int64)

    edges = np.unique(edges, axis=0)
    deg = np.bincount(
        np.concatenate([edges[:, 0], edges[:, 1]]),
        minlength=num_nodes,
    )
    nonzero_nodes = np.flatnonzero(deg)
    if nonzero_nodes.size == 0:
        return edges[:0], np.array([], dtype=np.int64)

    sorted_nodes = nonzero_nodes[np.argsort(-deg[nonzero_nodes], kind="stable")]
    seeds = sorted_nodes[: min(init_topk, sorted_nodes.size)]

    visited = np.zeros(num_nodes, dtype=bool)
    frontier = seeds.copy()
    visited[frontier] = True
    edge_indices = []
    for _ in range(init_hops):
        if frontier.size == 0:
            break
        frontier_mask = np.zeros(num_nodes, dtype=bool)
        frontier_mask[frontier] = True
        traversed = frontier_mask[edges[:, 0]] | frontier_mask[edges[:, 1]]
        traversed_idx = np.flatnonzero(traversed)
        if traversed_idx.size == 0:
            break
        edge_indices.append(traversed_idx)
        endpoints = edges[traversed].reshape(-1)
        neighbors = np.unique(endpoints[~visited[endpoints]])
        if neighbors.size == 0:
            break
        visited[neighbors] = True
        frontier = neighbors

    if not edge_indices:
        return edges[:0], seeds
    edge_indices = np.unique(np.concatenate(edge_indices))
    return edges[edge_indices], seeds


def _normalize_columns(A, format="csc"):
    A_csc = A.tocsc()
    D = np.array(A_csc.sum(axis=0)).flatten().astype(float)
    D_inv = np.zeros_like(D)
    nonzero_mask = D != 0
    D_inv[nonzero_mask] = 1.0 / D[nonzero_mask]
    P = A_csc.multiply(D_inv)
    return P.asformat(format)


def _log_init_graph(name, init_hops, init_topk, seeds, num_edges):
    if init_hops > 0 and init_topk > 0:
        seed_preview = seeds[:10].tolist()
        print(
            f"-> [{name}] init_hops={init_hops}, init_topk={init_topk}, "
            f"seeds={seed_preview}, init_edges={num_edges}",
            flush=True,
        )
        
class graph:
    def __init__(self, path):
        self.path = path
        
    def load(self):
        """
        Loads the graph data and initializes structural matrices
        """
        pass
    
    def get(self):
        """
        Retrieves the current state of graph structural components.
        """
        pass
    
    def update(self):
        """
        Updates the directed graph structure when a new link is formed.
        """
        pass
    

class MovieLens(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, num_users, num_items, path = None, init_hops=0, init_topk=0):
        if not path:
            path = self.path
        self.G = np.load(path)
        self.num_items = num_items
        self.num_users = num_users
        num_nodes = num_users + num_items
        if init_hops > 0 and init_topk > 0:
            entry_path = os.path.join(DATA_DIR, "MovieLens/movie_2000users_10000items_entry.npy")
            raw_edges = _positive_entry_edges(entry_path)
            full_edges = np.column_stack((raw_edges[:, 0], raw_edges[:, 1] + num_users))
            init_edges, seeds = _topk_hop_edges(num_nodes, full_edges, init_hops, init_topk)
            G = np.column_stack((init_edges[:, 0], init_edges[:, 1] - num_users))
        else:
            G = self.G
            seeds = np.array([], dtype=np.int64)
        A = sp.lil_matrix((num_nodes, num_nodes))
        for i in range(len(G)):
            if init_hops > 0 and init_topk > 0:
                user, item = G[i]
                A[item + num_users, user] = 1
                A[user, item + num_users] = 1
            else:
                user, item, weight = G[i]
                if weight == -1:
                    A[item + num_users, user] = 1
                    A[user, item + num_users] = 1
                #else: A[item + num_users , user] = 0
        P = _normalize_columns(A)
        num_edges = A.nnz // 2
        #num_nodes = A.shape[0] 
        
        self.A = A
        self.P = P
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.degree = np.sum(A, axis=1) 
        _log_init_graph("MovieLens", init_hops, init_topk, seeds, num_edges)
        
    def get(self):
        
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }
    def update(self,item_id, user_id):  
        item_node = item_id + self.num_users
        if self.A[item_node, user_id] == 0:
            self.A[item_node, user_id] = 1
        self.P[:, user_id] = self.A[:, user_id]/ self.A[:, user_id].sum()
        self.P = self.P.tocsr()
        self.num_edges = self.A.nnz
        self.degree = np.sum(self.A, axis=1) 

class Amazon_fashion(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, num_users, num_items, path = None, init_hops=0, init_topk=0):
        if not path:
            path = self.path
        self.G = np.load(path)
        self.num_items = num_items
        self.num_users = num_users
        num_nodes = num_users + num_items
        if init_hops > 0 and init_topk > 0:
            entry_path = os.path.join(DATA_DIR, "Amazon_fashion/new/amazon_fashion_4000users_entry.npy")
            raw_edges = _positive_entry_edges(entry_path)
            full_edges = np.column_stack((raw_edges[:, 0], raw_edges[:, 1] + num_users))
            init_edges, seeds = _topk_hop_edges(num_nodes, full_edges, init_hops, init_topk)
            G = np.column_stack((init_edges[:, 0], init_edges[:, 1] - num_users))
        else:
            G = self.G
            seeds = np.array([], dtype=np.int64)
        A = sp.lil_matrix((num_nodes, num_nodes))
        for i in range(len(G)):
            if init_hops > 0 and init_topk > 0:
                user, item = G[i]
                A[item + num_users, user] = 1
                A[user, item + num_users] = 1
            else:
                user, item, weight = G[i]
                if weight == 1:
                    A[item + num_users, user] = 1
                    A[user, item + num_users] = 1
                #else: A[item + num_users , user] = 0
        P = _normalize_columns(A)
        num_edges = A.nnz // 2
        num_nodes = A.shape[0] 
        
        self.A = A
        self.P = P
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.degree = np.sum(A, axis=1) 
        _log_init_graph("Amazon_fashion", init_hops, init_topk, seeds, num_edges)
            
        
    def get(self):
        
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
            
        }
    
    def update(self,item_id, user_id):  
        item_node = item_id + self.num_users
        if self.A[item_node, user_id] == 0:
            self.A[item_node, user_id] = 1
            self.A[user_id, item_node] = 1
        self.P[:, user_id] = self.A[:, user_id]/ self.A[:, user_id].sum()
        self.P[:, item_node] = self.A[:, item_node]/ self.A[:, item_node].sum()
        self.P = self.P.tocsr()
        self.num_edges = self.A.nnz // 2
        self.degree = np.sum(self.A, axis=1) 
        
class Facebook(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, num_users, path = None, init_hops=0, init_topk=0):
        if not path:
            path = self.path
        self.G = np.load(path)
        self.num_users = num_users
        num_nodes = num_users 
        if init_hops > 0 and init_topk > 0:
            entry_path = os.path.join(DATA_DIR, "Facebook/facebook_combined_ALLusers_entry.npy")
            G, seeds = _topk_hop_edges(num_nodes, _positive_entry_edges(entry_path), init_hops, init_topk)
        else:
            G = self.G
            seeds = np.array([], dtype=np.int64)
        A = sp.lil_matrix((num_nodes, num_nodes))
        for i in range(len(G)):
            if init_hops > 0 and init_topk > 0:
                user, item = G[i]
                A[item , user] = 1
                A[user, item] = 1
            else:
                user, item, weight = G[i]
                if weight == 1:
                    A[item , user] = 1
                    A[user , item] = 1
                #else: A[item, user] = 0
        P = _normalize_columns(A)
        num_edges = A.nnz // 2
        #num_nodes = A.shape[0] 
        
        self.A = A
        self.P = P
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.degree = np.sum(A, axis=1)
        _log_init_graph("Facebook", init_hops, init_topk, seeds, num_edges)
        
    def get(self):
        
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }
    
    def update(self, user_id1, user_id2):  
        if self.A[user_id1, user_id2] == 0 :
            self.A[user_id1, user_id2] = 1
            self.A[user_id2, user_id1] = 1
        self.P[:, user_id1] = self.A[:, user_id1]/ self.A[:, user_id1].sum()
        self.P[:, user_id2] = self.A[:, user_id2]/ self.A[:, user_id2].sum()
        self.P = self.P.tocsr()
        self.num_edges = self.A.nnz // 2
        self.degree = np.sum(self.A, axis=1) 

class Grqc(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, num_users, path = None, init_hops=0, init_topk=0):
        if not path:
            path = self.path
        self.G = np.load(path)
        self.num_users = num_users
        num_nodes = num_users 
        if init_hops > 0 and init_topk > 0:
            entry_path = os.path.join(DATA_DIR, "GrQc/Insert/GrQc_ALLusers_entry.npy")
            G, seeds = _topk_hop_edges(num_nodes, _positive_entry_edges(entry_path), init_hops, init_topk)
        else:
            G = self.G
            seeds = np.array([], dtype=np.int64)
        A = sp.lil_matrix((num_nodes, num_nodes))
        for i in range(len(G)):
            if init_hops > 0 and init_topk > 0:
                user, item = G[i]
                A[item , user] = 1
                A[user, item] = 1
            else:
                user, item, weight = G[i]
                if weight == 1:
                    A[item , user] = 1
                    A[user , item] = 1
                #else: A[item, user] = 0
        P = _normalize_columns(A)
        num_edges = A.nnz // 2
        #num_nodes = A.shape[0] 
        
        self.A = A
        self.P = P
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.degree = np.sum(A, axis=1)
        _log_init_graph("Grqc", init_hops, init_topk, seeds, num_edges)
        
    def get(self):
        
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }
    
    def update(self, user_id1, user_id2):  
        if self.A[user_id1, user_id2] == 0 :
            self.A[user_id1, user_id2] = 1 
            self.A[user_id2, user_id1] = 1
            
        self.P[:, user_id1] = self.A[:, user_id1]/ self.A[:, user_id1].sum() 
        self.P[:, user_id2] = self.A[:, user_id2]/ self.A[:, user_id2].sum()
        self.P = self.P.tocsr()
        self.num_edges = self.A.nnz // 2
        self.degree = np.sum(self.A, axis=1) 
        
class PPA(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, path = None, init_hops=0, init_topk=0):
        if not path:
            path = self.path
        dataset = LinkPropPredDataset(name='ogbl-ppa', root=path)
        graph_data = dataset[0]
        split_edge = dataset.get_edge_split()
        self.num_nodes = int(graph_data['num_nodes'])
        
        edge_index = _to_numpy(split_edge['train']['edge']).astype(np.int64)
        self.G = edge_index
        if init_hops > 0 and init_topk > 0:
            edge_index, seeds = _topk_hop_edges(self.num_nodes, edge_index, init_hops, init_topk)
        else:
            seeds = np.array([], dtype=np.int64)
        
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        data = np.ones(len(src), dtype=np.float32)
        A_coo = sp.coo_matrix((data, (src, dst)), shape=(self.num_nodes, self.num_nodes))
        A_csr = A_coo.tocsr()
        A = A_csr + A_csr.T
        A.data = np.ones_like(A.data) 
        
        # P
        self.A = A.tolil() 
        self.num_edges = self.A.nnz // 2
        A_csc = self.A.tocsc()
        D = np.array(A_csc.sum(axis=0)).flatten().astype(float)
        self.P = _normalize_columns(A_csc) 
        self.degree = D    
        _log_init_graph("PPA", init_hops, init_topk, seeds, self.num_edges)
        
    def get(self):
        
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }
    
    def update(self, u, v):  
        if self.A[u, v] == 0:
            self.A[u, v] = 1
            self.A[v, u] = 1
            
            self.P[:, u] = self.A[:, u] / self.A[:, u].sum()
            self.degree[u] = self.A[:, u].sum()
            self.P[:, v] = self.A[:, v] / self.A[:, v].sum()
            self.degree[v] = self.A[:, v].sum()
            self.num_edges = self.A.nnz 

class Collab(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, path = None, init_hops=0, init_topk=0):
        if not path:
            path = self.path
        dataset = LinkPropPredDataset(name='ogbl-collab', root=path)
        graph_data = dataset[0]
        split_edge = dataset.get_edge_split()
        self.num_nodes = int(graph_data['num_nodes'])
        
        edge_index = _to_numpy(split_edge['train']['edge']).astype(np.int64)
        self.G = edge_index
        if init_hops > 0 and init_topk > 0:
            edge_index, seeds = _topk_hop_edges(self.num_nodes, edge_index, init_hops, init_topk)
        else:
            seeds = np.array([], dtype=np.int64)
        
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        data = np.ones(len(src), dtype=np.float32)
        A_coo = sp.coo_matrix((data, (src, dst)), shape=(self.num_nodes, self.num_nodes))
        A_csr = A_coo.tocsr()
        A = A_csr + A_csr.T
        A.data = np.ones_like(A.data) 
        
        self.A = A.tolil() 
        self.num_edges = self.A.nnz // 2
        A_csc = self.A.tocsc()
        D = np.array(A_csc.sum(axis=0)).flatten().astype(float)
        self.P = _normalize_columns(A_csc) 
        self.degree = D    
        _log_init_graph("Collab", init_hops, init_topk, seeds, self.num_edges)
        
    def get(self):
        
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }
    
    def update(self, u, v):  
        if self.A[u, v] == 0:
            self.A[u, v] = 1
            self.A[v, u] = 1
            
            self.P[:, u] = self.A[:, u] / self.A[:, u].sum()
            self.degree[u] = self.A[:, u].sum()
            self.P[:, v] = self.A[:, v] / self.A[:, v].sum()
            self.degree[v] = self.A[:, v].sum()
            self.num_edges = self.A.nnz 

class Vessel(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, path = None, init_hops=0, init_topk=0):
        if not path:
            path = self.path
        dataset = LinkPropPredDataset(name='ogbl-vessel', root=path)
        graph_data = dataset[0]
        split_edge = dataset.get_edge_split()
        self.num_nodes = int(graph_data['num_nodes'])
        
        edge_index = _to_numpy(split_edge['train']['edge']).astype(np.int64)
        self.G = edge_index
        if init_hops > 0 and init_topk > 0:
            edge_index, seeds = _topk_hop_edges(self.num_nodes, edge_index, init_hops, init_topk)
        else:
            seeds = np.array([], dtype=np.int64)
        
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        data = np.ones(len(src), dtype=np.float32)
        A_coo = sp.coo_matrix((data, (src, dst)), shape=(self.num_nodes, self.num_nodes))
        A_csr = A_coo.tocsr()
        A = A_csr + A_csr.T
        A.data = np.ones_like(A.data) 
        
        self.A = A.tolil() 
        self.num_edges = self.A.nnz // 2
        A_csc = self.A.tocsc()
        D = np.array(A_csc.sum(axis=0)).flatten().astype(float)
        self.P = _normalize_columns(A_csc) 
        self.degree = D    
        _log_init_graph("Vessel", init_hops, init_topk, seeds, self.num_edges)
        
    def get(self):
        
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }
    
    def update(self, u, v):  
        if self.A[u, v] == 0:
            self.A[u, v] = 1
            self.A[v, u] = 1
            
            self.P[:, u] = self.A[:, u] / self.A[:, u].sum()
            self.degree[u] = self.A[:, u].sum()
            self.P[:, v] = self.A[:, v] / self.A[:, v].sum()
            self.degree[v] = self.A[:, v].sum()
            self.num_edges = self.A.nnz 
class DDI(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, path = None):
        if not path:
            path = self.path
        dataset = LinkPropPredDataset(name='ogbl-ddi', root=path)
        graph_data = dataset[0]
        split_edge = dataset.get_edge_split()
        self.num_nodes = int(graph_data['num_nodes'])
        
        edge_index = split_edge['train']['edge']
        self.G = edge_index
        
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        data = np.ones(len(src), dtype=np.float32)
        
        A_coo = sp.coo_matrix((data, (src, dst)), shape=(self.num_nodes, self.num_nodes))
        A_csr = A_coo.tocsr()
        A = A_csr + A_csr.T
        A.data = np.ones_like(A.data)
        self.A = A.tolil() 
        
        self.num_edges = self.A.nnz // 2
        
        A_csc = self.A.tocsc()
        D = np.array(A_csc.sum(axis=0)).flatten().astype(float)
        self.degree = D
        
        nonzero_mask = D != 0
        D_inv = np.zeros_like(D)
        D_inv[nonzero_mask] = 1.0 / D[nonzero_mask]
        
        D_mat = sp.diags(D_inv)
        self.P = A_csc.dot(D_mat).tocsr()
        
    def get(self):
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }
    
    def update(self, u, v):  
        if self.A[u, v] == 0:
            self.A[u, v] = 1
            self.A[v, u] = 1
            
            self.degree[u] += 1
            self.degree[v] += 1
            self.num_edges += 2
            
            D = self.degree
            nonzero_mask = D != 0
            D_inv = np.zeros_like(D)
            D_inv[nonzero_mask] = 1.0 / D[nonzero_mask]
            
            import scipy.sparse as sp
            D_mat = sp.diags(D_inv)
            
            self.P = self.A.dot(D_mat).tocsr()
            
class Citation2(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, path = None):
        if not path:
            path = self.path
        dataset = LinkPropPredDataset(name='ogbl-citation2', root=path)
        graph_data = dataset[0]
        split_edge = dataset.get_edge_split()
        self.num_nodes = int(graph_data['num_nodes'])
        train_edge = split_edge['train']
        src = train_edge['source_node']
        dst = train_edge['target_node']
        
        if hasattr(src, 'numpy'): src = src.numpy()
        if hasattr(dst, 'numpy'): dst = dst.numpy()
        
        self.G = (src, dst) 
        data = np.ones(len(src), dtype=np.float32)
        A_coo = sp.coo_matrix((data, (dst, src)), shape=(self.num_nodes, self.num_nodes))
        self.A = A_coo.tolil() 
        self.num_edges = self.A.nnz
        
        A_csc = self.A.tocsc()
        D = np.array(A_csc.sum(axis=0)).flatten().astype(float)
        nonzero_mask = D != 0
        D_inv = np.zeros_like(D)
        D_inv[nonzero_mask] = 1.0 / D[nonzero_mask]
        P = A_csc.multiply(D_inv)
        self.P = P.tocsc()
        self.degree = D
        
    def get(self):
        
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }
    
    def update(self, u, v):
        source, target = u, v
        if self.A[target, source] == 0:
            self.A[target, source] = 1
            deg_src = self.A[:, source].sum()
            if deg_src > 0: 
                self.P[:, source] = self.A[:, source] / deg_src
                self.degree[source] = deg_src
            self.num_edges = self.A.nnz

def to_prmatrix(P: sp.spmatrix):
    sums = P.sum(axis = 0)
    Q = sp.lil_matrix(P.shape)
    P_t = P.transpose()
    for i in range(P.shape[0]):
        if sums[0, i] != 0:
            Q[i, :] = P_t[i, :]/sums[0, i]
    Q = Q.transpose()
    return Q.tocsr()

if __name__ == "__main__":
    #TEST
    # MovieLens = MovieLens(os.path.join(DATA_DIR, "MovieLens/movie_2000users_10000items_noedge.npy"))
    # Amazon_fashion = Amazon_fashion(os.path.join(DATA_DIR, "Amazon_fashion/new/Insert/Amazon_fashion_4000users_noedge.npy"))
    # Facebook = Facebook(os.path.join(DATA_DIR, "Facebook/Insert/facebook_combined_ALLusers_noedge.npy"))
    # Grqc = Grqc(os.path.join(DATA_DIR, "GrQc/Insert/GrQc_ALLusers_noedge.npy"))
    # Facebook.load(5000)
    # print(Facebook.path,Facebook.G)
    # print(Facebook.G.shape)
    # print(Facebook.get())
    # print(Facebook.update(3000,3000))
    pass
    
