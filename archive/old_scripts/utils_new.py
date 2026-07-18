#1.写所有数据集的导入代码
#2.其他需要的工具
import numpy as np
import scipy.sparse as sp
from ogb.linkproppred import LinkPropPredDataset
import datetime
import os,sys
import os
import numpy as np

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
        
    def load(self, num_users, num_items, path = None):
        if not path:
            path = self.path
        self.G = np.load(path)
        self.num_items = num_items
        self.num_users = num_users
        num_nodes = num_users + num_items
        G = self.G
        A = sp.lil_matrix((num_nodes, num_nodes))
        for i in range(len(G)):
            user, item, weight = G[i]
            if weight == -1:
                A[item + num_users, user] = 1
                #A[user, item + num_users] = 1
            #else: A[item + num_users , user] = 0
        D = np.array(A.sum(axis=0)).flatten()  
        nonzero_mask = D != 0
        D_inv = np.zeros_like(D)
        D_inv[nonzero_mask] = 1.0 / D[nonzero_mask]
        D_inv = sp.diags(D_inv, format='csc')
        P = A @ D_inv
        num_edges = A.nnz
        #num_nodes = A.shape[0] 
        
        self.A = A
        self.P = P
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.degree = np.sum(A, axis=1) 
        
    def get(self):
        
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }
    def update(self,item_id, user_id):  
        if self.A[item_id + self.num_users, user_id] == 0:
            self.A[item_id + self.num_users, user_id] = 1 
            #self.A[user_id, item_id + self.num_users] = 1 
        self.P[:, user_id] = self.A[:, user_id]/ self.A[:, user_id].sum() 
        self.P = self.P.tocsr()
        self.num_edges = self.A.nnz    
        self.degree = np.sum(self.A, axis=1) 

class Amazon_fashion(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, num_users, num_items, path = None):
        if not path:
            path = self.path
        self.G = np.load(path)
        self.num_items = num_items
        self.num_users = num_users
        num_nodes = num_users + num_items
        G = self.G
        A = sp.lil_matrix((num_nodes, num_nodes))
        for i in range(len(G)):
            user, item, weight = G[i]
            if weight == 1:
                A[item + num_users, user] = 1
                #A[user, item + num_users] = 1
            #else: A[item + num_users , user] = 0
        D = np.array(A.sum(axis=0)).flatten()  
        nonzero_mask = D != 0
        D_inv = np.zeros_like(D)
        D_inv[nonzero_mask] = 1.0 / D[nonzero_mask]
        D_inv = sp.diags(D_inv, format='csc')
        P = A @ D_inv
        num_edges = A.nnz
        num_nodes = A.shape[0] 
        
        self.A = A
        self.P = P
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.degree = np.sum(A, axis=1) 
            
        
    def get(self):
        
        return {
            'A': self.A,
            'P': self.P,
            'degree': self.degree,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
            
        }
    
    def update(self,item_id, user_id):  
        if self.A[item_id + self.num_users, user_id] == 0:
            self.A[item_id + self.num_users, user_id] = 1
            #self.A[user_id, item_id + self.num_users] = 1 
        self.P[:, user_id] = self.A[:, user_id]/ self.A[:, user_id].sum() 
        #self.P[:, item_id + self.num_users] = self.A[:, item_id + self.num_users]/ self.A[:, item_id + self.num_users].sum()#
        #self.P = self.P.tocsr()
        self.num_edges = self.A.nnz    
        self.degree = np.sum(self.A, axis=1) 
        
class Facebook(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, num_users, path = None):
        if not path:
            path = self.path
        self.G = np.load(path)
        self.num_users = num_users
        num_nodes = num_users 
        G = self.G
        A = sp.lil_matrix((num_nodes, num_nodes))
        for i in range(len(G)):
            user, item, weight = G[i]
            if weight == 1:
                A[item , user] = 1
                #A[user , item] = 1
            #else: A[item, user] = 0
        D = np.array(A.sum(axis=0)).flatten()  
        nonzero_mask = D != 0
        D_inv = np.zeros_like(D)
        D_inv[nonzero_mask] = 1.0 / D[nonzero_mask]
        D_inv = sp.diags(D_inv, format='csc')
        P = A @ D_inv
        num_edges = A.nnz
        #num_nodes = A.shape[0] 
        
        self.A = A
        self.P = P
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.degree = np.sum(A, axis=1)
        
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
            #self.A[user_id2, user_id1] = 1 
        self.P[:, user_id1] = self.A[:, user_id1]/ self.A[:, user_id1].sum() 
        #self.P[:, user_id2] = self.A[:, user_id2]/ self.A[:, user_id2].sum() 
        #self.P = self.P.tocsr()
        self.num_edges = self.A.nnz    
        self.degree = np.sum(self.A, axis=1) 

class Grqc(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, num_users, path = None):
        if not path:
            path = self.path
        self.G = np.load(path)
        self.num_users = num_users
        num_nodes = num_users 
        G = self.G
        A = sp.lil_matrix((num_nodes, num_nodes))
        for i in range(len(G)):
            user, item, weight = G[i]
            if weight == 1:
                A[item , user] = 1
                #A[user , item] = 1#
            #else: A[item, user] = 0
        D = np.array(A.sum(axis=0)).flatten()  
        nonzero_mask = D != 0
        D_inv = np.zeros_like(D)
        D_inv[nonzero_mask] = 1.0 / D[nonzero_mask]
        D_inv = sp.diags(D_inv, format='csc')
        P = A @ D_inv
        num_edges = A.nnz
        #num_nodes = A.shape[0] 
        
        self.A = A
        self.P = P
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.degree = np.sum(A, axis=1)
        
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
            #self.A[user_id2, user_id1] = 1#
            
        self.P[:, user_id1] = self.A[:, user_id1]/ self.A[:, user_id1].sum() 
        #self.P[:, user_id2] = self.A[:, user_id2]/ self.A[:, user_id2].sum() 
        #self.P = self.P.tocsr()
        self.num_edges = self.A.nnz    
        self.degree = np.sum(self.A, axis=1) 
        
class PPA(graph):
    def __init__(self, path):
        super().__init__(path)
        
    def load(self, path = None):
        if not path:
            path = self.path
        dataset = LinkPropPredDataset(name='ogbl-ppa', root=path)
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
        
        # P
        self.A = A.tolil() 
        self.num_edges = self.A.nnz // 2
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
        
    def load(self, path = None):
        if not path:
            path = self.path
        dataset = LinkPropPredDataset(name='ogbl-collab', root=path)
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
        
    def load(self, path = None):
        if not path:
            path = self.path
        dataset = LinkPropPredDataset(name='ogbl-vessel', root=path)
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
    # MovieLens = MovieLens("/mnt/data/xinyu/bandits_pj/PRB/online_link_prediction/data/MovieLens/movie_2000users_10000items_noedge.npy")
    # Amazon_fashion = Amazon_fashion("/mnt/data/xinyu/bandits_pj/PRB/online_link_prediction/data/Amazon_fashion/new/Insert/Amazon_fashion_4000users_noedge.npy")
    # Facebook = Facebook("/mnt/data/xinyu/bandits_pj/PRB/online_link_prediction/data/Facebook/Insert/facebook_combined_ALLusers_noedge.npy")
    # Grqc = Grqc("/mnt/data/xinyu/bandits_pj/PRB/online_link_prediction/data/GrQc/Insert/GrQc_ALLusers_noedge.npy")
    # Facebook.load(5000)
    # print(Facebook.path,Facebook.G)
    # print(Facebook.G.shape)
    # print(Facebook.get())
    # print(Facebook.update(3000,3000))
    pass
    
