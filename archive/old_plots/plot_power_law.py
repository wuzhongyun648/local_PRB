import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import scipy.sparse as sp

# 路径 hack
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入必要模块
from EENet import EE_Net
import ppr_solver
import utils
from load_data import load_movielen, load_facebook, load_amazon_fashion, load_grqc, load_ogb_collab, load_ogb_ppa, load_ogb_vessel, load_ogb_citation2

# 绘图风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

PROJECT_ROOT = "/mnt/data/xinyu/bandits_pj/PRB"
SAVE_DIR = os.path.join(current_dir, "results", "plots")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ================= 配置区域 =================
# 模拟 main.py 的配置函数
def get_config(graph_name):
    if graph_name == 'MovieLens':
        path = os.path.join(PROJECT_ROOT, "online_link_prediction/data/MovieLens/movie_2000users_10000items_noedge.npy")
        return load_movielen, utils.MovieLens, path, 2000, 10000, 2000 # offset
    elif graph_name == 'Amazon_fashion':
        path = os.path.join(PROJECT_ROOT, "online_link_prediction/data/Amazon_fashion/new/Insert/Amazon_fashion_4000users_noedge.npy")
        return load_amazon_fashion, utils.Amazon_fashion, path, 4000, 4000, 4000
    elif graph_name == 'Facebook':
        path = os.path.join(PROJECT_ROOT, "online_link_prediction/data/Facebook/Insert/facebook_combined_ALLusers_noedge.npy")
        return load_facebook, utils.Facebook, path, 4039, 0, 0
    elif graph_name == 'Grqc':
        path = os.path.join(PROJECT_ROOT, "online_link_prediction/data/GrQc/Insert/GrQc_ALLusers_noedge.npy")
        return load_grqc, utils.Grqc, path, 5242, 0, 0
    elif graph_name == 'Collab':
        path = os.path.join(PROJECT_ROOT, "dataset")
        return load_ogb_collab, utils.Collab, path, 0, 0, 0
    # ... 其他 OGB 可以按需添加
    else:
        raise ValueError(f"Unknown: {graph_name}")

def collect_and_plot(graph_name, num_rounds=10000):
    print(f"-> Running simulation for {graph_name} ({num_rounds} rounds) ...")
    
    # 1. 初始化环境
    LoaderClass, GraphClass, data_path, n_users, n_items, offset = get_config(graph_name)
    
    # Loader
    b = LoaderClass()
    
    # Graph
    gm = GraphClass(data_path)
    if graph_name in ['MovieLens', 'Amazon_fashion']:
        gm.load(n_users, n_items)
    elif graph_name in ['Facebook', 'Grqc']:
        gm.load(n_users)
    else:
        gm.load()
        
    # Net
    ee_net = EE_Net(b.dim, b.n_arm, pool_step_size=50, lr_1=0.1, lr_2=0.01, hidden=100)
    
    # Solver Params
    alpha = 0.85
    # 动态计算 epsilon
    if hasattr(gm, 'num_nodes'):
        m = gm.num_nodes # 或者是边数，这里按 main.py 逻辑用节点数近似
    else:
        m = 10000
    eps = (1.0 / m) * 1e-3
    
    # 2. 收集数据容器
    # 存储每一轮排序后的 top-k 分数
    # Shape: (num_rounds, k)
    all_ranked_scores = [] 

    # 3. 运行循环
    for t in range(num_rounds):
        # Step
        step_res = b.step()
        if len(step_res) == 6:
            context, context_ind, rwd, _, user_id, _ = step_res
        else:
            context, context_ind, rwd, _, user_id, _ = step_res
            
        # Predict (Get h)
        _, h_observe = ee_net.predict(context, t)
        
        # Construct h dense
        h_indices = []
        h_values = []
        for i in range(len(context_ind)):
            raw_item = context_ind[i][1]
            real_node = raw_item + offset
            h_indices.append(real_node)
            val = h_observe[i]
            if isinstance(val, (list, np.ndarray)): val = val[0]
            h_values.append(val)
            
        # Run Solver (FastPRB) to get "Calculated y_t" (PPR Scores)
        # 我们这里用 Power Iteration 获得精确解，或者 APPR 均可
        # 为了画图平滑，这里用 Power Iteration 模拟理想分布
        P = gm.P
        h_dense = np.zeros(gm.num_nodes)
        h_dense[h_indices] = h_values
        
        p_scores_all = ppr_solver.power_iteration(P, alpha, h_dense, 30)
        
        # 提取 Candidate 的分数
        cand_scores = p_scores_all[h_indices]
        
        # === 核心逻辑: 取绝对值 -> 排序 ===
        # 1. 取绝对值
        abs_scores = np.abs(cand_scores)
        # 2. 从大到小排序
        sorted_scores = np.sort(abs_scores)[::-1]
        
        all_ranked_scores.append(sorted_scores)
        
        # 简单的 Update 逻辑防止报错 (不真正训练网络以节省时间)
        if t % 10 == 0:
            print(f"   Step {t}/{num_rounds}", end='\r')

    print("")
    
    # 4. 取平均 (平滑曲线)
    # 对所有轮次在相同 Rank 位置的值取平均
    mean_ranked_scores = np.mean(all_ranked_scores, axis=0)
    
    # 5. 绘图
    plt.figure(figsize=(8, 6))
    
    ranks = np.arange(1, len(mean_ranked_scores) + 1)
    
    # Log-Log Plot
    plt.loglog(ranks, mean_ranked_scores, 'o-', linewidth=2, markersize=5, label='Calculated $y_t$ (Mean)')
    
    # (可选) 拟合 Power Law 直线参考
    try:
        # 只拟合前 80% 的点，避开尾部噪音
        fit_len = int(len(ranks) * 0.8)
        log_x = np.log10(ranks[:fit_len])
        log_y = np.log10(mean_ranked_scores[:fit_len])
        coeffs = np.polyfit(log_x, log_y, 1)
        poly = np.poly1d(coeffs)
        y_fit = 10**(poly(np.log10(ranks)))
        plt.loglog(ranks, y_fit, 'r--', label=f'Power Law Fit (slope={coeffs[0]:.2f})', alpha=0.7)
    except:
        pass

    plt.title(f"Rank-Value Distribution: {graph_name} (Candidates)", fontsize=14, fontweight='bold')
    plt.xlabel("Rank (Log Scale)", fontsize=12)
    plt.ylabel("Absolute Score $|y_t|$ (Log Scale)", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    out_path = os.path.join(SAVE_DIR, f"{graph_name}_Rank_Distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved plot to: {out_path}\n")

if __name__ == "__main__":
    # 你想画哪些图
    datasets = ['MovieLens', 'Amazon_fashion', 'Facebook', 'Grqc'] # 'Collab'
    
    for ds in datasets:
        collect_and_plot(ds)