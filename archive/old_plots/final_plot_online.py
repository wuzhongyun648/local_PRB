import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ================= 1. 配置区域 =================

# 数据集列表
DATASETS = ['Vessel'] 
# MovieLens  Facebook  Amazon  GrQc  Collab  PPA Vessel
# 需要画的方法列表
METHODS = [
    'FastPRB', 
    'PRB', 
    'EE-Net', 
    'NeuralUCB', 
    'NeuralTS', 
    'Neural_epsilon'
]

# 绝对路径配置 (PRB / FastPRB)
CUSTOM_PATHS = {
    'MovieLens': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/MovieLens_FastPRB_alpha0.85_eps8.33e-05_T10000_lr10.0073_lr20.0004_202601252259/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/MovieLens_PRB_alpha0.85_powT50_T10000_lr10.0073_lr20.0004_202601252128/final_results.npy',
    },
    'Facebook': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Facebook_FastPRB_alpha0.85_eps0.000248_T10000_lr10.06_lr20.02_202601261327/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Facebook_PRB_alpha0.85_powT50_T10000_lr10.06_lr20.02_202601252130/final_results.npy',
    },
    'Amazon': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Amazon_fashion_FastPRB_alpha0.85_eps0.000125_T5000_lr10.1_lr20.01_202601261331/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Amazon_fashion_PRB_alpha0.85_powT50_T5000_lr10.1_lr20.01_202601261332/final_results.npy',
    },
    'GrQc': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Grqc_FastPRB_alpha0.85_eps0.000191_T10000_lr10.01_lr20.004_202601261327/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Grqc_PRB_alpha0.85_powT50_T10000_lr10.01_lr20.004_202601252130/final_results.npy',
    },
    'Collab': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Collab_FastPRB_alpha0.85_eps4.24e-06_T5000_lr10.01_lr20.004_202601242113/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Collab_PRB_alpha0.85_powT50_T5000_lr10.01_lr20.004_202601250106/final_results.npy',
    },
    'PPA': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/PPA_FastPRB_alpha0.85_eps0.000191_T5000_lr10.01_lr20.004_202601250049/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/PPA_PRB_alpha0.85_powT50_T5000_lr10.01_lr20.004_202601211613/final_results.npy',
    },
    'Vessel': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Vessel_FastPRB_alpha0.85_eps2.86e-07_T5000_lr10.01_lr20.004_202601242113/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Vessel_PRB_alpha0.85_powT50_T5000_lr10.01_lr20.004_202601242113/final_results.npy',
    }
}

# 基线结果目录
BASE_DIR_BASELINES = "./results/baselines"

# 样式配置
COLORS = {
    'FastPRB': "#E62B16",  # 朱红 (Vermilion) - 最显眼
    'PRB':     '#0072B2',  # 深蓝 (Blue) - 对比强
    'EE-Net':  '#009E73',  # 蓝绿 (Bluish Green)
    'NeuralUCB': '#E69F00',# 橙色 (Orange)
    'NeuralTS':  '#CC79A7',# 紫红 (Reddish Purple)
    'Neural_epsilon':    '#56B4E9',# 天蓝 (Sky Blue)
}
NAME_MAPPING = {
    'FastPRB': 'Fast-PRB',
    'Neural_epsilon': 'NeuralGreedy',
    # 你也可以在这里修改其他方法的显示名称，例如：
    # 'EE-Net': 'EE-Net (Ours)', 
}
def get_linestyle(method):
    if method == 'FastPRB':
        # FastPRB: 宽实线，作为底层背景
        return '-', 3 
    elif method == 'PRB':
        # PRB: 虚线 (Dashed)，稍细，叠加在上面
        # (0, (3, 2)) 是自定义虚线：3个点实线，2个点间隔，比默认 '--' 更清晰
        return (0, (3, 2)), 2.5 
    # 其他 Baseline: 细虚线/点划线
    return '--', 1.5

# ================= 2. 数据加载逻辑 (返回原始数据) =================

def load_raw_data(dataset, method):
    """
    返回原始数据矩阵，用于计算 Mean/Std
    Returns: 
        regret_data: np.array (runs, rounds)
        time_data: np.array (runs, rounds) representing time per step
    """
    
    # 1. 检查绝对路径 (FastPRB/PRB)
    if dataset in CUSTOM_PATHS and method in CUSTOM_PATHS[dataset]:
        file_path = CUSTOM_PATHS[dataset][method]
        if os.path.exists(file_path):
            try:
                # Format: (runs, rounds, 5) -> [time, regret, loss1, loss2, ppr_norm]
                raw = np.load(file_path)
                regret_data = raw[:, :, 1]  # Cumulative Regret
                time_data = raw[:, :, 0]    # Time per step
                return regret_data, time_data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return None, None
    
    # 2. 检查基线目录
    regret_path = os.path.join(BASE_DIR_BASELINES, f"{dataset}_{method}_regret.npy")
    time_path = os.path.join(BASE_DIR_BASELINES, f"{dataset}_{method}_time.npy")
    
    if os.path.exists(regret_path) and os.path.exists(time_path):
        try:
            r_data = np.load(regret_path)
            if r_data.ndim == 1: r_data = r_data.reshape(1, -1)
            
            t_data = np.load(time_path)
            if t_data.ndim == 1: t_data = t_data.reshape(1, -1)
            
            return r_data, t_data
        except Exception as e:
            print(f"Error loading baseline {method}: {e}")
            return None, None
            
    return None, None

# ================= 3. 绘图与统计 =================

def plot_dataset(dataset_name):
    # 存储绘图数据和统计数据
    plot_data = {}
    stats_list = [] # (method, mean, std)
    
    print(f"\n>>> Processing {dataset_name} ...")
    
    for method in METHODS:
        regret_raw, time_raw = load_raw_data(dataset_name, method)
        
        if regret_raw is not None:
            # --- A. 准备绘图数据 (均值曲线) ---
            regret_mean_curve = np.mean(regret_raw, axis=0)
            time_per_step_mean = np.mean(time_raw, axis=0)
            time_accum_curve = np.cumsum(time_per_step_mean)
            
            plot_data[method] = (regret_mean_curve, time_accum_curve)
            
            # --- B. 计算统计数据 (Regret & Time) ---
            
            # 1. Regret (取最后一轮累计值)
            final_regrets = regret_raw[:, -1] 
            reg_mean = np.mean(final_regrets)
            reg_std = np.std(final_regrets)
            
            # 2. 【新增】Time (计算每个 Run 的总耗时，然后取平均)
            # time_raw shape: (runs, rounds) -> 对 rounds 求和得到每个 run 的总时间
            total_times = np.sum(time_raw, axis=1) 
            time_mean = np.mean(total_times)
            time_std = np.std(total_times)
            
            stats_list.append((method, reg_mean, reg_std, time_mean, time_std))
        else:
            pass

    if not plot_data:
        print(f"No data found for {dataset_name}, skipping.")
        return

    # --- C. 输出统计表格 (增加 Time 列) ---
    print(f"\n{'='*75}") # 【修改】加长分割线
    print(f"Final Statistics (Regret & Total Time) - {dataset_name}")
    print(f"{'-'*75}")
    # 【修改】增加 Time 表头
    print(f"{'Method':<15} | {'Regret':<12} | {'Std':<10} | {'Time(s)':<10} | {'Std':<10}")
    print(f"{'-'*75}")
    
    # 按 Regret Mean 排序
    stats_list.sort(key=lambda x: x[1])
    
    for method, reg_mean, reg_std, time_mean, time_std in stats_list:
        # 获取显示名称 (Fast-PRB 等)
        display_name = NAME_MAPPING.get(method, method)
        
        # 【修改】增加 Time 输出
        if reg_mean == -1:
            print(f"{display_name:<15} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
        else:
            print(f"{display_name:<15} | {reg_mean:<12.0f} | {reg_std:<10.1f} | {time_mean:<10.2f} | {time_std:<10.2f}")
    
    print(f"{'='*75}\n")

    # --- D. 绘图 1: Regret vs Round ---
    plt.figure(figsize=(8, 6))
    for method, (regret, _) in plot_data.items():
        rounds = np.arange(len(regret))
        ls, lw = get_linestyle(method)
        display_name = NAME_MAPPING.get(method, method)
        
        # 【修改点】：定义 zorder
        if method == 'FastPRB': zorder = 10
        elif method == 'PRB': zorder = 20
        else: zorder = 5
        
        # 加入 zorder 参数
        plt.plot(rounds, regret, label=display_name, color=COLORS.get(method), 
                 linestyle=ls, linewidth=lw, zorder=zorder)
    
    
    #plt.title(f"{dataset_name}")
    plt.title(f"ogbl-{dataset_name}")
    #plt.title("Amazon Fashion")
    plt.xlabel("Round")
    plt.ylabel("Regret")
    plt.legend()
    
    plt.savefig(f"{dataset_name}_regret_round.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{dataset_name}_regret_round.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # --- E. 绘图 2: Regret vs Time ---
    plt.figure(figsize=(8, 6))
    for method, (regret, time_accum) in plot_data.items():
        ls, lw = get_linestyle(method)
        display_name = NAME_MAPPING.get(method, method)
        
        # 【修改点】：定义 zorder
        if method == 'FastPRB': zorder = 10
        elif method == 'PRB': zorder = 20
        else: zorder = 5

        # 加入 zorder 参数
        plt.plot(time_accum, regret, label=display_name, color=COLORS.get(method), 
                 linestyle=ls, linewidth=lw, zorder=zorder)
    #plt.title(f"{dataset_name}")
    #plt.title("Amazon Fashion")
    plt.title(f"ogbl-{dataset_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Regret")
    plt.legend()
    
    plt.savefig(f"{dataset_name}_regret_time.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{dataset_name}_regret_time.pdf", format='pdf', bbox_inches='tight')
    print(f"Plots saved for {dataset_name}.")
    plt.close()

if __name__ == "__main__":
    for ds in DATASETS:
        plot_dataset(ds)