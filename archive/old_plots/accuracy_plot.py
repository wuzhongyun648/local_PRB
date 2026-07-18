import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================= 配置区域 =================

# 需要画的数据集列表
DATASETS = ['GrQc'] 
# 可选: 'MovieLens', 'Facebook', 'Amazon', 'GrQc', 'Collab', 'PPA', 'Vessel'

# 需要画的方法列表
METHODS = ['FastPRB', 'PRB']

# 自定义文件路径配置 (直接指向 final_results.npy)
# 代码会自动去同级目录下找 worker_0_TimeAcc.npy
CUSTOM_PATHS = {
    'MovieLens': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/MovieLens_FastPRB_alpha0.85_eps8.33e-05_T10000_lr10.0073_lr20.0004_202601291141/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/MovieLens_PRB_alpha0.85_powT50_T10000_lr10.0073_lr20.0004_202601290025/final_results.npy',
    },
    'Facebook': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Facebook_FastPRB_alpha0.85_eps0.000248_T10000_lr10.06_lr20.02_202601291743/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Facebook_PRB_alpha0.85_powT50_T10000_lr10.06_lr20.02_202601291452/final_results.npy',
    },
    'Amazon': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Amazon_fashion_FastPRB_alpha0.85_eps0.000125_T5000_lr10.1_lr20.01_202601291742/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Amazon_fashion_PRB_alpha0.85_powT50_T5000_lr10.1_lr20.01_202601291452/final_results.npy',
    },
    'GrQc': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Grqc_FastPRB_alpha0.85_eps0.000191_T10000_lr10.01_lr20.004_202601291743/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Grqc_PRB_alpha0.85_powT50_T10000_lr10.01_lr20.004_202601291452/final_results.npy',
    },
    'Collab': {
        'FastPRB': '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Collab_FastPRB_alpha0.85_eps4.24e-06_T5000_lr10.01_lr20.004_202601291248/final_results.npy',
        'PRB':     '/mnt/data/xinyu/Fast_bandit/results/online_link_prediction/Collab_PRB_alpha0.9_powT50_T5000_lr10.01_lr20.004_202601290042/final_results.npy',
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

# 输出目录
OUTPUT_DIR = "./custom_plots_acc"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 显示名称映射
DATASET_NAME_MAP = {
    'MovieLens': 'MovieLens',
    'Amazon': 'AmazonFashion',
    'Facebook': 'Facebook',
    'GrQc': 'GrQc',
    'Collab': 'ogbl-Collab',
    'PPA': 'ogbl-PPA',
    'Vessel': 'ogbl-Vessel',
}

# 样式配置 (严格复刻原 plot.py)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

LINE_WIDTH = 6.0
TITLE_SIZE = 48
LABEL_SIZE = 40
TICK_SIZE = 36
LEGEND_SIZE = 36
FIG_SIZE = (12, 9)

def load_single_worker_data(final_result_path, worker_id=0):
    """
    根据 final_results.npy 的路径，去同级目录找 worker_{id}_TimeAcc.npy
    """
    if not os.path.exists(final_result_path):
        print(f" [Error] File not found: {final_result_path}")
        return None

    # 获取文件夹路径
    folder_path = os.path.dirname(final_result_path)
    
    # 构造目标文件名
    target_file = f"worker_{worker_id}_TimeAcc.npy"
    full_path = os.path.join(folder_path, target_file)
    
    # 如果 worker_0 不存在，尝试找其他 worker (兜底逻辑)
    if not os.path.exists(full_path):
        # 尝试搜索任何一个 TimeAcc 文件
        candidates = glob.glob(os.path.join(folder_path, "worker_*_TimeAcc.npy"))
        if candidates:
            candidates.sort() # 选第一个
            full_path = candidates[0]
            print(f"   [Warn] {target_file} not found. Using fallback: {os.path.basename(full_path)}")
        else:
            print(f"   [Error] No TimeAcc files found in {folder_path}!")
            return None
            
    try:
        data = np.load(full_path)
        # data shape: (N_points, 2) -> [Time, Accuracy]
        return data
    except Exception as e:
        print(f" [Error] Failed to load {full_path}: {e}")
        return None

def make_one_plot(ds_key, std_name, fig_size, suffix=""):
    """
    画单张图的核心函数
    ds_key: 数据集Key
    std_name: 数据集显示名
    fig_size: 图片尺寸 (w, h)
    suffix: 文件名后缀 (如 "" 或 "_Wide")
    """
    print(f"   -> Generating Plot: Size={fig_size}, Suffix='{suffix}'")
    plt.figure(figsize=fig_size)
    
    # 遍历方法画线
    for method in METHODS:
        if method not in CUSTOM_PATHS[ds_key]:
            continue
            
        path = CUSTOM_PATHS[ds_key][method]
        
        # 读取数据 (只读一个 worker)
        data = load_single_worker_data(path, worker_id=0)
        
        if data is None:
            continue
            
        # 提取 X, Y
        time_axis = data[:, 0]
        accuracy = data[:, 1] * 100 # 转为百分比
        
        # === 样式逻辑 ===
        label_name = 'LocPRB' if method == 'FastPRB' else method
        
        if method == 'FastPRB':
            line_color = 'tab:cyan'
            current_ls = '-'
            current_zorder = 5
        elif method == 'PRB':
            line_color = 'tab:blue'
            current_ls = '--'
            current_zorder = 10
        else:
            line_color = 'black'
            current_ls = '-'
            current_zorder = 1
        
        # 绘图
        plt.plot(time_axis, accuracy, label=label_name, color=line_color, 
                 linewidth=LINE_WIDTH, linestyle=current_ls, zorder=current_zorder)

    # 设置图表装饰
    plt.title(std_name, fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.xlabel("Time (s)", fontsize=LABEL_SIZE, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=LABEL_SIZE, fontweight='bold')
    
    # 刻度样式
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=3, length=10)
    
    # 图例样式
    plt.legend(frameon=True, fontsize=LEGEND_SIZE, loc='best')
    
    # 去除网格
    plt.grid(False)
    plt.tight_layout()
    
    # 保存
    out_png = os.path.join(OUTPUT_DIR, f"{std_name}_Accuracy_vs_Time{suffix}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"{std_name}_Accuracy_vs_Time{suffix}.pdf")
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"      Saved: {out_png}")
    plt.close()

def main():
    print(">>> Starting Custom Accuracy-Time Plotting...")
    print(f" -> Output Directory: {OUTPUT_DIR}")
    
    for ds_key in DATASETS:
        if ds_key not in CUSTOM_PATHS:
            print(f" [Skip] Dataset {ds_key} not configured in CUSTOM_PATHS.")
            continue
            
        print(f"\nProcessing Dataset: {ds_key} ...")
        std_name = DATASET_NAME_MAP.get(ds_key, ds_key)
        
        # 1. 生成原始尺寸图 (12, 9)
        make_one_plot(ds_key, std_name, fig_size=(12, 9), suffix="")
        
        # 2. 生成双倍宽度图 (24, 9)
        make_one_plot(ds_key, std_name, fig_size=(24, 9), suffix="_Wide")

    print("\nAll Done!")

if __name__ == "__main__":
    main()