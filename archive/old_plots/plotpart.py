import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# 同时控制「Regret_vs_Rounds」与「Regret_vs_Time」的坐标轴显示范围（同一组数值）。
# Rounds 图：横轴为 round 索引，纵轴为 regret；Time 图：横轴为累计时间（秒），纵轴为 regret。
# 若为 None：不限制该轴；若给定数值：绘图前会先把超过该范围的数据直接截断，再设置坐标轴范围。
# ---------------------------------------------------------------------------
REGRET_AXIS_X_MAX = 400
REGRET_AXIS_Y_MAX = 200

tz_cn = timezone(timedelta(hours=8))
CURRENT_TIME = datetime.now(tz_cn).strftime("%Y%m%d%H%M")

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results","online_link_prediction")
PLOT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")
if not os.path.exists(PLOT_OUTPUT_DIR):
    os.makedirs(PLOT_OUTPUT_DIR)


DATASET_NAME_MAP = {
    # 原始文件夹名关键词 : 目标显示Title
    'MovieLens': 'MovieLens',
    'Amazon': 'AmazonFashion',
    'Amazon_fashion': 'AmazonFashion', 
    'Facebook': 'Facebook',
    'Grqc': 'GrQc',       # 修改为 GrQc
    'GrQc': 'GrQc',
    'Collab': 'ogbl-Collab',
    'ogbl-collab': 'ogbl-Collab', # 修改首字母大写
    'PPA': 'ogbl-PPA',      # 修改为大写 PPA
    'ogbl-ppa': 'ogbl-PPA',
    'Vessel': 'ogbl-Vessel', # 修改首字母大写
    'ogbl-vessel': 'ogbl-Vessel',
    
}
def parse_args():
    """
    Parses command-line arguments for the visualization script.

    This function retrieves the list of result directory names provided by the user,
    which will be used to locate and load the experimental data for plotting.
    """
    parser = argparse.ArgumentParser(description="Plot PRB/FastPRB Results (plotpart: + Regret vs Time with fixed axis limits in code)")
    parser.add_argument('folders', nargs='+', help='List of result folder names to plot')
    return parser.parse_args()

    
def load_data_from_folder(folder_name):
    full_path = os.path.join(RESULTS_DIR, folder_name)
    
    if not os.path.exists(full_path):
        print(f" [Warning] Folder not found: {folder_name}")
        return None, None

    # --- Part 1: 加载 Regret 数据 (保持原样) ---
    final_path = os.path.join(full_path, "final_results.npy")
    target_file = None
    
    if os.path.exists(final_path):
        target_file = final_path
    else:
        checkpoints = glob.glob(os.path.join(full_path, "checkpoint_step_*.npy"))
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            target_file = checkpoints[-1]
            print(f"   -> Running experiment? Using latest checkpoint: {os.path.basename(target_file)}")
    
    if target_file is None:
        print(f" [Error] No .npy files found in {folder_name}")
        return None, None

    raw_data = np.load(target_file)
    
    if raw_data.ndim == 3:
        data_mean = np.mean(raw_data, axis=0) 
        std_regret = np.std(raw_data[:, :, 1], axis=0)
        final_mean_regret = data_mean[-1, 1]
        final_std_regret = std_regret[-1]
        total_times_per_run = np.sum(raw_data[:, :, 0], axis=1) 
        final_mean_time = np.mean(total_times_per_run)
        final_std_time = np.std(total_times_per_run)
    else:
        data_mean = raw_data
        std_regret = np.zeros(raw_data.shape[0]) 
        final_mean_regret = data_mean[-1, 1]
        final_std_regret = 0.0
        total_time = np.sum(raw_data[:, 0])
        final_mean_time = total_time
        final_std_time = 0.0

    # --- Part 2: 加载 Accuracy 数据 (新增) ---
    acc_files = glob.glob(os.path.join(full_path, "worker_*_TimeAcc.npy"))
    acc_data_all = None
    final_mean_acc = 0.0
    final_std_acc = 0.0
    
    if acc_files:
        # 按 worker id 排序
        acc_files.sort(key=lambda x: int(x.split('worker_')[-1].split('_')[0]))
        # 读取列表
        try:
            raw_acc_list = [np.load(f) for f in acc_files]
            # 堆叠成 (Runs, Points, 2)
            acc_data_all = np.stack(raw_acc_list, axis=0)
            
            # 计算最后的 Accuracy 统计 (取每个 Run 的最后一个点的 Acc)
            final_accs = acc_data_all[:, -1, 1]
            final_mean_acc = np.mean(final_accs)
            final_std_acc = np.std(final_accs)
        except Exception as e:
            print(f"   [Warning] Error loading TimeAcc files in {folder_name}: {e}")

    # --- Part 3: 解析 Metadata (保持原样) ---
    parts = folder_name.split('_')
    alpha_index = -1
    for i, part in enumerate(parts):
        if part.startswith("alpha"):
            alpha_index = i
            break
    
    if alpha_index > 0:
        method = parts[alpha_index - 1]
        dataset = "_".join(parts[:alpha_index - 1])
    else:
        if 'FastPRB' in parts:
            idx = parts.index('FastPRB')
            method = 'FastPRB'
            dataset = "_".join(parts[:idx])
        elif 'PRB' in parts:
            idx = parts.index('PRB')
            method = 'PRB'
            dataset = "_".join(parts[:idx])
        else:
            dataset = parts[0]
            method = parts[1]

    if 'FastPRB' in method:
        label = 'LocPRB'
    elif 'PRB' in method:
        label = 'PRB'
    else:
        label = method

    return data_mean, {
        "dataset": dataset, 
        "label": label, 
        "folder": folder_name, 
        "std": std_regret,
        "final_mean_regret": final_mean_regret, 
        "final_std_regret": final_std_regret,
        "final_mean_time": final_mean_time,
        "final_std_time": final_std_time,
        # 新增 Accuracy 字段
        "acc_data_all": acc_data_all,
        "final_mean_acc": final_mean_acc,
        "final_std_acc": final_std_acc
    }


def clip_curve_to_limits(x_values, y_values, x_max=None, y_max=None):
    """
    截断超过显示范围的数据，而不是只依赖 set_xlim/set_ylim 隐藏。

    这里默认 regret 与累计时间/round 都是单调不减的，因此可以安全地保留前缀。
    一旦某个点首次超过 x 或 y 上限，就在该点之前停止绘制。
    """
    x_arr = np.asarray(x_values)
    y_arr = np.asarray(y_values)

    if x_arr.size == 0:
        return x_arr, y_arr

    keep_mask = np.ones(x_arr.shape[0], dtype=bool)
    if x_max is not None:
        keep_mask &= (x_arr <= float(x_max))
    if y_max is not None:
        keep_mask &= (y_arr <= float(y_max))

    valid_indices = np.where(keep_mask)[0]
    if valid_indices.size == 0:
        return x_arr[:1], y_arr[:1]

    end_idx = valid_indices[-1] + 1
    return x_arr[:end_idx], y_arr[:end_idx]
    
def plot_dataset_group(dataset_name, experiments):
    std_name = DATASET_NAME_MAP.get(dataset_name, dataset_name)
    
    # --- 0. 创建本次绘图的总 Batch 文件夹 ---
    batch_dir_name = f"{CURRENT_TIME}_Batch_Output"
    batch_dir_path = os.path.join(PLOT_OUTPUT_DIR, batch_dir_name)
    if not os.path.exists(batch_dir_path):
        os.makedirs(batch_dir_path)
        print(f"\n>>> Created Batch Directory: {batch_dir_path}")

    print(f"\n" + "="*130)
    print(f" Dataset: {std_name}")
    # 修改表头，加入 Accuracy
    print(f"{'Method':<30} | {'Final Regret':<30} | {'Final Acc (%)':<20} | {'Total Time':<30}")
    print("-" * 130)
    
    for exp in experiments:
        info = exp['info']
        regret_str = f"{info['final_mean_regret']:.2f} ± {info['final_std_regret']:.2f}"
        time_str = f"{info['final_mean_time']:.2f} ± {info['final_std_time']:.2f} s"
        
        # Accuracy 字符串
        if info['acc_data_all'] is not None:
            acc_str = f"{info['final_mean_acc']*100:.2f} ± {info['final_std_acc']*100:.2f}"
        else:
            acc_str = "N/A"
            
        print(f"{info['label']:<30} | {regret_str:<30} | {acc_str:<20} | {time_str:<30}")
    print("="*130 + "\n")
    print(f"\n Drawing plots for dataset: {std_name} ...")
    
    # --- 严格复用原来的样式常量 ---
    LINE_WIDTH = 6.0
    TITLE_SIZE = 48
    LABEL_SIZE = 40
    TICK_SIZE = 36
    LEGEND_SIZE = 36
    FIG_SIZE = (12, 9)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    max_rounds_T = max(len(exp['data']) for exp in experiments)
    max_cum_time = max(float(np.max(np.cumsum(exp['data'][:, 0]))) for exp in experiments)
    max_regret_plot = max(float(np.max(exp['data'][:, 1])) for exp in experiments)

    def _regret_axis_xlim():
        if REGRET_AXIS_X_MAX is not None:
            return 0.0, float(REGRET_AXIS_X_MAX)
        return 0.0, max(0.0, float(max_rounds_T - 1))

    def _regret_axis_ylim_rounds():
        if REGRET_AXIS_Y_MAX is not None:
            return 0.0, float(REGRET_AXIS_Y_MAX)
        return 0.0, float(max_rounds_T)

    def _regret_axis_xlim_time():
        if REGRET_AXIS_X_MAX is not None:
            return 0.0, float(REGRET_AXIS_X_MAX)
        return 0.0, float(max_cum_time)

    def _regret_axis_ylim_time():
        if REGRET_AXIS_Y_MAX is not None:
            return 0.0, float(REGRET_AXIS_Y_MAX)
        return 0.0, float(max_regret_plot)

    # =========================================================
    # 图一：Regret vs Rounds (汇总图) -> 存放在 Batch 根目录
    # =========================================================
    fig_rr, ax_rr = plt.subplots(figsize=FIG_SIZE)

    for idx, exp in enumerate(experiments):
        data = exp['data']
        info = exp['info']
        rounds = np.arange(len(data))
        mean_regret = data[:, 1]
        plot_rounds, plot_regret = clip_curve_to_limits(
            rounds,
            mean_regret,
            x_max=REGRET_AXIS_X_MAX,
            y_max=REGRET_AXIS_Y_MAX,
        )
        
        # === 样式逻辑 (保持原样) ===
        current_ls = '-'       
        current_zorder = 5     
        
        if info['label'] == 'LocPRB':
            line_color = 'tab:cyan'
            current_ls = '-'
            current_zorder = 5
        elif info['label'] == 'PRB':
            line_color = 'tab:blue'
            current_ls = '--'
            current_zorder = 10
        else:
            line_color = colors[idx]

        ax_rr.plot(plot_rounds, plot_regret, label=info['label'], color=line_color,
                   linewidth=LINE_WIDTH, linestyle=current_ls, zorder=current_zorder)

    ax_rr.set_title(std_name, fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax_rr.set_xlabel("Rounds", fontsize=LABEL_SIZE, fontweight='bold')
    ax_rr.set_ylabel("Regret", fontsize=LABEL_SIZE, fontweight='bold')
    ax_rr.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=3, length=10)
    ax_rr.legend(frameon=True, fontsize=LEGEND_SIZE, loc='best')
    ax_rr.grid(False)
    fig_rr.tight_layout()
    fig_rr.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.90)
    ax_rr.set_xlim(*_regret_axis_xlim())
    ax_rr.set_ylim(*_regret_axis_ylim_rounds())

    # 保存到 Batch 根目录
    out_path_rounds_png = os.path.join(batch_dir_path, f"{std_name}_Regret_vs_Rounds.png")
    out_path_rounds_pdf = os.path.join(batch_dir_path, f"{std_name}_Regret_vs_Rounds.pdf")
    fig_rr.savefig(out_path_rounds_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    fig_rr.savefig(out_path_rounds_pdf, bbox_inches='tight', pad_inches=0.2)
    print(f"   -> Saved Regret Plot: {out_path_rounds_png}")
    plt.close(fig_rr)

    # =========================================================
    # 图二：Regret vs Time（累计每步时间；坐标轴与图一共用 REGRET_AXIS_*）
    # =========================================================
    fig_rt, ax_rt = plt.subplots(figsize=FIG_SIZE)
    for idx, exp in enumerate(experiments):
        data = exp['data']
        info = exp['info']
        mean_regret = data[:, 1]
        cum_time = np.cumsum(data[:, 0])
        plot_time, plot_regret = clip_curve_to_limits(
            cum_time,
            mean_regret,
            x_max=REGRET_AXIS_X_MAX,
            y_max=REGRET_AXIS_Y_MAX,
        )

        current_ls = '-'
        current_zorder = 5
        if info['label'] == 'LocPRB':
            line_color = 'tab:cyan'
            current_ls = '-'
            current_zorder = 5
        elif info['label'] == 'PRB':
            line_color = 'tab:blue'
            current_ls = '--'
            current_zorder = 10
        else:
            line_color = colors[idx]

        ax_rt.plot(
            plot_time,
            plot_regret,
            label=info['label'],
            color=line_color,
            linewidth=LINE_WIDTH,
            linestyle=current_ls,
            zorder=current_zorder,
        )

    ax_rt.set_title(std_name, fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax_rt.set_xlabel("Time (s)", fontsize=LABEL_SIZE, fontweight='bold')
    ax_rt.set_ylabel("Regret", fontsize=LABEL_SIZE, fontweight='bold')
    ax_rt.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=3, length=10)
    ax_rt.legend(frameon=True, fontsize=LEGEND_SIZE, loc='best')
    ax_rt.grid(False)
    fig_rt.tight_layout()
    fig_rt.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.90)
    # 在 tight_layout 之后再锁坐标轴，避免显示范围被挤回数据范围
    ax_rt.set_xlim(*_regret_axis_xlim_time())
    ax_rt.set_ylim(*_regret_axis_ylim_time())

    out_path_rt_png = os.path.join(batch_dir_path, f"{std_name}_Regret_vs_Time.png")
    out_path_rt_pdf = os.path.join(batch_dir_path, f"{std_name}_Regret_vs_Time.pdf")
    fig_rt.savefig(out_path_rt_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    fig_rt.savefig(out_path_rt_pdf, bbox_inches='tight', pad_inches=0.2)
    rx = REGRET_AXIS_X_MAX if REGRET_AXIS_X_MAX is not None else max_cum_time
    ry = REGRET_AXIS_Y_MAX if REGRET_AXIS_Y_MAX is not None else max_regret_plot
    print(f"   -> Saved Regret vs Time: {out_path_rt_png} (xlim [0,{rx}], ylim [0,{ry}])")
    plt.close(fig_rt)

    # =========================================================
    # 图三：Accuracy vs Time (分进程独立图) -> 存放在子目录
    # =========================================================
    print(f"   -> Generating Individual Accuracy Plots...")
    
    for exp in experiments:
        info = exp['info']
        acc_data = info['acc_data_all'] # (Runs, Points, 2)
        
        if acc_data is None:
            continue
            
        # 1. 创建子文件夹
        # 例如: results/plots/Batch_Output/MovieLens_FastPRB_Details
        sub_folder_name = f"{info['folder']}_Details"
        sub_dir_path = os.path.join(batch_dir_path, sub_folder_name)
        if not os.path.exists(sub_dir_path):
            os.makedirs(sub_dir_path)
            
        # 2. 遍历每个 Worker (Run) 画图
        num_runs = acc_data.shape[0]
        for run_id in range(num_runs):
            # column 0: time, column 1: accuracy
            this_run_time = acc_data[run_id, :, 0]
            this_run_acc  = acc_data[run_id, :, 1] * 100 # 转百分比
            
            final_acc_val = this_run_acc[-1]
            
            plt.figure(figsize=FIG_SIZE)
            
            # 使用紫色画单进程曲线，保持粗线条风格
            plt.plot(this_run_time, this_run_acc, 
                     color='tab:purple', linewidth=LINE_WIDTH, 
                     marker='o', markersize=10) # 加上点更清晰
            
            # 标题包含 Run ID 和最终准确率
            plt.title(f"Worker {run_id} | Final: {final_acc_val:.2f}%", fontsize=TITLE_SIZE, fontweight='bold', pad=20)
            plt.xlabel("Time (s)", fontsize=LABEL_SIZE, fontweight='bold')
            plt.ylabel("Test Accuracy (%)", fontsize=LABEL_SIZE, fontweight='bold')
            
            # 保持大字体刻度
            plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=3, length=10)
            plt.grid(False)
            plt.tight_layout()
            plt.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.90)
            
            # 保存到子目录
            out_name = f"worker_{run_id}_TimeAcc.png"
            plt.savefig(
                os.path.join(sub_dir_path, out_name),
                dpi=100,
                bbox_inches='tight',
                pad_inches=0.2,
            ) # 这里dpi不用太高，主要是为了预览挑选
            plt.close()
            
    print(f"   -> All Individual Plots saved in subfolders under: {batch_dir_path}")
        
def main():
    args = parse_args()
    grouped_data = {}
    print(f"-> Loading {len(args.folders)} folders...")
    
    for folder in args.folders:
        data, info = load_data_from_folder(folder)
        if data is not None:
            ds_name = info['dataset']
            if ds_name not in grouped_data:
                grouped_data[ds_name] = []
            grouped_data[ds_name].append({
                'data': data,
                'info': info
            })
            print(f"   -> Loaded: {folder} (Steps: {len(data)})")

    if not grouped_data:
        print("\n No valid data loaded. Exiting.")
        return

    for ds_name, experiments in grouped_data.items():
        plot_dataset_group(ds_name, experiments)
        
    print("\n All plots generated successfully!")

if __name__ == "__main__":
    main()
