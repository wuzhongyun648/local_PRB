import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timezone, timedelta

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.join(CURRENT_DIR, "results", "node_classification")
PLOT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "results", "plots", "node_classification")

if not os.path.exists(PLOT_OUTPUT_DIR):
    os.makedirs(PLOT_OUTPUT_DIR)

tz_cn = timezone(timedelta(hours=8))
CURRENT_TIME = datetime.now(tz_cn).strftime("%Y%m%d%H%M")
# ===========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Node Classification Results")
    parser.add_argument('folders', nargs='+', help='List of result folder names to plot')
    return parser.parse_args()

def load_data_from_folder(folder_name):
    """
    """
    full_path = os.path.join(BASE_DIR, folder_name)
    
    if not os.path.exists(full_path):
        print(f"  [Warning] Folder not found: {full_path}")
        return None, None

    final_path = os.path.join(full_path, "final_results.npy")
    target_file = None
    
    if os.path.exists(final_path):
        target_file = final_path
    else:
        checkpoints = glob.glob(os.path.join(full_path, "checkpoint_step_*.npy"))
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            target_file = checkpoints[-1]
            print(f"   -> Using checkpoint: {os.path.basename(target_file)}")
    
    if target_file is None:
        print(f"[Error] No .npy files found in {folder_name}")
        return None, None

    try:
        raw_data = np.load(target_file)
        
        if raw_data.ndim == 3:
            data_mean = np.mean(raw_data, axis=0)
            std_regret = np.std(raw_data[:, :, 1], axis=0)
        else:
            data_mean = raw_data
            std_regret = np.zeros(raw_data.shape[0])
            
        final_mean = data_mean[-1, 1]
        final_std = std_regret[-1]
            
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
            dataset = parts[0]
            method = parts[1]

        key_param = ""
        for p in parts:
            if p.startswith("eps"):
                key_param = f"eps={p.replace('eps', '')}"
                break
            elif p.startswith("powT"):
                key_param = f"T={p.replace('powT', '')}"
                break
        
        label = f"{method}"
        if key_param:
            label += f" ({key_param})"
            
        return data_mean, {
            "dataset": dataset, 
            "label": label, 
            "folder": folder_name, 
            "std": std_regret,
            "final_mean": final_mean,
            "final_std": final_std
        }

    except Exception as e:
        print(f" [Error] Failed to load {target_file}: {e}")
        return None, None

def plot_dataset_group(dataset_name, experiments):
    """
    绘图并保存 (PNG + PDF)
    """
    print(f"\n" + "="*80)
    print(f" Dataset: {dataset_name}")
    print(f"{'Method':<50} | {'Final Regret (Mean ± Std)':<25}")
    print("-" * 80)
    
    for exp in experiments:
        info = exp['info']
        print(f"{info['label']:<50} | {info['final_mean']:.2f} ± {info['final_std']:.2f}")
    print("="*80 + "\n")
    
    print(f" Drawing plots for {dataset_name} ...")
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    plt.figure(figsize=(10, 6))
    for idx, exp in enumerate(experiments):
        data = exp['data']
        info = exp['info']
        std = info.get('std', np.zeros(len(data)))
        
        rounds = np.arange(len(data))
        regret = data[:, 1]
        
        plt.plot(rounds, regret, label=info['label'], color=colors[idx], linewidth=2)
        plt.fill_between(rounds, regret - std, regret + std, color=colors[idx], alpha=0.2)

    plt.title(f"Node Classification Regret: {dataset_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Rounds", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    path_png = os.path.join(PLOT_OUTPUT_DIR, f"{dataset_name}_Regret_vs_Rounds_{CURRENT_TIME}.png")
    path_pdf = os.path.join(PLOT_OUTPUT_DIR, f"{dataset_name}_Regret_vs_Rounds_{CURRENT_TIME}.pdf")
    
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"   -> Saved: {path_png}")
    plt.close()

    plt.figure(figsize=(10, 6))
    for idx, exp in enumerate(experiments):
        data = exp['data']
        info = exp['info']
        std = info.get('std', np.zeros(len(data)))
        
        time_axis = np.cumsum(data[:, 0])
        regret = data[:, 1]
        
        plt.plot(time_axis, regret, label=info['label'], color=colors[idx], linewidth=2)
        plt.fill_between(time_axis, regret - std, regret + std, color=colors[idx], alpha=0.2)

    plt.title(f"Node Classification Regret (Time): {dataset_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    path_png = os.path.join(PLOT_OUTPUT_DIR, f"{dataset_name}_Regret_vs_Time_{CURRENT_TIME}.png")
    path_pdf = os.path.join(PLOT_OUTPUT_DIR, f"{dataset_name}_Regret_vs_Time_{CURRENT_TIME}.pdf")
    
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"   -> Saved: {path_png}")
    plt.close()

def main():
    args = parse_args()
    grouped_data = {}
    
    print(f"-> Searching in: {BASE_DIR}")
    
    for folder in args.folders:
        data, info = load_data_from_folder(folder)
        if data is not None:
            ds_name = info['dataset']
            if ds_name not in grouped_data:
                grouped_data[ds_name] = []
            
            grouped_data[ds_name].append({'data': data, 'info': info})

    if not grouped_data:
        print("\n No valid data loaded.")
        return

    # 绘图
    for ds_name, experiments in grouped_data.items():
        plot_dataset_group(ds_name, experiments)
        
    print("\nAll Node Classification plots generated!")

if __name__ == "__main__":
    main()