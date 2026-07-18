import argparse
import numpy as np
import sys 
import os 
import time
import multiprocessing as mp
import traceback
from collections import defaultdict

# ==============================================================================
# 路径配置 & 导入修复
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "online_link_prediction"))

# 导入模型
from KernelUCB import KernelUCB
from LinUCB import Linearucb
from Neural_epsilon import Neural_epsilon
from NeuralTS import NeuralTS
from NeuralUCB import NeuralUCBDiag
from NeuralNoExplore import NeuralNoExplore
try:
    from EENet import EE_Net
except ImportError:
    from online_link_prediction.EENet import EE_Net

# 导入数据加载器
from load_data import load_movielen, load_amazon_fashion, load_facebook, load_grqc

# 硬件设置
os.environ["OMP_NUM_THREADS"] = "1" # 防止 numpy 多线程与 multiprocessing 冲突
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # 如果基线不需要 GPU，建议禁用以省显存

# ==============================================================================
# Worker: 单个任务执行逻辑
# ==============================================================================
def run_single_task(dataset_name, method_name, run_id, args):
    """
    执行单元：跑 1 个数据集 的 1 个方法 的 第 run_id 次实验
    """
    try:
        # 设置随机种子
        seed = run_id * 100 + 42
        np.random.seed(seed)
        
        # 1. 加载数据集
        if dataset_name == 'MovieLens':
            b = load_movielen()
        elif dataset_name == 'Amazon':
            b = load_amazon_fashion()
        elif dataset_name == 'Facebook':
            b = load_facebook()
        elif dataset_name == 'GrQc':
            b = load_grqc()
        else:
            return None # 异常

        # 2. 模型初始化
        if method_name == "KernelUCB":
            model = KernelUCB(b.dim, args.lamdba, args.nu)
        elif method_name == "LinUCB":
            model = Linearucb(b.dim, args.lamdba, args.nu)
        elif method_name == "Neural_epsilon":
            model = Neural_epsilon(b.dim, 0.01) # NeuGreedy
        elif method_name == "NeuralTS":
            model = NeuralTS(b.dim, b.n_arm, m=100, sigma=args.lamdba, nu=args.nu)
        elif method_name == "NeuralUCB":
            model = NeuralUCBDiag(b.dim, lamdba=args.lamdba, nu=args.nu, hidden=100)
        elif method_name == "NeuralNoExplore":
            model = NeuralNoExplore(b.dim)
        elif method_name == "EE-Net":
            # 这里的 lr 根据数据集微调？或者统一使用命令行传入的
            model = EE_Net(b.dim, b.n_arm, pool_step_size=50, 
                           lr_1=args.lr1, lr_2=args.lr2, 
                           hidden=100, neural_decision_maker=False, kernel_size=40)
        
        regrets = []
        sum_regret = 0
        
        # 3. 训练循环
        # 为了减少子进程通信开销，我们不实时打印，只返回结果
        for t in range(10000):
            step_result = b.step()
            if len(step_result) == 6:
                context, context_ind, rwd, _, _, _ = step_result
            else:
                context, rwd = step_result

            # Decision
            if method_name == "EE-Net":
                arm_select, _ = model.predict(context, t)
            else:
                arm_select = model.select(context)
            
            reward = rwd[arm_select]

            # Update
            if method_name in ["LinUCB", "KernelUCB"]:
                model.train(context[arm_select], reward)
            elif method_name == "EE-Net":
                model.update(context, reward, t)
                if t < 2000:
                    if t % 50 == 0: model.train(t)
                else:
                    if t % 100 == 0: model.train(t)
            else:
                model.update(context[arm_select], reward)
                if t < 1000:
                    if t % 10 == 0: model.train(t)
                else:
                    if t % 100 == 0: model.train(t)

            regret = np.max(rwd) - reward
            sum_regret += regret
            regrets.append(sum_regret)

        print(f"✅ [Done] {dataset_name} | {method_name} | Run {run_id} | Final: {sum_regret:.0f}", flush=True)
        
        # 返回标识信息和数据
        return (dataset_name, method_name, run_id, regrets)

    except Exception as e:
        print(f"❌ [Error] {dataset_name} | {method_name} | Run {run_id}: {e}", flush=True)
        traceback.print_exc()
        return None

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ALL baselines in parallel')
    
    # 支持一次输入多个数据集，或者默认全部
    parser.add_argument('--datasets', nargs="+", default=['MovieLens', 'Amazon', 'Facebook', 'GrQc'],
                        help='List of datasets to run')
    
    # 支持一次输入多个方法
    parser.add_argument("--methods", nargs="+", 
                        default=["EE-Net", "NeuralUCB", "NeuralTS", "Neural_epsilon", "LinUCB", "KernelUCB"],
                        help='List of methods')
    
    parser.add_argument('--lamdba', default=0.1, type=float)
    parser.add_argument('--nu', default=0.1, type=float)
    parser.add_argument('--lr1', default=0.1, type=float, help='EE-Net lr1')
    parser.add_argument('--lr2', default=0.01, type=float, help='EE-Net lr2')
    parser.add_argument('--runs', default=5, type=int, help='Number of independent runs per method')
    parser.add_argument('--workers', default=10, type=int, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # 1. 生成任务列表
    # 笛卡尔积: Dataset x Method x Run
    tasks = []
    for ds in args.datasets:
        for method in args.methods:
            for run in range(args.runs):
                tasks.append((ds, method, run, args))
    
    print(f"=== Starting Parallel Baselines ===")
    print(f"Datasets: {args.datasets}")
    print(f"Methods:  {args.methods}")
    print(f"Total Tasks: {len(tasks)}")
    print(f"Workers: {args.workers}")
    print("=" * 50)
    
    # 2. 启动进程池
    start_time = time.time()
    
    # 必须用 spawn 启动，因为涉及 PyTorch CUDA (即使不用GPU，spawn也是最安全的)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    with mp.Pool(processes=args.workers) as pool:
        # 使用 starmap 将参数解包传给函数
        results = pool.starmap(run_single_task, tasks)
        
    # 3. 结果聚合
    # 结构: data_store[dataset][method] = [run0_data, run1_data, ...]
    data_store = defaultdict(lambda: defaultdict(list))
    
    for res in results:
        if res is not None:
            ds, met, run, regrets = res
            data_store[ds][met].append(regrets)
            
    # 4. 保存结果
    save_dir = "./results/baselines"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("\n=== Saving Results ===")
    for ds in data_store:
        for met in data_store[ds]:
            # 确保按 run_id 顺序并不重要，这里只是收集了所有 run 的列表
            runs_data = data_store[ds][met]
            
            # 转换为 numpy 数组 (Runs, Steps)
            runs_array = np.array(runs_data)
            
            # 保存路径: results/baselines/MovieLens_NeuralUCB_regret.npy
            filename = f"{ds}_{met}_regret.npy"
            path = os.path.join(save_dir, filename)
            
            np.save(path, runs_array)
            print(f"Saved: {filename} (Shape: {runs_array.shape})")
            
    total_time = time.time() - start_time
    print(f"\nAll tasks finished in {total_time/60:.2f} minutes.")