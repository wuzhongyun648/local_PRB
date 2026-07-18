import argparse
import numpy as np
import sys 
import os 
import time
import multiprocessing as mp
import traceback
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
warnings.filterwarnings("ignore", message=".*The use of `x.T` on tensors.*")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "online_link_prediction"))
sys.path.append(os.path.join(current_dir, "baselines"))
from baselines.KernelUCB import KernelUCB
from baselines.LinUCB import Linearucb
from baselines.Neural_epsilon import Neural_epsilon
from baselines.NeuralTS import NeuralTS
from baselines.NeuralUCB import NeuralUCBDiag
from baselines.NeuralNoExplore import NeuralNoExplore
from baselines.EENet_origin import EE_Net
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["OMP_NUM_THREADS"] = "1" 
from load_data import load_movielen, load_amazon_fashion, load_facebook, load_grqc, load_ogb_collab, load_ogb_ppa, load_ogb_vessel



def run_single_task(dataset_name, method_name, run_id, args):
    """
    """
    try:
        seed = run_id * 100 + 43
        np.random.seed(seed)
        
        if dataset_name == 'MovieLens':
            b = load_movielen()
        elif dataset_name == 'Amazon':
            b = load_amazon_fashion()
        elif dataset_name == 'Facebook':
            b = load_facebook()
        elif dataset_name == 'GrQc':
            b = load_grqc()
        elif dataset_name == 'Collab':
            b = load_ogb_collab()
        elif dataset_name == 'PPA':
            b = load_ogb_ppa()
        elif dataset_name == 'Vessel':
            b = load_ogb_vessel()    
        else:
            return None 

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
            k_size = 40
            if dataset_name == "Vessel":
                k_size = 5
            model = EE_Net(
                b.dim,
                b.n_arm,
                pool_step_size=50,
                lr_1=args.lr1,
                lr_2=args.lr2,
                lr_3=0.01,
                hidden=100,
                neural_decision_maker=False,
                kernel_size=k_size,
            )
            

        regrets = []
        time_records = []
        sum_regret = 0
        start_time_total = time.time()
        for t in range(args.T):
            step_start = time.time()
            step_result = b.step()
            if len(step_result) == 6:
                context, context_ind, rwd, _, _, _ = step_result
            else:
                context, rwd = step_result
            
            if method_name == "EE-Net":
                arm_select = model.predict(context, t)
                
            else:
                arm_select = model.select(context)
            
            reward = rwd[arm_select]

            if method_name in ["LinUCB", "KernelUCB"]:
                model.train(context[arm_select], reward)
            elif method_name == "EE-Net":
                model.update(context, reward, t)
                if t < 2000:
                    if t % 50 == 0: 
                        _ = model.train(t)
                else:
                    if t % 100 == 0: 
                        _ = model.train(t)
            else:
                model.update(context[arm_select], reward)
                if t < 2000:
                    if t % 50 == 0: model.train(t)
                else:
                    if t % 100 == 0: model.train(t)

            regret = np.max(rwd) - reward
            sum_regret += regret
            regrets.append(sum_regret)
            
        
            step_end = time.time()
            step_duration = step_end - step_start
            time_records.append(step_duration)
            if t % 100 == 0:
                total_elapsed = step_end - start_time_total
                print(f"[{dataset_name} | {method_name} | Run {run_id}] Step {t} | Regret: {sum_regret:.0f} | Time: {total_elapsed:.2f}s", flush=True)
        print(f" [Done] {dataset_name} | {method_name} | Run {run_id} | Final: {sum_regret:.0f}", flush=True)
        
        return (dataset_name, method_name, run_id, regrets, time_records)

    except Exception as e:
        print(f" [Error] {dataset_name} | {method_name} | Run {run_id}: {e}", flush=True)
        traceback.print_exc()
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ALL baselines in parallel')
    
    parser.add_argument('--datasets', nargs="+", default=['MovieLens', 'Amazon', 'Facebook', 'GrQc', 'Collab', 'PPA', 'Vessel'],help='List of datasets to run')
    
    parser.add_argument("--methods", nargs="+", 
                        default=["EE-Net", "NeuralUCB", "NeuralTS", "Neural_epsilon", "LinUCB", "KernelUCB"],
                        help='List of methods')
    parser.add_argument('--T', default=10000, type=int, help='Total number of rounds')
    parser.add_argument('--lamdba', default=0.1, type=float)
    parser.add_argument('--nu', default=0.001, type=float)
    parser.add_argument('--lr1', default=0.1, type=float, help='EE-Net lr1')
    parser.add_argument('--lr2', default=0.01, type=float, help='EE-Net lr2')
    parser.add_argument('--runs', default=10, type=int, help='Number of independent runs per method')
    parser.add_argument('--workers', default=10, type=int, help='Number of parallel workers')
    
    args = parser.parse_args()
    
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
    
    start_time = time.time()
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    with mp.Pool(processes=args.workers) as pool:
        results = pool.starmap(run_single_task, tasks)
        
    data_store = defaultdict(lambda: defaultdict(list))
    time_store = defaultdict(lambda: defaultdict(list))
    
    for res in results:
        if res is not None:
            ds, met, run, regrets, times = res
            data_store[ds][met].append(regrets)
            time_store[ds][met].append(times)
    save_dir = "./results/baselines"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("\n=== Saving Results ===")
    for ds in data_store:
        for met in data_store[ds]:
            runs_data = data_store[ds][met]
            runs_array = np.array(runs_data)
            
            filename_regret = f"{ds}_{met}_regret.npy"
            path_regret = os.path.join(save_dir, filename_regret)
            np.save(path_regret, runs_array)
            
            runs_time_data = time_store[ds][met]
            runs_time_array = np.array(runs_time_data)
            
            filename_time = f"{ds}_{met}_time.npy"
            path_time = os.path.join(save_dir, filename_time)
            np.save(path_time, runs_time_array)
            
            print(f"Saved: {filename_regret} & {filename_time}")
            
    total_time = time.time() - start_time
    print(f"\nAll tasks finished in {total_time/60:.2f} minutes.")