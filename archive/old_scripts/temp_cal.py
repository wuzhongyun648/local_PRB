import os
import sys
import numpy as np

# 路径 hack
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from load_data import (
    load_movielen, load_facebook, load_amazon_fashion, load_grqc,
    load_ogb_collab, load_ogb_ppa, load_ogb_vessel, load_ogb_citation2
)

# 定义项目根目录 (必须和你 main.py 里的一致)
PROJECT_ROOT = "/mnt/data/xinyu/bandits_pj/PRB"

def get_n_and_eps(name):
    print(f"正在加载 {name} ...", end="", flush=True)
    loader = None
    
    # 实例化 Loader (注意小数据集需要路径 hack，但 load_data 内部已经处理了 os.path)
    # 我们只需要利用 load_data 内部写死的路径逻辑
    try:
        if name == 'MovieLens': loader = load_movielen()
        elif name == 'Amazon_fashion': loader = load_amazon_fashion()
        elif name == 'Facebook': loader = load_facebook()
        elif name == 'Grqc': loader = load_grqc()
        elif name == 'Collab': loader = load_ogb_collab()
        elif name == 'PPA': loader = load_ogb_ppa()
        elif name == 'Vessel': loader = load_ogb_vessel()
        elif name == 'Citation2': loader = load_ogb_citation2()
        
        n = loader.num_nodes
        eps = 1.0 / n
        print(f" 完成")
        return n, eps
    except Exception as e:
        print(f" 失败: {e}")
        return 0, 0

if __name__ == "__main__":
    datasets = [
        'MovieLens', 'Amazon_fashion', 'Facebook', 'Grqc',
        'Collab', 'PPA', 'Vessel', 'Citation2'
    ]
    
    print(f"{'Dataset':<20} | {'Nodes (n)':<12} | {'Epsilon (1/n)':<20}")
    print("-" * 60)
    
    for name in datasets:
        n, eps = get_n_and_eps(name)
        # 格式化输出，方便复制
        # .2e 表示科学计数法，保留2位小数
        print(f"{name:<20} | {n:<12} | {eps:.3e} (或 {eps:.10f})")