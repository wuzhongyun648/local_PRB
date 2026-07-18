import numpy as np
import os
import sys

# 确保能导入当前目录下的 load_data
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from load_data import load_movielen, load_amazon_fashion, load_facebook, load_grqc

def print_dataset_table():
    print("\n" + "="*95)
    print(f"{'Dataset':<20} | {'#Nodes':<10} | {'#Edges':<10} | {'Type':<20} | {'Context dim. (d)':<15}")
    print("-" * 95)

    datasets = [
        # (Display Name, Loader Class, Manual Node Count, Graph Type)
        # Node counts are based on your main.py get_configurations function
        ("MovieLens", load_movielen, 12000, "Bipartite (U U I)"),     # 2000 Users + 10000 Items
        ("AmazonFashion", load_amazon_fashion, 8000, "Bipartite (U U I)"), # 4000 Users + 4000 Items
        ("Facebook", load_facebook, 4039, "Unipartite"),             # 4039 Users
        ("GrQc", load_grqc, 5242, "Unipartite")                      # 5242 Users
    ]

    for name, loader_cls, node_count, graph_type in datasets:
        try:
            # 实例化 Loader，这会读取 .npy 文件
            # 注意：Loader 初始化时会打印一些信息，我们暂时忽略
            loader = loader_cls()
            
            # 获取 Edges 数量
            # 逻辑：load_data.py 中 self.pos_index 存储了所有权重为 1 的正样本边
            num_edges = len(loader.pos_index)
            
            # 获取 Context Dimension
            dim = loader.dim
            
            # 打印一行数据
            # 使用 , 分隔千分位
            print(f"{name:<20} | {node_count:<10,} | {num_edges:<10,} | {graph_type:<20} | {dim:<15}")

        except Exception as e:
            print(f"{name:<20} | {'Error':<10} | {'Error':<10} | {graph_type:<20} | {'Error':<15}")
            print(f"  [Error Details]: {e}")
    
    print("="*95 + "\n")
    print("注: #Nodes 数据来源于 main.py 配置; #Edges 为 load_data.py 中读取的 pos_index 长度。")

if __name__ == "__main__":
    print_dataset_table()