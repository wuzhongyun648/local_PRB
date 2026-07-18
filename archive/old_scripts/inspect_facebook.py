import numpy as np
import os

# 路径根据你的服务器实际情况调整
PROJECT_ROOT = "/mnt/data/xinyu/bandits_pj/PRB"
feat_path = os.path.join(PROJECT_ROOT, "online_link_prediction/data/Facebook/facebook_combined_ALLusers_features.npy")

if os.path.exists(feat_path):
    print(f"-> Loading: {feat_path}")
    data = np.load(feat_path)
    
    print(f"Shape: {data.shape}")
    print(f"Min: {np.min(data)}, Max: {np.max(data)}")
    print(f"Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    
    # 检查稀疏度 (0 的比例)
    zeros = np.sum(data == 0)
    sparsity = zeros / data.size
    print(f"Sparsity (0的占比): {sparsity:.4%}")
    
    # 检查是否只有 0 和 1
    unique_vals = np.unique(data)
    if len(unique_vals) < 20:
        print(f"Unique values: {unique_vals}")
    else:
        print(f"Unique values count: {len(unique_vals)}")
else:
    print("文件不存在")