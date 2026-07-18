import torch
from ogb.linkproppred import LinkPropPredDataset
import numpy as np

def check_edge_location():
    print(">>> Loading ogbl-collab raw dataset...")
    # 加载官方数据
    dataset = LinkPropPredDataset(name='ogbl-collab')
    split_edge = dataset.get_edge_split()
    
    # 目标泄露边 (来自你的报错日志)
    target_u = 100860
    target_v = 136989
    
    print(f"\n>>> Hunting for edge ({target_u}, {target_v})...")
    
    # 1. 检查 Train Set
    print("Checking Train Set...")
    train_edges = split_edge['train']['edge']
    # Collab 的 train edge 是 (N, 2) 的 numpy array 或 tensor
    # 检查双向 (u,v) 或 (v,u)
    # 转换为 set 加速查找
    train_set = set()
    for i in range(len(train_edges)):
        u, v = train_edges[i][0], train_edges[i][1]
        train_set.add((u, v))
        train_set.add((v, u))
        
    if (target_u, target_v) in train_set:
        print(f"❌ FOUND in TRAIN set! (Index: {np.where((train_edges[:,0]==target_u) & (train_edges[:,1]==target_v))})")
        print("结论：训练集中包含了测试边。可能是 Collab 的'多重边'特性，或者是数据集版本问题。")
    else:
        print("✅ Not in Train set.")

    # 2. 检查 Validation Set
    print("Checking Validation Set...")
    valid_edges = split_edge['valid']['edge']
    valid_set = set()
    for i in range(len(valid_edges)):
        u, v = valid_edges[i][0], valid_edges[i][1]
        valid_set.add((u, v))
        valid_set.add((v, u))

    if (target_u, target_v) in valid_set:
        print("❌ FOUND in VALIDATION set!")
        print("结论：验证集中包含了测试边。请检查 Phase 2 的注入逻辑。")
    else:
        print("✅ Not in Validation set.")
        
    # 3. 检查 Test Set (确认它确实在测试集里)
    print("Checking Test Set...")
    test_edges = split_edge['test']['edge']
    test_set = set()
    for i in range(len(test_edges)):
        u, v = test_edges[i][0], test_edges[i][1]
        test_set.add((u, v))
        test_set.add((v, u))
        
    if (target_u, target_v) in test_set:
        print("ℹ️ Confirmed in TEST set.")
    else:
        print("❓ Not in Test set? 那你的程序为什么会测这条边？")

if __name__ == "__main__":
    check_edge_location()