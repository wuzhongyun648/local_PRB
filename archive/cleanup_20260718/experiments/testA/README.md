# A0-A10 Controlled Ablations

该目录用于逐项测试 `paper/CODE_PAPER_DIFFERENCES.md` 中能够在不改变数据任务定义的前提下独立评估的算法修改。活动代码 `main.py`、`src/main.py` 和 `src/main_dyn.py` 均不需要修改。

## 实验定义

| ID | 基准 | 唯一变化 |
|---|---|---|
| A0 | Current | 当前 `main.py` 行为，不改变模型或传播算法 |
| A1 | A0 | exploration 改为两层全连接 ReLU 网络 |
| A2 | A0 | 取消 gradient block pooling，使用完整梯度 |
| A3 | A0 | 使用论文形式的 Gaussian weight initialization，bias 置零 |
| A4 | A0 | 去掉 500 轮后的 exploration `1/sqrt(t)` 衰减 |
| A5 | A0 | 去掉 exploration target/output 的 `+1/-1` 平移 |
| A6 | A0 | 去掉前 1000 轮失败时对未选臂添加目标 `1.2` 的样本 |
| A7 | A0 | 每条 edge context 做 L2 normalization |
| A8 | A0 | personalization vector 做 L1 normalization |
| A9 | A0 | 每轮只用最新反馈对两个网络各做一步 SGD |
| A10 | A0 | 使用 `src/main_dyn.py` 中 invariant-correct 的 DYN-APPR |

每个 A1-A10 都只相对 A0 改变一项，不累计前面的修改。

## 随机数据流控制

Runner 为每个 loader 保存独立的 NumPy RNG 状态。模型训练中的 `np.random.shuffle` 等调用不会改变 loader 后续生成的正负候选。因此在 dataset、seed、T 和 `n_neg` 相同时，A0-A10 获得相同的固定测试样本和在线候选数据流。

动态图仍由算法实际选择决定。不同算法作出不同选择后，后续图状态发生分叉是在线问题本身的一部分，不应消除。

## 输出隔离

每项结果写入：

```text
testA/results/A0/online_link_prediction/...
testA/results/A1/online_link_prediction/...
...
testA/results/A10/online_link_prediction/...
```

因此同一分钟运行多个消融也不会互相覆盖。

## 单项运行

以下命令以 Collab、T=500 为例：

```bash
cd /mnt/data/xinyu/Fast_bandit

python -u testA/run_ablation.py \
  --ablation A0 \
  --graph_name Collab \
  --method FastPRB \
  --alpha 0.85 \
  --appr_eps 4.24e-06 \
  --T 500 \
  --lr1 0.01 \
  --lr2 0.004 \
  --runs 10 \
  --workers 5 \
  --seed 0
```

把 `--ablation A0` 改为 `A1` 至 `A10` 即可运行对应实验。

## 建议的两阶段运行方式

先用较小规模检查全部变体：

注意：A4 只在 `t > 500` 时与 A0 产生差异，因此研究 A4 时必须设置
`T > 501`。若统一筛选全部变体，建议使用 `T=1000`；下面保留
`T=500` 只是快速检查运行完整性的示例。

```bash
for a in A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10; do
  python -u testA/run_ablation.py \
    --ablation "$a" \
    --graph_name Collab \
    --method FastPRB \
    --alpha 0.85 \
    --appr_eps 4.24e-06 \
    --T 500 \
    --lr1 0.01 \
    --lr2 0.004 \
    --runs 10 \
    --workers 5 \
    --seed 0 \
    > "log/testA_${a}_Collab_T500.log" 2>&1
done
```

这里使用顺序执行，避免 11 组实验同时争用 CPU、GPU 和内存，导致 time 指标无法比较。

筛选后再将 `T` 提升到论文使用的 horizon。若需要 `nohup`，单项命令形式为：

```bash
nohup python -u testA/run_ablation.py \
  --ablation A4 \
  --graph_name Collab \
  --method FastPRB \
  --alpha 0.85 \
  --appr_eps 4.24e-06 \
  --T 500 \
  --lr1 0.01 \
  --lr2 0.004 \
  --runs 10 \
  --workers 5 \
  --seed 0 \
  > log/testA_A4_Collab_T500.log 2>&1 < /dev/null &
```

## 汇总结果

所有实验完成后执行：

```bash
python testA/summarize.py
```

输出字段包括：

- runs 和 T；
- 最终累计 regret 均值与标准差；
- 每个 run 的在线时间均值与标准差；
- 原始 `final_results.npy` 路径。

## 解释限制

A1-A10 用于隔离算法实现因素，不包括以下会改变任务定义的修改：

- cold graph；
- 候选共享 serving node；
- 官方 OGB test split；
- 移除 validation/test graph leakage；
- Vessel uniform negatives；
- propagated pseudo-regret。

这些项目应放在单独的 Paper-Strict protocol 中评估，不能只根据与当前任务的绝对 regret 大小决定是否采用。
