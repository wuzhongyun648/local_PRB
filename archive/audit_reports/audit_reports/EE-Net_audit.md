# EE-Net Audit Report

## 结论

当前 EE-Net baseline 基本复刻了作者 GitHub 代码的核心实现，尤其是 `predict/update/train` 里的 f1、f2、可选 f3、线性决策器、f2 shifted target 和早期伪样本逻辑。

但它不完全等同于论文 Algorithm 1 的干净算法形式：论文是 `f1 + f2`，f2 直接学习 residual `r - f1(x)`；代码实现是 `f1 + w(t) * (f2 - 1)`，f2 学习 `(r - f1(x)) + 1`，并额外加入早期失败样本的伪标签。

## 核对范围

- 当前 runner：`/mnt/data/xinyu/Fast_bandit/online_baselines_run.py`
- 当前实现：`/mnt/data/xinyu/Fast_bandit/baselines/EENet_origin.py`
- 当前实现：`/mnt/data/xinyu/Fast_bandit/baselines/EENetClass_origin.py`
- 原论文：`/mnt/data/xinyu/Fast_bandit/paper/baselines/EE-Net/【13】Neural Exploitation and Exploration of Contextual Bandits by Ban yikun.pdf`
- 原 GitHub 代码：`/mnt/data/xinyu/Fast_bandit/paper/baselines/EE-Net/EE-Net-ICLR-2022/`

## 一致点

- 当前 `baselines/EENet_origin.py` 与原始 `paper/baselines/EE-Net/EE-Net-ICLR-2022/EENet.py` 的核心流程一致：先用 f1 输出 exploitation score 和梯度，再用 f2 对聚合梯度输出 exploration score，最后根据线性或神经 decision maker 选臂。
- 当前默认线上 baseline 使用 `neural_decision_maker=False`，与原始 `EENet_run.py` 一致，实际走线性决策器，不启用 f3 选臂。
- f1/f2/f3 网络结构基本沿用原始 GitHub：f1 是 MLP，f2 是一层 Conv1d 加 FC，f3 是输入 `[f1_score, f2_score]` 的 MLP。
- 当前 runner 的 EE-Net 调用顺序是 `predict(context, t)` -> 取 reward -> `update(context, reward, t)` -> 周期性 `train(t)`，与原始 GitHub runner 的总体 online loop 一致。

## 不一致/风险点

- 论文 vs 代码：f2 target 与打分形式不同。论文 Algorithm 1 中 f2 学习 `r - f1(x)`，选择 `argmax(f1 + f2)`；原始 GitHub 和当前代码都用 `r_2 = (r - f1) + 1`，选择 `f1 + w(t) * (f2 - 1)`。这与论文数学形式不同，但当前代码与作者 GitHub 一致。
- 论文没有早期伪样本增强。原始 GitHub 和当前代码在 `t < 1000` 且选中臂 reward 为 0 时，会给其他未选臂的 f2 梯度追加伪标签 `1.2`。这是作者代码里的启发式，不在论文 Algorithm 1 中。
- 当前 runner 的训练频率不同于原始 GitHub runner。原始 `EENet_run.py` 是 `t < 1000` 每 10 轮 train，之后每 100 轮；当前 `online_baselines_run.py` 是 `t < 2000` 每 50 轮，之后每 100 轮。这是训练流程差异，不只是数值超参。
- f2 CNN kernel size 被当前 runner 改为数据集相关。原始 GitHub f2 默认 kernel size 100；当前 EE-Net 初始化传入 `kernel_size=40`，Vessel 为 5。机制仍是 Conv1d f2，但不是原始 GitHub 的精确结构。
- 论文 Algorithm 1 描述为每轮 SGD 更新 f1/f2。原始 GitHub 和当前代码都先缓存样本，然后周期性 `train()` 在历史样本上循环优化。因此当前代码更像作者发布实现，而不是论文伪代码的逐轮单步更新。

## 建议修正方向

- 如果目标是复现作者 GitHub baseline，当前实现核心算法可保留；主要应考虑把训练 schedule 和 f2 kernel 设置明确标注为当前在线链路预测适配，而非原始 GitHub 完全一致。
- 如果目标是严格贴论文 Algorithm 1，则应移除 shifted target/`-1` 平移、早期伪样本增强和加权 `w(t)`，改为 f2 直接学习 `r - f1(x)`，选择 `argmax(f1 + f2)`，并统一说明是否采用逐轮 SGD 还是作者代码式周期训练。
