# NeuralTS Audit Report

## 结论

当前 `NeuralTS` 不是严格复现论文 Algorithm 1，也不等价于下载的原始 GitHub `NeuralTSDiag` 实验实现。它保留了“对每个 arm 采样 reward、取最大样本 arm”的 Thompson Sampling 外壳，但 posterior/design 矩阵更新、`lambda/nu` 方差语义、训练正则化和在线训练时机存在算法性偏差。

## 核对范围

- 当前 runner：`/mnt/data/xinyu/Fast_bandit/online_baselines_run.py`
- 当前实现：`/mnt/data/xinyu/Fast_bandit/baselines/NeuralTS.py`
- 原论文：`/mnt/data/xinyu/Fast_bandit/paper/baselines/NeuralTS/TS.pdf`
- 原 GitHub 代码：`/mnt/data/xinyu/Fast_bandit/paper/baselines/NeuralTS/NeuralTS/`

## 一致点

- 当前代码、论文和原始代码都使用神经网络输出作为 reward posterior mean，并对每个 arm 的标量 reward 采样后 `argmax` 选臂。
- 当前实现和原始代码都只用被选 arm 的 context/reward 进入历史训练数据，而不是训练所有 arms。
- 网络形态大体一致：单隐藏层 `Linear-ReLU-Linear`，这是原始 GitHub 代码的实际实现方式。

## 不一致/风险点

- Diagonal design update 明显不一致。原始 `NeuralTSDiag` 是逐参数更新 `U += g * g`；当前代码用 `torch.matmul(g, g.T)`，对一维梯度会得到标量并广播加到所有 design entries，丢失逐参数 posterior 方差信息。
- `lambda` 没有进入当前 posterior/design 初始化与采样机制。论文为 `U0 = lambda I`，采样方差含 `lambda` 和 `/m`；原始代码为 `U = lambda * ones` 且 `sigma = sqrt(sum(lambda * nu * g^2 / U))`。当前 `Design = ones`，传入的 `args.lamdba` 进入了未使用的 `sigma` 参数，实际采样只用 `nu * sqrt(sum(g^2 / Design))`。
- 训练目标缺少原始实现的 L2/weight decay 正则化。论文目标包含围绕初始化参数的正则项；原始代码用 `weight_decay`；当前 SGD 没有 weight decay。
- 训练时机不同。原始 `train.py` 每轮 `select` 后立刻 `train(context[arm], reward)`，`delay` 默认 1；当前 runner 每轮先 `update` design/history，但只在前 2000 轮每 50 轮、之后每 100 轮训练网络。
- 论文顺序与两份代码都不完全一致。论文 Algorithm 1 是先用新 reward 训练得到 `theta_t`，再用 `g(x_t,a_t; theta_t)` 更新 `U_t`；原始 GitHub 和当前代码都在本轮训练前更新 design/U。当前点与原始代码接近，但与论文 Algorithm 1 不严格一致。

## 建议修正方向

- 若目标是对齐原始 GitHub 实验实现，应以 `NeuralTSDiag` 为准：使用 `U = lambda * ones`，逐参数 `U += g * g`，采样标准差按原始 `lambda/nu/U` 公式，并恢复训练中的 weight decay 与每轮训练/`delay` 语义。
- 若目标是更贴近论文 Algorithm 1，应调整顺序为：选臂、观测 reward、训练网络得到新参数、再用新参数梯度更新 design，并明确 `/m` scaling 与 `lambda` 的位置。
- 当前最优先修正项是 `Design += torch.matmul(g, g.T)`，这是算法机制级错误，不是超参数差异。

