# NeuralUCB Audit Report

## 结论

当前 `NeuralUCB` 不是论文 Algorithm 1 的严格实现；它更接近下载 GitHub 中的 `NeuralUCBDiag` 对角近似工程版。但当前 Online baseline 也没有完全复现下载 GitHub 代码的训练流程，主要差异在训练触发、样本入库方式和正则化。

## 核对范围

- 当前 runner：`/mnt/data/xinyu/Fast_bandit/online_baselines_run.py`
- 当前实现：`/mnt/data/xinyu/Fast_bandit/baselines/NeuralUCB.py`
- 原论文：`/mnt/data/xinyu/Fast_bandit/paper/baselines/NeuralUCB/UCB.pdf`
- 原 GitHub 代码：`/mnt/data/xinyu/Fast_bandit/paper/baselines/NeuralUCB/NeuralUCB/`

## 一致点

- 当前实现和原 GitHub `NeuralUCBDiag` 都使用单个 ReLU 网络预测奖励，并用每个候选 arm 的网络梯度构造 UCB：`f(x) + sigma`。
- 当前实现和原 GitHub 都使用对角形式的不确定性累计：`U = lambda * ones`，选择 arm 后执行 `U += g_selected * g_selected`。这与 GitHub `learner_diag.py` 一致，但只是论文 full-matrix `Z_t` 的对角近似。
- 当前 runner 的交互顺序大体是：观测候选 context、`select`、拿 reward、记录样本、间歇训练。

## 不一致/风险点

- 与论文不一致。论文 Algorithm 1 使用完整设计矩阵 `Z_t = Z_{t-1} + g g^T / m`，UCB 为 `f(x; theta) + gamma * sqrt(g^T Z^{-1} g / m)`；当前代码使用对角 `U` 和 `sqrt(sum(lambda * nu * g^2 / U))`，没有 full matrix、`/m`、也没有论文中的动态 `gamma_t` 公式。
- 与论文训练目标不一致。论文 `TrainNN` 最小化历史样本平方误差加 `m lambda ||theta - theta0||^2 / 2`，并以 `theta0` 为锚；当前代码只做平方误差 SGD，没有 `theta0` 锚定正则。
- 与原 GitHub 代码不一致。原 GitHub `learner_diag.py` 的 optimizer 使用 `weight_decay=self.lamdba`；当前 `NeuralUCB.py` 的 optimizer 没有 weight decay。这会改变训练目标机制，不只是数值超参差异。
- 与原 GitHub 训练流程不一致。原 GitHub `train.py` 在 `t < 2000` 时每轮调用 `l.train(context[arm], r)`，`t >= 2000` 时每 100 轮才调用；且样本是在 `train()` 内 append，所以未训练轮的样本不会进入历史。当前 runner 每轮先 `model.update(...)` append 样本，但 `t < 2000` 只每 50 轮训练，之后每 100 轮训练。因此当前会累计所有轮样本但早期训练频率更低，后期训练集也包含原 GitHub 会跳过的中间样本。
- 与论文训练时序不一致。论文每轮在收到 reward 后更新 `Z_t` 并训练 `theta_t`；当前 Online baseline 是间歇训练，不是每轮训练。

## 建议修正方向

- 若目标是复现下载 GitHub `NeuralUCBDiag`：把当前 `update/train` 流程改回“训练时才 append 样本”的语义，并恢复 optimizer 的 `weight_decay=self.lamdba`；训练触发也应按原 `train.py` 对齐。
- 若目标是贴近论文 Algorithm 1：需要实现 full `Z`/`Z^{-1}` 或明确声明使用 diagonal approximation，同时补上论文式 `gamma_t`、`/m` 缩放、`theta0` 锚定正则和每轮基于历史数据训练的流程。
