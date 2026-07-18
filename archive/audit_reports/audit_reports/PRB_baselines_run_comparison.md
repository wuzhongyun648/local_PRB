# PRB `baselines_run.py` Comparison Report

## 结论

`/mnt/data/xinyu/PRB/baselines/baselines_run.py` 更像一份早期或示例 baseline runner，而不是 PRB 论文 Online Link Prediction 表 1/图 2 的完整实验 runner。它固定跑 `load_mnist_1d()`，默认只做 5 次串行 run，训练调度为前 1000 轮每 10 轮训练，之后每 100 轮训练；这些都和 PRB 论文的 online link prediction 设置不一致。

相对当前 `/mnt/data/xinyu/Fast_bandit/online_baselines_run.py`，PRB runner 更接近 EE-Net 原 GitHub 代码里的 baseline 训练调度，但覆盖的数据集、方法、并行和保存协议都更窄。PRB 目录下的 `NeuralTS.py` 和 `NeuralUCB.py` 与 Fast 当前 baseline 文件高度相近，说明 Fast 当前脚本主要是把这套 PRB baseline 代码扩展到了更多数据集和并行运行，而不是回到 NeuralTS/NeuralUCB 原 GitHub 代码。

## 核对范围

- PRB runner：`/mnt/data/xinyu/PRB/baselines/baselines_run.py`
- PRB baseline 模块：`/mnt/data/xinyu/PRB/baselines/NeuralTS.py`、`/mnt/data/xinyu/PRB/baselines/NeuralUCB.py`、`/mnt/data/xinyu/PRB/baselines/load_data.py`
- 当前 Fast runner：`/mnt/data/xinyu/Fast_bandit/online_baselines_run.py`
- 当前 Fast baseline 模块：`/mnt/data/xinyu/Fast_bandit/baselines/NeuralTS.py`、`/mnt/data/xinyu/Fast_bandit/baselines/NeuralUCB.py`、`/mnt/data/xinyu/Fast_bandit/baselines/Neural_epsilon.py`
- PRB 论文：`/mnt/data/xinyu/Fast_bandit/paper/0922NeurIPS-2024-pagerank-bandits-for-link-prediction-Paper-Conference.pdf`
- 原 baseline GitHub 代码：`/mnt/data/xinyu/Fast_bandit/paper/baselines/`

## PRB Runner 的实际行为

- 虽然参数里有 `--dataset`，并且导入了 `load_yelp`、`load_mnist_1d`、`load_movielen`，但主循环每次 run 都固定执行 `b = load_mnist_1d()`。因此它不是 online link prediction 的 MovieLens/Amazon/Facebook/GrQc runner。
- 默认方法为 `["Neural_epsilon", "NeuralTS", "NeuralUCB", "NeuralNoExplore"]`，help 中提到 `KernelUCB`、`LinUCB`，但当前 `/mnt/data/xinyu/PRB/baselines` 目录只看到 `NeuralTS.py`、`NeuralUCB.py`、`load_data.py`、`packages.py` 等文件，缺少 `Neural_epsilon.py`、`NeuralNoExplore.py`、`KernelUCB.py`、`LinUCB.py`。按当前目录快照，这个 runner 不是自包含可运行的完整 baseline 包。
- 每个方法串行跑 5 次，循环固定 `range(10000)`，只保存 `./results/{method}_regret.npy`。
- 神经 baseline 的训练调度是：每轮先 `model.update(context[arm], reward)`；若 `t < 1000`，每 10 轮训练一次；否则每 100 轮训练一次。
- 交互逻辑只处理 `context, rwd = b.step()` 两个返回值，不使用 link prediction loader 返回的 `X_ind/user/item`，也不维护动态图 `G_t`。

## 与 PRB 论文的差异

- 数据集不一致。论文 Online Link Prediction 使用 MovieLens、AmazonFashion、Facebook、GrQc；还报告 Cora/Citeseer/Pubmed 的 online node classification。PRB runner 实际固定 MNIST 1D。
- 候选 arms 设置不一致。论文推荐/社交图设置是每轮随机 100 个候选节点，其中包含 10 个正样本；PRB runner 固定 MNIST 分类式 bandit，不是这套 100-arm link prediction 协议。PRB `load_data.py` 中的 link prediction loaders 也是 10 arms，而不是论文正文的 100 arms。
- 训练频率不一致。论文 Appendix A.1 明确写所有 bandit methods 在 `t < 2000` 时每 50 轮训练，之后每 100 轮训练；PRB runner 是 `t < 1000` 每 10 轮训练。
- runs 不一致。论文报告 10 runs 的 mean/std；PRB runner 固定 5 runs，且只保存 raw regret arrays。
- baseline 覆盖不一致。论文表 1 包含 EE-Net、NeuGreedy、NeuralUCB、NeuralTS、PRB；PRB runner 不包含 EE-Net 和 PRB 本身，只是若干 baseline。
- 超参搜索协议不一致。论文 Appendix A.1 说明 NeuralUCB/TS 对 `nu` 和 `lambda` 做 grid search，并对学习率做 grid search；PRB runner 只接收单组 `--lamdba` 和 `--nu`，学习率在模型文件里固定。

## 与原 GitHub Baseline 代码的差异

### NeuralTS

- 原 GitHub `NeuralTSDiag` 使用 `U = lambda * ones`，采样标准差来自 `sqrt(sum(lambda * nu * g^2 / U))`，并在 `train()` 中使用 weight decay 形式的正则。
- PRB `NeuralTS.py` 使用 `Design = ones`，采样为 `nu * sqrt(sum(g^2 / Design))`，传入的 `sigma/lambda` 没有进入 design 初始化或方差公式。
- PRB `NeuralTS.py` 的 `update()` 使用 `Design += torch.matmul(g, g.T)`；对一维梯度这会变成标量加到整个向量，而不是原 GitHub 的逐参数 `U += g * g`。
- 原 GitHub NeuralTS `train.py` 默认每轮调用 `train(context[arm], reward)`；PRB runner 改成外部 `update()` 后按 10/100 轮 schedule 间歇训练。

### NeuralUCB

- PRB `NeuralUCB.py` 的 UCB 选臂和对角 `U += g * g` 与原 GitHub `NeuralUCBDiag` 的工程版接近。
- 原 GitHub `NeuralUCB.py` 的 `train()` 会在训练函数里 append 当前样本，并使用 `weight_decay=self.lamdba`；PRB 版本拆成外部 `update()` append 样本，`train()` 只训练已有历史，且 optimizer 没有 weight decay。
- 原 GitHub runner 在前 2000 轮每轮训练，之后每 100 轮训练；PRB runner 是前 1000 轮每 10 轮训练，之后每 100 轮训练。

### NeuralGreedy

- 原 Deep Bayesian Bandits 的 Neural Greedy/RMS 是初始 round-robin 后纯 `argmax`，训练由 `PosteriorBNNSampling.update()` 按 `training_freq` 用 RMSProp/minibatch 触发。
- PRB runner 调用的 `Neural_epsilon` 是固定 `p=0.01` 的 epsilon-greedy 标量 arm-context PyTorch 实现，不是原 TensorFlow Neural Greedy/RMS 流程。
- 当前 PRB 目录快照中没有 `Neural_epsilon.py`，所以该 runner 对 NeuralGreedy 的实现依赖并不在 `PRB/baselines` 自身目录内。

### EE-Net

- PRB runner 完全没有 EE-Net。
- EE-Net 原 GitHub runner 的训练调度是 `t < 1000` 每 10 轮、之后每 100 轮，这一点与 PRB runner 的神经 baseline schedule 一致；但 PRB 论文最终实验写的是 `t < 2000` 每 50 轮、之后每 100 轮。

## 与当前 Fast `online_baselines_run.py` 的差异

- Fast runner 支持 7 个数据集：MovieLens、Amazon、Facebook、GrQc、Collab、PPA、Vessel；PRB runner 实际只跑 MNIST 1D。
- Fast runner 默认方法包含 EE-Net、NeuralUCB、NeuralTS、Neural_epsilon、LinUCB、KernelUCB；PRB runner 默认不包含 EE-Net、LinUCB、KernelUCB。
- Fast runner 按 dataset/method/run 构造任务并用 multiprocessing 并行；PRB runner 是 method 外层、5 runs 内层的串行循环。
- Fast runner 对 `b.step()` 的 6 元组返回做兼容，能跑 link prediction loaders；PRB runner 只接受 2 元组，不能直接运行 PRB `load_movielen()` 这类返回 6 元组的 loader。
- Fast runner 的训练调度与 PRB 论文一致：`t < 2000` 每 50 轮训练，之后每 100 轮训练；PRB runner 使用更早的 `t < 1000` 每 10 轮训练。
- Fast runner 保存每个数据集、每个方法、每个 run 的 regret/time 到 `./results/baselines/{Dataset}_{Method}_{regret,time}.npy`；PRB runner 只保存每个方法的 regret 到 `./results/{method}_regret.npy`。
- Fast runner 设置 `np.random.seed(run_id * 100 + 43)`；PRB runner 没有显式设置 seed。
- Fast runner 的 NeuralTS 在 PRB 版本基础上增加了 NaN/std guard 和 `torch.normal(..., size=(1,))`，但核心公式和 `Design += torch.matmul(g, g.T)` 问题仍然继承自 PRB 版本。
- Fast runner 的 NeuralUCB 与 PRB 版本核心几乎一致，主要差异来自 runner 调度、数据集和设备配置。

## 差异分类

### 算法机制级差异

- PRB `NeuralTS.py` 与原 NeuralTS GitHub 的 design/posterior 方差公式不同，且 `Design += torch.matmul(g, g.T)` 是机制级错误。
- PRB `NeuralUCB.py` 缺少原 GitHub `weight_decay=self.lamdba`，训练目标和正则化机制不同。
- PRB/当前 Fast 的 `Neural_epsilon` 不是原 TensorFlow NeuralGreedy/RMS，而是 epsilon-greedy PyTorch 改写。
- PRB runner 的训练频率和论文 Appendix A.1 不一致，会显著改变在线学习轨迹。

### 实验工程/协议差异

- PRB runner 固定 MNIST 1D，未使用 `--dataset` 切换。
- PRB runner 5 runs，论文 10 runs，Fast runner 默认 10 runs。
- PRB runner 串行，Fast runner 并行。
- PRB runner 保存格式更粗，不保存 dataset 维度和 time records。
- PRB runner 当前目录快照缺少若干 import 的 baseline 文件。

### 数据协议差异

- 论文正文描述 recommendation/social 每轮 100 arms 且包含 10 positives；PRB/Fast 当前 link prediction loaders 通常是 10 arms，即 1 positive + 9 negatives。
- PRB runner 实际使用的 MNIST 1D 分类 bandit 与 online link prediction 的 positive/negative edge candidate pool 不是同一数据协议。
- Fast 新增 OGB datasets，并含有随机负采样或 hard-negative 逻辑；这不在 PRB `baselines_run.py` 中。

## 后续修改建议优先级

1. 先决定要对齐哪个目标：PRB 论文实验协议、原 baseline GitHub 代码，还是 PRB repo 里的早期 runner。三者并不一致。
2. 如果目标是论文 Online Link Prediction，应以 Fast runner 的多数据集框架和 `t < 2000` 每 50 轮 schedule 为基础，而不是回退到 PRB `baselines_run.py`。
3. 优先修 NeuralTS 的 design/posterior 更新问题，因为它是最明确的算法错误。
4. 其次恢复 NeuralUCB/NeuralTS 的正则化语义，至少使 `lambda` 在 design 初始化、方差或 optimizer regularization 中按选定目标一致使用。
5. 再决定 NeuralGreedy 是否应改成原 TensorFlow NeuralGreedy/RMS 的 pure greedy + initial round-robin，还是保留并明确命名为 epsilon-greedy adaptation。
6. EE-Net 不应从 PRB `baselines_run.py` 推断，因为该 runner 不包含 EE-Net；应单独以 EE-Net 原 GitHub 与 PRB 论文 Appendix A.1 为准。
