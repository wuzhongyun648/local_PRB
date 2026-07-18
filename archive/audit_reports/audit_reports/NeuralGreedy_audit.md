# NeuralGreedy Audit Report

## 结论

当前 `/mnt/data/xinyu/Fast_bandit/baselines/Neural_epsilon.py` 不是严格复刻 Deep Bayesian Bandits Showdown 原始 TensorFlow Neural Greedy/RMS 流程。它保留了“神经网络预测奖励、按预测最大值选 arm、用已观察反馈训练”的核心贪心 baseline 思路，但加入了固定 epsilon 随机探索，且训练机制、warm-start、网络输出接口与官方代码不同。

## 核对范围

- 当前 runner：`/mnt/data/xinyu/Fast_bandit/online_baselines_run.py`
- 当前实现：`/mnt/data/xinyu/Fast_bandit/baselines/Neural_epsilon.py`
- 原论文：`/mnt/data/xinyu/Fast_bandit/paper/baselines/NeuralGreedy/Nerual Greedy.pdf`
- 原 GitHub 代码：`/mnt/data/xinyu/Fast_bandit/paper/baselines/NeuralGreedy/tensorflow-models/research/deep_contextual_bandits/`

## 一致点

- 主循环一致：观察候选 context，选 action，获得该 action reward，再更新/训练。
- 都是不维护 posterior/UCB covariance 的点估计神经网络 greedy baseline。
- 都只用已选 action 的反馈训练：当前只存 `context[arm_select], reward`；官方 `ContextualDataset.add(context, action, reward)` 用 one-hot weights 只反传被选 action。

## 不一致/风险点

- 探索机制不同。论文/README 描述 Neural Greedy “acts greedily”，epsilon-greedy 只是可选变体；官方 `PosteriorBNNSampling.action` 初始 round-robin 后直接 `argmax`。当前 `Neural_epsilon(p=0.01)` 每轮有 1% 随机选 arm，且没有官方的 initial pulls round-robin。
- 优化器机制不同。原始 README 调用 `PosteriorBNNSampling(..., 'RMSProp')`，官方 `NeuralBanditModel.select_optimizer()` 返回 `tf.train.RMSPropOptimizer`；当前每次 `train()` 创建 `optim.SGD`。这不是单纯超参值差异，而是训练算法不同。
- 训练过程不同。官方每 `training_freq` 在 `update()` 内训练固定 `training_epochs` 个随机 minibatch，batch 来自 `ContextualDataset.get_batch_with_weights()`；当前 runner 外部按 `t < 2000` 每 50、之后每 100 调 `train()`，训练为打乱历史数据后的逐样本 SGD，最多 2000 step 或 loss 阈值提前停。
- 模型接口不同。官方是“一个 context -> num_actions 输出”，用 action index 权重训练；当前 link prediction 适配为“每个候选 edge/context 一行 -> 标量输出”，再在候选行之间 `argmax`。这可能适合当前任务，但不是官方 TensorFlow 代码的同一接口。
- 官方有学习率 decay/reset 与 gradient clipping 机制；当前没有。若目标是复刻官方 RMS NeuralGreedy，这属于训练机制差异。

## 建议修正方向

- 若要对齐原始 Deep Bayesian Bandits Neural Greedy/RMS：移除固定 epsilon，加入 `initial_pulls` round-robin warm-start，使用 RMSProp、随机 minibatch、固定训练 epochs、LR reset/decay 与 gradient clipping。
- 若要保留当前 link prediction 标量 arm-context 形式：建议把方法明确标注为 `epsilon-greedy neural greedy adaptation`，不要声称与官方 TensorFlow Neural Greedy 完全一致。
- 最小机制修正优先级：先处理 epsilon vs pure greedy、initial round-robin、optimizer/training loop；网络宽度和具体学习率值可作为次级超参问题。
# Audit Report: NeuralGreedy

