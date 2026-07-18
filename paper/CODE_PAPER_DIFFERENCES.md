# Differences Between the Paper and the Current Code

本文档比较以下论文和当前实现：

- 论文：`paper/Local_Bandits_on_VLGs.pdf`
- 普通入口：`main.py` / `src/main.py`
- 历史动态入口：`archive/cleanup_20260718/code/src/main_dyn.py`（当前动态入口已合并到 `src/main.py`）

其中，历史 `main.py` 的 `FastPRB` 使用每轮从零开始的 APPR；已归档的 `main_dyn.py` 使用跨轮维护状态的 DYN-APPR。

## 1. 两个入口共同存在的差异

### 1.1 Exploration 网络结构不同

论文实验部分称 exploitation 和 exploration 均为隐藏宽度 100 的两层全连接 ReLU 网络。

当前代码中：

- exploitation 是两层全连接网络；
- exploration 是 `Conv1d + 全连接网络`。

相关代码：`src/EENetClass.py`。

### 1.2 Exploration 打分包含额外启发式

论文使用：

\[
s_t[v_{t,i}] = \mu_{t,i}+b_{t,i}.
\]

代码实际使用：

\[
s_t[v_{t,i}]
=\mu_{t,i}+w_t(b_{t,i}-1),
\]

其中：

\[
w_t=
\begin{cases}
1, & t\le 500,\\
1/\sqrt{t}, & t>500.
\end{cases}
\]

前 1000 轮选择错误时，代码还会给未选择候选加入目标值 `1.2` 的额外 exploration 训练样本。论文没有这些操作。

相关代码：`src/EENet.py`。

### 1.3 Exploration 目标发生平移

论文的 exploration 回归目标为：

\[
r-\mu.
\]

代码使用：

\[
(r-\mu)+1,
\]

并在打分时从 exploration 输出中减去 1。两者意图接近，但代码不是论文公式的直接实现。

### 1.4 梯度特征被聚合压缩

论文使用完整梯度：

\[
\phi(x)=\nabla_{\theta_1}f_1(x;\theta_1).
\]

代码通过 `block_reduce` 对梯度参数每 50 维取均值，再将压缩后的梯度输入 exploration 网络。

相关代码：`src/EENetClass.py`。

### 1.5 网络初始化方式不同

论文理论使用指定方差的 Gaussian initialization。代码使用 PyTorch `nn.Linear` 和 `nn.Conv1d` 的默认初始化，没有显式实现论文 Eq. (19) 的初始化。

因此，论文基于 NTK 的 regret 条件不能直接应用到当前网络实现。

### 1.6 Personalization vector 未归一化

论文假设：

\[
\|s_t\|_1=1,
\]

并说明不满足时应先执行：

\[
s_t\leftarrow s_t/\|s_t\|_1.
\]

代码直接传播 EE-Net 输出，不执行 L1 归一化，而且 `s_t` 可以包含负值。

这不会阻止线性 PPR 方程求解，但论文的 APPR locality/work bound 使用了该归一化条件。

### 1.7 Context 不满足单位范数假设

论文 regret 理论假设：

\[
\|x_{t,i}\|_2=1.
\]

代码将边两端节点特征直接拼接，没有对最终 edge context 做 L2 归一化。PPA/Vessel 的按特征维标准化也不等于单位向量归一化。

### 1.8 候选集合不总是共享 serving node

论文每轮候选边为：

\[
(v_t,v_{t,1}),\ldots,(v_t,v_{t,k}),
\]

即所有候选共享同一个 serving node `v_t`。

通用 OGB loader 的负样本是独立随机边 `(u,v)`，不同候选通常不共享起点。MovieLens、Facebook 和 GrQc 等 loader 也从全局负边集合采样，不能保证负候选与正边共享起点。

相关代码：`src/load_data.py`。

### 1.9 Vessel 使用 hard negatives

论文描述每轮包含一个正样本和九个均匀随机负样本。Vessel loader 使用 KDTree 从特征空间近邻中选择困难负样本，并非均匀随机采样。

### 1.10 OGB 初始图不是 cold graph

论文实验部分声称：

\[
G_0=(V,E_0=\varnothing).
\]

当前 Collab、PPA 和 Vessel graph manager 直接使用 OGB train edges 初始化图。与此同时，在线 loader 的正样本也来自 train edges。因此，成功选择的正边经常已经存在于初始图中，动态图不会真正插入新边。

相关代码：`src/utils.py` 和 `src/load_data.py`。

这也意味着 Collab 上的 DYN-APPR 实验主要测试了 personalization vector 的动态变化，未必实际触发 `INSERTUPDATE`。

### 1.11 OGB 负采样使用全部 split 的结构

OGB loader 构建 `adj_gt` 时合并了 train、validation 和 test edges，并用它排除负样本。

模型没有直接接收 validation/test 标签，但负采样过程使用了未来 split 的图结构信息，与严格在线协议和标准 OGB evaluation protocol 不一致。

### 1.12 固定测试集不是官方 test split

周期评测使用相同 loader 的 `step()` 随机生成 100 个样本。它通常仍使用训练正边和在线负采样机制，而不是 OGB 官方 test split。

### 1.13 实验 regret 与理论 regret 定义不同

代码记录：

\[
R_T^{\mathrm{code}}=\sum_{t=1}^T(1-r_t),
\]

即累计选择错误次数。

论文理论定义 oracle propagated score 下的 pseudo-regret：

\[
R_T^{\mathrm{paper}}
=\sum_{t=1}^T
\left(
\pi_t^\star[v_{t,i_t^\star}]
-\pi_t^\star[v_{t,\hat i_t}]
\right).
\]

两者不是同一个评价量，当前实验结果不能直接视为论文定理中的 pseudo-regret。

### 1.14 神经网络更新不是每轮单步 SGD

论文 Algorithm 1 描述根据当前反馈进行参数更新。代码定期对历史样本进行 replay 式训练，每次最多进行 2000 次 optimizer step：

- 前 2000 轮每 50 轮训练一次；
- 之后每 100 轮训练一次。

训练频率与论文实验附录接近，但具体优化过程不是 Algorithm 1 中的单步更新。

### 1.15 运行时间统计口径可能不同

代码明确从运行时间中排除了：

- 固定测试集构造时间；
- 每 50 轮执行的 100 样本评测时间。

论文表格使用 `total running time`，但没有同样明确地列出全部扣除项。因此代码和论文的时间结果未必采用完全相同的统计口径。

### 1.16 转移矩阵记号存在方向差异

论文写作：

\[
P=D^{-1}A,
\]

并将其称为 row-stochastic random-walk matrix。

代码通过列归一化构造等价于 `A D^{-1}` 的矩阵，并使用列向量方程 `P @ p`。对于无向图，这对应常见的列向量 PPR 约定，但与论文文字中的矩阵方向不完全一致。

论文正文、Algorithm 2 push 和附录证明中也混用了 row/column convention，需要统一符号后再逐项对照。

## 2. `main.py` 特有的差异

### 2.1 FastPRB 没有实现 DYN-APPR

`src/main.py` 每轮调用 `src/ppr_solver.py::appr()`。该函数每次都重新初始化：

```python
p = np.zeros(num_nodes)
r = np.zeros(num_nodes)
```

因此它只实现从零开始的 Algorithm 2 APPR，没有实现：

- 跨轮保存 `(p,r)`；
- `INSERTUPDATE`；
- `s_t-s_{t-1}` residual 更新；
- 动态图 warm start。

论文 Theorem 4.2 关于 DYN-APPR 累计工作量的结论不适用于 `main.py`。

### 2.2 FastPRB 名称与实际算法范围不同

论文将完整方法称为 LOCPRB，并在 Algorithm 1 中调用 DYN-APPR。`main.py` 的 `FastPRB` 更准确地应描述为：

> EE-Net + per-round scratch APPR

而不是完整的 LOCPRB/DYN-APPR。

### 2.3 PRB 使用有限次 power iteration

代码的 PRB 使用固定轮数 power iteration，实验通常设置为 50 次。它是 PPR 的有限迭代近似，而不是论文理论中方程的精确解。

论文实验也使用 50 次作为 PRB 对照，但理论 oracle PPR 与该有限迭代结果不同。

## 3. 已归档 `main_dyn.py` 特有的差异

### 3.1 通过 degree 差识别插入边

论文 Algorithm 3 显式接收新增边端点 `(a,b)`。历史 `main_dyn.py` 通过相邻轮 degree vector 的变化识别两个端点。

在“每轮至多插入一条无向边”的论文假设下，两种方式等价。但以下情况会被代码拒绝：

- 一轮插入多条边；
- 删除边；
- 只更新一个方向；
- degree 与邻接矩阵不同步。

### 3.2 零度端点需要额外规则

当前 `INSERTONEDIR` 要求端点插边前 degree 大于 0，否则会报错。这与论文理论附录假设一致。

但该假设和论文实验中的完全空图 `G_0=(V,\varnothing)` 冲突。第一次插边时旧 degree 为 0，`d_u/(d_u-1)` 无法使用。

论文需要明确选择以下一种设定：

- 初始图不是完全空图；
- 给每个节点加入 self-loop；
- 为零度端点定义单独的初始化更新规则。

### 3.3 周期评测不共享 DYN 状态

在线训练轮次维护 `(p,r,s,degree)`，但周期评测的 100 个测试样本分别使用 scratch APPR。

这样可以避免测试样本彼此污染动态状态，工程上合理，但不属于论文 Algorithm 3 的直接描述。

### 3.4 方法名称不同

动态入口将结果目录标记为 `DynFastPRB`，用于避免与 `main.py` 的 `FastPRB` 在同一分钟启动时写入相同目录。

论文方法名称为 LOCPRB。`DynFastPRB` 是仓库中的结果管理名称，而不是论文正式名称。

## 4. 理论结论的适用范围

### 4.1 可以对应的部分

历史 `main_dyn.py` 的传播状态满足：

\[
(I-\alpha P)p+(1-\alpha)r=(1-\alpha)s,
\]

并在每轮 push 后满足：

\[
\max_u\frac{|r(u)|}{d_u}<\epsilon.
\]

因此，在 degree 正、每轮至多一条无向插边、矩阵 convention 一致的前提下，可以对应论文的 APPR residual certificate 和 degree-normalized approximation guarantee。

### 4.2 不能直接对应的部分

以下原因使当前完整代码不能直接宣称满足论文 Theorem 4.4 的 neural-bandit regret bound：

- 网络结构不同；
- 初始化不同；
- context 没有单位范数；
- `s_t` 没有 L1 归一化；
- exploration bonus 使用额外启发式；
- 实验 regret 定义不同；
- 候选集合不满足共享 serving node 的协议。

论文 Theorem 4.2 的累计动态工作量结论还依赖：

- 正确的端点插入分布假设；
- 每轮至多插入一条边；
- 合适的 `s_t` 变化成本分析；
- `\|s_t\|_1=1`；
- 正 degree 假设。

当前 Collab 数据流中正边通常已位于初始 train graph，因此真实动态图插入行为与该理论设定不同。

## 5. 建议修复优先级

若目标是严格复现论文，应优先处理：

1. 统一并修正文中 DYN-APPR 公式、矩阵方向和 `s_t-s_{t-1}` 更新。
2. 让每轮候选边共享同一个 serving node。
3. 明确 cold graph、self-loop 和零度节点处理规则。
4. 避免使用 validation/test edges 辅助在线负采样。
5. 使用独立的官方 test split 进行评测。
6. 统一实验 regret 与论文理论 regret 的定义和名称。
7. 对 edge context 和 `s_t` 执行论文要求的归一化。
8. 将 exploration 网络、梯度输入和初始化改为论文描述的形式。
9. 去除或在论文中明确说明 exploration decay、目标平移和额外样本启发式。
10. 明确运行时间是否包含数据加载、测试集构造和周期评测。

## 6. 当前实现的准确描述

`main.py`：

> EE-Net-style contextual scoring with per-round scratch APPR.

历史 `main_dyn.py`：

> EE-Net-style contextual scoring with invariant-preserving stateful DYN-APPR, including endpoint insertion correction and sparse personalization updates.

两者都实现了“候选 contextual score + 图传播 + 在线反馈更新”的核心思想，但当前完整实验协议和神经网络实现仍不能视为论文所有公式与理论假设的逐项复现。
