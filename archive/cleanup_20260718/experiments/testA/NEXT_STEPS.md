# Workflow After the A0-A10 Ablations

本文档说明 A0-A10 完成后应如何检查、分析和使用结果，以及在修改正式代码和论文前仍需做出的决定。

## 1. 不要立即把最优项合并进正式代码

A0-A10 是单因素实验：A1-A10 都只相对 A0 改变一项。单项有效不代表多个修改叠加后仍然有效，原因包括：

- 网络结构与初始化可能产生交互；
- context normalization 和 source normalization 可能共同改变分数尺度；
- exploration decay、target shift 和辅助样本共同决定探索强度；
- 单步 SGD 对不同网络结构的效果可能不同；
- DYN-APPR 的收益依赖 source 和图变化模式。

因此，A0-A10 的作用是筛选候选修改和理解因果方向，不是直接生成最终版本。

## 2. 第一步：确认全部实验完整

查看 master 日志：

```bash
cd /mnt/data/xinyu/Fast_bandit
cat log/testA_all_Collab_T1000_20260712_080912.log
```

正常情况下应看到 A0-A10 各自一条 `START` 和一条 `END status=0`。

检查单项日志中的异常：

```bash
rg -n "Traceback|Error|Killed|out of memory|CUDA" \
  log/testA_A*_Collab_T1000_20260712_080912.log
```

检查每项结果形状：

```bash
python testA/summarize.py
```

每项应满足：

```text
Runs = 10
T = 1000
final_results.npy shape = (10, 1000, 5)
```

如某项失败，只重跑该项，不要重跑已经完成的实验。重跑时保持 dataset、T、runs、workers、seed 和超参数不变。

## 3. 第二步：生成统一结果表

运行：

```bash
python testA/summarize.py > testA/results/summary.tsv
```

至少保留以下指标：

- 最终累计 regret 均值与标准差；
- 每个 run 的累计在线时间均值与标准差；
- regret-versus-round 曲线；
- regret-versus-time 曲线；
- 每个 seed 相对 A0 的差值。

推荐对每项计算配对差值：

\[
\Delta R_i = R_i^{A_j}-R_i^{A0},
\]

\[
\Delta T_i = T_i^{A_j}-T_i^{A0},
\]

其中 `i` 是相同 seed。使用配对差值比只比较两组均值更可靠，因为每项使用相同的 loader 随机数据流。

建议输出：

- `mean(Delta R)`；
- `std(Delta R)`；
- 10 个 seed 中 regret 改善的数量；
- `mean(Delta T)` 和相对时间百分比；
- paired bootstrap confidence interval 或 Wilcoxon signed-rank test。

不要只根据单个 seed 或均值小于 1 的差异下结论。

## 4. 第三步：画三类图

### 4.1 Regret versus rounds

判断修改是否从早期开始改善，还是只在后期产生效果。

### 4.2 Regret versus online time

判断算法是否虽然每轮 regret 较低，但达到相同效果需要更长时间。

### 4.3 相对 A0 的配对差值

为每个 A 项画 10 个 seed 的最终 regret difference。该图可以区分：

- 稳定改善；
- 只改善少数 seed；
- 均值改善但方差明显增加；
- 实际没有可重复差异。

## 5. 第四步：按修改性质做决定

### 5.1 数学正确性修改

例如 invariant-correct DYN-APPR 和 `s_t-s_{t-1}` residual 更新。

决策规则：

- 正确性不能仅根据 regret 决定；
- 如果正确实现更慢，应优化实现或如实报告，不应退回数学错误实现；
- 如果不用 DYN-APPR，可以保留 scratch APPR，但论文不能再声称实验使用了 DYN-APPR。

### 5.2 模型和优化启发式

包括 A1-A6、A9：

- exploration 网络结构；
- gradient pooling；
- initialization；
- exploration decay；
- target shift；
- 辅助样本；
- replay training 或单步 SGD。

这些可以根据实验结果选择，但论文必须描述最终实际使用的形式。不能保留启发式却继续在论文中写另一套网络或训练规则。

### 5.3 理论假设相关的归一化

包括 A7 和 A8。

需要同时考虑：

- 是否改善或保持 regret；
- 是否明显改变 APPR support 和 push work；
- 是否让理论假设与代码一致；
- 是否改变现有超参数的有效尺度。

归一化后原学习率、epsilon 和 alpha 未必仍然最优。如果 A7/A8 初次结果较差，应先小范围重新调学习率或 epsilon，再判断归一化本身是否有害。

## 6. A0-A10 各项的具体判断问题

### A1：全连接 exploration

需要决定：

- 论文是否坚持两个网络均为 MLP；
- Conv1d 是否是有意设计；
- 如果保留 Conv1d，论文是否增加其结构和 kernel size 描述。

### A2：完整梯度

需要同时检查时间和内存。完整梯度可能提高表达能力，但显著增加 exploration 输入维度。

若 regret 没有稳定改善，应保留 gradient pooling，并在论文中把它作为可扩展性设计说明。

### A3：Gaussian initialization

A3 只改变当前网络中的权重初始化，没有同时把 exploration 改成 MLP。因此它测试的是 initialization 的独立影响，不是完整理论网络。

如果后续选择 A1，还需要测试 `A1 + A3`。

### A4：移除 exploration decay

A4 只在 `t > 500` 后与 A0 不同。当前 T=1000 实验只有后半段能观察差异。

如果 A4 的差异较小，应在 T=5000 上复验，而不是直接判断 decay 无效。

### A5：移除 target shift

理论上 `+1/-1` 可以相互抵消，但神经网络训练、初始化和非线性会使两种形式不完全等价。重点检查早期训练稳定性和 exploration 输出范围。

### A6：移除额外 1.2 样本

如果 A6 regret 变差，说明辅助样本可能有效，但它必须作为正式算法的一部分写进论文并进行消融。如果 A6 不变或更好，应删除该启发式以简化方法。

### A7：Context L2 normalization

如果采用 A7，需要重新确认学习率。归一化直接改变 exploitation 输入尺度和梯度尺度。

### A8：Source L1 normalization

重点记录：

- APPR push time；
- `ppr_norm`；
- regret；
- epsilon sensitivity。

固定 epsilon 下，source 尺度变化会直接改变 active residual 数量。因此 A8 的时间变化不能只解释为实现速度变化。

### A9：每轮单步 SGD

需要比较：

- train time 是否显著下降；
- regret 是否稳定；
- 后期模型是否欠拟合；
- 是否需要重新调学习率。

如果采用 A9，论文中的训练频率描述和代码中的 periodic replay 都要同步调整。

### A10：DYN-APPR

需要检查数据流中实际发生了多少条新边插入。当前 Collab 初始图包含 train edges，而在线正样本也来自 train edges，可能导致 `INSERTUPDATE` 很少触发。

如果没有真实插边，A10 主要测量的是 source residual reuse，不能充分支持论文关于动态图插边更新的结论。

应在真正 cold/warm-start graph 或隐藏边流上另外统计：

- edge insertions；
- `INSERTUPDATE` 次数和耗时；
- push 次数；
- processed degree；
- scratch APPR 和 DYN-APPR 的时间差。

## 7. 第五步：建立组合实验 B 系列

从 A0 开始，只加入已经通过单项筛选的修改，建立累计组合：

```text
B0 = A0
B1 = B0 + selected change 1
B2 = B1 + selected change 2
...
```

建议加入顺序：

1. 数学正确性和传播算法；
2. 数据尺度相关修改 A7/A8；
3. 网络结构和初始化 A1/A2/A3；
4. exploration 规则 A4/A5/A6；
5. 训练策略 A9。

每加入一项都和上一个 B 版本做配对比较。若某项单独有效、组合后失效，应记录交互作用，而不是只保留最终数字。

至少测试以下高概率交互组合：

```text
A1 + A3
A1 + A2
A4 + A5 + A6
A7 + A8
A7 + A9
A8 + A10
```

### 7.1 B 系列的正式定义

B 系列用于回答：多个单独有效的算法修改叠加以后，是否仍然改善 regret/time。B 系列保持 Current 数据协议不变，因此可以继续与 A0 做配对比较。

建议新建：

```text
testB/
  run_combination.py
  selected_changes.yaml
  summarize.py
  results/
```

`selected_changes.yaml` 记录由 A 系列筛选出的修改及加入顺序。例如：

```yaml
baseline: A0
sequence:
  - id: B1
    add: A10
    reason: invariant-correct dynamic propagation
  - id: B2
    add: A8
    reason: source normalization
  - id: B3
    add: A1
    reason: paper exploration architecture
```

具体选哪些 A 项必须由 A0-A10 结果决定，不能在看到结果前预设“全部采用”。

### 7.2 推荐的 B 编号

| ID | 定义 | 主要比较对象 |
|---|---|---|
| B0 | A0 原始实现 | A0，一致性检查 |
| B1 | B0 + 第一个入选修改 | B0 |
| B2 | B1 + 第二个入选修改 | B1、A0 |
| B3 | B2 + 第三个入选修改 | B2、A0 |
| ... | 继续逐项累计 | 上一个 B、A0 |
| B-final | 所有最终入选的算法修改 | A0、最佳单项 A |

每个 B 版本只比前一个版本多一项。不要直接从 B0 跳到包含五项修改的 B1，否则仍然无法归因。

### 7.3 必须额外运行的交互实验

除累计链外，建议单独编号：

| ID | 组合 | 原因 |
|---|---|---|
| BX1 | A1 + A3 | 网络结构和初始化高度相关 |
| BX2 | A1 + A2 | exploration 结构和梯度维度相关 |
| BX3 | A4 + A5 + A6 | 三项共同控制 exploration 行为 |
| BX4 | A7 + A8 | context/source 尺度共同变化 |
| BX5 | A7 + A9 | context 尺度影响单步 SGD |
| BX6 | A8 + A10 | source 尺度直接影响 DYN push work |

如果交互组合显著优于各单项，应将整个组合作为一个候选模块加入累计链，并在论文中联合描述。

### 7.4 B 系列运行规格

第一阶段：

```text
Dataset = Collab
T = 1000
Runs = 10
Workers = 10
Seeds = 0...9
```

第二阶段只对 `B-final`、A0 和一个最强竞争版本运行：

```text
T = 5000
Runs = 10
```

所有 B 实验继续使用与 A 系列相同的 loader RNG 隔离机制。输出目录和日志必须包含完整组合签名或 B 编号，避免无法判断某个结果包含哪些修改。

### 7.5 B 系列的接受标准

一个修改进入 `B-final`，至少满足以下之一：

- 属于数学正确性要求；
- regret 有稳定配对改善，且时间代价可接受；
- time 有明显改善，且 regret 没有实质退化；
- 让代码与论文理论假设一致，重新调参后性能不退化。

建议提前定义“实质退化”阈值。例如：

```text
Regret degradation tolerance: <= 1% of A0 final regret
Time degradation tolerance: <= 5%, unless required for correctness
```

阈值应在查看 B 结果前确定，避免事后选择标准。

## 8. 第六步：重新调参

单因素筛选阶段使用相同超参数是为了公平归因，但最终组合改变网络结构或输入尺度后，应重新调参。

建议只在选定组合上调：

- `lr1`；
- `lr2`；
- `epsilon`；
- `alpha`；
- exploration kernel size（若保留 Conv1d）；
- hidden width；
- training frequency（若保留 replay）。

调参数据和最终报告数据必须分开，避免用同一批 seeds 既选配置又报告最终结果。

## 9. 第七步：建立 C 系列 Paper-Strict protocol

A0-A10 没有修改会改变任务定义的数据协议。算法版本确定后，再单独建立 Paper-Strict 实验，包括：

- 每轮候选共享同一个 serving node；
- 明确 cold graph、warm-start graph 或 self-loop 规则；
- 在线正边来自尚未暴露的 hidden-edge stream；
- 负采样不使用 validation/test edges；
- 使用独立官方 test split；
- Vessel 是否统一为 uniform negatives；
- 同时报告错误次数和论文 propagated pseudo-regret。

Paper-Strict 与 Current protocol 的绝对 regret 不应直接解释为算法优劣，因为任务已经改变。应分别报告两套协议，或者明确选择其中一套作为论文正式实验。

### 9.1 C 系列的目标

C 系列固定使用 `B-final` 算法，只改变数据和评测协议，用于回答：论文描述的任务协议与当前仓库协议之间，每项差异会怎样改变任务难度、regret 和运行时间。

建议新建：

```text
testC/
  run_protocol.py
  generate_streams.py
  protocol_configs/
  summarize.py
  results/
```

与 B 系列不同，C 系列中的部分绝对 regret 不可直接横向解释为算法变好或变差，因为候选集合和图暴露方式已经改变。

### 9.2 C 系列单因素实验

所有 C1-C6 都从 C0 单独修改一项：

| ID | 基线 | 唯一变化 |
|---|---|---|
| C0 | B-final | Current 数据和评测协议 |
| C1 | C0 | 同一轮所有候选共享 serving node |
| C2 | C0 | 使用明确的 cold/warm-start graph，不把全部 train edges 作为初始图 |
| C3 | C0 | 负采样只使用当时可见图，不查询 validation/test edges |
| C4 | C0 | 周期评测改用独立官方 validation/test split |
| C5 | C0 | 所有数据集统一使用论文声明的 uniform negatives，取消 Vessel hard negatives |
| C6 | C0 | 增加论文 propagated pseudo-regret，同时保留错误次数 |

C6 不应删除现有错误次数，而应同时输出两种指标，便于连接历史实验与理论分析。

### 9.3 C 系列累计协议实验

在单因素 C1-C6 完成后，再建立累计协议：

| ID | 定义 |
|---|---|
| CP0 | C0，即 B-final + Current protocol |
| CP1 | CP0 + shared serving node |
| CP2 | CP1 + corrected initial graph/hidden-edge stream |
| CP3 | CP2 + leakage-free negative sampling |
| CP4 | CP3 + independent official evaluation split |
| CP5 | CP4 + uniform negative policy |
| CP6 | CP5 + dual regret reporting |
| C-final | 完整 Paper-Strict protocol，通常等于 CP6 |

累计顺序优先处理会影响后续数据定义的项目：serving node、initial graph、negative sampling、evaluation，最后再增加指标。

### 9.4 C2 的初始图必须先做决定

论文同时声称 cold graph，并在 DYN 理论中假设端点插边前 degree 为正，这两者冲突。运行 C2 前必须选择并写入论文：

1. **Self-loop cold graph**：每个节点只有 self-loop，真实边为空；
2. **Warm-start graph**：只暴露一小部分 train edges，并报告比例或构造规则；
3. **Zero-degree special case**：推导并实现零度端点第一次插边规则。

推荐 warm-start graph 或 self-loop cold graph，因为二者都能定义 PPR 转移并满足正 degree 条件。选择哪一种属于研究设定，不能根据哪个 regret 更低来决定。

### 9.5 C 系列必须预生成 stream

改变 serving node 和 negative sampling 后，仅依赖 RNG 隔离已经不够。应由 `generate_streams.py` 预先生成每个协议和 seed 的在线流，保存：

```text
seed
round
serving_node
positive_edge
negative_edges
candidate_order
contexts or feature indices
reward_vector
visible_graph_version
```

同一 C 配置下的不同算法必须读取同一 stream。不同 C 配置因为任务协议不同，应使用各自明确命名的 stream 集合。

### 9.6 C 系列的比较方式

允许的比较：

- C1 与 C0：任务协议变化造成的影响；
- C2 与 C0：初始图暴露方式造成的影响；
- CPk 与 CP(k-1)：新增协议约束的边际影响；
- C-final 内不同算法：在严格协议下的算法优劣。

不应直接声称：

> C-final regret 高于 A0，所以严格论文方法更差。

因为两者面对的候选、初始图和负样本可能完全不同。正确表述应是严格协议改变了任务难度；算法优劣必须在同一个 C 配置内比较。

### 9.7 C 系列运行规格

协议调试阶段：

```text
T = 100
Runs = 2
Workers = 2
```

验证数据流、无泄漏和图更新后：

```text
T = 1000
Runs = 10
Workers = 10
```

最终只对 C-final 和必要对照运行论文 horizon 及全部数据集。

### 9.8 C-final 的必要对照

在 C-final 协议内至少运行：

- B-final + scratch APPR；
- B-final + DYN-APPR；
- 原始 EE-Net/no graph propagation；
- 论文正式列出的其他 baselines。

这样才能说明严格协议下的收益来自 DYN/local propagation，而不是数据协议变化。

## 10. 第八步：扩大实验规模

推荐顺序：

1. Collab，T=1000：单因素筛选；
2. Collab，T=5000：确认长期效果；
3. MovieLens、Facebook/GrQc：检查不同图类型；
4. PPA、Vessel：检查大图时间和内存；
5. 最终配置使用独立 seeds 重跑 10 次。

不要仅依据 Collab 一个数据集决定所有模型结构。不同修改可能对二部图、社交图和超大稀疏图产生不同影响。

## 11. 最终需要做出的决定

完成 A 系列和 B 系列后，需要明确回答：

1. 正式方法使用 scratch APPR 还是 DYN-APPR？
2. exploration 使用 MLP 还是 Conv1d？
3. 是否保留 gradient pooling？
4. 是否采用论文 Gaussian initialization？
5. 是否保留 exploration decay？
6. 是否保留 target shift？
7. 是否保留辅助 1.2 样本？
8. context 和 source 是否归一化？
9. 使用 periodic replay 还是每轮单步 SGD？
10. 正式实验使用 Current protocol 还是 Paper-Strict protocol？
11. regret 报告错误次数、propagated pseudo-regret，还是两者都报告？
12. 时间是否包含测试集构造和周期评测？

## 12. 论文和代码同步规则

最终版本确定后，逐项检查：

- Algorithm 伪代码与实际执行顺序一致；
- 网络结构、初始化和 exploration 公式一致；
- normalization 与理论假设一致；
- candidate protocol 与 loader 一致；
- initial graph 与实验描述一致；
- regret 定义与结果文件列一致；
- time 统计口径明确；
- 所有未采用的启发式从代码或论文中删除；
- 所有保留的启发式在论文中披露并提供消融。

建议把 `paper/CODE_PAPER_DIFFERENCES.md` 作为最终核对清单：只有在论文、正式代码和实验协议三者一致后，才删除对应差异项。

## 13. 推荐的总体工作流

```text
完成 A0-A10
    -> 完整性检查
    -> 配对统计和曲线
    -> 单因素分类决策
    -> 建立 B0-Bn 累计算法组合和 BX 交互实验
    -> 重新调参
    -> 建立 C0-C6 单因素协议实验
    -> 建立 CP0-CP6 累计协议并得到 C-final
    -> 多数据集和长 horizon 验证
    -> 独立 seeds 最终实验
    -> 同步修改论文与正式代码
    -> 用差异文档做最终审计
```

原则是：数学正确性由 invariant 和测试决定；模型启发式由消融决定；数据协议由研究问题决定。不要用单一 regret 数字同时替代这三类判断。
