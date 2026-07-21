# 当前基线阻塞问题与 Test C 计划

本文档记录在完整运行 `PRB`、`LocPRB` 和 `dyn_locPRB` 前必须解决的问题，以及随后 Test C 的实验目的和设计。本文档只定义后续工作，不代表这些修改已经实现。

## 当前基线

后续工程基线统一从 `src/main.py` 启动，包含三个入口：

- `PRB`：power iteration，使用 EE-Net 产生的原始 personalization/source vector。
- `LocPRB`：scratch APPR，使用相同的原始 source vector。
- `dyn_locPRB`：DYN-APPR，使用与 `LocPRB` 相同的原始 source vector。

`main_B.py` 不再作为正式入口。A1-A9 和 B1-B5 只保留在 TestA/TestB 历史实验中；A10 对应的 DYN-APPR 保留为独立方法入口。

## 完整实验前的阻塞问题

### P0：在线流与测试集不独立

部分 loader 的 `testing_dataset()` 和在线 `step()` 共用全局 NumPy RNG。当前 RNG 状态保存与恢复方式可能让固定测试样本和在线流前若干样本重合，AmazonFashion 尤其需要检查。

完成条件：测试流、validation 流和在线训练流使用独立 seed/RNG，自动检查三者无样本重叠。

### P0：不同方法没有读取完全相同的数据流

loader、网络初始化、测试集构造和负采样可能共同消耗全局随机数。即使 run seed 相同，不同算法的运行路径也可能改变后续候选集合。

完成条件：按 dataset 和 seed 预生成在线事件，至少保存 round、serving node、正边、负边、candidate 顺序、reward、context 标识和图版本。所有方法只读同一份事件流。

### P0：历史 PRB 不能作为最终公平对照

历史 PRB 与当前 LocPRB 可能使用不同代码、随机流、测试协议和计时方式；AmazonFashion 的历史 regret 与 Accuracy-Time 还可能来自不同批次。

完成条件：在同一代码版本、事件流、seed、worker 设置和机器环境中重新运行 PRB。

### P0：时间口径不完整

当前主结果排除了测试集构建和周期 evaluation，但历史结果未必采用相同口径。Vessel 的 evaluation 会重复执行大量 scratch APPR，可能占绝大多数墙钟时间。

完成条件：每个 run 同时保存 Online Time 和 End-to-End Time，并单独记录 PPR、训练、数据加载、图更新和 evaluation 时间。

### P1：DYN 的动态更新覆盖不足

当初始图已经包含在线正边时，实验主要测到 source reuse，不一定触发真正的边插入和 `INSERTUPDATE`。

完成条件：保存 source update、实际新边、`INSERTUPDATE`、fallback/reset、push 次数、PPR 时间，以及 DYN 与 scratch APPR 的误差和决策分歧次数。

### P1：Vessel evaluation 成本不可控

当前每 50 轮对 100 个固定样本逐个计算 APPR。该流程不适合直接执行完整 Vessel 矩阵。

完成条件：在不改变 Accuracy-Time 定义的前提下确定评测频率、样本数和是否使用独立离线评测，并先估算单个 run 的总成本。

## Test C 的目的

TestA 衡量单个实现组件，TestB 衡量经验组合，Test C 用于确认结果在论文定义的数据协议、动态图过程和指标下是否仍然成立。Test C 不以继续搜索最好组合为主要目标。

## Test C 实验定义

| 实验 | 相对前一版本的修改 | 主要回答的问题 |
|---|---|---|
| C0 | 当前基线：`LocPRB` 与 `dyn_locPRB` | 修复阻塞问题后的工程基线是多少 |
| C1 | 每轮所有 candidate edge 共享同一个 serving node | 当前 arm 构造是否符合论文的局部 link prediction 定义 |
| C2 | 图从冷启动/明确 warm start 开始，并只按在线事件插边 | 方法是否真的在动态增长图上工作 |
| C3 | 负采样只使用当前时刻可见信息 | 完整图或未来边信息是否造成泄漏 |
| C4 | 使用独立 validation/test split | Accuracy 是否能衡量未见数据上的泛化 |
| C5 | Vessel 改为论文规定的 uniform negatives | hard negatives 是否改变了论文结论 |
| C6 | 同时报告 mistake regret 和 propagated pseudo-regret | 当前 regret 与论文理论指标是否一致 |

累计版本记为 `CP1` 至 `CP6`：`CPk` 包含 C1 到 Ck 的全部修改。最终重点比较 C0 与 CP6，单项 C1-C6 用于归因。

## C 组开始前必须决定的协议

1. 图初始化采用空图+self-loop、明确 warm start，还是为零度节点定义单独转移规则。若论文继续声称冷启动，建议空图+self-loop，并明确 self-loop 不计入 link edge。
2. 时间主表使用 Online Time 还是 End-to-End Time。建议两者都报告，Online Time 作为算法效率主指标。
3. regret 主表使用 mistake regret 还是论文 pseudo-regret。建议两者都保存，论文理论对照使用 pseudo-regret。
4. validation seeds 与最终 test seeds 必须互斥；TestA/TestB 使用过的 seeds 不再用于最终显著性结论。

## 建议执行顺序

1. 修复全部 P0 问题并增加自动一致性检查。
2. 每个数据集执行 `T=100, runs=1` 的 PRB/LocPRB/DYN smoke test。
3. 用 `T=1000` 逐项运行 C1-C6，检查方向、成本和实现正确性。
4. 只对 C0 和 CP6 执行完整 rounds、10 runs 的主实验。
5. 使用 paired per-seed delta、置信区间和决策分歧统计整理结果。
6. 根据 C0 与 CP6 的比较决定修改代码实现还是修改论文算法与实验描述。
