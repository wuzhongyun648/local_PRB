# Fast Bandit Milestones

更新时间：2026-07-23

## 2026-07-23 状态更正

提交 `ff3291c` / `83fd320` 的所谓最终 T=1000 结果不能作为 Milestone 3/4
完成证据：当时 DYN 在连续 3 次 reset 后永久锁定为 scratch，七个数据集均只有
冷启动及 reset、没有真实 DYN continuation；996/1000 轮只是复用 workspace 的
scratch 快路径。该结果证明的是 scratch 实现路径差异，不是 DYN 加速。

当前重新打开 Milestone 3/4，要求：

- 删除永久锁定，每轮都基于候选 residual 重新判断 DYN 或 scratch。
- 候选 residual 包含历史 residual、source delta 和有效单向/双向插边修正；
  predictor 在只读副本上完整影子传播两分支，得到精确候选 pushes/edge-visits。
- Python/Numba 共享同一预测与执行内核；Numba 路径整体编译核心数值判断及
  APPR 状态更新。
- DYN 预测判断单独计时，并从 PPR time、step time、online total time 中扣除；
  同时保留包含预测时间的原始对照字段。
- 两个方法在 execute 前都运行同一个双候选只读 predictor 并 scrub 48 MiB：
  DYN 使用预测分支，Loc 强制 scratch；分别记为 prediction/control 并从 adjusted
  时间扣除，以对齐 cache 与 CPU frequency；性能矩阵固定单 CPU core。
- Loc 与自适应 DYN 使用同一可复用 `p/r/queue/queued` scratch 执行内核。
- 重新运行七数据集的 Numba 与 Python、Loc 与 DYN，均为 T=1000。

## 当前阶段目标

本阶段只推进到 `T=1000` 验证，不运行最终完整实验。

最终希望在 PRB、LocPRB、DYN-LocPRB 的 regret 基本相当时实现：

```text
Numba-DYN-LocPRB < Numba-LocPRB < PRB
```

其中 `<` 表示在统一实验口径下运行时间更短。优先比较 PPR propagation time，
同时报告 online total time。若 Numba 路线经过诊断和优化仍不能满足目标，才评估：

```text
Python-DYN-LocPRB < Python-LocPRB < PRB
```

五个运行版本只是中间诊断手段。最终代码和正式结果只保留 PRB、LocPRB、
DYN-LocPRB 三个正式版本，并优先采用 Numba 后端。

## 本阶段不做的工作

- 不执行 Test C。
- 不运行最终完整 rounds、10 runs 的正式实验。
- 不反复重跑耗时较长的 PRB；本阶段暂时复用已有 PRB 数据和时间。
- 不以牺牲明显 regret 为代价换取速度排序。

## 版本与提交规则

1. 开始本阶段前先提交当前代码，建立可回退基线。
2. 每完成一个 milestone，立即创建独立 Git commit。
3. 每个完成记录必须写明 commit、验证命令、结果位置和主要结论。
4. 如果后续实现失败或结果恶化，最坏情况下回退到阶段基线。
5. 不擅自修改或清理无关的用户文件及 EE-Net 子模块状态。

阶段基线：

| 记录 | Commit | 说明 |
|---|---|---|
| Baseline | `92e466f` | 开始里程碑工作前的可回退版本 |

## 运行资源规则

- 性能实验运行前后检查 `htop`/进程状态和 `nvidia-smi`。
- 避免 CPU oversubscription、GPU 争用和多个性能任务互相干扰。
- 固定并记录 CPU 线程数、GPU、workers、runs 和 Python 环境。
- Numba 首次编译必须预热，编译时间不计入 PPR time。
- 发现资源拥塞时不采纳该次计时，待资源恢复后重跑。

## Milestone 0：选定暂用 PRB 参照

目标：盘点已有 `results/` 和 `archive/` 中的 PRB，暂定本阶段的七数据集
PRB regret 与时间参照，避免反复运行 PRB。

完成条件：

- 列出每个候选结果的路径、代码/协议时期、shape、最终 regret 和时间口径。
- 标记不完整、混合批次或无法确认口径的结果。
- 由项目负责人选定本阶段暂用的 PRB 参照版本。
- 将选择和限制写入本文件并提交。

选择：方案 A。

- MovieLens、AmazonFashion、Facebook、GrQc、Collab、Vessel 使用
  `results/online_link_prediction/` 中 2026-07-20 同批完整 PRB 结果的前
  1000 轮。
- PPA 使用同批 `log/dynamic_queue_20260719_162701/PPA_PRB.log` 中约
  1000 轮的 partial reference。
- 本阶段不重跑 PRB；最终完整实验仍需在统一、无资源争用的环境重跑。
- 结构化参照见 `benchmarks/prb_option_a_reference.json`。

状态：完成。

## Milestone 1：统一五版本诊断口径

目标：让同一正式入口可复现地运行五个中间版本：

1. PRB；
2. Numba-LocPRB；
3. Numba-DYN-LocPRB；
4. Python-LocPRB；
5. Python-DYN-LocPRB。

完成条件：

- 显式选择 PPR backend，不依赖环境中是否恰好安装 Numba。
- Python/Numba 后端共享同一算法逻辑和输入。
- 保存 PPR、训练、图更新、loader/other、evaluation、online total 和端到端时间。
- 保存 regret、pushes、edge visits、active nodes、source updates、insert updates 和 fallback。
- 保存完整配置、代码 commit 和运行环境信息。
- Numba warm-up 明确排除在计时之外。
- 增加并通过必要的单元/合同测试。

实现结果：

- 正式入口增加显式 `--ppr_backend {numba,python}`；PRB 固定为 SciPy。
- Python/Numba 通过同一 Numba dispatcher 及其 `.py_func` 选择，不复制算法。
- Numba 使用真实 CSR/degree dtype 在正式计时前逐 worker 预热。
- 保留旧 `final_results.npy` 五列，新增 `worker_*_metrics.json` 和
  `metrics_summary.json`。
- 新增 PPR、训练、图更新、loader、predict/source、decision、other、evaluation、
  setup、warmup 和 worker wall time，以及 scratch/DYN 工作量统计。
- 结果目录使用微秒级唯一时间戳；checkpoint 使用 worker-specific 文件名。
- 记录 Git dirty 状态、Python/NumPy/SciPy/Numba/Torch、CPU affinity 和 GPU 信息。

验证：

- `.venv-dyn-diagnosis/bin/python -m unittest discover -s tests -v`：7/7 通过。
- 无 Numba 的默认 Python 环境：3 个通用测试通过，4 个 Numba 合同测试按预期跳过。
- MovieLens `T=1` 的 Python/Numba Loc/DYN 四路真实入口 smoke test 均完成；同算法
  backend 的 regret、loss 和 PPR norm 完全一致，新旧结果与 JSON 均成功保存。

状态：完成。实现 commit：`e5ff896`。

## Milestone 2：七数据集短诊断

目标：在七个数据集上运行五个中间版本的短实验，定位速度排序失败的原因。

建议规模：`T=20~100, runs=1, workers=1`。

完成条件：

- 覆盖 MovieLens、AmazonFashion、Facebook、GrQc、Collab、PPA、Vessel。
- 同 dataset/seed 下使用一致配置。
- 输出五版本 regret、PPR time、online total、pushes 和 edge visits 对比表。
- 分别解释 Numba-DYN vs Numba-LocPRB、Numba-LocPRB vs PRB 的瓶颈。
- 运行期间无明显 CPU/GPU 资源争用。

结果：

- 完成七数据集、四个 Loc/DYN backend 的 `T=20` 串行矩阵，全部任务成功。
- 完成七数据集 Numba Loc/DYN 的 `T=100` timing run 和逐轮 scratch 对照。
- Python/Numba 同算法 regret 一致；T=20 Loc/DYN regret 在七数据集全部一致。
- T=100 时 Numba-DYN 仅 Vessel 快于 Numba-Loc；其余数据集慢 1.74x-1.98x。
- DYN push/edge 工作量普遍是 scratch 的 1.2x-1.94x；相邻 source 低重叠是主因。
- PPA 有 44/100 次候选决策分歧，regret 为 Loc 95、DYN 86，需要在优化中重点守护。
- 完整报告见 `benchmarks/BACKEND_DIAGNOSIS_T20_T100.md`。

状态：完成。诊断 commit：`62cd631`。

## Milestone 3：优先优化 Numba 路线

目标：在不明显改变 regret 的情况下，优先实现：

```text
Numba-DYN-LocPRB < Numba-LocPRB < PRB
```

可能的优化必须由 Milestone 2 的数据驱动，包括但不限于 active queue、全节点扫描、
source delta、历史 residual、数组分配、push/edge visits 和自适应 scratch fallback。

完成条件：

- 每项优化有优化前后相同输入的基准。
- Python/Numba 同算法输出合同继续通过。
- 记录 LocPRB/DYN 的 PPR 误差、决策分歧和 regret 变化。
- 七数据集短实验显示目标排序，或明确列出仍未满足的数据集与原因。

阶段性结果（仅实现优化，尚未改变求解策略）：

- scratch 与 DYN 均使用稀疏 source seed，避免每轮扫描完整 source/残差向量。
- DYN 使用已知插边端点验证 degree 变化，只更新端点 degree，并移除每轮完整
  `degree.astype(float64)` 与返回向量复制。
- Python/Numba 合同测试继续通过（Numba 环境 9/9；无 Numba 环境 5 通过、4 跳过）。
- `T=20`、seed 0、单进程下，七数据集 Loc/DYN regret 均完全一致；相较优化前，
  DYN PPR 时间均下降。当前 MovieLens 与 Vessel 已达到 DYN < Loc，其余五个仍未达到。
- 该结果说明实现开销已显著降低，但低 source overlap 导致的额外 push 仍是主要瓶颈；
  下一步进入可明确汇报的自适应 scratch reset。此项会改变 DYN 的求解策略，必须单独
  记录并验证输出、regret 与触发次数。
- 已实现基于 degree/epsilon 归一化 source pressure 的自适应 reset，并复用 scratch
  workspace。该版本是 adaptive hybrid，属于求解策略变化，不再等同于纯 DYN。
- `T=100` 七数据集 PPR 总时间比 LocPRB 低约 10.9%，但逐数据集仅 PPA、Vessel 达到
  DYN < Loc；其余五个仍未达到。除 PPA 外 regret/决策完全一致；PPA regret 相差 1，
  有 10/100 次决策分歧。详见 `benchmarks/ADAPTIVE_DYN_T100.md`。

此前“七数据集均达到”的结果已因永久 scratch 锁定而作废，报告
`benchmarks/T1000_FINAL_NUMBA_VALIDATION.md` 仅保留为错误路径审计材料。

最终结果：

- 真实可逆 adaptive DYN 已完成；每轮都重新判断 DYN/scratch，不存在永久锁定。
- predictor 精确覆盖 source delta、有效插边修正、候选传播和分支初始化；判断及状态
  更新进入 Numba 编译路径。
- Loc 运行等价 predictor control，并与 DYN 使用相同 cache scrub；prediction/control
  均从 adjusted 时间中扣除并单独报告。
- 七数据集 `T=1000` 的 Numba Loc/DYN regret 全部一致；DYN 分支合计 3027 轮，
  scratch 分支合计 3973 轮。
- Numba adjusted PPR 合计 Loc `34.427s`、DYN `25.307s`，DYN 快 `26.49%`。
  逐数据集 5/7 达标；Facebook 和 Grqc 分别慢 `1.91%`、`1.32%`。

状态：完成。最终实现 commit：`45099f4`。

## Milestone 4：`T=1000` 验证

目标：验证短实验中的速度关系能否持续到中等长度在线运行。

建议配置：`T=1000`、paired seeds；具体 runs 数在短实验成本确认后确定。

完成条件：

- 七数据集完成选定三版本的 `T=1000` 运行。
- PRB 原则上复用 Milestone 0 选定的数据；只有缺失或不可比时才单独决定是否补跑。
- 报告最终 regret、regret 曲线、PPR time、online total 和关键诊断统计。
- 判断是否满足 `Numba-DYN < Numba-LocPRB < PRB`。
- 若 Numba 路线未满足，给出是否进入 Python 备选路线的明确结论。
- 本阶段到此结束，不自动启动最终完整实验。

旧结果（已作废）：

- 完成七数据集、seed 0、Numba Loc/DYN 的 T=1000 串行配对验证。
- 七数据集均满足 DYN < Loc 的隔离 PPR 时间不等式；合计快 41.1%。
- Loc/DYN regret 七数据集完全一致；700 个诊断决策比较无分歧。
- 方案 A 的六个可用 PRB PPR 折算和 PPA online time 均远慢于 Loc，支持 interim
  `Loc < PRB`；但 PRB 尚不是统一 paired rerun，PPA regret 尤其只能标记为 partial。
- 选定 Numba 路线进入最终三版本；不启动 Test C 或完整论文实验。

新结果：

- Numba 和纯 Python 的七数据集 Loc/DYN 矩阵均完成。
- Numba adjusted PPR 合计达到 DYN < Loc；adjusted online total 也达到
  DYN `610.099s` < Loc `619.990s`。
- 纯 Python adjusted PPR 为 DYN `666.492s` > Loc `544.297s`，明确拒绝 Python
  备选路线。
- 方案 A 暂用参照支持 `Numba-Loc < PRB` 的速度关系，但 PRB 并非统一 paired
  rerun，不能据此声称最终三方法 regret 已公平对齐。
- 完整逐数据集与分项计时见 `benchmarks/T1000_REAL_ADAPTIVE_VALIDATION.md`。

状态：完成；选择 Numba 路线。本阶段停止，不启动 Test C 或最终完整实验。

## Milestone 完成记录

| Milestone | 状态 | Commit | 验证/结果 | 结论 |
|---|---|---|---|---|
| 0 PRB 参照 | 完成 | 待本次提交 | `benchmarks/prb_option_a_reference.json` | 采用方案 A；PPA 标记为 partial |
| 1 统一口径 | 完成 | `e5ff896` | 7 tests + MovieLens T=1 四路 smoke | 显式后端、预热、结构化计时已完成 |
| 2 短诊断 | 完成 | `62cd631` | `benchmarks/BACKEND_DIAGNOSIS_T20_T100.md` | 已定位 source delta 导致的额外 push |
| 3 真实 adaptive DYN | 完成 | `45099f4` | 19 tests + Numba/Python 正式入口 | 逐轮可逆判断，精确候选成本 |
| 4 T=1000 | 完成 | 本结果提交 | `benchmarks/T1000_REAL_ADAPTIVE_VALIDATION.md` | 选 Numba；聚合 DYN < Loc < PRB（PRB interim） |
