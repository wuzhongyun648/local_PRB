# Fast Bandit Milestones

更新时间：2026-07-22

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

状态：完成，等待本次提交记录 commit。

## Milestone 2：七数据集短诊断

目标：在七个数据集上运行五个中间版本的短实验，定位速度排序失败的原因。

建议规模：`T=20~100, runs=1, workers=1`。

完成条件：

- 覆盖 MovieLens、AmazonFashion、Facebook、GrQc、Collab、PPA、Vessel。
- 同 dataset/seed 下使用一致配置。
- 输出五版本 regret、PPR time、online total、pushes 和 edge visits 对比表。
- 分别解释 Numba-DYN vs Numba-LocPRB、Numba-LocPRB vs PRB 的瓶颈。
- 运行期间无明显 CPU/GPU 资源争用。

状态：未开始。

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

状态：未开始。

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

状态：未开始。

## Milestone 完成记录

| Milestone | 状态 | Commit | 验证/结果 | 结论 |
|---|---|---|---|---|
| 0 PRB 参照 | 完成 | 待本次提交 | `benchmarks/prb_option_a_reference.json` | 采用方案 A；PPA 标记为 partial |
| 1 统一口径 | 完成 | 待本次提交 | 7 tests + MovieLens T=1 四路 smoke | 显式后端、预热、结构化计时已完成 |
| 2 短诊断 | 未开始 | 待定 | 待定 | 待定 |
| 3 Numba 优化 | 未开始 | 待定 | 待定 | 待定 |
| 4 T=1000 | 未开始 | 待定 | 待定 | 待定 |
