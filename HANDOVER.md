# Fast Bandit 项目交接文档

更新时间：2026-07-23

## 0. 当前唯一任务（覆盖下文旧计划）

当前只做 T=1000 验证，不做 Test C 或最终完整实验。旧提交 `ff3291c` /
`83fd320` 使用“连续 3 次 reset 后永久锁定 scratch”的路径，导致七数据集没有
任何真实 DYN continuation；其速度结论已经作废。

当前实现位于 `src/adaptive_appr.py`：

- 每轮重新构造候选 residual 并选择 DYN/scratch，没有永久锁定。
- 候选 residual 精确包含历史 residual、source delta、有效无向插边修正。
- predictor 在只读副本上对 DYN 与 scratch 候选 residual 做完整影子传播，因此
  得到精确候选 pushes/edge-visits；不会修改 live `p/r/source`。成本为影子
  `pushes + edge_visits + branch-init`，scratch 连续数组写使用
  `RESET_WRITE_WEIGHT=0.1` 的校准系数。全部影子传播属于预测时间，正式调整后
  PPR/online time 排除它，包含预测的原始字段仍保留。
- predictor 后固定 scrub 48 MiB cache buffer，scrub 也计入预测时间；目的是避免
  被排除的影子传播把 CSR/残差留在 CPU cache 中、给随后 DYN execute 制造隐藏优势。
  正式 timing run 应将整个进程固定到一个 CPU core，避免跨 socket 迁移。
- Python 与 Numba 共享同一内核；APPR `p/r/source/queue` 状态更新在同一执行内核。
- Loc 与 adaptive DYN 复用同一种 scratch 执行路径和 caller-owned workspace。
- `adaptive_prediction_time` 从 `ppr_time`、逐轮时间和 `online_total_time` 中扣除；
  `ppr_time_including_prediction` 与 `online_wall_time_including_prediction` 保留原值。

下一步是串行运行七数据集四组 T=1000：
Numba Loc/DYN、Python Loc/DYN，并汇总 regret、整体、PPR、预测、训练、loader、
图更新和其他时间。PRB 暂用 `benchmarks/prb_option_a_reference.json`。

## 1. 当前目标

项目研究超大图在线 link prediction 中 PRB、LocPRB 和 DYN-LocPRB 的 regret、accuracy 与运行时间。当前工作路线是：

1. 固定回归原始 source 尺度后的工程基线。
2. 修复数据流、测试集和计时协议中的阻塞问题。
3. 完整运行 PRB、LocPRB、DYN-LocPRB。
4. 最后执行 Test C，判断严格论文协议对结果的影响。

论文文件：

- `paper/Local_Bandits_on_VLGs.pdf`
- `paper/0922NeurIPS-2024-pagerank-bandits-for-link-prediction-Paper-Conference.pdf`

协议问题和 Test C 的详细定义见 `paper/BLOCKERS_AND_TEST_C.md`。

## 2. 当前正式入口

统一从仓库根目录运行：

```bash
python main.py ...
```

`main.py` 是 `src.main.main()` 的薄封装。`src/main.py` 支持三个方法：

| CLI method | 实现 | Source |
|---|---|---|
| `PRB` | power iteration | EE-Net 原始输出 |
| `LocPRB` | 每轮 scratch APPR | EE-Net 原始输出 |
| `dyn_locPRB` | 跨轮维护状态的 DYN-APPR | EE-Net 原始输出 |

静态局部方法统一使用 `LocPRB`，`dyn-LocPRB` 和 `DynLocPRB` 会映射到 `dyn_locPRB`。

核心文件：

- `src/main.py`：实验主循环、方法入口、计时和结果保存。
- `src/dynamic_appr.py`：DYN-APPR、source delta、插边修正和 fallback。
- `src/ppr_solver.py`：scratch APPR 与 power iteration。
- `src/load_data.py`：数据 loader 和在线采样。
- `src/utils.py`：图管理和结果保存。
- `src/EENet.py`、`src/EENetClass.py`：EE-Net 模型。
- `src/experiment_configs.py`：数据路径和默认参数。

## 3. 已确定的实现决策

### 正式基线不使用 A8

PRB、LocPRB 和 DYN-LocPRB 均直接使用 EE-Net 输出构造的原始 personalization/source vector。正式路径不做 L1 normalization。A8 只保留在归档的 TestA 历史消融中。

### B5 不再是正式方法

`main_B.py` 已删除。A1、A3、A6 和 B5 仅作为历史消融保留在 archive 中。`src/main.py` 不再包含隐藏的 B5 variant 入口。

### DYN 的定位

DYN 是 LocPRB 的传播后端，不应被解释为新的学习模型。理想情况下，它应在容许的近似误差内保持与 scratch LocPRB 相同的决策和 regret，同时降低 PPR 时间。

当前 DYN 已实现 invariant-preserving source update、插边更新及异常情况 fallback，但现有数据流不保证频繁触发真实新边插入，因此还需要更新覆盖率和 scratch 对照验证。

## 4. 当前运行脚本

`run_locprb_matrix.sh` 按顺序运行四个数据集，每个数据集同时启动：

- `LocPRB`：5 workers，10 runs。
- `dyn_locPRB`：5 workers，10 runs。

参数如下：

| Dataset | T | epsilon | lr1 | lr2 |
|---|---:|---:|---:|---:|
| MovieLens | 10000 | 8.33e-05 | 0.0073 | 0.0004 |
| Amazon_fashion | 5000 | 1.25e-04 | 0.1 | 0.01 |
| Collab | 5000 | 4.24e-06 | 0.01 | 0.004 |
| Vessel | 5000 | 2.86e-07 | 0.01 | 0.004 |

共同参数为 `alpha=0.85`、`n_neg=9`（10 arms）、`seed=0`。

注意：该脚本目前只有 8 个 Loc/DYN 任务，不包含 PRB。它结束后调用的绘图脚本仍读取历史 PRB 目录，因此在公平协议修复前不能直接用于最终论文比较。

## 5. Results 当前状态

`results/online_link_prediction/` 只保留 7 个非测试、完整结果目录：

- MovieLens LocPRB、DYN-LocPRB：`10 x 10000 x 5`。
- AmazonFashion LocPRB、DYN-LocPRB：`10 x 5000 x 5`。
- Collab LocPRB、DYN-LocPRB、旧版局部方法控制组：`10 x 5000 x 5`。

六个 Loc/DYN 结果与当前未归一化 source 的算法设置一致，但仍存在数据流、测试集和计时协议差异，只能作为历史参考，不能作为最终公平结果。

其他保留结果：

- `results/final/`：已确认的历史论文完整结果、消融数据和图。
- `results/baselines_prb_style/20260710_184342/`：7 个数据集、6 个 baseline、10 runs、T=10000 的完整结果。
- `results/plots/`：当前为空，等待重新生成。
- `log/`：当前为空，等待新实验。

## 6. Archive

2026-07-18 清理内容位于：

```text
archive/cleanup_20260718/
```

其中包含：

- `experiments/testA/`：A0-A10 代码、说明和结果。
- `experiments/testB/`：B1-B5 代码、说明和结果。
- `results/`：B5、smoke test、未完成 Vessel、旧 baseline 和测试专用结果。
- `plots/`：历史 comparison、TestA/TestB 和调试图。
- `logs/`：历史运行日志。
- `code/src/`：旧 `main_dyn.py`、B5 组件和 adapted baseline runner。
- `MANIFEST.txt`：归档文件与大小清单。

不要删除 archive；此前 Git 中没有可用于恢复这些文件的提交。

## 7. 完整实验前必须解决的问题

### P0：测试集与在线流可能重叠

部分 loader 的 `testing_dataset()` 和 `step()` 共用全局 NumPy RNG。当前 RNG 保存/恢复方式可能导致测试样本与在线流前若干样本重合，AmazonFashion 风险最明显。

### P0：方法间数据流不严格相同

loader、网络初始化、负采样和测试构造共同消耗随机数。必须预生成按 dataset/seed 固定的在线事件，使 PRB、LocPRB 和 DYN 读取完全相同的 serving node、正负边、candidate 顺序和 reward。

### P0：历史 PRB 不是公平对照

历史 PRB 与当前代码的随机流、测试协议和计时口径不同。AmazonFashion 的历史 regret 和 Accuracy-Time 还可能来自不同批次。最终结果必须重新运行 PRB。

### P0：计时口径需要统一

应同时保存 Online Time 和 End-to-End Time，并拆分 PPR、训练、loader、图更新和 evaluation。当前代码报告的主要时间排除了测试集构建和周期 evaluation。

### P1：DYN 缺少覆盖率与等价性统计

需要记录 source update、新边、`INSERTUPDATE`、fallback/reset、push 数、PPR 时间、与 scratch PPR 的误差，以及决策分歧轮数。

### P1：Vessel evaluation 成本过高

当前每 50 轮评测 100 个样本，并对每个样本执行 scratch APPR。此前 Vessel 墙钟时间主要消耗在 evaluation，而不是 GPU、内存或磁盘阻塞。修改协议前不要直接重新启动完整 Vessel 矩阵。

## 8. Test C

Test C 的目的不是继续寻找最好组件，而是确认结果在论文的数据协议、动态图和 regret 定义下是否成立。

| 实验 | 修改 |
|---|---|
| C0 | 修复阻塞问题后的当前基线 |
| C1 | 每轮 candidate edge 共享 serving node |
| C2 | 冷启动/明确 warm start，并按在线事件插边 |
| C3 | 负采样只使用当前可见信息 |
| C4 | 独立 validation/test split |
| C5 | Vessel 使用论文 uniform negatives |
| C6 | 同时报告 mistake regret 和 propagated pseudo-regret |

累计版本 CP6 包含 C1-C6。最终重点比较 C0 和 CP6，逐项实验用于归因。

## 9. 推荐后续顺序

1. 先建立 Git 初始提交，当前 `main` 分支没有 commit，所有文件均显示为 untracked。
2. 修复 P0 数据协议问题并增加无重叠、一致事件流测试。
3. 给 DYN 增加更新覆盖率、误差和决策分歧统计。
4. 每个数据集执行 `T=100, runs=1, workers=1` smoke test。
5. 明确图初始化、时间和 regret 的最终口径。
6. 在同一协议下完整运行 PRB、LocPRB、DYN-LocPRB。
7. 运行 C1-C6 的 T=1000 消融，再完整运行 C0 与 CP6。
8. 用独立 test seeds 汇报 paired delta、置信区间、Regret-Rounds 和 Accuracy-Time。

## 10. 常用命令

LocPRB smoke test 示例：

```bash
python -u main.py \
  --graph_name MovieLens \
  --method LocPRB \
  --alpha 0.85 \
  --appr_eps 8.33e-05 \
  --T 100 \
  --lr1 0.0073 \
  --lr2 0.0004 \
  --runs 1 \
  --workers 1 \
  --seed 0
```

PRB 使用 `--power_T 50`，且不要传 `--appr_eps`：

```bash
python -u main.py \
  --graph_name MovieLens \
  --method PRB \
  --alpha 0.85 \
  --power_T 50 \
  --T 100 \
  --lr1 0.0073 \
  --lr2 0.0004 \
  --runs 1 \
  --workers 1 \
  --seed 0
```

绘制当前 LocPRB 对比图：

```bash
python -m src.plot_locprb_comparison \
  --result-root results/online_link_prediction \
  --output-dir results/plots/locprb_comparison
```

在重新运行公平 PRB 前，该绘图命令仍会混用历史 PRB，仅用于检查，不用于最终论文图。

## 11. 交接注意事项

- 不要恢复 `main_B.py` 或把 A1/A3/A6 悄悄加入正式基线。
- 不要把 DYN regret 的变化直接解释为加速；应先检查决策分歧和近似误差。
- 不要用 TestA/TestB 的 seeds 同时选择方法并报告最终显著性。
- 不要仅根据目录名判断实验完整性；检查 `config.txt` 和 `final_results.npy` 的实际 shape。
- 不要把 evaluation 时间与 online 时间混为同一口径。
- 修改数据协议后，现有历史结果不能与新结果直接作最终公平比较。

## 12. LocPRB/DYN 性能诊断与 SciPy 结论（2026-07-22）

> 2026-07-22 后续状态：当前工作树已在纯实现优化检查点之后加入 adaptive
> scratch reset。它会在动态 source pressure 不低于 fresh scratch 时重置，并复用
> workspace；这属于算法/求解策略变化，不能再按本节所述的纯 DYN 解读。短结果与
> 触发率见 `benchmarks/ADAPTIVE_DYN_T100.md`。

> 2026-07-23 更正：连续三次 reset 后锁定 scratch 的 T=1000 结果已经作废，
> 不得再作为 DYN < Loc 的证据。`benchmarks/T1000_FINAL_NUMBA_VALIDATION.md`
> 仅作为错误路径审计材料。

### 当前代码状态

- 当前性能诊断提交为 `519fd0f perf: compile and diagnose dynamic APPR`。
- `src/ppr_solver.py` 的 scratch LocPRB push 和 `src/dynamic_appr.py` 的 DYN push 都有 Numba `@njit(cache=True)` 实现，Numba 不可用时回退到 Python。
- 已加入 `--ppr_diagnostics` 和 `--ppr_diagnostic_every` 诊断参数。
- 曾实现“稀疏活动队列”DYN 优化：Vessel 有明显收益，但 Collab/PPA 仍慢于 scratch，因此已按决定回滚，当前正式代码不包含该实验性优化。

### scratch LocPRB 的含义

scratch LocPRB 就是普通 LocPRB：每一轮都从 `p=0, r=s_t` 开始重新计算 APPR。DYN-LocPRB 则保留上轮 `(p, r)`，并注入 source delta `s_t-s_(t-1)`。

DYN 不保证 push 数少于 scratch。当相邻轮的 source 支持集合重叠很低时，delta 同时包含移除旧 source 和加入新 source 的正负残差，再加上历史残差，可能比 scratch 触发更多 push。短诊断中 OGB 数据的 `Insert Updates=0`，原因是在线正边已存在于初始训练图，因此没有真正触发动态插边。

### Numba 与纯 Python 诊断

- 合成测试中，Numba DYN 内核比纯 Python DYN 快约 101–112 倍，两者输出完全一致。
- Collab，`T=100`，同进程/同 seed，两个局部方法均强制使用纯 Python：
  - scratch LocPRB PPR：56.002180 s；online total：60.511288 s。
  - DYN-LocPRB PPR：33.315273 s；online total：37.676929 s。
  - DYN 的 PPR 时间减少 40.51%，online total 减少 37.73%；两者 regret 均为 93。
- 这复现了 Test A/A10 中 DYN 快 30%–40% 的现象，但该结论是“两者都不使用 Numba”时的相对结果，不应为了保留此相对加速而关闭 Numba，因为绝对时间会大幅恶化。

### 七数据集纯 Python LocPRB/DYN 与 PRB 对照

诊断协议：`T=20, runs=1, workers=1, power_T=50`，使用真实数据和同一 seed；LocPRB/DYN 强制调用 `.py_func`；跳过不计入 online time 的周期 evaluation 传播。这是性能诊断，不是可用于论文的完整实验。

| Dataset | LocPRB PPR (s) | DYN PPR (s) | PRB PPR (s) | DYN 相对 LocPRB |
|---|---:|---:|---:|---:|
| MovieLens | 0.362889 | 0.073057 | 0.033764 | -79.9% |
| AmazonFashion | 0.250791 | 0.080782 | 0.026455 | -67.8% |
| Facebook | 0.422976 | 0.296077 | 0.039886 | -30.0% |
| GrQc | 0.282095 | 0.111795 | 0.046723 | -60.4% |
| Collab | 15.215712 | 10.067396 | 9.450419 | -33.8% |
| PPA | 36.913420 | 21.588538 | 244.236736 | -41.5% |
| Vessel | 101.406671 | 23.697892 | 88.318819 | -76.6% |

在该短诊断中，三种方法在每个数据集上的 regret 一致。纯 Python DYN 在七个数据集上都快于纯 Python scratch。PRB 在 MovieLens、AmazonFashion、Facebook、GrQc 和 Collab 上更快；DYN 在 PPA 和 Vessel 上分别比 PRB 快约 91.2% 和 73.2%。

### SciPy 稀疏矩阵加速的可行性

当前纯 Python scratch/DYN 只把 SciPy CSR 用作图存储（`indptr`/`indices`），APPR 核心仍是队列/push 循环；它们没有像 PRB 那样使用 SciPy sparse `P @ v` 执行核心传播。PRB 虽然没有 Numba，但 `P @ v` 实际在 SciPy 的编译内核中运行，因此在小/中图上仍然很快。

LocPRB 可以利用 SciPy，但不能在不改变计算方式的情况下直接把局部 push 替换为一次 `P @ v`：

1. LocPRB 是带动态阈值和活动队列的不规则局部传播，PRB 则是固定次数的全图 SpMV。
2. 全图 SciPy SpMV 需要每次扫描长度为 `n` 的向量和大量边，在 PPA/Vessel 上可能消除局部算法优势。
3. 把逐节点 Gauss–Seidel push 改成批量 Jacobi 更新会改变 push 顺序和近似路径，需重新验证 APPR 误差、决策分歧和 regret，不能默认与现有 LocPRB 完全等价。
4. 只对活动子图执行 `P[active].T @ r[active]` 理论可行，但频繁的 CSR 切片、临时稀疏矩阵和数组分配可能比 Numba 小队列 push 更慢。

### 推荐的后续实现

如果继续优化，优先实现自适应混合内核，不建议全面替换现有 Numba push：

- 活动节点/活动边很少时，使用 Numba local push。
- 活动集合足够大时，尝试 SciPy batch SpMV。
- DYN 保留上轮 `(p,r)`；source delta 过大、残差漂移或预计成本高于 scratch 时，自适应回退到 scratch LocPRB。
- 切换阈值应通过 MovieLens/Amazon/Facebook/GrQc/Collab/PPA/Vessel 的活动比例和时间曲线实测确定，不应只在单一数据集上调参。

任何 SciPy/混合内核变更都应先运行小规模合同测试，同时检查 APPR `L1/Linf` 误差、每轮推荐是否一致、regret、push 数以及排除 test/evaluation 后的 PPR 时间。
