# Fast Bandit 项目交接文档

更新时间：2026-07-18

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
