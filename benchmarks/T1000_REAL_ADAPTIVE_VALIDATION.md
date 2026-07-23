# T=1000 真实自适应 DYN 验证

日期：2026-07-23

## 结论

在七个数据集、单核、单 worker、seed 0、`T=1000` 的统一实验中：

- Numba-DYN 与 Numba-Loc 的七组 regret 全部完全一致。
- Numba 的 adjusted PPR 总时间为 DYN `25.307s`、Loc `34.427s`，
  因而 DYN 比 Loc 快 `26.49%`，聚合目标成立。
- 逐数据集有 5/7 严格满足 DYN < Loc；Facebook 慢 `1.91%`，Grqc 慢
  `1.32%`，两者属于接近持平但仍应如实记为未满足。
- 纯 Python 的 adjusted PPR 总时间为 DYN `666.492s`、Loc `544.297s`，
  DYN 慢 `22.45%`，因此拒绝 Python 路线。
- 方案 A 的暂用 PRB 时间支持 `Numba-Loc < PRB`；但它不是统一 paired rerun，
  且部分 regret 与本轮差异明显，因此只能支持本阶段的速度参照，不能作为最终论文
  的公平三方法结论。

本阶段选择 Numba 路线作为最终三版本候选。最终完整实验仍需统一重跑 PRB。

## 实验口径

- `T=1000`，`runs=1`，`workers=1`，seed 0。
- 固定到单个 CPU core；BLAS 线程数固定为 1；隐藏 GPU。
- Numba 正式计时前完成预热，编译时间不计入 PPR。
- Loc 和 DYN 每轮都执行相同的双候选 predictor 和 48 MiB cache scrub。
- DYN 使用 predictor 的选择，Loc 强制 scratch；两者均无永久锁定。
- predictor 精确构造候选 residual，包含 source delta、有效插边修正，并影子执行
  DYN 与 scratch 候选传播。
- 按任务要求，prediction/control 时间从 adjusted PPR、step 和 online total 中剔除，
  同时单独保存；inclusive 字段保留原始包含判断开销的时间。
- 运行日志同时记录主机负载。沙箱内 `nvidia-smi` 不可用；运行期间无其他实验并发，
  未观察到 CPU oversubscription。

Numba 日志：
`log/backend_matrix_T1000_matched_control_numba_T1000_20260723/`

Python 日志：

- `log/backend_matrix_T1000_pure_python_small4_T1000_20260723/`
- `log/backend_matrix_T1000_pure_python_collab_T1000_20260723/`
- `log/backend_matrix_T1000_pure_python_ppa_T1000_20260723/`
- `log/backend_matrix_T1000_pure_python_vessel_T1000_20260723/`

## Numba 结果

Adjusted PPR 不包含 prediction/control；inclusive PPR 包含它。

| Dataset | Regret Loc/DYN | Loc PPR (s) | DYN PPR (s) | DYN/Loc | Loc control (s) | DYN prediction (s) | Inclusive Loc/DYN (s) | Branch DYN/scratch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MovieLens | 313 / 313 | 0.1583 | 0.1348 | 0.8517 | 4.9462 | 4.9584 | 5.1045 / 5.0932 | 428 / 572 |
| AmazonFashion | 298 / 298 | 0.1158 | 0.1141 | 0.9856 | 4.6473 | 4.6523 | 4.7631 / 4.7664 | 295 / 705 |
| Facebook | 824 / 824 | 0.1092 | 0.1113 | 1.0191 | 4.3417 | 4.3742 | 4.4509 / 4.4855 | 160 / 840 |
| Grqc | 842 / 842 | 0.1194 | 0.1210 | 1.0132 | 4.2220 | 4.2279 | 4.3414 / 4.3489 | 191 / 809 |
| Collab | 785 / 785 | 3.4355 | 3.3719 | 0.9815 | 10.9365 | 11.0277 | 14.3720 / 14.3996 | 6 / 994 |
| PPA | 921 / 921 | 2.9712 | 2.0806 | 0.7003 | 6.2179 | 6.0695 | 9.1891 / 8.1501 | 948 / 52 |
| Vessel | 879 / 879 | 27.5176 | 19.3732 | 0.7040 | 24.1586 | 21.9616 | 51.6762 / 41.3348 | 999 / 1 |
| **总计** | — | **34.4269** | **25.3068** | **0.7351** | **59.4702** | **57.2717** | **93.8971 / 82.5785** | **3027 / 3973** |

Adjusted online total 为 Loc `619.990s`、DYN `610.099s`，DYN 快 `1.60%`。
即使把 prediction/control 加回 PPR，聚合 inclusive PPR 仍是 DYN 更快。

### Numba 整体分项

| 分项 | Loc (s) | DYN (s) | 口径 |
|---|---:|---:|---|
| Adjusted PPR | 34.4269 | 25.3068 | 排除 prediction/control |
| Prediction/control | 59.4702 | 57.2717 | 单独报告 |
| Inclusive PPR | 93.8971 | 82.5785 | 上述两项之和 |
| Train | 264.7784 | 264.3721 | 在线训练 |
| Loader | 65.6363 | 65.5183 | 数据加载/采样 |
| Predict/source | 23.3210 | 23.2552 | 网络预测与 source 构造 |
| Decision | 0.2185 | 0.2123 | 选臂 |
| Graph update | 223.9757 | 223.2680 | 图更新 |
| Other | 7.6012 | 8.1346 | 未归入以上项目 |
| Adjusted online total | 619.9900 | 610.0989 | 排除 prediction/control |

## 纯 Python 结果

Python resolver 已确保顶层和嵌套影子传播 helper 都走 `.py_func`，因此这不是
“Python execute + Numba predictor”的混合结果。

| Dataset | Regret Loc/DYN | Loc PPR (s) | DYN PPR (s) | DYN/Loc | Loc control (s) | DYN prediction (s) | Branch DYN/scratch |
|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens | 313 / 313 | 0.2215 | 0.2311 | 1.0433 | 4.4637 | 4.5135 | 428 / 572 |
| AmazonFashion | 298 / 298 | 6.9380 | 7.2325 | 1.0425 | 15.4242 | 15.4206 | 295 / 705 |
| Facebook | 824 / 824 | 7.4081 | 7.4414 | 1.0045 | 21.4407 | 21.2119 | 160 / 840 |
| Grqc | 842 / 842 | 4.6999 | 5.0750 | 1.0798 | 15.7881 | 16.0712 | 191 / 809 |
| Collab | 785 / 785 | 296.1027 | 300.3622 | 1.0144 | 797.1595 | 808.2309 | 6 / 994 |
| PPA | 921 / 921 | 53.4130 | 51.0217 | 0.9552 | 143.1532 | 139.4889 | 948 / 52 |
| Vessel | 879 / 879 | 175.5139 | 295.1274 | 1.6815 | 401.8237 | 403.4437 | 999 / 1 |
| **总计** | — | **544.2971** | **666.4915** | **1.2245** | **1399.2531** | **1408.3807** | **3027 / 3973** |

### Python 整体分项

| 分项 | Loc (s) | DYN (s) |
|---|---:|---:|
| Adjusted PPR | 544.2971 | 666.4915 |
| Prediction/control | 1399.2531 | 1408.3807 |
| Train | 265.0416 | 265.1507 |
| Loader | 65.2241 | 65.0160 |
| Predict/source | 27.2022 | 27.1160 |
| Decision | 0.2320 | 0.2241 |
| Graph update | 260.9841 | 258.4241 |
| Other | 10.1885 | 10.5653 |
| Adjusted online total | 1173.2028 | 1293.0220 |

## PRB 方案 A 暂用参照

PRB 不在本轮重跑。下表按已有完整实验的平均时间线性折算到 1000 轮；PPA 只有
约 1000 轮 partial online time，缺少可比 PPR。

| Dataset | PRB regret reference | PRB PPR T=1000 估算 (s) | Numba-Loc PPR (s) | 暂用速度判断 |
|---|---:|---:|---:|---|
| MovieLens | 314.2 ± 15.2 | 3.3232 | 0.1583 | Loc < PRB |
| AmazonFashion | 193.9 ± 37.5 | 2.8020 | 0.1158 | Loc < PRB |
| Facebook | 642.5 ± 135.4 | 3.3836 | 0.1092 | Loc < PRB |
| Grqc | 834.9 ± 36.3 | 2.6147 | 0.1194 | Loc < PRB |
| Collab | 784.8 ± 72.8 | 430.3022 | 3.4355 | Loc < PRB |
| PPA | 约 748.7（partial） | 无 PPR；online 约 13524.2 | 2.9712 | 仅 online 支持 |
| Vessel | 889.0 ± 8.6 | 3507.0060 | 27.5176 | Loc < PRB |

由于 AmazonFashion、Facebook、PPA 的 PRB regret 与本轮 Loc/DYN 不足以称为统一
paired 的“差不多”，本表不能替代最终 PRB rerun。它只实现用户指定的阶段策略：
避免反复运行昂贵 PRB，先判断优化方向。

## 最终判断

1. 选择 Numba 版本进入最终三方法候选；Python 版本不进入最终版本。
2. 本阶段在聚合口径达到 `Numba-DYN < Numba-Loc < PRB`。
3. 若要求每个数据集都严格满足，当前还有 Facebook 与 Grqc 两个轻微缺口。
4. 最终论文结论仍必须用统一事件流、paired seeds 和统一计时完整重跑 PRB 后确认。
5. 本阶段到此停止，不启动 Test C 或最终完整实验。
