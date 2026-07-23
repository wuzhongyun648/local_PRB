# T=1000 四方法结果汇总

日期：2026-07-23

实验覆盖七个数据集和四种方法：

- Numba-DYN-LocPRB
- Numba-LocPRB
- Python-DYN-LocPRB
- Python-LocPRB

统一配置为 `T=1000`、`runs=1`、`workers=1`、seed 0、固定单 CPU core，
BLAS 线程数为 1，关闭周期 evaluation。Numba 已在正式计时前预热。

时间口径：

- `Total`：完整实验端到端时间，包含 setup、warmup 和 1000 轮在线运行，但剔除
  DYN prediction 或 Loc predictor control。
- `Round`：1000 轮累计 online time，剔除 testset build、evaluation、diagnostics
  和 prediction/control；不是单轮平均时间。
- `PPR`：Round 中正式执行所选 PPR 分支的时间，不含 prediction/control。
- `Train`：Round 中神经网络训练时间。
- `Other`：`Round - PPR - Train`，合并 loader、predict/source、decision、
  graph update 和其余在线记账时间。

所有时间单位均为秒。表中 DYN 是每轮在真实动态继续和 scratch reset 之间重新选择的
adaptive DYN，不存在永久锁定。

## Numba

| Dataset | DYN Regret | Loc Regret | DYN Total | Loc Total | DYN Round | Loc Round | DYN PPR | Loc PPR | DYN Train | Loc Train | DYN Other | Loc Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MovieLens | 313 | 313 | 128.028 | 128.496 | 126.502 | 126.995 | 0.135 | 0.158 | 37.648 | 37.828 | 88.719 | 89.009 |
| AmazonFashion | 298 | 298 | 84.479 | 83.633 | 84.331 | 83.482 | 0.114 | 0.116 | 37.789 | 37.329 | 46.428 | 46.037 |
| Facebook | 824 | 824 | 54.210 | 53.030 | 53.839 | 52.667 | 0.111 | 0.109 | 38.394 | 37.476 | 15.334 | 15.082 |
| Grqc | 842 | 842 | 51.199 | 51.322 | 51.005 | 51.140 | 0.121 | 0.119 | 37.867 | 37.998 | 13.017 | 13.022 |
| Collab | 785 | 785 | 56.486 | 57.639 | 54.318 | 55.464 | 3.372 | 3.435 | 38.759 | 39.687 | 12.187 | 12.341 |
| PPA | 921 | 921 | 175.261 | 177.994 | 154.881 | 156.807 | 2.081 | 2.971 | 36.883 | 37.257 | 115.918 | 116.579 |
| Vessel | 879 | 879 | 97.853 | 106.152 | 85.222 | 93.435 | 19.373 | 27.518 | 37.032 | 37.203 | 28.816 | 28.715 |

Numba 七组 DYN/Loc regret 完全一致。按 PPR 时间，DYN 在 MovieLens、
AmazonFashion、Collab、PPA、Vessel 上更快，在 Facebook 和 Grqc 上略慢。

## Python

| Dataset | DYN Regret | Loc Regret | DYN Total | Loc Total | DYN Round | Loc Round | DYN PPR | Loc PPR | DYN Train | Loc Train | DYN Other | Loc Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MovieLens | 313 | 313 | 125.170 | 125.449 | 123.879 | 124.103 | 0.231 | 0.222 | 37.824 | 37.705 | 85.824 | 86.177 |
| AmazonFashion | 298 | 298 | 88.168 | 87.950 | 88.153 | 87.934 | 7.233 | 6.938 | 37.431 | 37.659 | 43.490 | 43.337 |
| Facebook | 824 | 824 | 60.226 | 60.400 | 60.116 | 60.290 | 7.441 | 7.408 | 37.445 | 37.638 | 15.229 | 15.244 |
| Grqc | 842 | 842 | 55.332 | 55.116 | 55.284 | 55.068 | 5.075 | 4.700 | 37.694 | 37.904 | 12.515 | 12.464 |
| Collab | 785 | 785 | 354.503 | 349.935 | 352.734 | 348.174 | 300.362 | 296.103 | 39.700 | 39.413 | 12.671 | 12.658 |
| PPA | 921 | 921 | 265.102 | 270.119 | 241.622 | 246.428 | 51.022 | 53.413 | 37.144 | 37.380 | 153.456 | 155.634 |
| Vessel | 879 | 879 | 385.139 | 264.940 | 371.234 | 251.205 | 295.127 | 175.514 | 37.913 | 37.342 | 38.193 | 38.349 |

Python 七组 DYN/Loc regret也完全一致。按 PPR 时间，只有 PPA 上 DYN 更快；
其余六个数据集均为 DYN 更慢，因此最终不采用 Python 路线。

原始结构化 metrics 位于各实验结果目录的 `metrics_summary.json`；运行日志位于：

- `log/backend_matrix_T1000_matched_control_numba_T1000_20260723/`
- `log/backend_matrix_T1000_pure_python_small4_T1000_20260723/`
- `log/backend_matrix_T1000_pure_python_collab_T1000_20260723/`
- `log/backend_matrix_T1000_pure_python_ppa_T1000_20260723/`
- `log/backend_matrix_T1000_pure_python_vessel_T1000_20260723/`
