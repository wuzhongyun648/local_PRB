# Adaptive DYN-LocPRB short validation

Date: 2026-07-22. Protocol: `T=100`, seed 0, one run, one worker,
`evaluation_every=0`, Numba backend, sequential execution. DYN diagnostics run
fresh scratch after the reported solver timer and are excluded from online time.

This checkpoint **changes the DYN solving strategy**. Before each continuation it
compares the degree/epsilon-normalized pressure of the source delta with the
pressure of a fresh source. If the dynamic pressure is not smaller, it resets to
scratch APPR using a caller-owned workspace. It is therefore an adaptive hybrid,
not the original pure state-maintaining DynamicAPPR.

| Dataset | Loc regret | Adaptive DYN regret | Loc PPR (s) | DYN PPR (s) | DYN/Loc | resets/99 | decision disagreements |
|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens | 81 | 81 | 0.012674 | 0.012898 | 1.018 | 99 | 0 |
| AmazonFashion | 22 | 22 | 0.007501 | 0.009642 | 1.285 | 99 | 0 |
| Facebook | 93 | 93 | 0.012397 | 0.013685 | 1.104 | 99 | 0 |
| GrQc | 91 | 91 | 0.009986 | 0.011941 | 1.196 | 99 | 0 |
| Collab | 92 | 92 | 0.427827 | 0.452554 | 1.058 | 99 | 0 |
| PPA | 95 | 94 | 0.532746 | 0.503841 | 0.946 | 50 | 10 |
| Vessel | 89 | 89 | 2.574978 | 2.182715 | 0.848 | 99 | 0 |

Across the seven datasets, summed PPR time is 3.5781s for LocPRB and 3.1873s
for adaptive DYN (DYN/Loc 0.891, about 10.9% faster). The strict per-dataset
inequality is not yet achieved: only PPA and Vessel pass at T=100. Six datasets
have identical regret and decisions. PPA differs by one regret and 10/100
decisions because 49 rounds continued the dynamic state; its maximum diagnostic
L1 distance to scratch is 0.0640.

Logs:

- Small datasets: `log/backend_matrix_T100_20260722_180437/`
- Collab/PPA/Vessel: `log/backend_matrix_T100_20260722_180728/`

The next decision is based on T=1000: check whether the aggregate advantage is
stable and whether PPA remains within the regret guardrail. The adaptive reset
rate and the fact that most datasets effectively use reusable scratch must be
reported in any final result.
