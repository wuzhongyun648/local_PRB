# T=1000 final Numba validation

Date: 2026-07-22. This is the requested T=1000 milestone, not the later full
paper experiment.

## Protocol

- One paired run, seed 0, one worker, sequential CPU execution.
- Numba backend; evaluation disabled for timing.
- Reported PPR time excludes JIT warmup, evaluation, diagnostics and work-counter
  instrumentation for both LocPRB and Adaptive-DYN.
- Adaptive-DYN checks source pressure initially. After three consecutive scratch
  resets it locks a reusable scratch fast path. This is an **algorithm/strategy
  change** and is not pure state-maintaining DYN. In these runs 996/1000 rounds
  used the locked path.
- DYN ran a fresh scratch diagnostic every 10 rounds (100 checks), excluded from
  timing.
- PRB was not rerun. It uses the user-selected interim Option A archive reference.

Final logs:

- `log/backend_matrix_T1000_20260722_190712/` (four smaller datasets)
- `log/backend_matrix_T1000_20260722_192003/` (Collab, PPA, Vessel)

## Paired LocPRB vs Adaptive-DYN result

| Dataset | Loc regret | DYN regret | Loc PPR (s) | DYN PPR (s) | DYN/Loc | PPR saving | diagnostic disagreements |
|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens | 313 | 313 | 0.160205 | 0.138083 | 0.862 | 13.8% | 0/100 |
| AmazonFashion | 298 | 298 | 0.091623 | 0.089824 | 0.980 | 2.0% | 0/100 |
| Facebook | 824 | 824 | 0.086118 | 0.083676 | 0.972 | 2.8% | 0/100 |
| GrQc | 842 | 842 | 0.093318 | 0.093060 | 0.997 | 0.3% | 0/100 |
| Collab | 785 | 785 | 3.638239 | 3.371236 | 0.927 | 7.3% | 0/100 |
| PPA | 921 | 921 | 2.990656 | 2.615889 | 0.875 | 12.5% | 0/100 |
| Vessel | 879 | 879 | 40.666589 | 21.724469 | 0.534 | 46.6% | 0/100 |

All seven datasets satisfy `Numba-DYN < Numba-LocPRB` in PPR time. Summed
PPR time is 47.7267s for Loc and 28.1162s for DYN, a 41.1% reduction. Regret is
exactly equal on every dataset; all 700 sampled decision comparisons agree and
the sampled L1 error is zero.

Online total is also retained in each metrics JSON, but it includes neural
training and host-load variation. It is not monotone on all datasets (most
notably MovieLens), despite only millisecond-scale PPR differences. The milestone
therefore uses the isolated PPR solver time for the method-speed inequality and
reports online total as a secondary systems measurement.

## LocPRB vs archived PRB Option A

The archive provides full-run PPR totals for six datasets. Linear T=1000
equivalents are 3.323s (MovieLens), 2.802s (AmazonFashion), 3.384s (Facebook),
2.615s (GrQc), 430.302s (Collab), and 3507.006s (Vessel), all far above the
corresponding LocPRB times. PPA lacks an archived PPR total, but its approximate
T=1000 online time is 13524.2s versus 212.7s for LocPRB. Thus Option A supports
the interim ordering `LocPRB < PRB` with very large margins.

PRB regret is not a fully paired comparison: six archive files contain 10 runs,
and PPA is only an interleaved partial log. Where seed-0 archive rows can be
read, PRB regret is 315, 273, 755, 858, 848 and 875 versus Loc regret 313, 298,
824, 842, 785 and 879. These differences are within one archived PRB standard
deviation for every complete dataset. PPA's archived mean is approximately
748.7 versus Loc 921, but no reliable paired run can be recovered from the
interleaved partial log. This limitation remains explicitly interim; the final
paper experiment must rerun PRB under the unified protocol.

## Milestone conclusion

At T=1000, the desired Numba route is selected for the final three-version
experiment:

```text
Adaptive Numba-DYN-LocPRB < Numba-LocPRB < PRB (Option A interim)
```

The first inequality is paired and verified on all seven datasets with identical
regret. The second is supported by the selected archived timing reference, not a
new fair PRB rerun. No Test C or full paper experiment was started.
