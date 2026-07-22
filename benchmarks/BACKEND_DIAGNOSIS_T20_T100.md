# Backend diagnosis at T=20 and T=100

Date: 2026-07-22

## Protocol

- `runs=1`, `workers=1`, `seed=0`, CPU execution, BLAS/OMP threads fixed to 1.
- Periodic evaluation disabled because it is excluded from online timing.
- All jobs ran sequentially through `scripts/run_backend_matrix.sh`.
- Resource snapshots stayed near 87% CPU idle on a 96-logical-core host. NVIDIA SMI was
  unavailable in the local sandbox, so these runs explicitly used `CUDA_VISIBLE_DEVICES=''`.
- PRB was not rerun; `benchmarks/prb_option_a_reference.json` remains the interim reference.

Raw logs:

- `log/backend_matrix_T20_20260722_165948/`
- `log/backend_matrix_T100_20260722_171149/`
- `log/backend_matrix_T100_20260722_171842/` (DYN correctness diagnostics)

## T=20 backend matrix

Python and Numba backends produced the same regret for every algorithm and dataset. Loc and
DYN also had identical regret on all seven datasets at this length.

| Dataset | Regret | Python Loc PPR | Python DYN PPR | Numba Loc PPR | Numba DYN PPR | Numba DYN/Loc |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens | 19 | 0.237384s | 0.055912s | 0.003926s | 0.007068s | 1.800 |
| AmazonFashion | 8 | 0.197054s | 0.062576s | 0.001861s | 0.003723s | 2.001 |
| Facebook | 18 | 0.266836s | 0.161862s | 0.002499s | 0.004919s | 1.968 |
| GrQc | 17 | 0.165164s | 0.073195s | 0.001809s | 0.003861s | 2.134 |
| Collab | 18 | 12.563593s | 6.593840s | 0.098484s | 0.180817s | 1.836 |
| PPA | 19 | 27.853916s | 16.293692s | 0.209536s | 0.386993s | 1.847 |
| Vessel | 18 | 71.607297s | 18.065877s | 1.113734s | 1.120427s | 1.006 |

The Python result is caused partly by implementation asymmetry: Python Loc performs its
full-node initialization loop in Python, while DYN performs several preparation scans in
NumPy before entering its Python push loop. It does not imply that DYN performs less graph
work.

## T=100 Numba timing

| Dataset | Loc regret | DYN regret | Loc PPR | DYN PPR | DYN/Loc time | DYN/Loc pushes |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens | 81 | 81 | 0.019709s | 0.034274s | 1.739 | 1.809 |
| AmazonFashion | 22 | 22 | 0.010221s | 0.020104s | 1.967 | 1.236 |
| Facebook | 93 | 93 | 0.012898s | 0.024465s | 1.897 | 1.276 |
| GrQc | 91 | 91 | 0.010492s | 0.019236s | 1.833 | 1.722 |
| Collab | 92 | 92 | 0.469403s | 0.864240s | 1.841 | 1.706 |
| PPA | 95 | 86 | 0.664004s | 1.314444s | 1.980 | 1.838 |
| Vessel | 89 | 89 | 5.790507s | 5.425310s | 0.937 | 1.935 |

Only Vessel currently satisfies Numba DYN < Numba Loc at T=100. PPA has a regret difference
of 9 because Loc and DYN disagree on 44 of 100 candidate decisions; DYN happens to have lower
regret in this seed, but it is outside the provisional absolute-difference guardrail.

## DYN versus scratch diagnostics at T=100

| Dataset | Decision disagreements | Mean L1 error | Max L1 error | DYN/scratch pushes | Fallbacks | Inserts |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens | 0 | 0.000000 | 0.000000 | 1.809 | 18 | 0 |
| AmazonFashion | 0 | 0.001880 | 0.022429 | 1.236 | 66 | 9 |
| Facebook | 0 | 0.833747 | 1.456711 | 1.276 | 1 | 6 |
| GrQc | 0 | 0.250568 | 0.437357 | 1.722 | 3 | 6 |
| Collab | 0 | 1.666450 | 2.640036 | 1.706 | 0 | 0 |
| PPA | 44 | 2.142719 | 2.896689 | 1.838 | 0 | 0 |
| Vessel | 0 | 0.023477 | 0.050272 | 1.935 | 0 | 0 |

## Root cause and next optimization order

Consecutive source supports are almost disjoint. The current DYN update propagates
`s_t-s_(t-1)`, normally close to 20 support entries, while scratch Loc propagates only the 10
current candidates. Signed removal residuals therefore make DYN perform 1.2x-1.9x as many
pushes/edge visits.

The non-algorithmic overheads should be removed first:

1. represent current/previous source sparsely instead of scanning full dense vectors;
2. initialize the DYN active queue from changed support/endpoints rather than scanning all
   nodes;
3. reuse queue/state buffers and avoid returning a full `p.copy()` each round;
4. pass graph-change information explicitly instead of discovering it through full degree
   scans.

These changes preserve the APPR invariant and push rule. If they are insufficient, an
adaptive scratch fallback is the most plausible route to the target inequality, but that
would make DYN a hybrid implementation and must be reported explicitly before final use.
