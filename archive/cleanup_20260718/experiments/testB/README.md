# Test B Combinations

| ID | Changes relative to A0 |
|---|---|
| B1 | A6 |
| B2 | A1 + A6 |
| B3 | A3 + A6 |
| B4 | A1 + A3 + A6 |
| B5 | A1 + A3 + A6 + A8 |

The runner reuses Test A's isolated loader RNG. Results are stored under
`testB/results/B1` through `testB/results/B5`.

Example:

```bash
python -u testB/run_combination.py \
  --combination B1 \
  --graph_name Collab \
  --method FastPRB \
  --alpha 0.85 \
  --appr_eps 4.24e-06 \
  --T 10000 \
  --lr1 0.01 \
  --lr2 0.004 \
  --runs 10 \
  --workers 10 \
  --seed 0
```
