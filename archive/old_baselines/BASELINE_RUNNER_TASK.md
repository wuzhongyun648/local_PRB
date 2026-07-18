# Online Baseline Runner Task

## Status Update

The two-runner audit was completed. After comparing results, the project kept the PRB-style runner as the active baseline runner and renamed it to:

```text
online_baselines_run.py
```

The original-code-aligned runner and its outputs were moved to:

```text
archive/original_style_baselines_20260710/
```

## Goal

Add online baseline runners for LocPRB experiments while preserving the older adapted implementation under `src/online_baselines_run_new.py` for reference.

The two new runners will use the same online link prediction protocol and the same experiment hyperparameters, but different baseline implementation sources:

- `online_baselines_run.py`: use the released PRB baseline implementations as directly as possible, changing only compatibility concerns such as imports, device handling, dataset adapters, multiprocessing, and result saving.
- `archive/original_style_baselines_20260710/scripts/online_baselines_original_style.py`: archived runner aligned with the original baseline papers/code after audit, while keeping the same dataset protocol, training schedule, and hyperparameters as the PRB-style runner.

## Shared Experiment Protocol

Both runners must support these datasets:

- `MovieLens`
- `AmazonFashion`
- `Facebook`
- `GrQc`
- `Collab`
- `PPA`
- `Vessel`

Each online round uses a 10-arm candidate set:

- 1 positive candidate
- 9 negative candidates

Both runners should expose the same command-line surface where practical:

- `--datasets`
- `--methods`
- `--T`
- `--runs`
- `--workers`
- `--lamdba`
- `--nu`
- `--lr1`
- `--lr2`
- `--seed`

The training frequency and hyperparameters must be unified across both runners so the comparison isolates implementation differences rather than schedule or tuning differences.

Default training schedule:

- before round 2000: train every 50 rounds
- from round 2000 onward: train every 100 rounds

Default candidate count:

- `n_neg = 9`
- `n_arm = 10`

## Expected Outputs

Each runner should write results under a separate result root:

- `results/baselines_prb_style/<timestamp>/`
- `results/baselines_original_style/<timestamp>/`

For each successful dataset/method pair, save:

- `<dataset>_<method>_regret.npy`
- `<dataset>_<method>_time.npy`

Each run directory should also include:

- `metadata.json`

The metadata should record:

- runner style
- datasets
- methods
- runs
- workers
- horizon `T`
- candidate count
- training schedule
- hyperparameters
- seed rule
- successes and failures

## Acceptance Criteria

The task is complete when:

1. `online_baselines_run.py` exists and runs the PRB-style baseline implementations on all supported dataset adapters.
2. The archived `online_baselines_original_style.py` exists under `archive/original_style_baselines_20260710/scripts/` and runs the audited original-code-aligned baseline implementations on the same dataset adapters.
3. The older adapted implementation under `src/online_baselines_run_new.py` is not the active baseline entry point.
4. Both new runners use 10-arm candidate sets by default.
5. Both new runners share the same default training schedule and hyperparameters.
6. Both new runners save regrets, per-step wall-clock times, and metadata in separate result roots.
7. At minimum, both new files pass Python syntax/import validation.
