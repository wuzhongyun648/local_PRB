# Fast_bandit Handover

## Current Focus

We audited and split the online baseline experiments for LocPRB into separate runner styles so future experiments can distinguish:

- the baseline implementation style inherited from the PRB/EE-Net released code, and
- the implementation style aligned more closely with the original baseline authors' code.

The active baseline entry point is now:

- `online_baselines_run.py`

The older adapted baseline implementation remains under `src/online_baselines_run_new.py` for reference. It should not be treated as the active baseline entry point.

## Important Files

- `BASELINE_RUNNER_TASK.md`
  - Defines the task goal, shared experiment protocol, expected outputs, and acceptance criteria.

- `online_baselines_run.py`
  - New standalone online baseline runner.
  - Uses PRB/EE-Net released baseline logic as directly as practical, with compatibility edits for imports, device handling, dataset adapters, multiprocessing, and saving.
  - Does not import `src.online_baselines_run_new`.

- `online_baselines_original_style.py`
  - This was created during the audit, run once, and then archived.
  - Current location: `archive/original_style_baselines_20260710/scripts/online_baselines_original_style.py`.
  - It reused the common execution/saving/data-loading logic from the PRB-style runner before that runner was renamed to `online_baselines_run.py`.
  - It overrides `NeuralUCB` and `NeuralTS` with implementations aligned to the official baseline repositories checked under `paper/baselines/`.
  - `NeuralTS` is implemented with per-arm gradients instead of `backpack`, to avoid adding a hard dependency while preserving the selection/update logic.

## Baseline Audit Summary

The existing Fast_bandit runner:

```bash
python online_baselines_run.py
```

runs the active PRB-style baseline implementation.

It should not be described as:

- strict PRB released baseline code, or
- strict original-author baseline code.

The PRB repository's `baselines/baselines_run.py` is not a direct online link-prediction runner: it imports `load_movielen`, but the actual loop uses `load_mnist_1d()`. The released PRB baseline classes are effectively copied from the EE-Net baseline code.

Important differences found:

- PRB `NeuralUCB` is close to official NeuralUCB, but differs in training interface and regularization details.
- PRB `NeuralTS` differs substantially from official NeuralTS, especially around batch gradients, uncertainty updates, and training.
- The original-style runner therefore overrides `NeuralUCB` and `NeuralTS`.

## Shared Protocol For New Runners

Both new runners support:

- `MovieLens`
- `AmazonFashion`
- `Facebook`
- `GrQc`
- `Collab`
- `PPA`
- `Vessel`

The current online baseline run used:

- datasets: `MovieLens Collab Vessel PPA`
- candidate set size: 10 arms
- candidate composition: 1 positive + 9 negatives
- `--runs 10`
- `--workers 10`
- default `T=10000`
- default methods:
  - `EE-Net`
  - `NeuralUCB`
  - `NeuralTS`
  - `NeuralGreedy`
  - `LinUCB`
  - `KernelUCB`

Default training schedule:

- before round 2000: train every 50 rounds
- from round 2000 onward: train every 100 rounds

## Historical Commands Used

These are the equivalent foreground commands from the audit. The PRB-style command uses the current active entry point. The original-style command now lives under the archive path.

```bash
cd /mnt/data/xinyu/Fast_bandit

python online_baselines_run.py \
  --datasets MovieLens Collab Vessel PPA \
  --runs 10 \
  --workers 10

python archive/original_style_baselines_20260710/scripts/online_baselines_original_style.py \
  --datasets MovieLens Collab Vessel PPA \
  --runs 10 \
  --workers 10
```

The execution environment did not preserve ordinary `nohup` or `setsid` child processes from tool calls, so the long jobs were run through detached `tmux` sessions:

```bash
tmux new-session -d -s locprb_prb_style_20260710 \
  'cd /mnt/data/xinyu/Fast_bandit && python -u online_baselines_run.py --datasets MovieLens Collab Vessel PPA --runs 10 --workers 10 > log/online_baselines_prb_style_20260710_063629.log 2>&1'

tmux new-session -d -s locprb_original_style_20260710 \
  'cd /mnt/data/xinyu/Fast_bandit && python -u archive/original_style_baselines_20260710/scripts/online_baselines_original_style.py --datasets MovieLens Collab Vessel PPA --runs 10 --workers 10 > log/online_baselines_original_style_20260710_063629.log 2>&1'
```

Both `tmux` sessions have finished. `tmux ls` currently reports no server.

## Completed Run Outputs

PRB-style output:

```text
results/baselines_prb_style/20260710_083837/
log/online_baselines_prb_style_20260710_063629.log
```

Original-style output:

```text
archive/original_style_baselines_20260710/results/baselines_original_style/20260710_083440/
archive/original_style_baselines_20260710/logs/online_baselines_original_style_20260710_063629.log
```

Both metadata files report:

- successes: 240
- failures: 0

The 240 tasks are:

```text
4 datasets * 6 methods * 10 runs = 240
```

Elapsed time from metadata:

- PRB-style: about 7089 seconds
- Original-style, now archived: about 6787 seconds

Each result directory contains one regret file and one time file per dataset/method pair plus `metadata.json`.

## Validation Already Done

Syntax/import validation:

```bash
python -m py_compile online_baselines_run.py archive/original_style_baselines_20260710/scripts/online_baselines_original_style.py
```

Smoke tests run before full jobs:

```bash
python online_baselines_run.py --datasets MovieLens --methods NeuralUCB --T 1 --runs 1 --workers 1
python archive/original_style_baselines_20260710/scripts/online_baselines_original_style.py --datasets MovieLens --methods NeuralUCB --T 1 --runs 1 --workers 1
python online_baselines_run.py --datasets MovieLens --methods NeuralTS --T 1 --runs 1 --workers 1
python archive/original_style_baselines_20260710/scripts/online_baselines_original_style.py --datasets MovieLens --methods NeuralTS --T 1 --runs 1 --workers 1
```

## Fixes Made During Run

`online_baselines_original_style.py` initially had a fallback recursion bug for methods other than `NeuralUCB` and `NeuralTS`.

Symptom:

```text
RecursionError: maximum recursion depth exceeded
```

Fix:

- saved the original `common.build_model` as `COMMON_BUILD_MODEL`
- fallback now calls `COMMON_BUILD_MODEL(...)` instead of the monkeypatched `common.build_model(...)`

The original-style job was stopped, fixed, and restarted cleanly. Final output has 0 failures.

## Notes For Future Agents

- Do not use `src/online_baselines_run_new.py` as the active baseline entry point unless explicitly asked; it is the older adapted runner retained for reference.
- Use `online_baselines_run.py` when the intended baseline is "PRB released/PRB-style".
- The active baseline runner to use going forward is `online_baselines_run.py`.
- The original-code-aligned runner and its results are archived under `archive/original_style_baselines_20260710/`.
- For long-running jobs in this environment, prefer detached `tmux` over plain `nohup`.
- The log files are verbose because workers print every 100 steps.
- Result arrays are saved as `.npy`; shapes should generally be `(10, 10000)` for each dataset/method regret/time file from the completed full run.
- The result roots also include earlier short smoke-test directories. Use the full-run directories listed above for analysis.
