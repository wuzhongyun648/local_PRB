# Fast Bandit Repository Migration Plan

## Goal

Clean the repository without losing experiment history. The target state is a small, reproducible core for the paper experiments, with logs, failed runs, old scripts, and external baseline material separated into `archive/`.

## Core Code To Keep Active

These files are the first-class experiment surface:

- `main.py`: main online PRB/FastPRB experiments, including eps and alpha ablations through command-line arguments.
- `online_baselines_run_new.py`: online baseline runner. These experiments are not in the paper, but the runner is still useful.
- `ppr_solver.py`: power iteration and APPR/local push PPR solvers.
- `EENet.py`, `EENetClass.py`: EE-Net model and component networks.
- `load_data.py`: online data loaders and candidate generation.
- `utils.py`: graph managers, graph updates, result saving helpers.
- `requirements.txt`: environment dependencies.

## First-Phase Changes

The first phase should be conservative:

1. Add `README.md` with the commands needed for main experiments, eps ablations, alpha ablations, and baselines.
2. Replace hardcoded absolute paths with paths derived from the repository root.
3. Keep all generated outputs under `results/`.
4. Move unused scripts, old plot scripts, old logs, and failed results into `archive/`; do not delete them.
5. Verify that `main.py` and `online_baselines_run_new.py` still import and pass a small smoke test.

## Proposed Directory Shape

```text
Fast_bandit/
  README.md
  MIGRATION_PLAN.md
  requirements.txt

  main.py
  online_baselines_run_new.py
  ppr_solver.py
  EENet.py
  EENetClass.py
  load_data.py
  utils.py
  baselines_new/

  data/
  dataset/
  paper/
  results/

  archive/
    logs/
    old_scripts/
    old_plots/
    old_baselines/
    failed_results/
    audit_reports/
```

## Archive Policy

Move files to `archive/` when they are not needed by the current active runners but may still contain useful history.

Candidates for `archive/old_scripts/`:

- `run_node_classification.py`
- `run_offline_link_prediction.py`
- `main_dmax_track.py`
- `offline_debug.py`
- `test_movie_nodes.py`
- `temp_cal.py`
- `inspect_facebook.py`
- `print_dataset_degree_stats.py`
- `data_ogb.py`
- `load_data_new.py`
- `utils_new.py`
- `online_baselines_run.py`

Candidates for `archive/old_plots/`:

- `plot.py`
- `plotpart.py`
- `plot_baseline.py`
- `plot_power_law.py`
- `plot_node_classification.py`
- `final_plot_online.py`
- `accuracy_plot.py`

Candidates for `archive/logs/`:

- Top-level `*.log`
- `nohup.out`
- Old baseline text logs

Candidates for `archive/old_baselines/`:

- `baselines/`

Candidates for `archive/failed_results/`:

- `results_failed/`

## Result Data Policy

Do not delete result data in the first phase. Later cleanup should create a result manifest that marks each result directory as one of:

- paper final
- ablation
- baseline
- tuning sweep
- failed or obsolete

Only after that manifest exists should old `.npy` result directories be deleted.

## Path Policy

All active code should derive paths from the repository root:

```python
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
OGB_ROOT = os.path.join(CURRENT_DIR, "dataset")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
```

Avoid absolute paths such as `/mnt/data/xinyu/bandits_pj/PRB` in active scripts.

## Verification Policy

After each cleanup phase:

1. Run import checks for active entry points.
2. Run a small smoke test with `T=1`, `runs=1`, and `workers=1`.
3. Confirm outputs land under `results/`.
4. Confirm archived files are not imported by active runners.
