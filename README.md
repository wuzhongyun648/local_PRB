# Fast Bandit

This repository contains the code for local bandits on virtual label graphs. Active implementation code lives under `src/`, with top-level scripts for the main experiments and online baselines:

- `main.py`: online PRB, LocPRB, and dynamic LocPRB experiments.
- `online_baselines_run.py`: online PRB-style baseline experiments.

## Code Structure

```text
main.py                         # compatibility wrapper for src.main
online_baselines_run.py         # active PRB-style online baseline runner
src/
  main.py                       # main PRB/LocPRB experiment runner
  dynamic_appr.py               # stateful DYN-APPR solver
  experiment_configs.py         # shared paths, dataset configs, defaults
  load_data.py                  # dataset loaders
  utils.py                      # graph utilities and result saving
  ppr_solver.py                 # PPR/APPR propagation utilities
  EENet.py, EENetClass.py       # neural scoring models for PRB/FastPRB
  baselines_new/                # baseline model implementations
  plot_results.py               # generic plotting for active results
  plot_paper_figures.py         # regenerate paper figures from results/final/data
data/                           # local preprocessed graph/input arrays
dataset/                        # OGB/raw dataset cache
results/final/                  # confirmed paper results and generated figures
archive/                        # old scripts, non-final results, logs, audit files
archive/maintenance_scripts/    # migration/result-audit scripts
paper/                          # paper-side auxiliary files
```

Normal experiment entry points are still the top-level wrappers:

```bash
python main.py ...
python online_baselines_run.py ...
```

Module-style execution is supported for the main PRB/FastPRB runner:

```bash
python -m src.main ...
```

## Environment

Install dependencies from:

```bash
pip install -r requirements.txt
```

Run commands from this directory:

```bash
cd /mnt/data/xinyu/Fast_bandit
```

Shared paths, active dataset metadata, default run counts, baseline method lists, and training schedule constants live in:

```text
src/experiment_configs.py
```

Command-line arguments override these defaults where applicable.

GPU selection is controlled by the shell environment. For CPU smoke tests:

```bash
CUDA_VISIBLE_DEVICES="" python main.py ...
```

For a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py ...
```

## Main Experiments

FastPRB example:

```bash
python main.py \
  --graph_name MovieLens \
  --method FastPRB \
  --alpha 0.85 \
  --appr_eps 8.33e-05 \
  --T 10000 \
  --lr1 0.0073 \
  --lr2 0.0004 \
  --runs 10 \
  --workers 10
```

PRB example:

```bash
python main.py \
  --graph_name MovieLens \
  --method PRB \
  --alpha 0.85 \
  --power_T 50 \
  --T 10000 \
  --lr1 0.0073 \
  --lr2 0.0004 \
  --runs 10 \
  --workers 10
```

Outputs are saved under:

```text
results/online_link_prediction/
```

Archived paper-confirmed outputs are stored separately under:

```text
results/final/data/table1/
```

## Eps Ablation

Run `main.py` with the same dataset and hyperparameters while changing `--appr_eps`.

```bash
python main.py --graph_name MovieLens --method FastPRB --alpha 0.85 --appr_eps 8.33e-04 --T 10000 --lr1 0.0073 --lr2 0.0004 --runs 10 --workers 10
python main.py --graph_name MovieLens --method FastPRB --alpha 0.85 --appr_eps 8.33e-05 --T 10000 --lr1 0.0073 --lr2 0.0004 --runs 10 --workers 10
python main.py --graph_name MovieLens --method FastPRB --alpha 0.85 --appr_eps 8.33e-06 --T 10000 --lr1 0.0073 --lr2 0.0004 --runs 10 --workers 10
```

## Alpha Ablation

Run `main.py` with the same dataset and hyperparameters while changing `--alpha`.

```bash
python main.py --graph_name MovieLens --method FastPRB --alpha 0.60 --appr_eps 8.33e-05 --T 10000 --lr1 0.0073 --lr2 0.0004 --runs 10 --workers 10
python main.py --graph_name MovieLens --method FastPRB --alpha 0.70 --appr_eps 8.33e-05 --T 10000 --lr1 0.0073 --lr2 0.0004 --runs 10 --workers 10
python main.py --graph_name MovieLens --method FastPRB --alpha 0.85 --appr_eps 8.33e-05 --T 10000 --lr1 0.0073 --lr2 0.0004 --runs 10 --workers 10
python main.py --graph_name MovieLens --method FastPRB --alpha 0.90 --appr_eps 8.33e-05 --T 10000 --lr1 0.0073 --lr2 0.0004 --runs 10 --workers 10
```

## Baseline Experiments

Run all default datasets and baseline methods:

```bash
python online_baselines_run.py --T 10000 --runs 10 --workers 10
```

Run a smaller selected baseline job:

```bash
python online_baselines_run.py \
  --datasets MovieLens Facebook GrQc \
  --methods EE-Net NeuralUCB NeuralTS NeuralGreedy \
  --T 10000 \
  --runs 10 \
  --workers 10
```

Outputs are saved under:

```text
results/baselines_prb_style/
```

The active baseline runner follows the PRB/EE-Net released baseline implementation style with compatibility fixes for this repository. The older adapted implementation is retained at `archive/cleanup_20260718/code/src/online_baselines_run_new.py` for reference.

## Results Manifest

Generate a result inventory without moving any result files:

```bash
python archive/maintenance_scripts/results_manifest.py
```

This writes:

```text
RESULTS_MANIFEST.md
results_manifest.csv
```

The manifest marks detected smoke tests, eps/alpha ablations, tuning runs, baseline files, plot files, and paper-candidate online runs.

## Plotting

Use the unified plotting entry point for active result formats:

```bash
python -m src.plot_results baseline --result-dir results/baselines_prb_style --datasets MovieLens --methods EE-Net NeuralUCB NeuralTS
```

For online PRB/FastPRB runs, pass one or more run directories or `final_results.npy` files:

```bash
python -m src.plot_results online \
  results/online_link_prediction/MovieLens_FastPRB_alpha0.85_eps8.33e-05_T10000_lr10.0073_lr20.0004_202601291141 \
  results/online_link_prediction/MovieLens_PRB_alpha0.85_powT50_T10000_lr10.0073_lr20.0004_202601290025 \
  --name MovieLens_PRB_vs_FastPRB
```

Plots are written to `results/plots/` by default.

Generate paper figures directly from archived final data:

```bash
python -m src.plot_paper_figures
```

This reads `results/final/data/` and writes PDF/PNG files to:

```text
results/final/plots/generated/
```

The generated paper figures include main regret curves, the ogbl-Collab eps ablation, and the ogbl-Collab alpha ablation.

## Final Results Archive

The cleaned final-result layout is:

```text
results/final/data/table1/       # main paper table/results
results/final/data/table3_eps/   # eps ablation results
results/final/data/table4_alpha/ # alpha ablation results
results/final/plots/exact/       # PDFs with exact stream matches to the paper PDF
results/final/plots/candidates/  # same-name candidate PDFs kept for manual review
results/final/plots/generated/   # regenerated figures from final data
results/final/metadata/          # copied audit CSV/MD reports
```

Non-final historical result files were moved to:

```text
archive/results_nonfinal_20260709_070314/
```

The archive operation is recorded in:

```text
results/final/FINAL_ARCHIVE_MANIFEST.md
```

## Result Confirmation

Confirm paper table values against raw `final_results.npy` files:

```bash
python archive/maintenance_scripts/select_paper_results.py --top 5
```

This scans `results/online_link_prediction/` when raw runs are present and writes:

```text
PAPER_RESULT_SELECTION.md
paper_result_selection.csv
all_online_result_stats.csv
```

Confirm PDF-embedded figure names and exact stream matches:

```bash
python archive/maintenance_scripts/confirm_paper_results.py
```

Archive confirmed paper results and non-final results:

```bash
python archive/maintenance_scripts/archive_paper_results.py
```

The archive script is intended for maintenance use. Review `paper_result_selection.csv` before running it on a new result set.

## Smoke Tests

Use these checks after structural changes:

```bash
python -m py_compile main.py online_baselines_run.py src/*.py src/baselines_new/*.py
python archive/maintenance_scripts/results_manifest.py
python main.py --graph_name MovieLens --method FastPRB --alpha 0.85 --appr_eps 8.33e-05 --T 1 --lr1 0.0073 --lr2 0.0004 --runs 1 --workers 1
python online_baselines_run.py --datasets MovieLens --methods EE-Net --T 1 --runs 1 --workers 1
```

After the final archive, these smoke tests should recreate `results/online_link_prediction/` and `results/baselines_prb_style/` for new runs. They should not write into `results/final/`.

## Archive

Historical scripts, old plot scripts, logs, old baselines, and failed runs are kept under `archive/` for traceability. They are not part of the active experiment surface.
