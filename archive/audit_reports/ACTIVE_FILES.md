# Active Files

This document records the active experiment surface after moving implementation code under `src/`.

## Entry Points

- `main.py`: top-level compatibility wrapper for `src/main.py`.
- `online_baselines_run_new.py`: top-level compatibility wrapper for `src/online_baselines_run_new.py`.
- `src/main.py`: primary online PRB/FastPRB runner. It also supports eps and alpha ablations through command-line arguments.
- `src/online_baselines_run_new.py`: online baseline runner for methods in `src/baselines_new/`.

## Shared Core

- `src/experiment_configs.py`: repository paths, active dataset path metadata, and shared default constants.
- `src/ppr_solver.py`: power iteration and APPR/local push PPR implementations.
- `src/EENet.py`: active EE-Net wrapper used by `src/main.py`.
- `src/EENetClass.py`: active EE-Net network components.
- `src/load_data.py`: active online data loaders for MovieLens, AmazonFashion, Facebook, GrQc, Collab, PPA, and Vessel.
- `src/utils.py`: graph manager classes, graph update logic, kernel-size helper, and result saving helper.

## Baseline Package

`src/online_baselines_run_new.py` imports these modules from `src/baselines_new/`:

- `EENet.py`
- `EENetClass.py`
- `KernelUCB.py`
- `LinUCB.py`
- `NeuralGreedy.py`
- `NeuralNoExplore.py`
- `NeuralTS.py`
- `NeuralUCB.py`
- `packages.py`
- `__init__.py`

## Data And Outputs

- `data/`: local `.npy` data for non-OGB datasets.
- `dataset/`: OGB dataset root.
- `results/`: generated experiment outputs.
- `paper/`: paper PDFs and reference material.

## Archive

`archive/` contains historical files that are intentionally not part of the active experiment surface:

- `archive/old_scripts/`: old runners, debug scripts, and alternative loaders.
- `archive/old_plots/`: old plotting scripts.
- `archive/old_baselines/`: old baseline package.
- `archive/logs/`: historical top-level logs.
- `archive/failed_results/`: failed result directories.
- `archive/audit_reports/`: previous implementation audit notes.

Files in `archive/` should not be imported by active entry points.
