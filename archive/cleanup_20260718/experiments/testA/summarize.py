"""Summarize final_results.npy files produced by the A0-A10 experiments."""

import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()

    print("Ablation\tRuns\tT\tFinalRegretMean\tFinalRegretStd\tTimeMeanSec\tTimeStdSec\tPath")
    for index in range(11):
        ablation = f"A{index}"
        pattern = os.path.join(args.root, ablation, "**", "final_results.npy")
        for path in sorted(glob.glob(pattern, recursive=True)):
            data = np.load(path)
            final_regret = data[:, -1, 1]
            total_time = data[:, :, 0].sum(axis=1)
            print(
                f"{ablation}\t{data.shape[0]}\t{data.shape[1]}\t"
                f"{final_regret.mean():.3f}\t{final_regret.std():.3f}\t"
                f"{total_time.mean():.3f}\t{total_time.std():.3f}\t{path}"
            )


if __name__ == "__main__":
    main()
