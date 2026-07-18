"""Run one selected B1-B5 combination using the controlled Test A harness."""

import argparse
import os
import sys


TESTB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTB_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _extract_combination(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--combination", choices=[f"B{i}" for i in range(1, 6)])
    known, remaining = parser.parse_known_args(argv[1:])
    argv[:] = [argv[0], *remaining]
    combination = known.combination or os.environ.get("TESTB_COMBINATION")
    if combination not in {f"B{i}" for i in range(1, 6)}:
        parser.error("--combination B1...B5 is required")
    os.environ["TESTB_COMBINATION"] = combination
    return combination


COMBINATION = _extract_combination(sys.argv)

# Import the Test A harness in its unmodified A0 mode. It provides the isolated
# loader RNG and the individual, already-tested installers used below.
os.environ["TESTA_ABLATION"] = "A0"
from testA import run_ablation as harness


COMBINATIONS = {
    "B1": ("A6",),
    "B2": ("A1", "A6"),
    "B3": ("A3", "A6"),
    "B4": ("A1", "A3", "A6"),
    "B5": ("A1", "A3", "A6", "A8"),
}

for change in COMBINATIONS[COMBINATION]:
    harness.INSTALLERS[change]()

RESULTS_ROOT = os.path.join(TESTB_DIR, "results", COMBINATION)
harness.base_main.RESULTS_DIR = RESULTS_ROOT


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    harness.base_main.main()


if __name__ == "__main__":
    main()
