"""Backward-compatible entry point for the integrated DYN-LocPRB runner."""

import sys

from src import main as base_main


def main():
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "dyn_locPRB"])
    else:
        index = sys.argv.index("--method") + 1
        if sys.argv[index] in ("FastPRB", "LocPRB"):
            sys.argv[index] = "dyn_locPRB"
    base_main.main()


if __name__ == "__main__":
    main()
