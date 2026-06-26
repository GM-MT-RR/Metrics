"""de_eval CLI — modular ΔE matching benchmark.

Usage:
  python main.py run   -c configs/homflow_sift_flow.yaml
  python main.py batch -c configs/all_matchers_Batch.yaml
  python main.py list
"""
import argparse
import sys
import warnings
from pathlib import Path

# Allow `python main.py ...` from anywhere: put this dir on sys.path so the
# `de_eval` package imports cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from de_eval import cli
from de_eval.core import Logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("de_eval CLI")
    subparser = parser.add_subparsers(title="commands", required=True)
    cli.register_parsers(subparser)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.func(args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        main()
    except KeyboardInterrupt:
        Logger.warn("Interrupted by user.")
