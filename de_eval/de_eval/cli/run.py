import argparse
import sys
import traceback

from ..core import Logger, load_config, run_experiment


def add_parser(subparser) -> None:
    parser = subparser.add_parser("run", help="Run a single ΔE-benchmark experiment from a YAML config.")
    parser.add_argument("-c", "--config", type=str, default="config.yaml",
                        help="Path to the experiment YAML config.")
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    Logger.info(f"Loading config from '{args.config}'")
    try:
        config = load_config(args.config)
        run_experiment(config)
    except Exception:
        Logger.error("Run crashed — aborting.")
        traceback.print_exc()
        sys.exit(1)
