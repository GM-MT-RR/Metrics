import argparse
import sys
import traceback

from ..core import Logger, load_batch_config, run_batch


def add_parser(subparser) -> None:
    parser = subparser.add_parser("batch", help="Run many matchers against the same data + synth.")
    parser.add_argument("-c", "--config", type=str, default="config.yaml",
                        help="Path to the batch YAML config.")
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    Logger.info(f"Loading batch config from '{args.config}'")
    try:
        batch = load_batch_config(args.config)
        run_batch(batch)
    except Exception:
        Logger.error("Batch crashed — aborting.")
        traceback.print_exc()
        sys.exit(1)
