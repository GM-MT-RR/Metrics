import argparse

from ..core import Logger
from ..synth import SYNTH_REGISTRY
from ..pipeline import ALIGNER_REGISTRY, REFINER_REGISTRY, EXTRACTOR_REGISTRY


def add_parser(subparser) -> None:
    parser = subparser.add_parser(
        "list", help="List registered synthesizers and the three pipeline stages.")
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    Logger.info("[bold]Synthesizers[/bold]: " + ", ".join(sorted(SYNTH_REGISTRY)))
    Logger.info("[bold]Aligners (stage 1)[/bold]: " + ", ".join(sorted(ALIGNER_REGISTRY)))
    Logger.info("[bold]Refiners (stage 2)[/bold]: " + ", ".join(sorted(REFINER_REGISTRY)))
    Logger.info("[bold]Extractors (stage 3)[/bold]: " + ", ".join(sorted(EXTRACTOR_REGISTRY)))
    Logger.info("Detectors (keypoint_homography align / sgbm extract): akaze, sift, loftr, roma")
