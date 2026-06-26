"""Staged ΔE matcher: align -> refine -> extract_patches.

Each stage is an independently swappable, registered component:

- aligners   (``aligners.py``)   : keypoint_homography | roma_dense
- refiners   (``refiners.py``)   : none | flow | raft
- extractors (``extractors.py``) : gradient | sgbm

Importing the stage modules here runs their ``@register_*`` decorators so the
registries are populated at import time (same pattern as ``matchers/__init__.py``).
Build a runnable matcher from config with ``Pipeline.from_matcher_config``.
"""
from .context import (  # noqa: F401
    AlignContext,
    BaseAligner, BaseRefiner, BaseExtractor,
    ALIGNER_REGISTRY, REFINER_REGISTRY, EXTRACTOR_REGISTRY,
    get_aligner, get_refiner, get_extractor,
    register_aligner, register_refiner, register_extractor,
)
from . import aligners      # noqa: F401  (registers keypoint_homography, roma_dense)
from . import refiners      # noqa: F401  (registers none, flow, raft)
from . import extractors    # noqa: F401  (registers gradient, sgbm)
from .runner_pipeline import Pipeline  # noqa: F401

__all__ = [
    "AlignContext", "Pipeline",
    "ALIGNER_REGISTRY", "REFINER_REGISTRY", "EXTRACTOR_REGISTRY",
    "get_aligner", "get_refiner", "get_extractor",
    "register_aligner", "register_refiner", "register_extractor",
]
