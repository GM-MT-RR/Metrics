"""Inter-stage carrier + the three component registries for the staged pipeline.

The ΔE matcher is decomposed into three independently swappable stages, run in
order by ``Pipeline`` (``runner_pipeline.py``):

    align  -> refine -> extract_patches

Each stage is a registered component (``@register_aligner`` / ``@register_refiner``
/ ``@register_extractor``) selected by name from config, so any combination is a
YAML edit rather than a new class. State flows between stages through one mutable
``AlignContext``:

- ``align``   fills ``a_frame`` / ``b_aligned`` / ``valid_mask`` (and ``certainty``
              for dense aligners) plus the debug carriers (``kps_*``, ``img_*_full``).
- ``refine``  rewrites ``b_aligned`` / ``valid_mask`` in place (residual-flow remap).
- ``extract`` reads the context and produces the matched RGB ``pairs_a`` / ``pairs_b``
              (N×3) the ΔE is computed on — this is what the runner consumes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..core.registry import make_registry

ALIGNER_REGISTRY, register_aligner, get_aligner = make_registry("aligner")
REFINER_REGISTRY, register_refiner, get_refiner = make_registry("refiner")
EXTRACTOR_REGISTRY, register_extractor, get_extractor = make_registry("extractor")


@dataclass
class AlignContext:
    """Mutable state threaded through align -> refine -> extract."""
    # Working frame (both co-registered, (H, W, 3) RGB in [0, 1]).
    a_frame: np.ndarray
    b_aligned: np.ndarray
    valid_mask: np.ndarray                      # (H, W) bool: valid overlap region
    certainty: Optional[np.ndarray] = None      # (H, W) confidence (dense aligners)

    # --- Debug carriers (drive the notebook-style per-pair visuals) ---
    kps_a: Optional[np.ndarray] = None          # matched (row, col) keypoints in the
    kps_b: Optional[np.ndarray] = None          # two FULL-res views (for the match plot)
    img_a_full: Optional[np.ndarray] = None     # reference view used for matching
    img_b_full: Optional[np.ndarray] = None     # synthetic view used for matching

    debug: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> image to save
    info: Dict[str, Any] = field(default_factory=dict)


class BaseAligner(ABC):
    """Stage 1: feature/keypoint extraction + geometry -> co-registered frame."""
    registered_name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)

    @abstractmethod
    def align(self, img_a: np.ndarray, img_b: np.ndarray,
              valid_mask: Optional[np.ndarray] = None) -> AlignContext:
        ...


class BaseRefiner(ABC):
    """Stage 2: the refinement process (residual-flow remap of ``b_aligned``)."""
    registered_name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)

    @abstractmethod
    def refine(self, ctx: AlignContext) -> AlignContext:
        ...


class BaseExtractor(ABC):
    """Stage 3: patch extraction -> matched RGB ``pairs_a`` / ``pairs_b`` (N×3)."""
    registered_name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)

    @abstractmethod
    def extract(self, ctx: AlignContext) -> "AlignResult":  # noqa: F821
        ...
