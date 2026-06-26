"""Compose the three swappable stages into one matcher.

``Pipeline`` runs align -> refine -> extract on a (source, synthetic) view pair
and returns the sparse ``AlignResult`` (matched RGB pairs) the runner computes ΔE
on. It is built from a ``MatcherConfig`` (``align`` / ``refine`` / ``patches``),
each naming a registered component, so any combination is a config choice.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..matchers.base import AlignResult
from .context import get_aligner, get_extractor, get_refiner


class Pipeline:
    def __init__(self, align: dict, refine: dict, patches: dict) -> None:
        self.aligner = get_aligner(align["type"])(**align.get("params", {}))
        self.refiner = get_refiner(refine["type"])(**refine.get("params", {}))
        self.extractor = get_extractor(patches["type"])(**patches.get("params", {}))

    def run(self, img_a: np.ndarray, img_b: np.ndarray,
            valid_mask: Optional[np.ndarray] = None) -> AlignResult:
        ctx = self.aligner.align(img_a, img_b, valid_mask=valid_mask)
        ctx = self.refiner.refine(ctx)
        return self.extractor.extract(ctx)

    @classmethod
    def from_matcher_config(cls, matcher) -> "Pipeline":
        """Build from a ``MatcherConfig`` (pydantic) or an equivalent dict."""
        def _stage(s):
            if isinstance(s, dict):
                return {"type": s["type"], "params": s.get("params", {})}
            return {"type": s.type, "params": s.params}
        return cls(_stage(matcher.align), _stage(matcher.refine), _stage(matcher.patches))
