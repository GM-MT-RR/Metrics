"""Base class + result container for synthetic-view generators.

A synthesizer takes the source view (iPhone colours) and the target view
(Samsung pose) and produces a *misaligned-but-same-colour* synthetic view of the
source. Because the synthetic view is a pure geometric warp of the source, the
ground-truth ΔE between corresponding pixels is exactly 0 — so any ΔE a matcher
reports is its own registration error.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class SynthResult:
    """Output of a synthesizer.

    Attributes:
        image:      synthetic view, float32 (H, W, 3) in [0, 1] (same colours as src).
        valid_mask: bool (H, W); True where the synthetic warp produced real content.
        info:       free-form metadata (e.g. the homography used), for debugging.
    """
    image: np.ndarray
    valid_mask: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)


class BaseSynth(ABC):
    registered_name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)

    @abstractmethod
    def synthesize(self, src: np.ndarray, tgt: np.ndarray) -> SynthResult:
        """Return a synthetic same-colour-as-``src`` view, geometrically warped."""
        ...

    def summary(self) -> Dict[str, Any]:
        return {"synth": self.registered_name, "params": self.params}
