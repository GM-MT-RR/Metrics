"""Result container for the staged pipeline's patch-extraction stage.

The pipeline (``de_eval.pipeline``) re-aligns the synthetic view onto the source
and the extractor reduces the match to **sparse RGB correspondences**: matched
pixel pairs ΔE is computed on. ``AlignResult`` is that single, sparse output form
the runner consumes — there is no longer a dense / image-aligned variant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AlignResult:
    # Matched RGB pairs ΔE is computed on (every extractor fills these):
    pairs_a: Optional[np.ndarray] = None       # (N, 3) RGB sampled in img_a
    pairs_b: Optional[np.ndarray] = None       # (N, 3) RGB sampled in img_b
    # The pixel COORDS (N, 2) (x, y) each pair was sampled at, in each view.
    coords_a: Optional[np.ndarray] = None
    coords_b: Optional[np.ndarray] = None

    # --- Debug carriers (drive the notebook-style per-pair visuals) ---
    kps_a: Optional[np.ndarray] = None         # matched (row, col) keypoints between
    kps_b: Optional[np.ndarray] = None         # the two views (for the match plot)
    img_a_full: Optional[np.ndarray] = None    # reference view the patches came from
    img_b_full: Optional[np.ndarray] = None    # synthetic view the patches came from

    debug: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> image to save
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return int(self.pairs_a.shape[0]) if self.pairs_a is not None else 0
