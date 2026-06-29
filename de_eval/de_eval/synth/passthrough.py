"""Pass-through synthetic-view generator.

For datasets where the distorted view is *real* and already on disk (e.g. the
simulatron iphone-s / iphone-e PNGs: iPhone-X colours forward-warped into another
viewpoint), there is nothing to synthesise. This synth returns the target view
unchanged as the "synthetic" view, so the matcher is tested against the real warp
instead of a fabricated homography+flow.

The valid mask marks pixels that hold real content. The simulatron writes holes as
exactly 0 (no inpaint), so any pixel that is non-zero in any channel is treated as
valid; fully-dense targets therefore yield an all-True mask.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseSynth, SynthResult
from .registry import register_synth


@register_synth("pass")
class PassthroughSynth(BaseSynth):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def synthesize(self, src: np.ndarray, tgt: np.ndarray) -> SynthResult:
        valid_mask = np.any(tgt > 0.0, axis=-1)
        return SynthResult(
            image=tgt.astype(np.float32),
            valid_mask=valid_mask,
            info={"synth": "pass"},
        )
