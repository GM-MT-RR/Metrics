"""ΔE₀₀ helpers, reusing the numpy CIEDE2000 in ``thesis_lib_shared.metrics``.

Two entry points:
- ``delta_e_dense(img_a, img_b)`` for dense / image-aligned matchers (per-pixel map).
- ``delta_e_pairs(rgb_a, rgb_b)`` for sparse / dense correspondence lists (N×3 RGB).

Both return ΔE in the same units; ``summarize`` gives the distribution stats used
in the report.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .. import _paths  # noqa: F401
from thesis_lib_shared.metrics import (
    delta_e_images,
    delta_e_cie2000,
    rgb_to_lab,
)


def delta_e_dense(img_a: np.ndarray, img_b: np.ndarray, linearize: bool = False) -> np.ndarray:
    """Per-pixel ΔE₀₀ between two aligned (H, W, 3) images in [0, 1]."""
    if linearize:
        lab_a = rgb_to_lab(img_a, linearize=True)
        lab_b = rgb_to_lab(img_b, linearize=True)
        return delta_e_cie2000(lab_a, lab_b)
    return delta_e_images(img_a, img_b)


def delta_e_pairs(rgb_a: np.ndarray, rgb_b: np.ndarray, linearize: bool = False) -> np.ndarray:
    """ΔE₀₀ over matched RGB pairs. Inputs (N, 3) float in [0, 1]; output (N,)."""
    rgb_a = np.asarray(rgb_a, dtype=np.float32).reshape(-1, 3)
    rgb_b = np.asarray(rgb_b, dtype=np.float32).reshape(-1, 3)
    if rgb_a.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    lab_a = rgb_to_lab(rgb_a, linearize=linearize)
    lab_b = rgb_to_lab(rgb_b, linearize=linearize)
    return delta_e_cie2000(lab_a, lab_b).astype(np.float32)


def summarize(de_values: np.ndarray) -> Dict[str, float]:
    """Distribution stats for a ΔE array (empty/NaN-safe)."""
    v = np.asarray(de_values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: float("nan") for k in
                ("mean", "median", "p25", "p75", "p95", "p99", "max", "n")}
    return {
        "mean":   float(np.mean(v)),
        "median": float(np.median(v)),
        "p25":    float(np.percentile(v, 25)),
        "p75":    float(np.percentile(v, 75)),
        "p95":    float(np.percentile(v, 95)),
        "p99":    float(np.percentile(v, 99)),
        "max":    float(np.max(v)),
        "n":      int(v.size),
    }
