"""Flat-pixel selection via L* gradient magnitude.

Ported from ``MetricDeProof/synthetic_delta_e.py`` (``gradient_magnitude``,
lines 43-47) and ``metric.ipynb`` cell 9. ΔE on textured/edge pixels is
dominated by residual misregistration; restricting to low-gradient (flat)
pixels isolates the genuine colour difference, which for a same-colour
synthetic pair should be ≈ 0.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import _paths  # noqa: F401
from thesis_lib_shared.metrics import rgb_to_lab


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of the L* channel (0-100 scale)."""
    l_channel = rgb_to_lab(img)[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def build_keep_mask(
    img_a: np.ndarray,
    img_b: np.ndarray,
    valid_mask: np.ndarray,
    gradient_threshold: float,
) -> np.ndarray:
    """Pixels that are valid (inside both warps) AND flat in both images."""
    grad = np.maximum(gradient_magnitude(img_a), gradient_magnitude(img_b))
    return (grad < gradient_threshold) & valid_mask
