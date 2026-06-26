"""Homography(+optical-flow) synthetic-view generator.

Pipeline (extends ``MetricDeProof/synthetic_delta_e.py`` lines 93-104 and the
synthetic section of ``metric.ipynb``):

1. Estimate the iPhone↔Samsung homography ``H`` from AKAZE matches.
2. Warp the source by a *noisy* ``H⁻¹`` to give it the Samsung-like geometry
   (projective misalignment vs. the original source).
3. Optionally add a smooth, low-frequency random displacement field via
   ``cv2.remap`` to introduce *non-linear* local warp (the new optical-flow
   variant the user asked for). ``flow_amplitude_px == 0`` ⇒ pure-homography
   baseline.

The colours are never touched, so corresponding pixels share identical RGB and
ground-truth ΔE is 0.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .. import _paths  # noqa: F401
from thesis_lib_shared.metrics import akaze_keypoint_matches

from .base import BaseSynth, SynthResult
from .registry import register_synth


def _smooth_flow_field(h: int, w: int, amplitude: float, sigma: float,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-smoothed random displacement (dx, dy) in pixels, shape (H, W)."""
    ksize = int(max(3, round(sigma * 3)) | 1)  # odd kernel
    dx = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    dy = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (ksize, ksize), sigma)
    dy = cv2.GaussianBlur(dy, (ksize, ksize), sigma)
    # Normalize so the largest displacement equals `amplitude` px.
    scale = amplitude / (max(np.abs(dx).max(), np.abs(dy).max()) + 1e-8)
    return dx * scale, dy * scale


@register_synth("homography_flow")
class HomographyFlowSynth(BaseSynth):
    def __init__(
        self,
        homography_noise_std: float = 0.05,
        flow_amplitude_px: float = 6.0,
        flow_smooth_sigma: float = 40.0,
        ransac_threshold: float = 1.0,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            homography_noise_std=homography_noise_std,
            flow_amplitude_px=flow_amplitude_px,
            flow_smooth_sigma=flow_smooth_sigma,
            ransac_threshold=ransac_threshold,
            seed=seed,
            **kwargs,
        )

    def synthesize(self, src: np.ndarray, tgt: np.ndarray) -> SynthResult:
        p = self.params
        rng = np.random.default_rng(p["seed"])
        h, w = src.shape[:2]

        # 1. iPhone↔Samsung homography from AKAZE matches (keypoints are row,col).
        kps_a, kps_b = akaze_keypoint_matches(src, tgt, use_ransac=True)
        if kps_a.shape[0] < 4:
            raise RuntimeError("Not enough AKAZE matches to estimate homography for synth.")
        pts_a = kps_a[:, ::-1].astype(np.float32)
        pts_b = kps_b[:, ::-1].astype(np.float32)
        H_mat, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, p["ransac_threshold"])
        if H_mat is None:
            raise RuntimeError("findHomography returned None for synth.")

        # 2. Noisy H⁻¹ warp -> projective misalignment with Samsung-like geometry.
        H_inv = np.linalg.inv(H_mat)
        H_noise = 1.0 + rng.normal(0.0, p["homography_noise_std"], size=H_inv.shape)
        H_inv = H_inv * H_noise
        H_inv /= H_inv[2, 2]

        synth = cv2.warpPerspective(src, H_inv, (w, h))
        valid = cv2.warpPerspective(
            np.ones((h, w), dtype=np.float32), H_inv, (w, h)
        ) > 0.999

        info = {"H": H_mat.tolist(), "H_inv": H_inv.tolist(), "flow": False}

        # 3. Optional smooth optical-flow field -> non-linear local warp.
        if p["flow_amplitude_px"] > 0.0:
            dx, dy = _smooth_flow_field(h, w, p["flow_amplitude_px"], p["flow_smooth_sigma"], rng)
            grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                         np.arange(h, dtype=np.float32))
            map_x = grid_x + dx
            map_y = grid_y + dy
            synth = cv2.remap(synth, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            flow_valid = cv2.remap(valid.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
            valid = valid & flow_valid
            info["flow"] = True
            info["flow_amplitude_px"] = float(p["flow_amplitude_px"])

        return SynthResult(image=np.clip(synth, 0.0, 1.0), valid_mask=valid, info=info)
