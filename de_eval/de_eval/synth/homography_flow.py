"""Homography(+measured-optical-flow) synthetic-view generator.

Pipeline (extends ``MetricDeProof/synthetic_delta_e.py`` lines 93-104 and the
synthetic section of ``metric.ipynb``):

1. Estimate the iPhone↔Samsung homography ``H`` from AKAZE matches.
2. Warp the source by a *noisy* ``H⁻¹`` to give it the Samsung-like geometry
   (projective misalignment vs. the original source).
3. Optionally add a *measured* dense optical flow (RAFT or Farneback) estimated
   from the **warped source -> target (Samsung)** as a non-linear local warp via
   ``cv2.remap``. The flow is estimated on the *warped* source so it lives in the
   same grid it is remapped on — estimating it from the original source would mix
   two grids and warp the wrong pixels. ``flow_backend == "none"`` ⇒
   pure-homography baseline.

The colours are never touched, so corresponding pixels share identical RGB and
ground-truth ΔE is 0; the residual ΔE a matcher reports is its own registration
error against the real Samsung deformation.

RAFT runs on the GPU and lazy-imports torch/torchvision; use ``n_workers: 1`` in
the config (same constraint as the RAFT refiner — the runner runs inline then).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .. import _paths  # noqa: F401
from thesis_lib_shared.metrics import akaze_keypoint_matches

from ..pipeline import flow as flow_est
from .base import BaseSynth, SynthResult
from .registry import register_synth


@register_synth("homography_flow")
class HomographyFlowSynth(BaseSynth):
    def __init__(
        self,
        homography_noise_std: float = 0.05,
        flow_backend: str = "raft",          # "raft" | "farneback" | "none"
        raft_h: int = 520,
        raft_w: int = 960,
        farneback_params: dict | None = None,
        ransac_threshold: float = 1.0,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            homography_noise_std=homography_noise_std,
            flow_backend=flow_backend,
            raft_h=raft_h,
            raft_w=raft_w,
            farneback_params=farneback_params or {},
            ransac_threshold=ransac_threshold,
            seed=seed,
            **kwargs,
        )

    def _measure_flow(self, warped_src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """Dense flow warped-source -> target, in the warped-source grid."""
        backend = self.params["flow_backend"]
        if backend == "raft":
            return flow_est.raft_flow(
                warped_src, tgt,
                raft_h=self.params["raft_h"], raft_w=self.params["raft_w"],
            )
        if backend == "farneback":
            return flow_est.farneback_flow(warped_src, tgt, **self.params["farneback_params"])
        raise ValueError(
            f"Unknown flow_backend '{backend}'. Use 'raft', 'farneback', or 'none'."
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

        info = {"H": H_mat.tolist(), "H_inv": H_inv.tolist(),
                "flow_backend": p["flow_backend"]}

        # 3. Measured optical-flow field (warped source -> target) -> non-linear
        #    local warp. Estimated on `synth` so the flow lives in the grid it is
        #    remapped on.
        if p["flow_backend"] != "none":
            flow = self._measure_flow(synth, tgt)
            grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                         np.arange(h, dtype=np.float32))
            map_x = grid_x + flow[..., 0]
            map_y = grid_y + flow[..., 1]
            synth = cv2.remap(synth, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            flow_valid = cv2.remap(valid.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
            valid = valid & flow_valid
            info["flow_mag_mean"] = float(np.hypot(flow[..., 0], flow[..., 1]).mean())

        return SynthResult(image=np.clip(synth, 0.0, 1.0), valid_mask=valid, info=info)


@register_synth("homography")
class HomographySynth(HomographyFlowSynth):
    """Homography-only synthetic view: the projective misalignment (steps 1-2) with
    NO non-linear flow (step 3). Pure-homography baseline — pins ``flow_backend``
    to ``"none"`` so ``synthesize`` skips the optical-flow remap regardless of config."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["flow_backend"] = "none"      # force homography-only
        super().__init__(**kwargs)
