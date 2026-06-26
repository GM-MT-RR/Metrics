"""Refinement stage: correct the residual misalignment the aligner left behind.

After the projective (or RoMa) alignment, a nonlinear local displacement can
remain. A refiner estimates a dense residual flow A->B on the histogram-equalized
grays and remaps ``b_aligned`` by it, pulling B back onto A. Histogram
equalization first because the two views differ in exposure/tone, which violates
the brightness-constancy the flow estimators assume (metric.ipynb cell 8).

Interchangeable refiners:
- ``none``  — pass-through (homography-only baseline).
- ``flow``  — Farneback dense optical flow (cv2, CPU).
- ``raft``  — RAFT (torchvision ``raft_large``), GPU; estimates flow at a fixed
              working resolution then upsamples+rescales it (matching_refinement
              RAFT cell). torch/torchvision lazy-imported; use ``n_workers: 1``.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .context import AlignContext, BaseRefiner, register_refiner


def _equalized_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    u8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.equalizeHist(u8).astype(np.float32) / 255.0


def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """HSV colour-wheel encoding of a flow field (hue=direction, val=magnitude)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0


def _apply_residual_flow(ctx: AlignContext, flow: np.ndarray) -> AlignContext:
    """Remap ``b_aligned`` (and ``valid_mask``) by an A->B residual flow field.

    flow[...,0]=u maps a->b, so sampling b_aligned at (x+u, y+v) brings B onto A.
    Shared by every flow-based refiner; mutates and returns ``ctx``.
    """
    h, w = ctx.a_frame.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    b_flowed = cv2.remap(ctx.b_aligned, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    flow_valid = cv2.remap(ctx.valid_mask.astype(np.float32), map_x, map_y,
                           cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
    ctx.b_aligned = np.clip(b_flowed, 0, 1)
    ctx.valid_mask = ctx.valid_mask & flow_valid
    ctx.debug["residual_flow"] = _flow_to_rgb(flow)
    ctx.info["flow_mag_mean"] = float(np.hypot(flow[..., 0], flow[..., 1]).mean())
    return ctx


@register_refiner("none")
class NoRefiner(BaseRefiner):
    """Homography-only baseline: leave ``b_aligned`` untouched."""

    def refine(self, ctx: AlignContext) -> AlignContext:
        return ctx


@register_refiner("flow")
class FarnebackRefiner(BaseRefiner):
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 6,
        winsize: int = 25,
        iterations: int = 5,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        flags: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pyr_scale=pyr_scale, levels=levels, winsize=winsize,
            iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma,
            flags=flags, **kwargs,
        )

    def refine(self, ctx: AlignContext) -> AlignContext:
        p = self.params
        gray_a = _equalized_gray(ctx.a_frame)
        gray_b = _equalized_gray(ctx.b_aligned)
        flow = cv2.calcOpticalFlowFarneback(
            (gray_a * 255).astype(np.uint8),
            (gray_b * 255).astype(np.uint8),
            None,
            p["pyr_scale"], p["levels"], p["winsize"], p["iterations"],
            p["poly_n"], p["poly_sigma"], p["flags"],
        )
        return _apply_residual_flow(ctx, flow)


@register_refiner("raft")
class RaftRefiner(BaseRefiner):
    def __init__(self, raft_h: int = 520, raft_w: int = 960, **kwargs: Any) -> None:
        super().__init__(raft_h=raft_h, raft_w=raft_w, **kwargs)

    def refine(self, ctx: AlignContext) -> AlignContext:
        import torch
        import torchvision.transforms as T
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        p = self.params
        rh, rw = int(p["raft_h"]), int(p["raft_w"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        h, w = ctx.a_frame.shape[:2]

        def _to_raft_batch(g):
            """Equalized gray (H, W) -> (1, 3, rh, rw) tensor in [-1, 1]."""
            t = torch.from_numpy(np.ascontiguousarray(g, dtype=np.float32))[None, None]
            t = t.repeat(1, 3, 1, 1)
            t = T.Resize(size=(rh, rw), antialias=True)(t)
            return (t - 0.5) / 0.5

        raft_a = _to_raft_batch(_equalized_gray(ctx.a_frame)).to(device)
        raft_b = _to_raft_batch(_equalized_gray(ctx.b_aligned)).to(device)

        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
        with torch.no_grad():
            flow_lr = model(raft_a, raft_b)[-1]                  # (1, 2, rh, rw) A->B px

        flow = torch.nn.functional.interpolate(
            flow_lr, size=(h, w), mode="bilinear", align_corners=False
        )
        flow[:, 0] *= w / rw                                     # dx scaled to full width
        flow[:, 1] *= h / rh                                     # dy scaled to full height
        flow = flow[0].permute(1, 2, 0).cpu().numpy()            # (h, w, 2)
        return _apply_residual_flow(ctx, flow)
