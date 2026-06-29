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

from . import flow as flow_est
from .context import AlignContext, BaseRefiner, register_refiner


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
    ctx.debug["residual_flow"] = flow_est.flow_to_rgb(flow)
    ctx.debug["step__3_b_refined"] = ctx.b_aligned          # actual post-refine B
    ctx.debug["step__3_valid_mask"] = ctx.valid_mask.astype(np.float32)
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
        flow = flow_est.farneback_flow(
            ctx.a_frame, ctx.b_aligned,
            pyr_scale=p["pyr_scale"], levels=p["levels"], winsize=p["winsize"],
            iterations=p["iterations"], poly_n=p["poly_n"], poly_sigma=p["poly_sigma"],
            flags=p["flags"],
        )
        return _apply_residual_flow(ctx, flow)


@register_refiner("raft")
class RaftRefiner(BaseRefiner):
    def __init__(self, raft_h: int = 520, raft_w: int = 960, **kwargs: Any) -> None:
        super().__init__(raft_h=raft_h, raft_w=raft_w, **kwargs)

    def refine(self, ctx: AlignContext) -> AlignContext:
        p = self.params
        flow = flow_est.raft_flow(
            ctx.a_frame, ctx.b_aligned, raft_h=p["raft_h"], raft_w=p["raft_w"]
        )
        return _apply_residual_flow(ctx, flow)
