"""Shared dense optical-flow estimation (Farneback / RAFT).

Pure estimators: ``(img_a, img_b) -> (H, W, 2)`` float32 displacement field in
A's pixel grid, direction A->B (``flow[...,0]=u``, sampling B at ``(x+u, y+v)``
brings B onto A). Histogram equalization first because the two views differ in
exposure/tone, which violates the brightness-constancy the flow estimators assume
(metric.ipynb cell 8).

Used by both ``pipeline/refiners.py`` (residual re-alignment of ``b_aligned``)
and ``synth/homography_flow.py`` (measured nonlinear warp of the synthetic view),
so the RAFT recipe lives in exactly one place.
"""
from __future__ import annotations

import cv2
import numpy as np


def equalized_gray(img: np.ndarray) -> np.ndarray:
    """RGB float [0,1] -> histogram-equalized grayscale float [0,1]."""
    gray = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    u8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.equalizeHist(u8).astype(np.float32) / 255.0


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """HSV colour-wheel encoding of a flow field (hue=direction, val=magnitude)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0


def farneback_flow(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 6,
    winsize: int = 25,
    iterations: int = 5,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
) -> np.ndarray:
    """Dense Farneback optical flow A->B (cv2, CPU). Returns (H, W, 2) float32."""
    gray_a = equalized_gray(img_a)
    gray_b = equalized_gray(img_b)
    flow = cv2.calcOpticalFlowFarneback(
        (gray_a * 255).astype(np.uint8),
        (gray_b * 255).astype(np.uint8),
        None,
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags,
    )
    return flow.astype(np.float32)


def raft_flow(
    img_a: np.ndarray,
    img_b: np.ndarray,
    raft_h: int = 520,
    raft_w: int = 960,
) -> np.ndarray:
    """Dense RAFT optical flow A->B (torchvision ``raft_large``, GPU).

    Estimates at a fixed working resolution ``(raft_h, raft_w)`` then upsamples +
    rescales the flow back to A's full resolution. torch/torchvision are
    lazy-imported; the GPU model means callers should use a single worker.
    Returns (H, W, 2) float32 in A's pixel grid.
    """
    import torch
    import torchvision.transforms as T
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = img_a.shape[:2]
    rh, rw = int(raft_h), int(raft_w)

    def _to_raft_batch(g):
        """Equalized gray (H, W) -> (1, 3, rh, rw) tensor in [-1, 1]."""
        t = torch.from_numpy(np.ascontiguousarray(g, dtype=np.float32))[None, None]
        t = t.repeat(1, 3, 1, 1)
        t = T.Resize(size=(rh, rw), antialias=True)(t)
        return (t - 0.5) / 0.5

    raft_a = _to_raft_batch(equalized_gray(img_a)).to(device)
    raft_b = _to_raft_batch(equalized_gray(img_b)).to(device)

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
    with torch.no_grad():
        flow_lr = model(raft_a, raft_b)[-1]                  # (1, 2, rh, rw) A->B px

    flow = torch.nn.functional.interpolate(
        flow_lr, size=(h, w), mode="bilinear", align_corners=False
    )
    flow[:, 0] *= w / rw                                     # dx scaled to full width
    flow[:, 1] *= h / rh                                     # dy scaled to full height
    return flow[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (h, w, 2)
