"""Keypoint-detector dispatch shared by the keypoint_* matchers.

Returns matched keypoints as two (N, 2) arrays in (row, col) order — the
convention used throughout ``thesis_lib_shared`` and the notebooks.

- ``akaze`` / ``sift``: classical, reuse thesis_lib_shared (work under .venv).
- ``loftr``: kornia LoFTR (lazy import).
- ``roma``:  tiny RoMa, confidence-sampled sparse matches (lazy import).

LoFTR/RoMa import their heavy deps *inside* ``detect_matches`` so classical
configs never trigger them. They require the user-level python 3.13 env where
``kornia`` / ``romatch`` are installed.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .. import _paths  # noqa: F401
from thesis_lib_shared.metrics import akaze_keypoint_matches, sift_keypoint_matches


def _loftr_matches(img_a: np.ndarray, img_b: np.ndarray,
                   resize: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    import kornia as K
    import kornia.feature as KF

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _prep(img: np.ndarray) -> "torch.Tensor":
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * resize), int(h * resize)))
        t = torch.from_numpy(small).permute(2, 0, 1)[None].float().to(device)
        return K.color.rgb_to_grayscale(t)

    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    with torch.inference_mode():
        out = matcher({"image0": _prep(img_a), "image1": _prep(img_b)})
        
    mkpts0 = out["keypoints0"].cpu().numpy() / resize  # back to full-res (x, y)
    mkpts1 = out["keypoints1"].cpu().numpy() / resize
    # (x, y) -> (row, col)
    return mkpts0[:, ::-1].astype(np.float32), mkpts1[:, ::-1].astype(np.float32)


def _roma_matches(img_a: np.ndarray, img_b: np.ndarray,
                  max_keypoints: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    from PIL import Image
    from romatch import tiny_roma_v1_outdoor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _to_pil(img: np.ndarray) -> "Image.Image":
        return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))

    roma = tiny_roma_v1_outdoor(device=device)
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    warp, certainty = roma.match(_to_pil(img_a), _to_pil(img_b), batched=False)
    matches, _ = roma.sample(warp, certainty, num=max_keypoints)
    mkpts0, mkpts1 = roma.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
    mkpts0 = mkpts0.cpu().numpy()
    mkpts1 = mkpts1.cpu().numpy()
    # (x, y) -> (row, col)
    return mkpts0[:, ::-1].astype(np.float32), mkpts1[:, ::-1].astype(np.float32)


def detect_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    detector: str = "sift",
    use_ransac: bool = True,
    ransac_threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return matched (row, col) keypoints for the requested detector."""
    det = detector.lower()
    if det == "akaze":
        return akaze_keypoint_matches(
            img_a, img_b, use_ransac=use_ransac, ransac_threshold=ransac_threshold
        )
    if det == "sift":
        return sift_keypoint_matches(
            img_a, img_b, use_ransac=use_ransac, ransac_threshold=ransac_threshold
        )
    if det == "loftr":
        return _loftr_matches(img_a, img_b)
    if det == "roma":
        return _roma_matches(img_a, img_b)
    raise ValueError(f"Unknown detector '{detector}'. Use akaze | sift | loftr | roma.")
