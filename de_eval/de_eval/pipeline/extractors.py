"""Patch-extraction stage: select the pixels ΔE is computed on, emitting matched
RGB ``pairs_a`` / ``pairs_b`` (N×3).

Two interchangeable extractors, both DENSE (one ΔE per kept pixel):
- ``gradient`` — DENSE per-pixel ΔE (metric.ipynb cell 10): every flat (low
  L*-gradient), valid pixel of the co-registered frame becomes one RGB pair, taken
  directly at that pixel. A and B share coords.
- ``sgbm``     — keypoints -> fundamental matrix -> uncalibrated stereo rectify ->
  SGBM disparity -> dense back-projection of the WHOLE disparity map into the
  original frames (matching_*_SGM cell 13/15): every valid-disparity pixel yields a
  pixel-aligned A/B RGB pair, optionally restricted to flat pixels.
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from ..eval.gradient import build_keep_mask
from ..matchers.base import AlignResult
from ..matchers.keypoints import detect_matches
from .context import AlignContext, BaseExtractor, register_extractor


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    return (np.clip(g, 0, 1) * 255).astype(np.uint8)


@register_extractor("gradient")
class GradientExtractor(BaseExtractor):
    def __init__(
        self,
        resize: float = 0.10,
        gradient_threshold: float = 25.0,
        **kwargs: Any,
    ) -> None:
        # window_radius / grid_k / max_pairs / seed are accepted (via **kwargs)
        # for backward-compatible configs but ignored: ΔE is dense per-pixel.
        super().__init__(
            resize=resize, gradient_threshold=gradient_threshold, **kwargs,
        )

    def extract(self, ctx: AlignContext) -> AlignResult:
        p = self.params
        h, w = ctx.a_frame.shape[:2]
        r = float(p["resize"])
        new_w, new_h = max(1, int(round(w * r))), max(1, int(round(h * r)))
        a = cv2.resize(ctx.a_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(ctx.b_aligned, (new_w, new_h), interpolation=cv2.INTER_AREA)
        v = cv2.resize(ctx.valid_mask.astype(np.float32), (new_w, new_h),
                       interpolation=cv2.INTER_NEAREST) > 0.999

        keep = build_keep_mask(a, b, v, p["gradient_threshold"])
        ys, xs = np.nonzero(keep)
        if ys.size == 0:
            return _empty_result(ctx, p["gradient_threshold"])

        # Dense per-pixel ΔE (metric.ipynb cell 10): every kept pixel's RGB is a
        # pair directly — no window/grid sampling. A and B are co-registered, so
        # the same (x, y) indexes both.
        pairs_a = a[ys, xs].astype(np.float32)                      # (N, 3)
        pairs_b = b[ys, xs].astype(np.float32)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)      # (N, 2) (x, y)

        overlay = a.copy()
        overlay[~keep] *= 0
        return AlignResult(
            pairs_a=pairs_a, pairs_b=pairs_b, coords_a=coords, coords_b=coords,
            img_a_full=a, img_b_full=b,
            kps_a=ctx.kps_a, kps_b=ctx.kps_b,
            debug={**ctx.debug,
                   "gradient_keep": overlay.astype(np.float32),
                   # actual images the ΔE is computed on (resized A/B + keep mask)
                   "step__4_a_resized": a.astype(np.float32),
                   "step__4_b_resized": b.astype(np.float32),
                   "step__5_keep_mask": keep.astype(np.float32),
                   "step__6_a_kept": overlay.astype(np.float32)},
            info={**ctx.info, "n_pairs": int(pairs_a.shape[0]),
                  "gradient_threshold": p["gradient_threshold"]},
        )


@register_extractor("sgbm")
class SgbmExtractor(BaseExtractor):
    def __init__(
        self,
        detector: str = "akaze",
        min_disparity: int = -64,
        num_disparities: int = 192,
        block_size: int = 7,
        uniqueness_ratio: int = 15,
        speckle_window_size: int = 150,
        speckle_range: int = 2,
        disp12_max_diff: int = 1,
        clahe_clip: float = 2.0,
        fundamental_threshold: float = 1.0,
        gradient_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        # window_radius / grid_k / max_pairs / seed are accepted (via **kwargs) for
        # backward-compatible configs but ignored: ΔE is dense over the disparity
        # map (notebook matching_*_SGM cell 13/15), not sparse keypoint windows.
        # gradient_threshold (optional) additionally restricts to flat pixels.
        super().__init__(
            detector=detector, min_disparity=min_disparity,
            num_disparities=num_disparities, block_size=block_size,
            uniqueness_ratio=uniqueness_ratio, speckle_window_size=speckle_window_size,
            speckle_range=speckle_range, disp12_max_diff=disp12_max_diff,
            clahe_clip=clahe_clip, fundamental_threshold=fundamental_threshold,
            gradient_threshold=gradient_threshold, **kwargs,
        )

    def _make_sgbm(self):
        p = self.params
        bs = int(p["block_size"])
        return cv2.StereoSGBM_create(
            minDisparity=int(p["min_disparity"]),
            numDisparities=int(p["num_disparities"]),
            blockSize=bs,
            P1=8 * bs * bs,
            P2=64 * bs * bs,
            disp12MaxDiff=int(p["disp12_max_diff"]),
            uniquenessRatio=int(p["uniqueness_ratio"]),
            speckleWindowSize=int(p["speckle_window_size"]),
            speckleRange=int(p["speckle_range"]),
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_HH,
        )

    def extract(self, ctx: AlignContext) -> AlignResult:
        p = self.params
        img_a, img_b = ctx.a_frame, ctx.b_aligned
        h, w = img_a.shape[:2]

        kps_a, kps_b = detect_matches(
            img_a, img_b, detector=p["detector"],
            use_ransac=False, ransac_threshold=p["fundamental_threshold"],
        )
        if kps_a.shape[0] < 8:
            raise RuntimeError(f"{p['detector']} produced <8 matches; cannot fit F.")
        pts_a = kps_a[:, ::-1].astype(np.float32)  # (x, y)
        pts_b = kps_b[:, ::-1].astype(np.float32)

        F, mask = cv2.findFundamentalMat(
            pts_a, pts_b, cv2.USAC_MAGSAC,
            ransacReprojThreshold=p["fundamental_threshold"],
            confidence=0.99999, maxIters=100000,
        )
        if F is None:
            raise RuntimeError("findFundamentalMat returned None.")
        if mask is not None:
            inl = mask.ravel().astype(bool)
            pts_a, pts_b = pts_a[inl], pts_b[inl]

        ok, H1, H2 = cv2.stereoRectifyUncalibrated(
            pts_a.reshape(-1, 1, 2), pts_b.reshape(-1, 1, 2), F, imgSize=(w, h)
        )
        if not ok:
            raise RuntimeError("stereoRectifyUncalibrated failed.")

        clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=(8, 8))
        rect_a = clahe.apply(cv2.warpPerspective(_to_gray_u8(img_a), H1, (w, h)))
        rect_b = clahe.apply(cv2.warpPerspective(_to_gray_u8(img_b), H2, (w, h)))

        disparity = self._make_sgbm().compute(rect_a, rect_b).astype(np.float32) / 16.0
        min_disp = int(p["min_disparity"])
        disp_vis = np.clip((disparity - min_disp) / max(p["num_disparities"], 1), 0, 1)

        # --- Dense, pixel-aligned correspondences (notebook matching_*_SGM cell 13) ---
        # disparity lives in the RECTIFIED-A frame; SGM treats rect-A as LEFT, so
        # disparity = x_A - x_B  =>  rect-A (x, y) <-> rect-B (x - d, y). Invert the
        # rectifying homographies to bring each endpoint back to the ORIGINAL frames
        # and remap the colour images -> two full images aligned pixel-for-pixel.
        H1inv, H2inv = np.linalg.inv(H1), np.linalg.inv(H2)
        ys_grid, xs_grid = np.indices((h, w), dtype=np.float32)
        valid = disparity >= min_disp                       # SGM marks invalid as < min_disp

        def _apply_H(Hm, x, y):
            den = Hm[2, 0] * x + Hm[2, 1] * y + Hm[2, 2]
            u = (Hm[0, 0] * x + Hm[0, 1] * y + Hm[0, 2]) / den
            v = (Hm[1, 0] * x + Hm[1, 1] * y + Hm[1, 2]) / den
            return np.ascontiguousarray(u, np.float32), np.ascontiguousarray(v, np.float32)

        mapAx, mapAy = _apply_H(H1inv, xs_grid, ys_grid)            # rect-A -> orig-A
        mapBx, mapBy = _apply_H(H2inv, xs_grid - disparity, ys_grid)  # rect-B(x-d) -> orig-B

        in_a = (mapAx >= 0) & (mapAx <= w - 1) & (mapAy >= 0) & (mapAy <= h - 1)
        in_b = (mapBx >= 0) & (mapBx <= w - 1) & (mapBy >= 0) & (mapBy <= h - 1)
        paired = valid & in_a & in_b

        # The mask carried through the pipeline (B-validity + warp/flow validity) is
        # in A's frame; bring it into the rectified-A frame to drop known-wrong B
        # pixels from the dense ΔE as well.
        vm_rect = cv2.warpPerspective(ctx.valid_mask.astype(np.float32), H1, (w, h),
                                      flags=cv2.INTER_NEAREST) > 0.5
        paired &= vm_rect

        # Dense colour images, remapped into the rectified-A frame, then sampled.
        patches_a = cv2.remap(np.clip(img_a, 0, 1), mapAx, mapAy, cv2.INTER_LINEAR)
        patches_b = cv2.remap(np.clip(img_b, 0, 1), mapBx, mapBy, cv2.INTER_LINEAR)

        # Optional flat-pixel restriction (config: gradient_threshold). Notebook SGM
        # has no gradient filter; off by default (None) keeps the notebook behaviour.
        if p.get("gradient_threshold") is not None:
            paired &= build_keep_mask(patches_a, patches_b, paired, float(p["gradient_threshold"]))

        ys_k, xs_k = np.nonzero(paired)
        if ys_k.size == 0:
            return _empty_result(ctx, info_extra={"detector": p["detector"], "n_disp": 0})

        pairs_a = patches_a[ys_k, xs_k].astype(np.float32)         # (N, 3)
        pairs_b = patches_b[ys_k, xs_k].astype(np.float32)
        coords = np.stack([xs_k, ys_k], axis=1).astype(np.float32)  # rect-A frame (x, y)

        a_kept = patches_a.copy(); a_kept[~paired] = 0.0
        b_kept = patches_b.copy(); b_kept[~paired] = 0.0
        return AlignResult(
            pairs_a=pairs_a, pairs_b=pairs_b, coords_a=coords, coords_b=coords,
            img_a_full=patches_a.astype(np.float32), img_b_full=patches_b.astype(np.float32),
            kps_a=ctx.kps_a, kps_b=ctx.kps_b,
            debug={**ctx.debug,
                   "sgbm_disparity": disp_vis.astype(np.float32),
                   # actual per-stage images for the steps/ dump (dense extractor):
                   "step__4_a_frame": np.clip(img_a, 0, 1).astype(np.float32),
                   "step__4_b_aligned": np.clip(img_b, 0, 1).astype(np.float32),
                   "step__4_disparity": disp_vis.astype(np.float32),
                   "step__5_valid_mask": paired.astype(np.float32),
                   # the dense pixel-aligned colour images ΔE is actually computed on
                   "step__6_a_patches": a_kept.astype(np.float32),
                   "step__6_b_patches": b_kept.astype(np.float32)},
            info={**ctx.info, "detector": p["detector"],
                  "n_pairs": int(pairs_a.shape[0])},
        )


def _empty_result(ctx: AlignContext, gradient_threshold: Optional[float] = None,
                  info_extra: Optional[dict] = None) -> AlignResult:
    z3 = np.zeros((0, 3), np.float32)
    z2 = np.zeros((0, 2), np.float32)
    info = {**ctx.info, "n_centers": 0, "n_pairs": 0}
    if gradient_threshold is not None:
        info["gradient_threshold"] = gradient_threshold
    if info_extra:
        info.update(info_extra)
    return AlignResult(
        pairs_a=z3, pairs_b=z3, coords_a=z2, coords_b=z2,
        img_a_full=ctx.a_frame, img_b_full=ctx.b_aligned,
        kps_a=ctx.kps_a, kps_b=ctx.kps_b, debug=dict(ctx.debug), info=info,
    )
