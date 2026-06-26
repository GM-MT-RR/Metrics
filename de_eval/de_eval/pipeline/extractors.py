"""Patch-extraction stage: pick matched centers, then sample a sub-pixel window
around each one, emitting matched RGB ``pairs_a`` / ``pairs_b`` (N×3) for ΔE.

Every center is expanded into a ``grid_k × grid_k`` grid of sub-pixel offsets over
a window of radius ``window_radius`` px (the shared ``sample_windows`` helper); each
grid point becomes its own RGB pair (denser, more stable colour sampling than a
single pixel). So a method that picks ``C`` centers yields ``C * grid_k**2`` pairs.

Two interchangeable extractors, differing only in how they pick the centers:
- ``gradient`` — flat (low L*-gradient) pixels of the co-registered frame; A and B
  share the same center coords (build_keep_mask, reused from eval.gradient).
- ``sgbm``     — keypoints -> fundamental matrix -> uncalibrated stereo rectify ->
  SGBM disparity -> back-project to matched centers; A and B centers differ, so the
  identical window offsets are applied around each independently.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from ..eval.gradient import build_keep_mask
from ..matchers.base import AlignResult
from ..matchers.keypoints import detect_matches
from .context import AlignContext, BaseExtractor, register_extractor


def sample_windows(
    img_a: np.ndarray,
    centers_a: np.ndarray,
    img_b: np.ndarray,
    centers_b: np.ndarray,
    window_radius: float,
    grid_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sub-pixel ``grid_k × grid_k`` window around each (paired) center.

    centers_a / centers_b are (C, 2) (x, y). For each center, lay a grid of
    offsets spanning [-window_radius, +window_radius] in x and y, bilinearly
    sample img_a at centers_a + offsets and img_b at centers_b + offsets, and
    return one RGB pair per grid point.

    Returns (pairs_a, pairs_b, coords_a, coords_b), each (C*grid_k**2, ·).
    With grid_k == 1 the window collapses to the centers themselves.
    """
    k = max(1, int(grid_k))
    offs = (np.array([0.0]) if k == 1
            else np.linspace(-window_radius, window_radius, k)).astype(np.float32)
    ox, oy = np.meshgrid(offs, offs)                      # (k, k)
    ox = ox.ravel()[None, :]                              # (1, k*k)
    oy = oy.ravel()[None, :]

    # cv2.remap caps each map dimension at SHRT_MAX, so a single (N, 1) column
    # overflows once C*k*k > 32767. Reshape the N samples into a near-square 2D
    # map (pad to a full last row, remap, then trim back to N).
    MAXDIM = 32000

    def _remap_flat(img, xs_flat, ys_flat):
        n = xs_flat.shape[0]
        width = min(MAXDIM, n) if n > 0 else 1
        rows = int(np.ceil(n / width))
        pad = rows * width - n
        xs2 = np.pad(xs_flat, (0, pad)).reshape(rows, width).astype(np.float32)
        ys2 = np.pad(ys_flat, (0, pad)).reshape(rows, width).astype(np.float32)
        out = cv2.remap(img, xs2, ys2, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)            # (rows, width, 3)
        return out.reshape(-1, 3)[:n].astype(np.float32)

    def _sample(img, centers):
        xs = (centers[:, 0:1] + ox).reshape(-1).astype(np.float32)  # (C*k*k,)
        ys = (centers[:, 1:2] + oy).reshape(-1).astype(np.float32)
        cols = _remap_flat(img, xs, ys)                            # (C*k*k, 3)
        coords = np.stack([xs, ys], axis=1)                       # (C*k*k, 2)
        return cols, coords.astype(np.float32)

    pairs_a, coords_a = _sample(img_a, centers_a.astype(np.float32))
    pairs_b, coords_b = _sample(img_b, centers_b.astype(np.float32))
    return pairs_a, pairs_b, coords_a, coords_b


def _subsample_centers(centers: np.ndarray, n_keep: int, seed: int) -> np.ndarray:
    if centers.shape[0] <= n_keep:
        return centers
    idx = np.random.default_rng(seed).choice(centers.shape[0], n_keep, replace=False)
    return centers[idx]


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    return (np.clip(g, 0, 1) * 255).astype(np.uint8)


@register_extractor("gradient")
class GradientExtractor(BaseExtractor):
    def __init__(
        self,
        resize: float = 0.10,
        gradient_threshold: float = 25.0,
        window_radius: float = 2.0,
        grid_k: int = 3,
        max_pairs: int = 50000,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            resize=resize, gradient_threshold=gradient_threshold,
            window_radius=window_radius, grid_k=grid_k,
            max_pairs=max_pairs, seed=seed, **kwargs,
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

        centers = np.stack([xs, ys], axis=1).astype(np.float32)     # (C, 2) (x, y)
        n_centers_cap = max(1, int(p["max_pairs"]) // max(1, int(p["grid_k"]) ** 2))
        centers = _subsample_centers(centers, n_centers_cap, int(p["seed"]))

        # A and B are co-registered here, so the same center serves both views.
        pairs_a, pairs_b, coords_a, coords_b = sample_windows(
            a, centers, b, centers, p["window_radius"], p["grid_k"]
        )

        overlay = a.copy()
        overlay[~keep] *= 0
        return AlignResult(
            pairs_a=pairs_a, pairs_b=pairs_b, coords_a=coords_a, coords_b=coords_b,
            img_a_full=a, img_b_full=b,
            kps_a=ctx.kps_a, kps_b=ctx.kps_b,
            debug={**ctx.debug, "gradient_keep": overlay.astype(np.float32)},
            info={**ctx.info, "n_centers": int(centers.shape[0]),
                  "n_pairs": int(pairs_a.shape[0]),
                  "gradient_threshold": p["gradient_threshold"],
                  "grid_k": int(p["grid_k"]), "window_radius": float(p["window_radius"])},
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
        window_radius: float = 2.0,
        grid_k: int = 3,
        max_pairs: int = 50000,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            detector=detector, min_disparity=min_disparity,
            num_disparities=num_disparities, block_size=block_size,
            uniqueness_ratio=uniqueness_ratio, speckle_window_size=speckle_window_size,
            speckle_range=speckle_range, disp12_max_diff=disp12_max_diff,
            clahe_clip=clahe_clip, fundamental_threshold=fundamental_threshold,
            window_radius=window_radius, grid_k=grid_k,
            max_pairs=max_pairs, seed=seed, **kwargs,
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

        disp = self._make_sgbm().compute(rect_a, rect_b).astype(np.float32) / 16.0

        valid_disp = disp > (p["min_disparity"] + 0.5)
        yr, xr = np.nonzero(valid_disp)
        if yr.size == 0:
            return _empty_result(ctx, info_extra={"detector": p["detector"], "n_disp": 0})

        d = disp[yr, xr]
        rect_a_pts = np.stack([xr, yr, np.ones_like(xr)], axis=1).astype(np.float64)
        rect_b_pts = np.stack([xr + d, yr, np.ones_like(xr)], axis=1).astype(np.float64)

        H1inv, H2inv = np.linalg.inv(H1), np.linalg.inv(H2)
        orig_a = (H1inv @ rect_a_pts.T).T
        orig_b = (H2inv @ rect_b_pts.T).T
        orig_a = orig_a[:, :2] / orig_a[:, 2:3]
        orig_b = orig_b[:, :2] / orig_b[:, 2:3]

        in_a = (orig_a[:, 0] >= 0) & (orig_a[:, 0] < w) & (orig_a[:, 1] >= 0) & (orig_a[:, 1] < h)
        in_b = (orig_b[:, 0] >= 0) & (orig_b[:, 0] < w) & (orig_b[:, 1] >= 0) & (orig_b[:, 1] < h)
        ok_pts = in_a & in_b
        centers_a = orig_a[ok_pts].astype(np.float32)               # (C, 2) (x, y) in A
        centers_b = orig_b[ok_pts].astype(np.float32)               # paired centers in B
        if centers_a.shape[0] == 0:
            return _empty_result(ctx, info_extra={"detector": p["detector"], "n_disp": 0})

        # Subsample the matched center PAIRS together (keep A/B aligned).
        n_centers_cap = max(1, int(p["max_pairs"]) // max(1, int(p["grid_k"]) ** 2))
        if centers_a.shape[0] > n_centers_cap:
            idx = np.random.default_rng(int(p["seed"])).choice(
                centers_a.shape[0], n_centers_cap, replace=False)
            centers_a, centers_b = centers_a[idx], centers_b[idx]

        pairs_a, pairs_b, coords_a, coords_b = sample_windows(
            img_a, centers_a, img_b, centers_b, p["window_radius"], p["grid_k"]
        )

        disp_vis = np.clip((disp - p["min_disparity"]) / max(p["num_disparities"], 1), 0, 1)
        return AlignResult(
            pairs_a=pairs_a, pairs_b=pairs_b, coords_a=coords_a, coords_b=coords_b,
            img_a_full=img_a, img_b_full=img_b,
            kps_a=ctx.kps_a, kps_b=ctx.kps_b,
            debug={**ctx.debug, "sgbm_disparity": disp_vis.astype(np.float32)},
            info={**ctx.info, "detector": p["detector"],
                  "n_centers": int(centers_a.shape[0]), "n_pairs": int(pairs_a.shape[0]),
                  "grid_k": int(p["grid_k"]), "window_radius": float(p["window_radius"])},
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
