"""Notebook-style debug visualisations for one aligned pair.

These reproduce the figures the matching notebooks / ``synthetic_delta_e.py``
draw, so a run with ``save_debug: true`` writes the *same* diagnostics per pair:

- ``save_keypoint_matches``  — the two views side-by-side with lines joining the
  matched keypoints (``metric.ipynb`` cell 3, ``synthetic_delta_e.save_keypoint_match_debug``).
- ``save_aligned_overlay``   — checkerboard A / B-aligned + |A − B| abs-diff (cell 6).
- ``save_gradient_keep``     — L* gradient map + the kept (flat) pixels (cell 9).
- ``save_de_histogram``      — per-pixel ΔE map + histogram (cell 10).
- ``save_sparse_pairs``      — for sparse matchers: the extracted match points on
  each view + a swatch strip of the matched RGB pairs ΔE is computed on.

All saved through matplotlib (Agg) so they work headless in worker processes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import _paths  # noqa: F401
from thesis_lib_shared.metrics import rgb_to_lab


def _ensure(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)


def _clip(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 1.0)


def save_keypoint_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    kps_a: np.ndarray,
    kps_b: np.ndarray,
    out_path: Path,
    stride: int = 5,
    title: Optional[str] = None,
) -> None:
    """Side-by-side views with lines joining matched (row, col) keypoints."""
    _ensure(out_path)
    h = max(img_a.shape[0], img_b.shape[0])
    canvas = np.zeros((h, img_a.shape[1] + img_b.shape[1], 3), dtype=np.float32)
    canvas[: img_a.shape[0], : img_a.shape[1]] = _clip(img_a)
    canvas[: img_b.shape[0], img_a.shape[1]:] = _clip(img_b)

    n = int(kps_a.shape[0])
    title = title or f"matched keypoints: {n} (showing 1/{stride})"
    plt.figure(figsize=(18, 8))
    plt.imshow(canvas)
    for (ra, ca), (rb, cb) in zip(kps_a[::stride], kps_b[::stride]):
        plt.plot([ca, cb + img_a.shape[1]], [ra, rb], lw=0.5)
    plt.scatter(kps_a[::stride, 1], kps_a[::stride, 0], s=4, c="lime", marker="x")
    plt.scatter(kps_b[::stride, 1] + img_a.shape[1], kps_b[::stride, 0],
                s=4, c="lime", marker="x")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_aligned_overlay(
    img_a: np.ndarray,
    b_aligned: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
    tile: int = 32,
) -> None:
    """Checkerboard(A, B-aligned) + |A − B-aligned| greyscale (metric cell 6).

    Only the **valid (kept) pixels** are shown; everything outside ``valid_mask``
    is discarded (blanked) so the residual panel reads only over the zone ΔE is
    actually computed on — a high |A − B| then clearly localises *where* the
    misregistration is, instead of being diluted by the unused frame.
    """
    _ensure(out_path)
    h, w = img_a.shape[:2]
    valid = valid_mask.astype(bool)
    gray_a = cv2.cvtColor(_clip(img_a), cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(_clip(b_aligned), cv2.COLOR_RGB2GRAY)
    yy, xx = np.mgrid[:h, :w]
    checker = ((yy // tile + xx // tile) % 2).astype(bool)
    # Blank (NaN) the discarded pixels so they render as background, not as a
    # spurious "zero difference" that a flat *mask multiply would imply.
    check_img = np.where(valid, np.where(checker, gray_a, gray_b), np.nan)
    resid = np.where(valid, np.abs(gray_a - gray_b), np.nan)
    mean = float(np.nanmean(resid)) if valid.any() else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(check_img, cmap="gray")
    axes[0].set_title(f"checkerboard A / B-aligned  (valid {valid.mean() * 100:.0f}%)")
    axes[0].axis("off")
    im = axes[1].imshow(resid, cmap="magma", vmin=0)
    axes[1].set_title(f"|A − B-aligned| over valid px  (mean={mean:.3f})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_gradient_keep(
    img_a: np.ndarray,
    b_aligned: np.ndarray,
    keep_mask: np.ndarray,
    gradient_threshold: float,
    out_path: Path,
) -> None:
    """L* gradient magnitude + the kept (flat) pixels overlay (metric cell 9)."""
    _ensure(out_path)

    def _grad(img):
        l = rgb_to_lab(_clip(img))[:, :, 0].astype(np.float32)
        gx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    grad = np.maximum(_grad(img_a), _grad(b_aligned))
    overlay = _clip(img_a).copy()
    overlay[~keep_mask] *= 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(grad, cmap="viridis")
    axes[0].set_title("L* gradient magnitude")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].imshow(overlay)
    axes[1].set_title(f"patch extracted: pixels kept (grad < {gradient_threshold:g})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_masked_overlay(
    img_a: np.ndarray,
    coords_a: np.ndarray,
    pairs_b: np.ndarray,
    de_vals: np.ndarray,
    out_path: Path,
) -> None:
    """Image view of a SPARSE matcher with the correspondence mask applied.

    Mirrors the notebooks' "A / B-resampled-onto-A / |A − B|" panel (matching_RoMa
    cell 11, matching_ROMA_SGM cells 22-23) but for sparse correspondences: each
    matched B colour is rasterised into A's pixel grid at its A-coordinate, giving
    a dense ``b_on_a`` image; pixels with no correspondence are masked (blacked
    out). So you see the *actual image* of the matched region, not just scattered
    points.
    """
    _ensure(out_path)
    h, w = img_a.shape[:2]
    xs = np.clip(np.round(coords_a[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(coords_a[:, 1]).astype(int), 0, h - 1)

    b_on_a = np.zeros((h, w, 3), dtype=np.float32)
    cover = np.zeros((h, w), dtype=bool)
    b_on_a[ys, xs] = _clip(pairs_b)
    cover[ys, xs] = True

    de_img = np.full((h, w), np.nan, dtype=np.float32)
    if de_vals.size == coords_a.shape[0]:
        de_img[ys, xs] = de_vals
    mean = float(de_vals.mean()) if de_vals.size else float("nan")

    # Sparse single-pixel matches read as near-empty at full-frame scale; dilate
    # the rasterised correspondences a touch so the mask-applied image is legible.
    k = np.ones((3, 3), np.uint8)
    cover_vis = cv2.dilate(cover.astype(np.uint8), k, iterations=1) > 0
    b_on_a = cv2.dilate(b_on_a, k, iterations=1)
    de_img = np.where(cover_vis, cv2.dilate(np.nan_to_num(de_img), k, iterations=1),
                      np.nan).astype(np.float32)

    a_masked = _clip(img_a).copy()
    a_masked[~cover_vis] = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(a_masked)
    axes[0].set_title(f"A — mask applied ({int(cover.sum())} matched px)")
    axes[0].axis("off")
    axes[1].imshow(b_on_a)
    axes[1].set_title("B colours resampled onto A (mask applied)")
    axes[1].axis("off")
    im = axes[2].imshow(de_img, cmap="inferno", vmin=0)
    axes[2].set_title(f"ΔE map — matched px (mean={mean:.2f})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_de_histogram(
    de_map: np.ndarray,
    keep_mask: np.ndarray,
    out_path: Path,
) -> None:
    """ΔE map over kept pixels + histogram of the kept ΔE values (metric cell 10)."""
    _ensure(out_path)
    vals = de_map[keep_mask]
    de_show = np.where(keep_mask, de_map, np.nan)
    mean = float(np.nanmean(vals)) if vals.size else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(de_show, cmap="inferno", vmin=0)
    axes[0].set_title(f"ΔE map — kept pixels (mean={mean:.2f})")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    if vals.size:
        axes[1].hist(vals, bins=40, color="steelblue")
        axes[1].axvline(mean, color="crimson", label=f"mean = {mean:.2f}")
        axes[1].legend()
    axes[1].set_title("per-pixel ΔE (kept)")
    axes[1].set_xlabel("ΔE2000")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_sparse_pairs(
    img_a: np.ndarray,
    img_b: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    de_vals: np.ndarray,
    out_path: Path,
    max_points: int = 4000,
    max_swatches: int = 64,
) -> None:
    """Sparse matchers: the extracted match points on each view + RGB-pair swatches.

    Top row: img_a and img_b with the matched coords scattered (the "patches
    extracted" for the ΔE pairs). Bottom row: a swatch strip — for a subset of
    pairs, the A colour over the B colour — so you can eyeball the colour
    agreement the sparse ΔE is measuring.
    """
    _ensure(out_path)
    n = int(coords_a.shape[0])
    idx = np.arange(n)
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(_clip(img_a))
    ax0.scatter(coords_a[idx, 0], coords_a[idx, 1], s=3, c="lime", marker=".")
    ax0.set_title(f"A: extracted match points ({n})")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(_clip(img_b))
    ax1.scatter(coords_b[idx, 0], coords_b[idx, 1], s=3, c="lime", marker=".")
    ax1.set_title("B (synthetic): extracted match points")
    ax1.axis("off")

    # Swatch strip of matched RGB pairs (A over B), ordered by ΔE if available.
    ax2 = fig.add_subplot(gs[1, :])
    m = min(max_swatches, n)
    if m > 0:
        order = np.argsort(de_vals)[:: max(1, n // m)][:m] if de_vals.size == n \
            else np.linspace(0, n - 1, m).astype(int)
        strip = np.zeros((2, m, 3), dtype=np.float32)
        strip[0] = _clip(pairs_a[order])
        strip[1] = _clip(pairs_b[order])
        ax2.imshow(strip, aspect="auto")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["A", "B"])
        ax2.set_xticks([])
        mean = float(de_vals.mean()) if de_vals.size else float("nan")
        ax2.set_title(f"matched RGB pairs — top/bottom = A/B  (sparse ΔE mean={mean:.2f})")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
