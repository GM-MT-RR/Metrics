"""
Build a *simulated iPhone* in the Samsung-S9 view and measure how well a
homography + Farneback optical flow can re-align it back to the real iPhone, in
ΔE2000 terms. Mirrors the *matching refinement flow* pipeline of
``matching_refinement.ipynb`` (TinyRoMa matching → homography → masked-equalized
Farneback flow → remap), for both the simulation and the re-alignment halves.

The simulated iPhone is made by resampling the **iPhone itself** into the
Samsung's geometry, so it carries the iPhone's exact colours: if the geometric
re-alignment were perfect, ΔE(iPhone, simulated) would be 0. The residual ΔE
therefore measures *geometric-resampling* error only — no colour transfer is
involved, only the homography + flow.

For every iPhone .mat image in a dng folder:
1. Match it against the paired Samsung with TinyRoMa to estimate a homography
   H (B -> A), exactly like the notebook's TinyRoMa cell.
2. Warp the iPhone into the Samsung's view with H^-1 (coarse alignment) — this
   is ``img_a_omo``: iPhone colours, Samsung geometry.
3. Refine ``img_a_omo`` -> Samsung with a dense Farneback flow on *masked*
   histogram-equalized greys (equalized over the valid overlap only, as the
   notebook does), then remap ``img_a_omo`` by that flow to get ``img_a_sim``,
   the simulated iPhone in the Samsung view.
4. Re-align ``img_a_sim`` onto the real iPhone with a TinyRoMa homography *and*
   a second masked-equalized Farneback flow (symmetric to step 2+3), then
   compute ΔE2000 on flat (low-gradient) pixels only. This measures how well
   homography + flow together recover the alignment.
Logs per-image results to a text file.

The flow is estimated on masked-equalized grayscale because iPhone/Samsung
differ in exposure/tone, which violates the brightness-constancy Farneback
assumes (otherwise the polynomial-expansion gradients vanish and the flow comes
back ~0).
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/run/media/gabriele/discone/Files/Magistrale/TESI/Codice")
sys.path.insert(0, str(ROOT / "Tests"))

from thesis_lib_shared.io import read_mat_image
from thesis_lib_shared.metrics import tiny_roma_keypoint_matches, rgb_to_lab
from delta_e import delta_e_images, delta_e_summary

PAIRED = ROOT / "data/iphone2samsung_or/pair-np-dv/paired"
IPHONE_DNG = PAIRED / "iphone-x/dng"
SAMSUNG_DNG = PAIRED / "samsung-s9/dng"

WORK_LONG = 1024          # long side of the working frame (short side M follows the aspect ratio)
RES = 0.25
GRAD_THRESHOLD = 25

# --- Optical-flow (Farneback) parameters, copied verbatim from the notebook's
#     calculate_and_plot_optical_flow defaults (it is called with only step=128, so
#     every flow param stays at its default). -----------------------------------
FLOW_SCALE = 1.0          # fraction of the estimated flow to apply (0 disables it)
FARNEBACK_PARAMS = dict(  # cv2.calcOpticalFlowFarneback kwargs (notebook defaults)
    pyr_scale=0.5,
    levels=6,
    winsize=25,
    iterations=5,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)
# Notebook erodes the valid-overlap mask by `winsize` before flow/equalization.
ROI_ERODE = FARNEBACK_PARAMS["winsize"]

# This variant writes alongside itself so it never clobbers the baseline run.
OUT_ROOT = Path(__file__).parent
LOG_PATH = OUT_ROOT / "synthetic_delta_e_flow_log.txt"
EXPERIMENTS_DIR = OUT_ROOT / "Experiments_flow"


def resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    """Resize ``img`` so its longer side is ``long_side`` px, keeping the aspect ratio."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    new_w, new_h = round(w * s), round(h * s)
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    l_channel = rgb_to_lab(img)[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


def homography_from_matches(img_a: np.ndarray, img_b: np.ndarray):
    """TinyRoMa matches + RANSAC homography mapping B -> A (notebook's recipe).

    Returns (H_mat, kps_a, kps_b) with the (row, col) keypoint arrays kept for the
    debug plot. ``cv2.findHomography(pts_b, pts_a)`` -> H maps B's pixels onto A.
    """
    kps_a, kps_b = tiny_roma_keypoint_matches(img_a, img_b)
    pts_a = kps_a[:, ::-1].astype(np.float32)   # (row, col) -> (x, y)
    pts_b = kps_b[:, ::-1].astype(np.float32)
    H_mat, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 1.0)
    return H_mat, kps_a, kps_b


# --- Notebook flow path: masked-CDF equalization + Farneback -------------------

def _equalize_masked(img: np.ndarray, mask: np.ndarray, nbins: int = 256) -> np.ndarray:
    """Histogram-equalize grayscale to [0, 1] using only the masked pixels.

    Matches the notebook's ``_equalize_masked``: equalizing over the valid overlap
    only (not the whole frame) keeps the warp's zeroed background from skewing the
    CDF, which is what makes the flow lock onto the real residual misalignment.
    """
    gray = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    out = np.zeros_like(gray, dtype=np.float32)
    if not mask.any():
        return out
    vals = gray[mask].astype(np.float64)
    hist, edges = np.histogram(vals, bins=nbins, range=(vals.min(), vals.max()))
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1]
    centers = (edges[:-1] + edges[1:]) / 2
    out[mask] = np.interp(vals, centers, cdf).astype(np.float32)
    return out


def _flow_to_uint8_gray(eq: np.ndarray) -> np.ndarray:
    """Equalized float map [0, 1] -> 8-bit grey, as the notebook's flow cell wants.

    Farneback needs 8-bit grayscale; passing floats in [0, 1] straight in makes the
    polynomial-expansion gradients vanish and the flow comes back ~0.
    """
    return np.clip(eq * 255.0, 0, 255).astype(np.uint8)


def refinement_flow(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Dense Farneback flow A -> B on masked-equalized greys, in A's pixel grid.

    The homography is coarse; this flow refines the residual misalignment it left
    behind. Equalization is done over the valid overlap (non-zero in both images,
    eroded by ``ROI_ERODE``) exactly like the notebook, then Farneback runs with the
    notebook's params (levels=6, winsize=25). Returns (H, W, 2) float32 (dx, dy).
    """
    h, w = img_a.shape[:2]
    if img_b.shape[:2] != (h, w):
        img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)

    valid = (img_a.max(axis=2) > 0) & (img_b.max(axis=2) > 0)
    roi = cv2.erode(valid.astype(np.uint8),
                    np.ones((ROI_ERODE, ROI_ERODE), np.uint8)).astype(bool)

    gray_a = _flow_to_uint8_gray(_equalize_masked(img_a, roi))
    gray_b = _flow_to_uint8_gray(_equalize_masked(img_b, roi))

    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, **FARNEBACK_PARAMS)
    return flow.astype(np.float32)


def apply_flow(img: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
    """Warp ``img`` by ``scale * flow`` using cv2.remap (non-linear displacement)."""
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = grid_x + scale * flow[:, :, 0]
    map_y = grid_y + scale * flow[:, :, 1]
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# --- Debug figure helpers ------------------------------------------------------

def save_keypoint_match_debug(img_a, img_b, kps_a, kps_b, title, out_path):
    h = max(img_a.shape[0], img_b.shape[0])
    canvas = np.zeros((h, img_a.shape[1] + img_b.shape[1], 3), dtype=np.float32)
    canvas[: img_a.shape[0], : img_a.shape[1]] = img_a
    canvas[: img_b.shape[0], img_a.shape[1] :] = img_b

    plt.figure(figsize=(18, 8))
    plt.imshow(canvas)
    for (ra, ca), (rb, cb) in zip(kps_a, kps_b):
        plt.plot([ca, cb + img_a.shape[1]], [ra, rb], lw=0.5)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_de_keep_mask_debug(img, keep_mask, title, out_path):
    overlay = img.copy()
    overlay[~keep_mask] *= 0
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_flow_debug(flow, title, out_path):
    """Quiver/colour visualisation of the optical-flow field magnitude."""
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(mag, cmap="magma")
    plt.colorbar(label="flow magnitude (px)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_image_debug(img, title, out_path):
    """Save a single RGB image as a labelled debug figure."""
    plt.figure(figsize=(8, 6))
    plt.imshow(np.clip(img, 0, 1))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_diff_debug(img_a, img_b, title, out_path):
    """Save the per-pixel absolute difference between two RGB images."""
    diff = np.abs(np.clip(img_a, 0, 1) - np.clip(img_b, 0, 1)).mean(axis=2)
    plt.figure(figsize=(8, 6))
    plt.imshow(diff, cmap="inferno")
    plt.colorbar(label="mean |Δ| (RGB)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_gray_diff_debug(img_a, img_b, title, out_path):
    """Save |A - B| on masked-equalized greys (geometry-only, colour-blind).

    Diffing iPhone vs Samsung in RGB is dominated by their exposure/tone gap; on
    equalized greys the difference reflects geometric misalignment instead, which
    is what the flow refinement is meant to shrink (cf. the notebook's show_diff).
    Equalization uses the same masked CDF the flow does, over the valid overlap.
    """
    valid = (img_a.max(axis=2) > 0) & (img_b.max(axis=2) > 0)
    roi = cv2.erode(valid.astype(np.uint8),
                    np.ones((ROI_ERODE, ROI_ERODE), np.uint8)).astype(bool)
    ga = _equalize_masked(img_a, roi)
    gb = _equalize_masked(img_b, roi)
    diff = np.where(roi, np.abs(ga - gb), np.nan)
    plt.figure(figsize=(8, 6))
    plt.imshow(diff, cmap="inferno")
    plt.colorbar(label="|Δ| (equalized grey)")
    plt.title(f"{title}  (mean {np.nanmean(diff):.3f})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_scalar_map_debug(field, title, out_path, label="ΔE2000"):
    """Save a (H, W) scalar field (e.g. a ΔE map) as a labelled heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(field, cmap="magma")
    plt.colorbar(label=label)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def process_image(mat_path: Path) -> str:
    pair_name = mat_path.stem
    out_dir = EXPERIMENTS_DIR / pair_name / "images"

    samsung_path = SAMSUNG_DNG / mat_path.name
    if not samsung_path.exists():
        return f"{mat_path.name}: SKIPPED (no matching Samsung file)"

    out_dir.mkdir(parents=True, exist_ok=True)

    img_a = read_mat_image(mat_path)
    img_b = read_mat_image(samsung_path)

    # Work at a fixed resolution (long side WORK_LONG, e.g. 1024xM) so matching,
    # flow and ΔE all run on a smaller, consistent grid regardless of input size.
    img_a = resize_long_side(img_a, WORK_LONG)
    img_b = resize_long_side(img_b, WORK_LONG)
    h_orig, w_orig = img_a.shape[:2]

    # STEP 01/02: raw inputs (iPhone reference + paired Samsung).
    save_image_debug(img_a, f"01 iPhone (reference) — {pair_name}",
                     out_dir / "step_01_iphone.png")
    save_image_debug(img_b, f"02 Samsung S9 (paired) — {pair_name}",
                     out_dir / "step_02_samsung.png")

    # ===== Simulate: iPhone -> Samsung view (homography + refinement flow) =====
    H_mat, kps_a, kps_b = homography_from_matches(img_a, img_b)
    # STEP 03: TinyRoMa matches that drive the first homography (B -> A).
    save_keypoint_match_debug(
        img_a, img_b, kps_a[::5], kps_b[::5],
        f"03 TinyRoMa matches: iPhone vs Samsung — {len(kps_a)}",
        out_dir / "step_03_keypoints_iphone_vs_samsung.png",
    )

    # Warp the iPhone into the Samsung's view (coarse alignment). H_mat maps
    # B -> A, so H_inv maps A -> B: img_a_omo is the iPhone resampled onto the
    # Samsung geometry, but keeping the iPhone's exact colours.
    H_inv = np.linalg.inv(H_mat)
    img_a_omo = cv2.warpPerspective(img_a, H_inv, (w_orig, h_orig))
    # STEP 04: iPhone warped into the Samsung view (homography only, coarse).
    save_image_debug(
        img_a_omo,
        f"04 iPhone warped into Samsung view (homography only) — {pair_name}",
        out_dir / "step_04_homography_warp.png",
    )
    # STEP 04b: residual the flow is asked to refine away (equalized-grey).
    save_gray_diff_debug(
        img_a_omo, img_b,
        f"04b Residual: img_a_omo vs Samsung (pre-flow) — {pair_name}",
        out_dir / "step_04b_residual_pre_flow.png",
    )

    # Refine img_a_omo -> Samsung with the notebook's masked-equalized Farneback
    # flow. Notebook convention: flow = Farneback(ref, mov) lives on ref's grid and
    # points into mov, so the *mov* argument is the one later remapped by it. Here
    # the output must live on Samsung's grid but keep img_a_omo's colours, so
    # Samsung is "ref" and img_a_omo is "mov": flow = Farneback(img_b, img_a_omo).
    flow = refinement_flow(img_b, img_a_omo)
    # STEP 05: Farneback flow magnitude refining img_a_omo -> Samsung.
    save_flow_debug(
        flow, f"05 Farneback flow magnitude: img_a_omo -> Samsung (px) — {pair_name}",
        out_dir / "step_05_samsung_optical_flow.png",
    )
    img_a_sim = apply_flow(img_a_omo, flow, FLOW_SCALE) if FLOW_SCALE != 0.0 else img_a_omo
    # STEP 06: the simulated iPhone (iPhone colours, Samsung view, flow-refined).
    save_image_debug(
        img_a_sim,
        f"06 Simulated iPhone (homography + flow, Samsung view) — {pair_name}",
        out_dir / "step_06_simulated_iphone.png",
    )
    # STEP 07: residual vs Samsung after the flow — should be smaller than 04b.
    save_gray_diff_debug(
        img_a_sim, img_b,
        f"07 Residual: simulated iPhone vs Samsung (post-flow) — {pair_name}",
        out_dir / "step_07_residual_post_flow.png",
    )

    # ===== Re-align: simulated iPhone -> real iPhone (homography + flow) =====
    # Symmetric to the simulate half. Both images carry the iPhone's colours, so
    # the residual ΔE is geometric-resampling error only (no colour transfer).
    H_mat2, kps_a2, kps_b2 = homography_from_matches(img_a, img_a_sim)
    # STEP 08: TinyRoMa matches that drive the re-alignment homography.
    save_keypoint_match_debug(
        img_a, img_a_sim, kps_a2[::5], kps_b2[::5],
        f"08 TinyRoMa matches: iPhone vs simulated iPhone — {len(kps_a2)}",
        out_dir / "step_08_keypoints_iphone_vs_simulated.png",
    )

    # H_mat2 maps img_a_sim -> img_a, so it re-aligns the simulated image directly.
    img_a_sim_homo = cv2.warpPerspective(img_a_sim, H_mat2, (w_orig, h_orig))
    valid_mask_homo = cv2.warpPerspective(
        np.ones((h_orig, w_orig), dtype=np.float32), H_mat2, (w_orig, h_orig)
    )
    # STEP 09: simulated iPhone after the re-alignment homography (coarse).
    save_image_debug(
        img_a_sim_homo,
        f"09 Simulated iPhone re-aligned onto iPhone (homography only) — {pair_name}",
        out_dir / "step_09_aligned_homography.png",
    )

    # Refine the re-alignment with the same recipe, symmetric to the simulate half:
    # output must live on img_a's grid with img_a_sim_homo's content, so img_a is
    # "ref" and img_a_sim_homo is "mov": flow2 = Farneback(img_a, img_a_sim_homo).
    # The flow is also applied to the validity mask so pixels pushed out of frame
    # become invalid.
    flow2 = refinement_flow(img_a, img_a_sim_homo)
    # STEP 10: Farneback flow magnitude refining the re-alignment.
    save_flow_debug(
        flow2, f"10 Farneback flow magnitude: re-alignment refine (px) — {pair_name}",
        out_dir / "step_10_realign_flow.png",
    )
    if FLOW_SCALE != 0.0:
        img_a_sim_aligned = apply_flow(img_a_sim_homo, flow2, FLOW_SCALE)
        valid_mask_full = apply_flow(valid_mask_homo, flow2, FLOW_SCALE) > 0.999
    else:
        img_a_sim_aligned = img_a_sim_homo
        valid_mask_full = valid_mask_homo > 0.999
    # STEP 11/12: simulated iPhone re-aligned (homography + flow) + residual vs iPhone.
    save_image_debug(
        img_a_sim_aligned,
        f"11 Simulated iPhone re-aligned onto iPhone (homography + flow) — {pair_name}",
        out_dir / "step_11_aligned.png",
    )
    save_diff_debug(
        img_a, img_a_sim_aligned,
        f"12 Residual: real iPhone vs re-aligned simulated — {pair_name}",
        out_dir / "step_12_residual_vs_iphone.png",
    )

    # ===== ΔE2000 on flat, valid pixels =====
    new_size = (int(round(w_orig * RES)), int(round(h_orig * RES)))
    img_a_small = cv2.resize(img_a, new_size, interpolation=cv2.INTER_AREA)
    img_a_sim_aligned_small = cv2.resize(
        img_a_sim_aligned, new_size, interpolation=cv2.INTER_AREA
    )
    valid_mask = cv2.resize(
        valid_mask_full.astype(np.float32), new_size, interpolation=cv2.INTER_NEAREST
    ) > 0.999

    grad = np.maximum(
        gradient_magnitude(img_a_small), gradient_magnitude(img_a_sim_aligned_small)
    )
    keep_mask = (grad < GRAD_THRESHOLD) & valid_mask

    # STEP 13: pixels kept for the ΔE evaluation (flat + valid).
    save_de_keep_mask_debug(
        img_a_small, keep_mask,
        f"13 Pixels used for DE eval (grad < {GRAD_THRESHOLD})",
        out_dir / "step_13_de_eval_keep_mask.png",
    )

    de_map = delta_e_images(img_a_small, img_a_sim_aligned_small)

    # STEP 14/15: full ΔE2000 map and the ΔE map masked to the kept pixels.
    save_scalar_map_debug(
        de_map,
        f"14 ΔE2000 map (all valid) — {pair_name}",
        out_dir / "step_14_delta_e_map.png",
    )
    de_map_kept = np.where(keep_mask, de_map, 0.0)
    save_scalar_map_debug(
        de_map_kept,
        f"15 ΔE2000 map (kept pixels only) — {pair_name}",
        out_dir / "step_15_delta_e_map_kept.png",
    )

    stats_kept = delta_e_summary(de_map[keep_mask])
    stats_all = delta_e_summary(de_map[valid_mask])

    return (
        f"{mat_path.name}: "
        f"flat_px={keep_mask.sum()}/{valid_mask.sum()} "
        f"de_flat={ {k: round(v, 3) for k, v in stats_kept.items()} } "
        f"de_all={ {k: round(v, 3) for k, v in stats_all.items()} }"
    )


def main():
    mat_files = sorted(IPHONE_DNG.glob("*.mat"))

    with open(LOG_PATH, "w") as log_file:
        for mat_path in mat_files:
            try:
                line = process_image(mat_path)
            except Exception as e:
                line = f"{mat_path.name}: ERROR {e}"
            print(line)
            log_file.write(line + "\n")
            log_file.flush()


if __name__ == "__main__":
    main()
