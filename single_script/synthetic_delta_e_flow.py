"""
Variant of ``synthetic_delta_e.py`` that adds a *real* optical-flow distortion
estimated from the paired Samsung S9 image (Gradient/Farneback flow).

For every iPhone .mat image in a dng folder:
1. Match it against the paired Samsung image to estimate a homography H (B -> A).
2. Warp the iPhone image itself by a noisy inverse of H to synthesize a distorted view.
3. NEW: estimate a dense optical flow (Farneback, gradient-based) from the iPhone
   image towards the paired Samsung S9 image, and apply that flow as a
   non-linear local displacement on top of the homography warp. This injects the
   *real* local geometry of the Samsung device instead of a purely synthetic /
   random warp.
4. Re-match/re-align the distorted view onto the original, then compute ΔE2000
   on flat (low-gradient) pixels only.
Logs per-image results to a text file and stops.

The colours of the distorted view are never altered for the ΔE computation, so
the residual ΔE measures geometric-resampling error only (homography + Samsung
optical-flow + re-alignment).
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/run/media/gabriele/discone/Files/Magistrale/TESI/Codice")
sys.path.insert(0, str(ROOT / "Tests"))

from thesis_lib_shared.io import read_mat_image
from thesis_lib_shared.metrics import akaze_keypoint_matches, rgb_to_lab
from delta_e import delta_e_images, delta_e_summary

sys.path.insert(0, str(ROOT / "Standard"))
from standard.data.matchers.sift import SIFTMatcher
from standard.data.quantizers.chroma_grid import ChromaGridQuantizer
from standard.methods.poly_batch import PolynomialColorTransfer

PAIRED = ROOT / "data/iphone2samsung_or/pair-np-dv/paired"
IPHONE_DNG = PAIRED / "iphone-x/dng"
SAMSUNG_DNG = PAIRED / "samsung-s9/dng"

RES = 0.10
GRAD_THRESHOLD = 25
NOISE_STD = 0.05

# --- Optical-flow (Samsung S9 gradient flow) parameters --------------------
FLOW_SCALE = 1.0          # fraction of the estimated Samsung flow to apply (0 disables it)
FLOW_PYR_SCALE = 0.5      # Farneback pyramid scale
FLOW_LEVELS = 3           # Farneback pyramid levels
FLOW_WINSIZE = 31         # Farneback averaging window
FLOW_ITERATIONS = 3       # Farneback iterations per level
FLOW_POLY_N = 7           # Farneback pixel neighbourhood for polynomial expansion
FLOW_POLY_SIGMA = 1.5     # Farneback Gaussian sigma for the polynomial expansion

# This variant writes alongside itself so it never clobbers the baseline run.
OUT_ROOT = Path(__file__).parent
LOG_PATH = OUT_ROOT / "synthetic_delta_e_flow_log.txt"
EXPERIMENTS_DIR = OUT_ROOT / "Experiments_flow"


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    l_channel = rgb_to_lab(img)[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


def samsung_optical_flow(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Dense gradient-based (Farneback) optical flow from iPhone A -> Samsung B.

    Returns a (H, W, 2) float32 displacement field (dx, dy) in pixels, sized to
    ``img_a``. ``img_b`` is resized to A's resolution first so the flow is
    expressed in A's pixel grid.
    """
    h, w = img_a.shape[:2]
    if img_b.shape[:2] != (h, w):
        img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)

    gray_a = cv2.cvtColor(np.clip(img_a, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(np.clip(img_b, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    # Farneback expects 8-bit single-channel input.
    gray_a = (gray_a * 255.0).astype(np.uint8)
    gray_b = (gray_b * 255.0).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        FLOW_PYR_SCALE, FLOW_LEVELS, FLOW_WINSIZE, FLOW_ITERATIONS,
        FLOW_POLY_N, FLOW_POLY_SIGMA, 0,
    )
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
    """Quiver/colour visualisation of the Samsung optical-flow field."""
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(mag, cmap="magma")
    plt.colorbar(label="flow magnitude (px)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def process_image(mat_path: Path, seed: int) -> str:
    rng = np.random.default_rng(seed)
    pair_name = mat_path.stem
    out_dir = EXPERIMENTS_DIR / pair_name / "images"

    samsung_path = SAMSUNG_DNG / mat_path.name
    if not samsung_path.exists():
        return f"{mat_path.name}: SKIPPED (no matching Samsung file)"

    out_dir.mkdir(parents=True, exist_ok=True)

    img_a = read_mat_image(mat_path)
    img_b = read_mat_image(samsung_path)

    kps_a, kps_b = akaze_keypoint_matches(img_a, img_b, use_ransac=True)
    pts_a = kps_a[:, ::-1].astype(np.float32)
    pts_b = kps_b[:, ::-1].astype(np.float32)
    H_mat, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 1.0)

    h_orig, w_orig = img_a.shape[:2]
    H_inv = np.linalg.inv(H_mat)
    H_noise = 1.0 + rng.normal(0.0, NOISE_STD, size=H_inv.shape)
    H_inv = H_inv * H_noise
    H_inv /= H_inv[2, 2]

    img_a_distorted = cv2.warpPerspective(img_a, H_inv, (w_orig, h_orig))

    # NEW: estimate the real Samsung-S9 optical flow (gradient/Farneback) and add
    # it as a non-linear local displacement on top of the homography warp.
    flow = samsung_optical_flow(img_a, img_b)
    save_flow_debug(
        flow, f"Samsung S9 optical flow magnitude (px) — {pair_name}",
        out_dir / "samsung_optical_flow.png",
    )
    if FLOW_SCALE != 0.0:
        img_a_distorted = apply_flow(img_a_distorted, flow, FLOW_SCALE)

    # Color-transfer img_a -> img_a_distorted (keypoint patches + chroma grid + poly deg 3)
    # to simulate extra color noise. Used ONLY to search the second AKAZE matches;
    # geometric alignment and DeltaE below still use the uncolored img_a_distorted.
    sift_matcher = SIFTMatcher(patch_size=5, use_ransac=True)
    X, Y = sift_matcher.match(img_a, img_a_distorted)
    quantizer = ChromaGridQuantizer(grid_size=50)
    X_q, Y_q = quantizer.quantize(X, Y)
    color_method = PolynomialColorTransfer(degree=3)
    color_method.fit([(X_q, Y_q)])
    img_a_distorted_colored = color_method.apply(img_a_distorted)

    kps_a2, kps_b2 = akaze_keypoint_matches(img_a, img_a_distorted_colored, use_ransac=True)
    save_keypoint_match_debug(
        img_a, img_a_distorted_colored, kps_a2[::5], kps_b2[::5],
        f"AKAZE matches: iPhone vs simulated-Samsung (colored+flow) — {len(kps_a2)}",
        out_dir / "keypoints_iphone_vs_simulated_colored.png",
    )

    pts_a2 = kps_a2[:, ::-1].astype(np.float32)
    pts_b2 = kps_b2[:, ::-1].astype(np.float32)
    H_mat2, _ = cv2.findHomography(pts_b2, pts_a2, cv2.RANSAC, 1.0)

    img_a_distorted_aligned = cv2.warpPerspective(img_a_distorted, H_mat2, (w_orig, h_orig))
    valid_mask_full = cv2.warpPerspective(
        np.ones((h_orig, w_orig), dtype=np.float32), H_mat2, (w_orig, h_orig)
    ) > 0.999

    new_size = (int(round(w_orig * RES)), int(round(h_orig * RES)))
    img_a_small = cv2.resize(img_a, new_size, interpolation=cv2.INTER_AREA)
    img_a_distorted_aligned_small = cv2.resize(
        img_a_distorted_aligned, new_size, interpolation=cv2.INTER_AREA
    )
    valid_mask = cv2.resize(
        valid_mask_full.astype(np.float32), new_size, interpolation=cv2.INTER_NEAREST
    ) > 0.999

    grad = np.maximum(
        gradient_magnitude(img_a_small), gradient_magnitude(img_a_distorted_aligned_small)
    )
    keep_mask = (grad < GRAD_THRESHOLD) & valid_mask

    save_de_keep_mask_debug(
        img_a_small, keep_mask,
        f"Pixels used for DE eval (grad < {GRAD_THRESHOLD})",
        out_dir / "de_eval_keep_mask.png",
    )

    de_map = delta_e_images(img_a_small, img_a_distorted_aligned_small)
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

    with open(LOG_PATH, "w") as log_file, ProcessPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(process_image, mat_path, seed): mat_path
            for seed, mat_path in enumerate(mat_files)
        }
        for future in as_completed(futures):
            mat_path = futures[future]
            try:
                line = future.result()
            except Exception as e:
                line = f"{mat_path.name}: ERROR {e}"
            print(line)
            log_file.write(line + "\n")
            log_file.flush()


if __name__ == "__main__":
    main()
