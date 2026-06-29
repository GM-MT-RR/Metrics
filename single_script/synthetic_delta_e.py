"""
For every iPhone .mat image in a dng folder:
1. Match it against the paired Samsung image to estimate a homography H (B -> A).
2. Warp the iPhone image itself by a noisy inverse of H to synthesize a distorted view.
3. Re-match/re-align the distorted view onto the original, then compute ΔE2000
   on flat (low-gradient) pixels only.
Logs per-image results to a text file and stops.
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
LOG_PATH = Path(__file__).parent / "synthetic_delta_e_log.txt"
EXPERIMENTS_DIR = Path("./Experiments")


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    l_channel = rgb_to_lab(img)[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


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
        f"AKAZE matches: iPhone vs simulated-Samsung (colored) — {len(kps_a2)}",
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
