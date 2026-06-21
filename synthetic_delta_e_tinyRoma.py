"""
For every iPhone .mat image in a dng folder:
1. Match it against the paired Samsung image to estimate a homography H (B -> A).
2. Warp the iPhone image itself by a noisy inverse of H to synthesize a distorted view
   (img_a_distorted, pure geometry), then poly-color-transfer it to add realistic color
   noise -> img_b_sim (the simulated "other camera" view).
3. Match img_a <-> img_b_sim densely with tinyRoma (RoMa), sample pixelwise correspondences
   from the dense field, and compute ΔE2000 between img_a and the *uncolored* img_a_distorted
   at those correspondences. ΔE therefore measures matching/alignment error as color error
   (a perfect match shares the same source color -> ΔE ~ 0).
Logs per-image results to a text file, writes a metrics CSV, and renders a summary figure.

Execution: the CPU simulation (step 1-2) runs in a process pool; tinyRoma is loaded once and
runs the matching (step 3) in batches on the GPU.
"""

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/run/media/gabriele/discone/Files/Magistrale/TESI/Codice")
sys.path.insert(0, str(ROOT / "Tests"))

from thesis_lib_shared.io import read_mat_image
from thesis_lib_shared.metrics import akaze_keypoint_matches
from delta_e import delta_e_images, delta_e_summary

sys.path.insert(0, str(ROOT / "Standard"))
from standard.data.matchers.sift import SIFTMatcher
from standard.data.quantizers.chroma_grid import ChromaGridQuantizer
from standard.methods.poly_batch import PolynomialColorTransfer

from romatch import tiny_roma_v1_outdoor

PAIRED = ROOT / "data/iphone2samsung_or/pair-np-dv/paired"
IPHONE_DNG = PAIRED / "iphone-x/dng"
SAMSUNG_DNG = PAIRED / "samsung-s9/dng"

NOISE_STD = 0.05
LOG_PATH = Path(__file__).parent / "synthetic_delta_e_log.txt"
CSV_PATH = Path(__file__).parent / "synthetic_delta_e_roma_metrics.csv"
SUMMARY_PATH = Path(__file__).parent / "synthetic_delta_e_roma_summary.png"
EXPERIMENTS_DIR = Path("./Experiments")

# tinyRoma matching parameters
ROMA_W, ROMA_H = 560, 450   # tinyRoma working resolution (floored to a multiple of 32 internally)
MAX_KEYPOINTS = 5000        # correspondences sampled from the dense warp, per pair
BATCH_SIZE = 8              # image pairs matched per tinyRoma forward pass
DEVICE = "cuda"             # "cuda" if torch.cuda.is_available() else "cpu"

DE_COLUMNS = ["mean", "median", "p25", "p75", "p95", "p99"]


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


# --- Phase 1: CPU simulation of the cross-camera view -----------------------------------

def simulate_pair(mat_path: Path, seed: int) -> dict:
    """Build the simulated pair for one iPhone image. Returns the arrays needed for matching.

    On success: {"pair", "status": "OK", "img_a", "img_a_distorted", "img_b_sim"}.
    Otherwise:  {"pair", "status": <reason>} (no arrays).
    """
    rng = np.random.default_rng(seed)
    pair_name = mat_path.stem

    samsung_path = SAMSUNG_DNG / mat_path.name
    if not samsung_path.exists():
        return {"pair": pair_name, "status": "SKIPPED (no matching Samsung file)"}

    img_a = read_mat_image(mat_path)
    img_b = read_mat_image(samsung_path)

    # iPhone <-> Samsung homography, used to drive a realistic geometric distortion of img_a.
    kps_a, kps_b = akaze_keypoint_matches(img_a, img_b, use_ransac=True)
    pts_a = kps_a[:, ::-1].astype(np.float32)
    pts_b = kps_b[:, ::-1].astype(np.float32)
    H_mat, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 1.0)

    h_orig, w_orig = img_a.shape[:2]
    H_inv = np.linalg.inv(H_mat)
    H_noise = 1.0 + rng.normal(0.0, NOISE_STD, size=H_inv.shape)
    H_inv = H_inv * H_noise
    H_inv /= H_inv[2, 2]

    # Pure geometric warp (no color change) -- this is the ΔE reference image.
    img_a_distorted = cv2.warpPerspective(img_a, H_inv, (w_orig, h_orig))

    # Color-transfer img_a -> img_a_distorted (keypoint patches + chroma grid + poly deg 3)
    # to simulate cross-camera color noise. This colored view is what tinyRoma matches against;
    # the ΔE below still uses the uncolored img_a_distorted.
    sift_matcher = SIFTMatcher(patch_size=5, use_ransac=True)
    X, Y = sift_matcher.match(img_a, img_a_distorted)
    quantizer = ChromaGridQuantizer(grid_size=50)
    X_q, Y_q = quantizer.quantize(X, Y)
    color_method = PolynomialColorTransfer(degree=3)
    color_method.fit([(X_q, Y_q)])
    img_b_sim = color_method.apply(img_a_distorted)

    return {
        "pair": pair_name,
        "status": "OK",
        "img_a": img_a,
        "img_a_distorted": img_a_distorted,
        "img_b_sim": img_b_sim,
    }


# --- Phase 2: tinyRoma dense matching + ΔE ----------------------------------------------

def load_tiny_roma(device: str):
    model = tiny_roma_v1_outdoor(device=torch.device(device))
    model.train(False)
    return model


def to_roma_tensor(img: np.ndarray) -> torch.Tensor:
    """RGB float [0,1] (H,W,3) -> (3, ROMA_H, ROMA_W) float tensor in [0,1] (ToTensor scale)."""
    resized = cv2.resize(img.astype(np.float32), (ROMA_W, ROMA_H))
    return torch.from_numpy(resized).permute(2, 0, 1).contiguous()


def roma_match_batch(model, imgs_a, imgs_b, device):
    """Batched dense match. Returns warp (B,H,W,4) and certainty (B,H,W)."""
    batch_a = torch.stack([to_roma_tensor(a) for a in imgs_a]).to(device)
    batch_b = torch.stack([to_roma_tensor(b) for b in imgs_b]).to(device)
    warp, certainty = model.match(batch_a, batch_b, batched=True)
    return warp, certainty


def correspondences_from_warp(model, warp_i, cert_i, h, w):
    """Sample pixelwise correspondences and map them to full-res (row, col) integer indices.

    Returns (pts_a_rc, pts_b_rc), each (N, 2) int = (row, col), filtered to in-bounds points.
    """
    matches, _ = model.sample(warp_i, cert_i, num=MAX_KEYPOINTS)
    mkpts0, mkpts1 = model.to_pixel_coordinates(matches, h, w, h, w)
    mkpts0 = mkpts0.cpu().numpy()  # (x, y) = (col, row) in A
    mkpts1 = mkpts1.cpu().numpy()  # (x, y) = (col, row) in B

    pts_a_rc = np.round(mkpts0[:, ::-1]).astype(np.int32)  # (row, col) in A
    pts_b_rc = np.round(mkpts1[:, ::-1]).astype(np.int32)  # (row, col) in B

    in_bounds = (
        (pts_a_rc[:, 0] >= 0) & (pts_a_rc[:, 0] < h)
        & (pts_a_rc[:, 1] >= 0) & (pts_a_rc[:, 1] < w)
        & (pts_b_rc[:, 0] >= 0) & (pts_b_rc[:, 0] < h)
        & (pts_b_rc[:, 1] >= 0) & (pts_b_rc[:, 1] < w)
    )
    return pts_a_rc[in_bounds], pts_b_rc[in_bounds]


def delta_e_at_points(img_ref, img_tgt, pts_a_rc, pts_b_rc):
    """ΔE2000 between img_ref at A-points and img_tgt at B-points (pixelwise). Returns stats dict."""
    colors_a = img_ref[pts_a_rc[:, 0], pts_a_rc[:, 1]]
    colors_b = img_tgt[pts_b_rc[:, 0], pts_b_rc[:, 1]]
    # delta_e_images expects (H, W, 3); treat the N correspondences as an (N, 1, 3) image.
    de_map = delta_e_images(colors_a[:, None, :], colors_b[:, None, :])
    return delta_e_summary(de_map.ravel())


def match_and_score(model, sim: dict) -> dict:
    """tinyRoma-match one simulated pair (already matched in a batch) and score ΔE.

    `sim` must carry warp/certainty under "warp"/"cert" (added by the batch loop).
    """
    img_a = sim["img_a"]
    h, w = img_a.shape[:2]

    pts_a_rc, pts_b_rc = correspondences_from_warp(model, sim["warp"], sim["cert"], h, w)
    stats = delta_e_at_points(img_a, sim["img_a_distorted"], pts_a_rc, pts_b_rc)

    out_dir = EXPERIMENTS_DIR / sim["pair"] / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(pts_a_rc) // 200)  # ~200 lines max for readability
    save_keypoint_match_debug(
        img_a, sim["img_b_sim"], pts_a_rc[::step], pts_b_rc[::step],
        f"tinyRoma matches: iPhone vs simulated view — {len(pts_a_rc)} correspondences",
        out_dir / "roma_matches.png",
    )

    record = {
        "pair": sim["pair"],
        "status": "OK",
        "roma_px": int(len(pts_a_rc)),
        "sampled_px": int(MAX_KEYPOINTS),
    }
    record.update({k: stats[k] for k in DE_COLUMNS})
    return record


def match_all(sims: list, device: str) -> list:
    """Load tinyRoma once and match all simulated pairs in batches."""
    model = load_tiny_roma(device)
    records = []
    for start in range(0, len(sims), BATCH_SIZE):
        chunk = sims[start : start + BATCH_SIZE]
        with torch.inference_mode():
            warp, cert = roma_match_batch(
                model, [s["img_a"] for s in chunk], [s["img_b_sim"] for s in chunk], device
            )
        for i, sim in enumerate(chunk):
            sim["warp"], sim["cert"] = warp[i], cert[i]
            try:
                records.append(match_and_score(model, sim))
            except Exception as e:  # noqa: BLE001 - keep the sweep going
                records.append({"pair": sim["pair"], "status": f"ERROR {e}"})
    return records


# --- Reporting --------------------------------------------------------------------------

def format_log_line(record: dict) -> str:
    if record.get("status") != "OK":
        return f"{record['pair']}: {record.get('status')}"
    stats = {k: round(record[k], 3) for k in DE_COLUMNS}
    return (
        f"{record['pair']}: "
        f"roma_px={record['roma_px']}/{record['sampled_px']} "
        f"de_roma={stats}"
    )


def write_csv(records: list) -> None:
    ok = [r for r in records if r.get("status") == "OK"]
    if not ok:
        return
    fields = ["pair", "roma_px", "sampled_px"] + DE_COLUMNS
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(ok, key=lambda x: x["pair"]):
            writer.writerow({k: r[k] for k in fields})


def write_summary(records: list) -> None:
    ok = sorted((r for r in records if r.get("status") == "OK"), key=lambda x: x["pair"])
    if not ok:
        return

    pairs = [r["pair"] for r in ok]
    de_mean = np.array([r["mean"] for r in ok], dtype=np.float64)

    # Aggregate row: mean of each metric column across pairs (+ mean coverage).
    agg = {k: float(np.mean([r[k] for r in ok])) for k in DE_COLUMNS}
    cov = np.array([r["roma_px"] / max(r["sampled_px"], 1) for r in ok], dtype=np.float64)

    fig, (ax_tbl, ax_bar) = plt.subplots(
        2, 1, figsize=(max(10, 0.5 * len(ok)), 10),
        gridspec_kw={"height_ratios": [len(ok) + 2, max(4, len(ok))]},
    )

    # Per-pair metrics table + aggregate row
    col_labels = ["pair", "roma_px", "sampled_px", "cov%"] + DE_COLUMNS
    cell_text = []
    for r, c in zip(ok, cov):
        cell_text.append(
            [r["pair"], r["roma_px"], r["sampled_px"], f"{100 * c:.1f}"]
            + [f"{r[k]:.3f}" for k in DE_COLUMNS]
        )
    cell_text.append(
        ["MEAN", "", "", f"{100 * cov.mean():.1f}"] + [f"{agg[k]:.3f}" for k in DE_COLUMNS]
    )
    ax_tbl.axis("off")
    table = ax_tbl.table(cellText=cell_text, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax_tbl.set_title("tinyRoma ΔE2000 per-pair metrics (last row = mean across pairs)")

    # Bar plot of per-pair mean ΔE
    ax_bar.bar(range(len(pairs)), de_mean, color="steelblue")
    ax_bar.axhline(agg["mean"], color="crimson", ls="--", lw=1, label=f"mean={agg['mean']:.3f}")
    ax_bar.set_xticks(range(len(pairs)))
    ax_bar.set_xticklabels(pairs, rotation=90, fontsize=7)
    ax_bar.set_ylabel("mean ΔE2000")
    ax_bar.set_title("Per-pair mean ΔE2000 at tinyRoma correspondences")
    ax_bar.legend()

    plt.tight_layout()
    plt.savefig(str(SUMMARY_PATH), dpi=120, bbox_inches="tight")
    plt.close()


def main():
    mat_files = sorted(IPHONE_DNG.glob("*.mat"))

    # Phase 1 (CPU, parallel): build the simulated pairs.
    sims, records = [], []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(simulate_pair, mat_path, seed): mat_path
            for seed, mat_path in enumerate(mat_files)
        }
        for future in as_completed(futures):
            mat_path = futures[future]
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001
                records.append({"pair": mat_path.stem, "status": f"ERROR {e}"})
                continue
            if result.get("status") == "OK":
                sims.append(result)
            else:
                records.append(result)

    # Phase 2 (GPU, batched): tinyRoma matching + ΔE.
    sims.sort(key=lambda s: s["pair"])
    records.extend(match_all(sims, DEVICE))

    with open(LOG_PATH, "w") as log_file:
        for record in sorted(records, key=lambda r: r["pair"]):
            line = format_log_line(record)
            print(line)
            log_file.write(line + "\n")

    write_csv(records)
    write_summary(records)


if __name__ == "__main__":
    main()
