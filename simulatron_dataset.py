#!/usr/bin/env python3
"""Dataset-wide port of MetricDeProof/simulatron.ipynb.

Runs the notebook pipeline over the whole paired iPhone-X / Samsung-S9 folder and emits a new
paired dataset with three image folders (+ two validity masks):

    iphone        - original iPhone-X image          (linear RAW float32, fully valid)
    iphone-s      - synthesized Samsung view          (cell-11 warp, holes = 0)
    iphone-e      - one random slight-extreme view     (cell-13 warp, holes = 0)
    iphone-s-mask - bool validity mask for iphone-s    (True = real reprojected pixel)
    iphone-e-mask - bool validity mask for iphone-e

Images are saved as 16-bit PNGs holding LINEAR RGB (no sRGB encode; 65535 levels keep it
effectively lossless). The only sRGB conversion in the whole pipeline happens inside
depth_anything(), which the model needs. NO inpaint is applied: only real warped pixels are kept;
the masks (16-bit PNG, 0/65535) tell evaluation which pixels are real so the holes can be masked
out rather than filled.

The function bodies (read_mat_fullres, depth_anything, build_K/scale_K, relative_pose_a_to_b,
disparity_to_depth, synthesize_view, rot_xyz/look_pose/VIEW_BANK) are copied verbatim from the
notebook cells 1/4/6/7/9/11/13; only the per-cell driver code and the inpaint blocks are dropped.

Run with the interpreter where `romatch`, `transformers`, and Depth Anything are installed
(the notebook's py3.13 user env, not the repo .venv):

    python MetricDeProof/simulatron_dataset.py            # full dataset
    python MetricDeProof/simulatron_dataset.py --limit 2  # smoke test on first 2 stems
"""

# ===================== cell 0 (verbatim imports + path setup) =====================
import sys
import argparse
import csv
import gc
from pathlib import Path

from PIL import Image
import torch
import cv2
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

ROOT = Path("/run/media/gabriele/discone/Files/Magistrale/TESI/Codice")
sys.path.insert(0, str(ROOT / "Tests"))
sys.path.insert(0, str(ROOT / "thesis_lib_shared"))   # the package lives here, not under Tests

from thesis_lib_shared.io import read_mat_image, read_dng_image  # noqa: F401 (parity with notebook)
from thesis_lib_shared.metrics import *  # noqa: F401,F403  (roma_keypoint_matches, delta_e_*)

device = "cuda"

# ===================== dataset I/O config =====================
PAIRED   = ROOT / "data/iphone2samsung_or/pair-np-dv/paired"
OUT_ROOT = ROOT / "data/iphone2samsung_or/pair-np-dv/simulatron"
OUT_IPH    = OUT_ROOT / "iphone"          # original iPhone-X, linear float32 (fully valid)
OUT_IPH_S  = OUT_ROOT / "iphone-s"        # synthesized Samsung view (cell 11), holes = 0
OUT_IPH_E  = OUT_ROOT / "iphone-e"        # one random extreme view (cell 13 subset), holes = 0
OUT_MASK_S = OUT_ROOT / "iphone-s-mask"   # bool validity mask for iphone-s (True = real pixel)
OUT_MASK_E = OUT_ROOT / "iphone-e-mask"   # bool validity mask for iphone-e
CACHE_DIR  = OUT_ROOT / "_depth_cache"    # cached Depth Anything output_a per stem (Pass 1 -> Pass 2)

SEED = 0                                   # rng for the per-image extreme-view choice
EXTREMITY = 1                              # notebook nominal value (cell 13), kept verbatim
E_VIEW_SUBSET = ["yaw sinistra", "yaw destra", "avvicinata"]


# ===================== cell 1: full-resolution RGGB demosaic (verbatim) =====================
def read_mat_fullres(path):
    """Load .mat RGGB and demosaic to full-resolution float32 (2H, 2W, 3) RGB in [0,1]."""
    d = loadmat(str(path))
    key = [k for k in d if not k.startswith("__")][0]
    rggb = np.asarray(d[key], dtype=np.float32)          # (H, W, 4): R, Gr, Gb, B
    h, w, _ = rggb.shape

    bayer = np.zeros((h * 2, w * 2), dtype=np.float32)    # full-res Bayer mosaic
    bayer[0::2, 0::2] = rggb[..., 0]   # R  at (even, even)
    bayer[0::2, 1::2] = rggb[..., 1]   # Gr at (even, odd)
    bayer[1::2, 0::2] = rggb[..., 2]   # Gb at (odd,  even)
    bayer[1::2, 1::2] = rggb[..., 3]   # B  at (odd,  odd)

    # demosaic in 16-bit for precision. Physical layout is RGGB, but OpenCV's Bayer naming is
    # off-by-one: an R-first (RGGB) sensor is decoded with COLOR_BayerBG2RGB (verified against the
    # per-plane means: R~0.16, G~0.23, B~0.09). Using BayerRG2RGB swaps R<->B.
    b16 = (np.clip(bayer, 0.0, 1.0) * 65535.0).astype(np.uint16)
    rgb = cv2.cvtColor(b16, cv2.COLOR_BayerBG2RGB).astype(np.float32) / 65535.0
    return rgb


# ===================== cell 4: Depth Anything V2 (verbatim, model loaded lazily) =====================
def depth_anything(img01, da_processor, da_model):
    """img01: float32 RGB [0,1] linear -> relative depth map (H,W), larger = NEARER."""
    # sRGB-encode + uint8 RGB, like a normal photo the model expects
    srgb = np.where(img01 <= 0.0031308, img01 * 12.92,
                    1.055 * np.power(np.clip(img01, 0, 1), 1 / 2.4) - 0.055)
    pil = Image.fromarray((np.clip(srgb, 0, 1) * 255).astype(np.uint8))

    inputs = da_processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = da_model(**inputs).predicted_depth          # (1, h, w)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=img01.shape[:2], mode="bicubic", align_corners=False
    ).squeeze()
    # Depth Anything outputs disparity-like depth: larger = nearer (same as MiDaS). Keep it.
    return pred.cpu().numpy()


# ===================== cell 6: camera intrinsics (verbatim) =====================
PIXEL_PITCH_MM = 1.4e-3            # 1.4 um
SENSOR_W, SENSOR_H = 4032, 3024   # full-resolution pixel grid from EXIF


def build_K(focal_mm, sensor_wh=(SENSOR_W, SENSOR_H), pixel_pitch_mm=PIXEL_PITCH_MM):
    """Pinhole K at full sensor resolution from physical focal length and pixel pitch."""
    f_px = focal_mm / pixel_pitch_mm
    w, h = sensor_wh
    return np.array([[f_px, 0.0,  w / 2.0],
                     [0.0,  f_px, h / 2.0],
                     [0.0,  0.0,  1.0]], dtype=np.float64)


def scale_K(K_full, sensor_wh, target_wh):
    """Rescale K from sensor_wh to the working image resolution target_wh=(W,H)."""
    sx = target_wh[0] / sensor_wh[0]
    sy = target_wh[1] / sensor_wh[1]
    K = K_full.copy()
    K[0, :] *= sx   # fx, cx
    K[1, :] *= sy   # fy, cy
    return K


# EXIF focal lengths from the DNGs
K_iphone_full  = build_K(4.2)   # iPhone XS
K_samsung_full = build_K(4.3)   # Galaxy S9 (SM-G960F)


def intrinsics_for(img_a, img_b):
    """K_a / K_b at the working image resolution (verbatim from cell 6)."""
    H, W = img_a.shape[:2]
    target_wh = (W, H)
    K_a = scale_K(K_iphone_full,  (SENSOR_W, SENSOR_H), target_wh)
    K_b = scale_K(K_samsung_full, (SENSOR_W, SENSOR_H), target_wh)
    return K_a, K_b


# ===================== cell 7: relative pose A -> B (verbatim) =====================
def relative_pose_a_to_b(img_a, img_b, K_a, K_b=None, depth_a=None,
                         matcher=roma_keypoint_matches,  # noqa: F405
                         ransac_thresh=0.8, conf=0.999):
    """Estimate relative pose (R, t) mapping camera A to camera B.

    Returns dict:
      R        : (3,3) rotation A->B
      t        : (3,) unit translation direction A->B (scale ambiguous)
      n_in     : RANSAC-pose inliers
      pts_a/pts_b : inlier matches in (x, y) pixel coords
      t_scaled : t * median relative depth (only if depth_a given; NOT metric)
    """
    if K_b is None:
        K_b = K_a

    # matches come back as (row, col); OpenCV wants (x, y) = (col, row)
    kps_a_rc, kps_b_rc = matcher(img_a, img_b)
    if len(kps_a_rc) < 8:
        raise ValueError(f"Not enough matches: {len(kps_a_rc)}")
    pts_a = kps_a_rc[:, ::-1].astype(np.float64)
    pts_b = kps_b_rc[:, ::-1].astype(np.float64)

    # Fundamental matrix (handles the two different intrinsics), then E = K_b^T F K_a
    F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, ransac_thresh, conf)
    if F is None:
        raise ValueError("Fundamental matrix estimation failed")
    mask = mask.ravel().astype(bool)
    pts_a, pts_b = pts_a[mask], pts_b[mask]
    E = K_b.T @ F @ K_a  # Diverse signature quindi devo ricosturire Essential assume stesse intrinisc

    # normalize points by each K, then recoverPose on identity intrinsics
    def _norm(pts, K):
        ph = np.hstack([pts, np.ones((len(pts), 1))])
        return (np.linalg.inv(K) @ ph.T).T[:, :2]
    na, nb = _norm(pts_a, K_a), _norm(pts_b, K_b)
    n_in, R, t, mask_pose = cv2.recoverPose(E, na, nb)
    t = t.ravel()

    out = {"R": R, "t": t, "n_in": int(mask_pose.sum()),
           "pts_a": pts_a, "pts_b": pts_b, "n_matches": len(mask)}

    if depth_a is not None:
        inv = np.clip(depth_a[pts_a[:, 1].astype(int), pts_a[:, 0].astype(int)], 1e-6, None)
        median_depth = float(np.median(1.0 / inv))   # relative, not metric
        out["t_scaled"] = t * median_depth
        out["median_rel_depth"] = median_depth

    return out


# ===================== cell 9: disparity -> depth (verbatim; only this fn is needed) =====================
def disparity_to_depth(disp, near=1.0, far=8.0, p_lo=2, p_hi=98):
    """Robustly map a disparity-like map (large=near) to depth Z (small=near)."""
    d = disp.astype(np.float64)
    lo, hi = np.percentile(d, [p_lo, p_hi])           # ignore extreme outliers
    d = np.clip((d - lo) / (hi - lo + 1e-12), 0, 1)   # 0..1, 1 = nearest
    # interpolate inverse-depth linearly between 1/far and 1/near  -> avoids the long far tail
    inv_z = d * (1.0 / near - 1.0 / far) + 1.0 / far
    return 1.0 / inv_z


# ===================== cell 11: forward warp / view synthesis (verbatim) =====================
def synthesize_view(img_a, depth_a, K_a, R, t, K_b, out_hw):
    """Proietta i pixel di A nel piano immagine di B usando depth + (R,t).
    img_a:  (H,W,3) RGB [0,1] di A
    depth_a:(H,W)   profondita' Z di A (stessa scala di t)
    K_a/K_b:(3,3) intrinsics; R,t: posa A->B (mondo->cam_B applicata a punti in cam_A)
    Ritorna (warped RGB, mask validi, depth in B, src_idx):
      src_idx (Ho,Wo) int64 = indice flat del pixel di A che ha vinto lo z-buffer (-1 se vuoto).
      Serve alla prova del round trip per tracciare la provenienza; chi non lo usa puo' ignorarlo."""

    H, W = depth_a.shape
    Ho, Wo = out_hw

    fx, fy, cx, cy = K_a[0, 0], K_a[1, 1], K_a[0, 2], K_a[1, 2]

    # 1) back-project A -> 3D nel frame camera di A
    vs, us = np.mgrid[0:H, 0:W]
    Z = depth_a.astype(np.float64)
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)          # (N,3)

    # 2) trasforma nel frame di B:  P_b = R P_a + t
    Pb = P @ R.T + t.reshape(1, 3)
    zb = Pb[:, 2]
    front = zb > 1e-6                                        # davanti alla camera B

    # 3) proietta con K_b
    u = (K_b[0, 0] * Pb[:, 0] / zb + K_b[0, 2])
    v = (K_b[1, 1] * Pb[:, 1] / zb + K_b[1, 2])
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    inb = front & (ui >= 0) & (ui < Wo) & (vi >= 0) & (vi < Ho)

    src_flat = np.arange(H * W, dtype=np.int64)              # indice del pixel sorgente di A
    cols = img_a.reshape(-1, 3)
    ui, vi, zb_v, cols, src_v = ui[inb], vi[inb], zb[inb], cols[inb], src_flat[inb]

    # 4) z-buffer: ad ogni pixel di B tieni il punto piu' vicino (gestisce le occlusioni)
    out = np.zeros((Ho, Wo, 3), np.float32)
    zbuf = np.full((Ho, Wo), np.inf, np.float64)
    sidx = np.full((Ho, Wo), -1, np.int64)
    flat = vi * Wo + ui

    order = np.argsort(-zb_v)                                # dal piu' lontano al piu' vicino
    out.reshape(-1, 3)[flat[order]] = cols[order]           # i vicini sovrascrivono i lontani
    zbuf.reshape(-1)[flat[order]] = zb_v[order]
    sidx.reshape(-1)[flat[order]] = src_v[order]
    mask = np.isfinite(zbuf)

    return out, mask, zbuf, sidx


# ===================== cell 13: extreme virtual views (verbatim helpers; NO inpaint) =====================
def rot_xyz(pitch_deg, yaw_deg, roll_deg):
    """Rotazione mondo->camera dagli angoli di Eulero (gradi)."""
    return Rotation.from_euler("xyz", [pitch_deg, yaw_deg, roll_deg], degrees=True).as_matrix()


def look_pose(pitch_deg, yaw_deg, roll_deg, tx, ty, tz, scene_z):
    """Costruisce (R, t) per una camera virtuale che ruota e trasla (t in unita' di SCENE_Z)."""
    R = rot_xyz(pitch_deg, yaw_deg, roll_deg)
    # vogliamo orbitare attorno alla scena: ruota intorno al centro scena, non all'origine camera
    center = np.array([0.0, 0.0, scene_z])      # punto guardato (di fronte alla camera A)
    t = -R @ center + center + np.array([tx, ty, tz]) * scene_z
    return R, t


# Banco di viste estreme: (nome, pitch, yaw, roll, tx, ty, tz). Angoli/traslazioni scalati da EXTREMITY.
VIEW_BANK = [
    ("yaw sinistra",     0,  -25,  0,  -0.30,  0.0,  0.0),
    ("yaw destra",       0,   25,  0,   0.30,  0.0,  0.0),
    ("dal basso",       22,    0,  0,   0.0,   0.30, 0.0),
    ("avvicinata",       0,    0,  0,   0.0,   0.0,  0.55),
    ("3/4 + roll",     -15,   30, 12,   0.25, -0.20, 0.15),
]
VIEW_BY_NAME = {v[0]: v for v in VIEW_BANK}


# ===================== shared no-inpaint wrapper =====================
def warp_view(img_a, depth_a, K_src, R, t, K_dst, out_hw):
    """Forward-warp img_a into the target view and return (warp_linear, mask) with NO inpaint.

    warp is linear float32 [0,1] with holes left at 0; mask is bool (True = a real pixel landed).
    Used identically by the Samsung-view and the extreme-view paths so they cannot diverge."""
    warp, mask, _, _ = synthesize_view(img_a, depth_a, K_src, R, t, K_dst, out_hw=out_hw)
    return warp.astype(np.float32), mask.astype(bool)


# ===================== 16-bit PNG writers (linear, lossless) =====================
def save_rgb16(path, img01):
    """Write a LINEAR RGB float [0,1] image as a 16-bit PNG (no sRGB encode, lossless 65535 levels)."""
    u16 = (np.clip(img01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    cv2.imwrite(str(path), cv2.cvtColor(u16, cv2.COLOR_RGB2BGR))


def save_mask16(path, mask_bool):
    """Write a bool validity mask as a 16-bit single-channel PNG (0 / 65535)."""
    cv2.imwrite(str(path), (mask_bool.astype(np.uint16) * 65535))


# ===================== driver =====================
def stems():
    """Sorted .mat stems from the iPhone folder (canonical pairing key)."""
    return sorted(p.stem for p in (PAIRED / "iphone-x/dng").glob("*.mat"))


def pass1_depth(stem_list):
    """Load Depth Anything once, cache disparity-like output_a per stem to disk, then free VRAM."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    DA_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
    print(f"[pass1] loading Depth Anything: {DA_MODEL}")
    da_processor = AutoImageProcessor.from_pretrained(DA_MODEL)
    da_model = AutoModelForDepthEstimation.from_pretrained(DA_MODEL).to(device).eval()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(stem_list, 1):
        img_a = read_mat_fullres(PAIRED / f"iphone-x/dng/{s}.mat")
        output_a = depth_anything(img_a, da_processor, da_model)
        np.save(CACHE_DIR / f"{s}.npy", output_a.astype(np.float32))
        print(f"[pass1] {i:>3}/{len(stem_list)}  {s}  depth range "
              f"[{float(output_a.min()):.2f}, {float(output_a.max()):.2f}]")

    # --- free Depth Anything from VRAM before RoMa loads (cell 5, verbatim spirit) ---
    del da_model, da_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        free, total = torch.cuda.mem_get_info()
        print(f"[pass1] Depth Anything deallocated. CUDA free: {free/1e9:.2f} / {total/1e9:.2f} GB")
    else:
        print("[pass1] Depth Anything deallocated (CPU run).")


def pass2_warp(stem_list):
    """Load RoMa (cached on first match) and produce the warped views + masks for every stem."""
    for d in (OUT_IPH, OUT_IPH_S, OUT_IPH_E, OUT_MASK_S, OUT_MASK_E):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    manifest = []
    n_ok = n_skip = 0

    for i, s in enumerate(stem_list, 1):
        img_a = read_mat_fullres(PAIRED / f"iphone-x/dng/{s}.mat")
        img_b = read_mat_fullres(PAIRED / f"samsung-s9/dng/{s}.mat")
        output_a = np.load(CACHE_DIR / f"{s}.npy")              # cached Depth Anything output
        depth_a = disparity_to_depth(output_a, near=1.0, far=8.0)  # cell 9, verbatim params

        K_a, K_b = intrinsics_for(img_a, img_b)

        # pose A->B; passing output_a (disparity-like) exactly as the notebook does -> t_scaled
        try:
            pose = relative_pose_a_to_b(img_a, img_b, K_a, K_b, depth_a=output_a)
        except (ValueError, cv2.error) as e:
            print(f"[pass2] {i:>3}/{len(stem_list)}  {s}  SKIP (pose failed: {e})")
            manifest.append((s, "", "", "", "", "skip"))
            n_skip += 1
            continue

        # --- iphone-s: synthesized Samsung view (cell 11) ---
        t_scaled = pose.get("t_scaled", pose["t"])
        iph_s, mask_s = warp_view(img_a, depth_a, K_a, pose["R"], t_scaled, K_b, img_b.shape[:2])

        # --- iphone-e: one random slight-extreme view from the subset (cell 13) ---
        scene_z = float(np.median(depth_a))
        view = VIEW_BY_NAME[E_VIEW_SUBSET[int(rng.integers(len(E_VIEW_SUBSET)))]]
        name, pitch, yaw, roll, tx, ty, tz = view
        R_e, t_e = look_pose(pitch * EXTREMITY, yaw * EXTREMITY, roll * EXTREMITY,
                             tx * EXTREMITY, ty * EXTREMITY, tz * EXTREMITY, scene_z)
        iph_e, mask_e = warp_view(img_a, depth_a, K_a, R_e, t_e, K_a, img_a.shape[:2])

        # --- write 16-bit PNGs (linear RGB, lossless; masks 0/65535) ---
        save_rgb16(OUT_IPH    / f"{s}.png", img_a)
        save_rgb16(OUT_IPH_S  / f"{s}.png", iph_s)
        save_rgb16(OUT_IPH_E  / f"{s}.png", iph_e)
        save_mask16(OUT_MASK_S / f"{s}.png", mask_s)
        save_mask16(OUT_MASK_E / f"{s}.png", mask_e)

        s_cov, e_cov = 100 * float(mask_s.mean()), 100 * float(mask_e.mean())
        manifest.append((s, pose["n_in"], f"{s_cov:.1f}", name, f"{e_cov:.1f}", "ok"))
        n_ok += 1
        print(f"[pass2] {i:>3}/{len(stem_list)}  {s}  inliers={pose['n_in']:<4} "
              f"s_cov={s_cov:4.1f}%  e='{name}' e_cov={e_cov:4.1f}%")

    # --- manifest ---
    with open(OUT_ROOT / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "samsung_pose_inliers", "s_coverage_pct",
                    "e_view_name", "e_coverage_pct", "status"])
        w.writerows(manifest)

    print(f"\n[done] processed={n_ok}  skipped={n_skip}  -> {OUT_ROOT}")
    print(f"       folders: {OUT_IPH.name}, {OUT_IPH_S.name}, {OUT_IPH_E.name}, "
          f"{OUT_MASK_S.name}, {OUT_MASK_E.name}  (+ manifest.csv)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N stems (smoke test)")
    args = ap.parse_args()

    stem_list = stems()
    if args.limit is not None:
        stem_list = stem_list[:args.limit]
    print(f"stems: {len(stem_list)}  (EXTREMITY={EXTREMITY}, SEED={SEED})")

    pass1_depth(stem_list)
    pass2_warp(stem_list)


if __name__ == "__main__":
    main()
