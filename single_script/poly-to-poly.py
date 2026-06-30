"""Round-trip proof: iPhone-X -> simulated Samsung-S9 -> simulated iPhone-X.

For every (iPhone-X, Samsung-S9) RAW pair whose filename timestamps are
within DT_THRESHOLD_S seconds of each other:

  1. Load both RAW RGGB images.
  2. Map the iPhone-X image with its own mapping_matrix (iPhone -> S9) to
     obtain a simulated S9 image.
  3. Map that simulated S9 image with the paired Samsung-S9 image's own
     mapping_matrix (S9 -> iPhone) to obtain a simulated iPhone-X image.
  4. Compute Delta E 2000 between the simulated iPhone-X (round-trip) and
     the original real iPhone-X image.

The two mapping_matrix kernels are the raw2raw polynomial mappings from
Afifi & Abuolaim, "Semi-Supervised Raw-to-Raw Mapping" (arXiv:2106.13883),
each stored under <camera>/mapping/<own_timestamp>.mat.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import rawpy
from scipy.io import loadmat

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thesis_lib_shared"))
from thesis_lib_shared.metrics import delta_e_images, delta_e_summary

DATASET_ROOT = Path(__file__).resolve().parent.parent / "data/iphone2samsung_or/paired"
IPHONE_DIR = DATASET_ROOT / "iphone-x"
SAMSUNG_DIR = DATASET_ROOT / "samsung-s9"
DT_THRESHOLD_S = 60.0
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

META_IPHONE = {"white_level": (2 ** 12) - 1, "black_level": 528,
               "cfa_pattern": np.array([0, 1, 1, 2])}
META_SAMSUNG = {"white_level": (2 ** 10) - 1, "black_level": 0,
                "cfa_pattern": np.array([1, 0, 2, 1])}


def parse_timestamp(dng_path: Path) -> datetime:
    return datetime.strptime(dng_path.stem, TIMESTAMP_FORMAT)


def find_dt_pairs(iphone_dir: Path, samsung_dir: Path, dt_threshold_s: float):
    """Pair iPhone/Samsung dng files whose timestamps differ by < dt_threshold_s."""
    iphone_files = sorted((iphone_dir / "dng").glob("*.dng"))
    samsung_files = sorted((samsung_dir / "dng").glob("*.dng"))
    samsung_ts = [(parse_timestamp(f), f) for f in samsung_files]

    pairs = []
    for iphone_path in iphone_files:
        ts_a = parse_timestamp(iphone_path)
        best = min(samsung_ts, key=lambda ts_f: abs((ts_f[0] - ts_a).total_seconds()))
        dt = abs((best[0] - ts_a).total_seconds())
        if dt < dt_threshold_s:
            pairs.append((iphone_path, best[1], dt))
    return pairs


def pack_rggb(raw_image: np.ndarray, cfa: np.ndarray) -> np.ndarray:
    height, width = raw_image.shape
    cfa = cfa.copy()
    cfa[cfa == 2] += 1
    cfa[2:][cfa[2:] == 1] += 1
    idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    channels = [raw_image[idx[c][0]:height:2, idx[c][1]:width:2] for c in cfa]
    return np.stack(channels, axis=-1)


def load_raw_rggb(dng_path: Path, meta: dict) -> np.ndarray:
    raw_bayer = rawpy.imread(str(dng_path)).raw_image_visible.astype(np.float32)
    raw_norm = (raw_bayer - meta["black_level"]) / (meta["white_level"] - meta["black_level"])
    raw_norm = np.clip(raw_norm, 0.0, 1.0)
    return pack_rggb(raw_norm, meta["cfa_pattern"])


def kernelP(rggb: np.ndarray) -> np.ndarray:
    r, gr, gb, b = np.split(rggb, 4, axis=1)
    return np.concatenate(
        [rggb, rggb ** 2, r * gr, r * gb, r * b, gr * gb, gr * b, gb * b,
         r * gr * gb * b, np.ones_like(r)],
        axis=1,
    )


def apply_mapping(img: np.ndarray, mapping_matrix: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    flat = np.reshape(img, (-1, 4))
    mapped = kernelP(flat) @ mapping_matrix
    mapped = np.reshape(mapped, (h, w, c))
    return np.clip(mapped, 0.0, 1.0)


def load_mapping_matrix(camera_dir: Path, dng_path: Path) -> np.ndarray:
    mat = loadmat(str(camera_dir / "mapping" / (dng_path.stem + ".mat")))
    return mat["mapping_matrix"]


def from_rggb_to_rgb(rggb: np.ndarray) -> np.ndarray:
    rgb = rggb.copy()
    rgb[:, :, 1] = (rggb[:, :, 1] + rggb[:, :, 2]) / 2
    return rgb[:, :, [0, 1, 3]]


def main():
    pairs = find_dt_pairs(IPHONE_DIR, SAMSUNG_DIR, DT_THRESHOLD_S)
    print(f"Found {len(pairs)} pairs with dt < {DT_THRESHOLD_S:.0f}s")

    for iphone_path, samsung_path, dt in pairs:
        iphone_to_s9 = load_mapping_matrix(IPHONE_DIR, iphone_path)
        s9_to_iphone = load_mapping_matrix(SAMSUNG_DIR, samsung_path)

        iphone_rggb = load_raw_rggb(iphone_path, META_IPHONE)

        simulated_s9_rggb = apply_mapping(iphone_rggb, iphone_to_s9)
        simulated_iphone_rggb = apply_mapping(simulated_s9_rggb, s9_to_iphone)

        iphone_rgb = from_rggb_to_rgb(iphone_rggb)
        simulated_iphone_rgb = from_rggb_to_rgb(simulated_iphone_rggb)

        de_map = delta_e_images(simulated_iphone_rgb, iphone_rgb)
        stats = delta_e_summary(de_map)

        print(f"{iphone_path.stem} - {samsung_path.stem} dt={dt:.1f}s: "
              f"{ {k: round(v, 3) for k, v in stats.items()} }")


if __name__ == "__main__":
    main()
