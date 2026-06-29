"""Paired-view dataset and image loaders.

``source`` holds the iPhone-X views (whose colours we keep) and ``target`` the
Samsung-S9 views (whose pose/geometry we copy). Files are paired by stem; .mat
captures are demosaiced via ``thesis_lib_shared.io.read_mat_image``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .. import _paths  # noqa: F401  (shared libs on path)
from thesis_lib_shared.io import read_mat_image


def _resize_shorter_side(img: np.ndarray, imsize: int,
                         interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize so the shorter side == imsize (aspect-preserving), like load_view."""
    h, w = img.shape[:2]
    scale = imsize / min(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interpolation)


def load_mask(path: str | Path, imsize: Optional[int] = None,
              erode: int = 0) -> np.ndarray:
    """Load a B-validity mask as a (H, W) bool array (True == VALID / white pixel).

    Read as-is, collapsed to a single channel, resized by the same ``imsize`` rule
    as ``load_view`` (NEAREST, to keep it a clean binary mask), and thresholded so
    any non-zero pixel is VALID. This rides through the pipeline in lockstep with
    image B (it seeds ``valid_mask``), so invalid B pixels are excluded from ΔE.

    ``erode`` > 0 shrinks the VALID (white) region by that many pixels via a
    morphological erosion (square kernel of side ``2*erode + 1``), discarding a rim
    around every invalid (black) area. The downstream resizes (imsize here, then
    the extractor's own resize) interpolate and bleed invalid content a pixel or
    two into neighbouring valid pixels; eroding trims that contaminated rim so it
    never reaches the ΔE. The erosion is applied AFTER the imsize resize, so the
    kernel is measured in working-resolution pixels.
    """
    path = Path(path)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if raw.ndim == 3:                       # any colour/alpha -> single channel
        raw = raw[..., 0]
    if imsize is not None:
        raw = _resize_shorter_side(raw, imsize, interpolation=cv2.INTER_NEAREST)
    mask = raw > 0
    if erode > 0:
        k = 2 * int(erode) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        # erode the VALID (white) region -> grows the discarded (black) region
        mask = cv2.erode(mask.astype(np.uint8), kernel) > 0
    return mask


def load_view(path: str | Path, imsize: Optional[int] = None) -> np.ndarray:
    """Load one view as float32 (H, W, 3) in [0, 1].

    .mat files are demosaiced via read_mat_image; everything else via OpenCV.
    8-bit and 16-bit PNG/TIFF are supported: values are normalized by the true
    bit depth (uint8 -> 255, uint16 -> 65535) and returned as linear RGB
    (no sRGB decode), matching read_dng_image and the simulatron's linear PNGs.
    """
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return read_mat_image(path, imsize=imsize)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # Collapse to 3-channel BGR -> RGB, keeping the (H, W, 3) invariant.
    if raw.ndim == 2:                       # grayscale -> broadcast to 3 channels
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif raw.shape[2] == 4:                 # BGRA -> drop alpha, then BGR->RGB
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    else:                                   # BGR -> RGB
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    # Normalize by the true bit depth; float inputs are assumed already [0, 1].
    maxval = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(raw.dtype)
    img = raw.astype(np.float32)
    if maxval is not None:
        img /= maxval
    if imsize is not None:
        img = _resize_shorter_side(img, imsize, interpolation=cv2.INTER_AREA)
    return img


class PairedViewDataset:
    """Index of (source, target) view pairs matched by filename stem."""

    def __init__(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        extensions: Sequence[str] = (".mat",),
        sample_indices: Optional[Sequence[int]] = None,
        mask_dir: Optional[str | Path] = None,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.extensions = tuple(e.lower() for e in extensions)

        src_files = self._list(self.source_dir)
        tgt_by_stem = {p.stem: p for p in self._list(self.target_dir)}
        # Masks paired by SOURCE stem; tolerate any image extension and a missing
        # file per pair (so a partially-masked dataset still runs -> None).
        mask_by_stem = (
            {p.stem: p for p in sorted(self.mask_dir.glob("*")) if p.is_file()}
            if self.mask_dir is not None else {}
        )

        pairs: List[Tuple[Path, Path, Optional[Path]]] = []
        for sp in src_files:
            tp = tgt_by_stem.get(sp.stem)
            if tp is not None:
                pairs.append((sp, tp, mask_by_stem.get(sp.stem)))

        if sample_indices is not None:
            pairs = [pairs[i] for i in sample_indices if 0 <= i < len(pairs)]

        self.pairs = pairs

    def _list(self, d: Path) -> List[Path]:
        return sorted(p for p in d.glob("*") if p.suffix.lower() in self.extensions)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[Tuple[Path, Path, Optional[Path]]]:
        return iter(self.pairs)

    def relpath(self, path: str | Path) -> Path:
        path = Path(path)
        try:
            return path.relative_to(self.source_dir)
        except ValueError:
            return Path(path.name)
