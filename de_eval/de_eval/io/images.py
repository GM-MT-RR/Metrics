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


def load_view(path: str | Path, imsize: Optional[int] = None) -> np.ndarray:
    """Load one view as float32 (H, W, 3) in [0, 1].

    .mat files are demosaiced via read_mat_image; everything else via OpenCV.
    """
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return read_mat_image(path, imsize=imsize)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if imsize is not None:
        h, w = img.shape[:2]
        scale = imsize / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


class PairedViewDataset:
    """Index of (source, target) view pairs matched by filename stem."""

    def __init__(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        extensions: Sequence[str] = (".mat",),
        sample_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.extensions = tuple(e.lower() for e in extensions)

        src_files = self._list(self.source_dir)
        tgt_by_stem = {p.stem: p for p in self._list(self.target_dir)}

        pairs: List[Tuple[Path, Path]] = []
        for sp in src_files:
            tp = tgt_by_stem.get(sp.stem)
            if tp is not None:
                pairs.append((sp, tp))

        if sample_indices is not None:
            pairs = [pairs[i] for i in sample_indices if 0 <= i < len(pairs)]

        self.pairs = pairs

    def _list(self, d: Path) -> List[Path]:
        return sorted(p for p in d.glob("*") if p.suffix.lower() in self.extensions)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[Tuple[Path, Path]]:
        return iter(self.pairs)

    def relpath(self, path: str | Path) -> Path:
        path = Path(path)
        try:
            return path.relative_to(self.source_dir)
        except ValueError:
            return Path(path.name)
