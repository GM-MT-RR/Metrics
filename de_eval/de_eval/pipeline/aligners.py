"""Alignment stage: bring ``img_b`` into ``img_a``'s frame.

Two interchangeable aligners:

- ``keypoint_homography`` — matched keypoints (akaze|sift|loftr|roma) ->
  ``cv2.findHomography`` -> ``warpPerspective``. Undoes a projective misalignment.
- ``roma_dense`` — TinyRoMa's dense warp resamples the *whole* of B into A's frame
  via one ``grid_sample`` (RoMa's own ``visualize_warp`` recipe) and also yields a
  per-pixel ``certainty``. This replaces the entire keypoint+homography geometry,
  so it is an alignment option, not a refiner.

Both leave A and B co-registered on the same pixel grid; the downstream refine and
patch-extraction stages assume that.
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from ..matchers.keypoints import detect_matches
from .context import AlignContext, BaseAligner, register_aligner

# Long side of the aspect-preserving working frame RoMa matches on (matching_RoMa
# cell 6: square resizes distort aspect; keep pixels square so fx == fy).
ROMA_LONG = 1024


@register_aligner("keypoint_homography")
class KeypointHomographyAligner(BaseAligner):
    def __init__(
        self,
        detector: str = "sift",
        use_ransac: bool = True,
        ransac_threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            detector=detector,
            use_ransac=use_ransac,
            ransac_threshold=ransac_threshold,
            **kwargs,
        )

    def align(self, img_a, img_b, valid_mask: Optional[np.ndarray] = None) -> AlignContext:
        p = self.params
        kps_a, kps_b = detect_matches(
            img_a, img_b,
            detector=p["detector"],
            use_ransac=p["use_ransac"],
            ransac_threshold=p["ransac_threshold"],
        )
        if kps_a.shape[0] < 4:
            raise RuntimeError(f"{p['detector']} produced <4 matches; cannot align.")
        pts_a = kps_a[:, ::-1].astype(np.float32)  # (row,col)->(x,y)
        pts_b = kps_b[:, ::-1].astype(np.float32)
        H, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, p["ransac_threshold"])
        if H is None:
            raise RuntimeError("findHomography returned None during alignment.")
        h, w = img_a.shape[:2]
        b_aligned = cv2.warpPerspective(img_b, H, (w, h))
        valid = cv2.warpPerspective(np.ones((h, w), np.float32), H, (w, h)) > 0.999
        if valid_mask is not None and valid_mask.shape == img_b.shape[:2]:
            # restrict to the synth's valid region too (warped into A's frame)
            valid = valid & (cv2.warpPerspective(
                valid_mask.astype(np.float32), H, (w, h)
            ) > 0.999)

        a_frame = np.clip(img_a.astype(np.float32), 0, 1)
        b_frame = np.clip(b_aligned.astype(np.float32), 0, 1)
        return AlignContext(
            a_frame=a_frame,
            b_aligned=b_frame,
            valid_mask=valid,
            kps_a=kps_a, kps_b=kps_b, img_a_full=img_a, img_b_full=img_b,
            debug={
                "step__0_input_b": np.clip(img_b.astype(np.float32), 0, 1),
                "step__1_a_frame": a_frame,
                "step__2_b_aligned": b_frame,
                "step__2_valid_mask": valid.astype(np.float32),
            },
            info={"n_keypoints": int(kps_a.shape[0]), "detector": p["detector"]},
        )


@register_aligner("roma_dense")
class RomaDenseAligner(BaseAligner):
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        smooth_certainty: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            confidence_threshold=confidence_threshold,
            smooth_certainty=smooth_certainty,
            **kwargs,
        )

    def align(self, img_a, img_b, valid_mask: Optional[np.ndarray] = None) -> AlignContext:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from romatch import tiny_roma_v1_outdoor

        p = self.params
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _to_pil(img):
            arr = np.clip(img.astype(np.float32), 0, 1) * 255.0
            return Image.fromarray(arr.astype(np.uint8), mode="RGB")

        # Aspect-preserving working frame (square pixels) for both views.
        h0, w0 = img_a.shape[:2]
        s = ROMA_LONG / max(h0, w0)
        rw, rh = round(w0 * s), round(h0 * s)
        small_a = cv2.resize(img_a, (rw, rh))
        small_b = cv2.resize(img_b, (rw, rh))

        roma = tiny_roma_v1_outdoor(device=device)
        roma.train(False)
        # warp (h_a, w_a, 4): [..,0:2]=this A-pixel's own coords, [..,2:4]=matched B
        # loc (both normalized [-1, 1]); certainty (h_a, w_a) = confidence over A.
        warp, certainty = roma.match(_to_pil(small_a), _to_pil(small_b), batched=False)

        # --- Resample B into A's frame (RoMa's own visualize_warp, tiny.py:160) ---
        b_t = torch.from_numpy(np.ascontiguousarray(small_b)).permute(2, 0, 1)[None].float().to(device)
        grid = warp[..., 2:][None]                               # (1, h_a, w_a, 2)
        b_on_a = F.grid_sample(b_t, grid, mode="bilinear", align_corners=False)[0]
        b_on_a = b_on_a.permute(1, 2, 0).cpu().numpy()           # (h_a, w_a, 3) B in A frame

        # Resample the incoming B-validity mask with the SAME grid as B, so known-
        # wrong B pixels land in A's frame exactly where B's content did. Without
        # this the mask (seeded into valid_mask upstream) would be silently dropped.
        mask_on_a = None
        if valid_mask is not None:
            vm_small = cv2.resize(valid_mask.astype(np.float32), (rw, rh),
                                  interpolation=cv2.INTER_NEAREST)
            vm_t = torch.from_numpy(np.ascontiguousarray(vm_small))[None, None].float().to(device)
            vm_on_a = F.grid_sample(vm_t, grid, mode="nearest", align_corners=False)[0, 0]
            mask_on_a = vm_on_a.cpu().numpy() > 0.5             # (h_a, w_a) in A frame

        cert = certainty
        if p["smooth_certainty"]:
            cert = F.avg_pool2d(cert[None, None], kernel_size=5, stride=1, padding=2)[0, 0]
        cert = cert.cpu().numpy()                                # (h_a, w_a) in A frame

        a_frame = np.clip(small_a.astype(np.float32), 0, 1)      # A in its own frame
        b_frame = np.clip(b_on_a.astype(np.float32), 0, 1)       # B resampled onto A
        confident = cert > p["confidence_threshold"]             # valid where confident
        if mask_on_a is not None:                                # & known-valid B pixels
            confident = confident & mask_on_a

        return AlignContext(
            a_frame=a_frame, b_aligned=b_frame, valid_mask=confident,
            certainty=np.clip(cert, 0, 1).astype(np.float32),
            img_a_full=a_frame, img_b_full=b_frame,
            debug={
                "certainty": np.clip(cert, 0, 1).astype(np.float32),
                "step__1_a_frame": a_frame,
                "step__2_b_aligned": b_frame,
                "step__2_valid_mask": confident.astype(np.float32),
            },
            info={"confidence_threshold": p["confidence_threshold"],
                  "n_confident": int(confident.sum())},
        )
