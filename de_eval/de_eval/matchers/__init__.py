"""Matcher building blocks shared by the staged pipeline.

The matcher is no longer a single fused class — it is composed from three
swappable stages in ``de_eval.pipeline`` (align / refine / extract). This package
now only holds the pieces those stages reuse:

- ``AlignResult`` — the sparse (matched RGB pairs) result the extractor returns.
- ``keypoints.detect_matches`` — keypoint-detector dispatch (akaze|sift|loftr|roma)
  used by the keypoint_homography aligner and the sgbm extractor.
"""
from .base import AlignResult  # noqa: F401
from . import keypoints  # noqa: F401

__all__ = ["AlignResult"]
