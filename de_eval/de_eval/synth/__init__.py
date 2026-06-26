"""Synthetic-view generators.

Register a new generator with ``@register_synth("name")`` and import its module
here so registration runs at import time.
"""
from .base import BaseSynth, SynthResult  # noqa: F401
from .registry import SYNTH_REGISTRY, get_synth, register_synth  # noqa: F401
from . import homography_flow  # noqa: F401

__all__ = ["BaseSynth", "SynthResult", "SYNTH_REGISTRY", "get_synth", "register_synth"]
