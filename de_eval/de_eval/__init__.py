"""de_eval — modular, YAML-configurable ΔE matching benchmark.

Synthesizes a misaligned-but-same-color view of an iPhone capture (copying the
Samsung-S9 geometry plus an optional nonlinear optical-flow field), then runs a
configured matcher to re-align it and measures the residual ΔE₀₀. A perfect
matcher recovers ΔE ≈ 0; any residual is geometric error leaking into color.

Public surface mirrors ``Standard``: registries for synthesizers, matchers, and
(implicitly) experiments, all driven by a pydantic ``Config`` loaded from YAML.
"""
from __future__ import annotations

from . import _paths  # noqa: F401  (side effect: shared libs on sys.path)

__all__ = ["_paths"]
