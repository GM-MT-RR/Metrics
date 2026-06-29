"""Pydantic config models for the ΔE matching benchmark.

A single config picks one ``synth`` (how the misaligned-same-color view is made)
and one ``matcher`` (how it is re-aligned), plus shared ``data`` / ``eval``
settings. A batch config shares everything and sweeps a list of matchers, exactly
like ``Standard/standard/core/config.py``'s ``BatchConfig.to_single_configs``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Source = view to keep colours from (iPhone); target = pose donor (Samsung)."""
    source: str
    target: str
    mask: Optional[str] = None   # dir of B validity masks; one file per pair, named by source stem
                                 # (non-zero pixel == VALID). null => no per-pair masking.
    mask_erode: int = 0          # px to shrink the VALID region by (morphological erosion),
                                 # discarding a rim around invalid areas to absorb resize bleed.


class SynthConfig(BaseModel):
    """How to synthesize the misaligned-but-same-color view from ``source``."""
    type: str = "homography_flow"
    params: Dict[str, Any] = Field(default_factory=dict)


class StageConfig(BaseModel):
    """One pipeline stage: a registered component name + its params."""
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class MatcherConfig(BaseModel):
    """Staged re-alignment matcher: align -> refine -> extract patches.

    Each stage names a registered component (see ``de_eval.pipeline``), so any
    combination is a config choice rather than a bespoke class:
      align:   keypoint_homography | roma_dense
      refine:  none | flow | raft
      patches: gradient | sgbm
    """
    align: StageConfig = Field(
        default_factory=lambda: StageConfig(type="keypoint_homography",
                                            params={"detector": "sift"}))
    refine: StageConfig = Field(default_factory=lambda: StageConfig(type="none"))
    patches: StageConfig = Field(default_factory=lambda: StageConfig(type="gradient"))


class EvalConfig(BaseModel):
    """ΔE evaluation settings shared by all matchers."""
    gradient_threshold: float = 25.0   # keep only flat (low-L*-gradient) pixels
    linearize: bool = False            # sRGB->linear before Lab (gamma-encoded inputs)


class Config(BaseModel):
    """Single experiment: one synth + one matcher over one image set."""
    experiment: str
    save_dir: str = "Experiments_de_eval"
    overwrite: bool = False
    data: DataConfig
    synth: SynthConfig = Field(default_factory=SynthConfig)
    matcher: MatcherConfig = Field(default_factory=MatcherConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    image_ext: List[str] = Field(default_factory=lambda: [".mat"])
    imsize: Optional[int] = None
    sample_indices: Optional[List[int]] = None
    n_workers: int = 8
    save_debug: bool = False


class ExperimentEntry(BaseModel):
    """One matcher inside a batch; may override the shared matcher/synth/eval."""
    name: str
    matcher: MatcherConfig
    synth: Optional[SynthConfig] = None
    eval: Optional[EvalConfig] = None
    save_debug: Optional[bool] = None


class BatchConfig(BaseModel):
    """Run many matchers against the same data + synth in one go."""
    save_dir: str = "Experiments_de_eval"
    overwrite: bool = False
    data: DataConfig
    synth: SynthConfig = Field(default_factory=SynthConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    experiments: List[ExperimentEntry]
    image_ext: List[str] = Field(default_factory=lambda: [".mat"])
    imsize: Optional[int] = None
    sample_indices: Optional[List[int]] = None
    n_workers: int = 8
    save_debug: bool = False

    def to_single_configs(self) -> List[Config]:
        out: List[Config] = []
        for exp in self.experiments:
            out.append(Config(
                experiment=exp.name,
                save_dir=self.save_dir,
                overwrite=self.overwrite,
                data=self.data,
                synth=exp.synth if exp.synth is not None else self.synth,
                matcher=exp.matcher,
                eval=exp.eval if exp.eval is not None else self.eval,
                image_ext=self.image_ext,
                imsize=self.imsize,
                sample_indices=self.sample_indices,
                n_workers=self.n_workers,
                save_debug=exp.save_debug if exp.save_debug is not None else self.save_debug,
            ))
        return out


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "experiments" in raw and "matcher" not in raw:
        raise ValueError(
            f"'{path}' looks like a batch config (has 'experiments', no top-level 'matcher'). "
            "Use `python main.py batch -c <config>` instead of `run`."
        )
    return Config(**raw)


def load_batch_config(path: str | Path) -> BatchConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return BatchConfig(**raw)
