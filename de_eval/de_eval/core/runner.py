"""Orchestrates one experiment (and batches of them).

Per image pair:
  load src(iPhone)+tgt(Samsung) -> synth misaligned-same-colour view of src
  -> matcher re-aligns synth onto src -> ΔE over kept/ matched pixels.

A perfect matcher recovers ΔE ≈ 0; the residual is the matcher's geometric error.

Experiment layout (under ``<save_dir>/<experiment>/``), mirroring Standard:
    config.yaml         snapshot of the config used
    run.log             rich run log
    metrics/de.csv      per-image ΔE rows + summary row
    summary.json        aggregate ΔE stats
    <pair>/images/*.png debug visualisations (when save_debug)
"""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, BatchConfig
from .logger import Logger
from ..io import PairedViewDataset, load_view
from ..synth import get_synth
from ..eval import delta_e_pairs, summarize
from ..report import (
    write_experiment_csv,
    aggregate_experiment,
    write_batch_summary,
    save_keypoint_matches,
    save_sparse_pairs,
    save_masked_overlay,
)


def _save_debug_image(img: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    cmap = "inferno" if img.ndim == 2 else None
    plt.imshow(np.clip(img, 0, 1) if img.ndim == 3 else img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close()


def _process_pair(args: Tuple) -> Dict:
    """Worker: synth + align + ΔE for one pair. Returns a per-image row dict."""
    from ..pipeline import Pipeline  # lazy: avoids core<->pipeline import cycle

    (src_path, tgt_path, rel, cfg_dict) = args
    cfg = Config(**cfg_dict)

    synth = get_synth(cfg.synth.type)(**cfg.synth.params)
    pipeline = Pipeline.from_matcher_config(cfg.matcher)

    src = load_view(src_path, imsize=cfg.imsize)
    tgt = load_view(tgt_path, imsize=cfg.imsize)

    sr = synth.synthesize(src, tgt)
    res = pipeline.run(src, sr.image, valid_mask=sr.valid_mask)

    # Single ΔE path: every extractor normalizes to matched RGB pairs.
    de_vals = delta_e_pairs(res.pairs_a, res.pairs_b, linearize=cfg.eval.linearize)

    stats = summarize(de_vals)
    row = {
        "image": str(rel),
        "de_mean": stats["mean"],
        "de_median": stats["median"],
        "de_p95": stats["p95"],
        "de_p99": stats["p99"],
        "de_max": stats["max"],
        "n_pairs": res.n_pairs,
    }

    if cfg.save_debug:
        out_dir = Path(cfg.save_dir) / cfg.experiment / Path(str(rel)).stem / "images"
        _save_debug_visuals(out_dir, src, sr, res, de_vals)

    return row


def _save_debug_visuals(out_dir, src, sr, res, de_vals) -> None:
    """Notebook-style diagnostics for one pair (gated by save_debug).

    Every extractor now returns the sparse correspondence form (matched RGB
    pairs), so the diagnostics are the sparse set: matched keypoints, extracted
    patches + pair swatches, and the correspondence-mask overlay, plus any named
    stage debug images (residual flow, disparity, certainty, gradient keep…).
    """
    # Inputs: the original src view + the synthetic misaligned-same-colour view.
    _save_debug_image(src, "src (iPhone, reference colours)", out_dir / "src.png")
    _save_debug_image(sr.image, "synthetic view (same colours, warped)", out_dir / "synth.png")

    # Matched keypoints between the two full-res views (lines + markers).
    if res.kps_a is not None and res.kps_b is not None and res.img_a_full is not None:
        save_keypoint_matches(
            res.img_a_full, res.img_b_full, res.kps_a, res.kps_b,
            out_dir / "keypoint_matches.png",
            title=f"matched keypoints: {res.kps_a.shape[0]} "
                  f"({res.info.get('detector', '')}, showing 1/5)",
        )

    # The extracted match points + RGB-pair swatches, plus the mask-applied view.
    if res.coords_a is not None and res.img_a_full is not None:
        save_sparse_pairs(
            res.img_a_full, res.img_b_full, res.coords_a, res.coords_b,
            res.pairs_a, res.pairs_b, np.asarray(de_vals),
            out_dir / "extracted_patches.png",
        )
        save_masked_overlay(
            res.img_a_full, res.coords_a, res.pairs_b, np.asarray(de_vals),
            out_dir / "masked_overlay.png",
        )

    # Any extra named debug images the stages produced (flow, disparity, keep…).
    for name, dbg in res.debug.items():
        _save_debug_image(dbg, name, out_dir / f"{name}.png")


def run_experiment(config: Config) -> Dict:
    exp_dir = Path(config.save_dir) / config.experiment
    if exp_dir.exists() and config.overwrite:
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False)
    Logger.attach_file(str(exp_dir / "run.log"))
    Logger.info(f"[bold]Experiment[/bold]: {config.experiment}")
    Logger.info(f"Synth: {config.synth.type}  {config.synth.params}")
    m = config.matcher
    Logger.info(
        f"Pipeline: align={m.align.type}{m.align.params}  "
        f"refine={m.refine.type}{m.refine.params}  patches={m.patches.type}{m.patches.params}"
    )

    ds = PairedViewDataset(
        config.data.source, config.data.target,
        extensions=config.image_ext, sample_indices=config.sample_indices,
    )
    Logger.info(f"Pairs: {len(ds)}")
    if len(ds) == 0:
        raise ValueError(f"No paired views found under {config.data.source}.")

    cfg_dict = config.model_dump()
    jobs = [(str(sp), str(tp), str(ds.relpath(sp)), cfg_dict) for sp, tp in ds]

    rows: List[Dict] = []
    # Single worker => run inline (cheaper, easier to debug; also for GPU matchers).
    if config.n_workers <= 1:
        for j in jobs:
            rows.append(_safe_process(j))
    else:
        with ProcessPoolExecutor(max_workers=config.n_workers) as pool:
            futures = {pool.submit(_safe_process, j): j for j in jobs}
            for fut in as_completed(futures):
                rows.append(fut.result())

    rows.sort(key=lambda r: r["image"])
    ok_rows = [r for r in rows if "error" not in r]
    for r in rows:
        if "error" in r:
            Logger.warn(f"{r['image']}: {r['error']}")

    write_experiment_csv(ok_rows, exp_dir / "metrics" / "de.csv")
    agg = aggregate_experiment(ok_rows)
    summary = {
        "experiment": config.experiment,
        "align": config.matcher.align.type,
        "refine": config.matcher.refine.type,
        "patches": config.matcher.patches.type,
        "detector": config.matcher.align.params.get("detector")
        or config.matcher.patches.params.get("detector", ""),
        "synth": config.synth.type,
        "n_images": len(ok_rows),
        **agg,
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    Logger.info(
        f"[{config.experiment}] mean ΔE={agg['de_mean']:.4f}  "
        f"median={agg['de_median']:.4f}  p95={agg['de_p95']:.4f}  p99={agg['de_p99']:.4f}"
    )
    return summary


def _safe_process(job: Tuple) -> Dict:
    rel = job[2]
    try:
        return _process_pair(job)
    except Exception as e:  # keep the batch alive; record the failure
        return {"image": rel, "error": f"{type(e).__name__}: {e}",
                "de_mean": float("nan"), "de_median": float("nan"),
                "de_p95": float("nan"), "de_p99": float("nan"),
                "de_max": float("nan"), "n_pairs": 0}


def run_batch(batch: BatchConfig) -> List[Dict]:
    summaries: List[Dict] = []
    for cfg in batch.to_single_configs():
        summaries.append(run_experiment(cfg))
    out_dir = Path(batch.save_dir)
    write_batch_summary(summaries, out_dir, title="ΔE matching benchmark")
    Logger.info(f"Wrote batch summary -> {out_dir}/results.csv, summary.md, summary.png")
    return summaries
