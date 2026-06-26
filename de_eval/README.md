# de_eval — modular ΔE matching benchmark

The goal of the matching algorithm is to obtain **perfect pixel pairs** — the same
physical scene point in two views — so that ΔE₀₀ measures only **colour**
difference, not misregistration.

This package validates a matcher by **synthesizing a view we control**: it takes an
iPhone-X capture, copies the Samsung-S9 *geometry* (a noisy inverse homography) and
adds a smooth **nonlinear optical-flow** field, while keeping the **iPhone colours
untouched**. The synthetic view is therefore *misaligned but pixel-for-pixel the same
colour*, so the ground-truth ΔE is **0**. Whatever ΔE a matcher reports is its own
geometric error leaking into the colour metric.

```
src (iPhone, colours)              synth (same colours, Samsung-like + flow warp)
        └──────────── matcher re-aligns synth → src ───────────┘
                         ΔE over kept/matched pixels   →  expect ≈ 0
```

## Layout (mirrors `../../Standard`, registry + pydantic-config + CLI)

```
de_eval/
  main.py                 # CLI: run / batch / list
  de_eval/
    core/   config.py registry.py logger.py runner.py
    io/     images.py           # PairedViewDataset (iPhone src, Samsung tgt)
    synth/  homography_flow.py   # noisy H⁻¹ warp + optional smooth flow field
    matchers/
      keypoint_homography.py     # AKAZE|SIFT|LoFTR|RoMa kp → H → warp → resize → grad-mask (dense ΔE)
      keypoint_flow.py           # + Farneback optical-flow residual refine
      roma_dense.py              # RoMa dense warp → resample A into B's grid → conf+grad mask (dense ΔE)
      stereo_sgbm.py             # kp → fundamental → rectify → SGBM disparity → pairs (sparse ΔE)
      keypoints.py               # detector dispatch akaze|sift|loftr|roma
    eval/   delta_e.py gradient.py
    report/ tables.py debug.py   # results.csv + summary.md + summary.png; per-pair debug figures
  configs/*.yaml
```

## Usage

```bash
cd MetricDeProof/de_eval

# one experiment
python main.py run   -c configs/homflow_sift_flow.yaml

# sweep every matcher, emit combined tables
python main.py batch -c configs/all_matchers_Batch.yaml

# list registered components
python main.py list
```

Outputs land in `Experiments_de_eval/<experiment>/`:
`config.yaml`, `run.log`, `metrics/de.csv` (per-image + summary row), `summary.json`,
and (when `save_debug: true`) `<pair>/images/*.png`.
A batch also writes `results.csv`, `summary.md`, `summary.png` at the batch `save_dir`.

### Debug images (`save_debug: true`)

Per pair, the runner reproduces the matching-notebook figures:

- `src.png`, `synth.png` — the iPhone reference and the synthetic (same-colour, warped) view.
- `keypoint_matches.png` — the two views side-by-side with lines joining the matched
  keypoints (`metric.ipynb` cell 3).

Dense matchers (`keypoint_homography`, `keypoint_flow`, `roma_dense`):
- `aligned_overlay.png` — checkerboard A / B-aligned + `|A − B|` residual (cell 6).
- `gradient_keep.png` — L* gradient map + the **patch extracted** (kept flat pixels, cell 9).
- `de_map.png` — ΔE map over kept pixels + histogram (cell 10).
- `residual_flow.png` — colour-wheel of the residual flow removed (`keypoint_flow` only).
- `certainty.png` — RoMa dense confidence map warped into B's frame (`roma_dense` only).

Sparse matchers (`stereo_sgbm`):
- `extracted_patches.png` — the matched points scattered on each view + an A-over-B
  swatch strip of the matched RGB pairs ΔE is computed on.
- `masked_overlay.png` — image view with the correspondence mask applied:
  A masked, B-colours resampled onto A masked, and the ΔE map over matched pixels
  (notebook `matching_ROMA_SGM` cells 22-23).

## What the numbers mean

On the homography+flow synth (`flow_amplitude_px > 0`), a 3-image smoke run gives:

| matcher | mean ΔE | reading |
| --- | --- | --- |
| `keypoint_flow` (sift / akaze) | ~0.14 | recovers the nonlinear warp → **DE ≈ 0** ✅ |
| `keypoint_homography` (sift / akaze) | ~0.52 | undoes the projective part only; flow residual remains |
| `stereo_sgbm` (akaze) | ~8.4 | disparity model mismatched to a non-stereo warp |

Set `synth.params.flow_amplitude_px: 0.0` for the **pure-homography baseline** (then
even `keypoint_homography` should be ≈ 0).

## Configuration

Every knob is YAML. A single config picks one `synth` + one `matcher`; a batch shares
`data`/`synth`/`eval` and lists matchers under `experiments:`. See the comments in
`core/config.py` and the example configs.

- `synth.homography_flow`: `homography_noise_std`, `flow_amplitude_px`,
  `flow_smooth_sigma`, `seed`.
- `matcher` (keypoint_*): `detector` (akaze|sift|loftr|roma), `use_ransac`,
  `ransac_threshold`, `resize` (the 0.10 downscale), `gradient_threshold`,
  and `farneback` (flow matcher only).
- `matcher.roma_dense`: `confidence_threshold`, `gradient_threshold`, `smooth_certainty`.
- `matcher.stereo_sgbm`: SGBM + fundamental-matrix params.

### Adding a component

Drop a class in `synth/` or `matchers/`, decorate it (`@register_synth("name")` /
`@register_matcher("name")`), import its module from the package `__init__`, and it's
selectable by `type:` in YAML. No other code changes.

## Deep-learning matchers (lazy imports)

`loftr` (kornia) and `roma`/`roma_dense` (romatch) import their heavy deps **inside**
`align()`, so classical AKAZE/SIFT configs run under the project `.venv`. The DL ones
require the user-level **python 3.13** env where `kornia` / `romatch` are installed:

```bash
/usr/bin/python3.13 main.py run -c configs/homflow_roma_dense.yaml   # n_workers: 1
```
