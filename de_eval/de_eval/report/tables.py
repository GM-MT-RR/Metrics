"""Summary table writers: per-image CSV, combined results CSV, Markdown, PNG.

Per-experiment the runner calls ``write_experiment_csv``. Across a batch,
``write_batch_summary`` aggregates one row per experiment into ``results.csv``,
``summary.md`` (a markdown table), and ``summary.png`` (a matplotlib table plus a
ΔE-mean bar chart) so the DE≈0 expectation is visible at a glance.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Columns reported per image and aggregated per experiment.
DE_COLS: Sequence[str] = ("de_mean", "de_median", "de_p95", "de_p99", "de_max", "n_pairs")


def write_experiment_csv(rows: List[Dict], path: Path) -> None:
    """Per-image rows + a mean summary row for one experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = ["image"] + list(DE_COLS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        summary = {"image": "summary"}
        for c in DE_COLS:
            vals = np.array([r[c] for r in rows if c in r and np.isfinite(r[c])], dtype=np.float64)
            summary[c] = f"{vals.mean():.6f}" if vals.size else ""
        writer.writerow(summary)


def aggregate_experiment(rows: List[Dict]) -> Dict[str, float]:
    """Mean of each DE column across images (one experiment -> one summary dict)."""
    out: Dict[str, float] = {}
    for c in DE_COLS:
        vals = np.array([r[c] for r in rows if c in r and np.isfinite(r[c])], dtype=np.float64)
        out[c] = float(vals.mean()) if vals.size else float("nan")
    return out


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(x) for x in r) + " |")
    return "\n".join(lines) + "\n"


def write_batch_summary(
    experiment_summaries: List[Dict],
    out_dir: Path,
    title: str = "ΔE matching benchmark",
) -> None:
    """Write results.csv + summary.md + summary.png for a batch of experiments.

    Each entry of ``experiment_summaries`` is
    ``{"experiment", "align", "refine", "patches", "detector", "synth", **DE_COLS}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not experiment_summaries:
        return

    meta_cols = ["experiment", "align", "refine", "patches", "detector", "synth"]
    cols = meta_cols + list(DE_COLS)

    # CSV
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for s in experiment_summaries:
            writer.writerow({k: s.get(k, "") for k in cols})

    # Markdown
    headers = meta_cols + list(DE_COLS)
    md_rows = [[s.get(k, "") for k in headers] for s in experiment_summaries]
    md = f"# {title}\n\nExpectation: same-colour synthetic pairs ⇒ **ΔE ≈ 0** after alignment.\n\n"
    md += _markdown_table(headers, md_rows)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    # PNG: table + ΔE-mean bar chart
    labels = [s["experiment"] for s in experiment_summaries]
    de_means = [s.get("de_mean", float("nan")) for s in experiment_summaries]

    fig, (ax_tbl, ax_bar) = plt.subplots(
        2, 1, figsize=(max(8, len(cols) * 1.2), 2 + 0.5 * len(experiment_summaries) + 4),
        gridspec_kw={"height_ratios": [len(experiment_summaries) + 1, 4]},
    )
    ax_tbl.axis("off")
    cell_text = [[(f"{s.get(k):.3f}" if isinstance(s.get(k), float) else str(s.get(k, "")))
                  for k in headers] for s in experiment_summaries]
    tbl = ax_tbl.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax_tbl.set_title(title, fontsize=12, pad=10)

    ax_bar.bar(range(len(labels)), de_means, color="steelblue")
    ax_bar.axhline(0, color="black", lw=0.8)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_bar.set_ylabel("mean ΔE₀₀ (kept pixels)")
    ax_bar.set_title("Lower is better — 0 = perfect re-alignment")
    plt.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
