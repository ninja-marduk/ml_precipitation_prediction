"""Show the anchor the paper's central result is stated against.

A reader meeting "per-cell monthly climatology" in the abstract has to take on
trust both what it is and why a predictor with no parameters scores 0.730. This
draws it. Panel (a) is the estimator: for three cells at different elevations,
every training year's annual cycle in grey, and the mean of them, which is the
climatology, in colour. Panel (b) is the prediction: the validation months of one
cell, with what the climatology returns for them.

Together they answer the question the definition alone does not. The cycle
repeats, the climatology is its average, and most of the variance in monthly
Andean precipitation is that cycle. Nothing is left for a learned model to
explain except the departures, which is what the deseasonalised test of the
manuscript scores and where no model here shows detectable skill.

Everything is read from the same NetCDF and the same training cutoff the baseline
script uses, so the figure and the reported 0.730 cannot come apart.

Usage: python models/scripts/figures/analysis/climatology_anchor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "models" / "scripts" / "figures"))
from _config import setup_paper_style, save_figure, OUTPUT_DPI  # noqa: E402

NC = (ROOT / "notebooks" / "data" / "output" /
      "complete_dataset_with_features_with_clusters_elevation_windows_imfs"
      "_with_onehot_elevation_clean.nc")
OUT = ROOT / ".docs" / "papers" / "5" / "figures" / "climatology_anchor.png"
SPLIT = 414                    # training steps; the manuscript's T_0
# The scored target months, from naive_baselines.txt: "validation target months
# span idx 474..517". Hard-coding them here would let this figure drift from the
# baselines, so they are asserted against that record below.
SCORED = (474, 517)
PROV = "models/provenance/naive_baselines.txt"
LABELS = ("J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")
# Okabe-Ito, matching the elevation strata used throughout the manuscript
BAND = (("Low ($<$1500 m)", "#0072B2"),
        ("Medium (1500-2800 m)", "#E69F00"),
        ("High ($>$2800 m)", "#009E73"))


def pick_cells(elev):
    """One representative cell per elevation stratum: the median of each band."""
    flat = elev.reshape(-1)
    out = []
    for lo, hi in ((-np.inf, 1500), (1500, 2800), (2800, np.inf)):
        idx = np.where(np.isfinite(flat) & (flat >= lo) & (flat < hi))[0]
        if idx.size == 0:
            continue
        # the cell whose elevation is the band median, so it is typical of it
        out.append(int(idx[np.argsort(flat[idx])[idx.size // 2]]))
    return out


def main():
    if not NC.exists():
        print(f"missing {NC}")
        return 1
    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)
    elev = ds["elevation"].values.astype(np.float64)
    elev = elev[0] if elev.ndim == 3 else elev
    months = (ds["month"].values.astype(int) if "month" in ds else
              np.array([int(str(x)[5:7]) for x in ds["time"].values]))
    if months.ndim > 1:
        months = months[:, 0, 0]
    ds.close()

    T = P.shape[0]
    flat = P.reshape(T, -1)
    cells = pick_cells(elev)
    print(f"{T} steps, training to {SPLIT}; cells chosen: {cells}")

    # Fail rather than draw a shaded band that disagrees with the baselines.
    rec = ROOT / PROV
    if rec.exists():
        import re
        m = re.search(r"target months span idx (\d+)\.\.(\d+) \(split at (\d+)\)",
                      rec.read_text(encoding="utf-8", errors="replace"))
        if m:
            got = (int(m.group(1)), int(m.group(2)))
            if got != SCORED or int(m.group(3)) != SPLIT:
                print(f"FATAL: {PROV} says scored={got} split={m.group(3)}, "
                      f"this figure assumes {SCORED} and {SPLIT}")
                return 1
            print(f"scored window {SCORED} agrees with {PROV}")

    setup_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.3),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.45]})

    # ---- (a) the estimator: every training year, and their mean ----------------
    for k, (cell, (name, colour)) in enumerate(zip(cells, BAND)):
        ax = axes[k]
        series = flat[:SPLIT, cell]
        mm = months[:SPLIT]
        clim = np.array([np.nanmean(series[mm == m]) for m in range(1, 13)])
        n_years = 0
        for y0 in range(0, SPLIT - 11, 12):
            seg_m, seg_v = mm[y0:y0 + 12], series[y0:y0 + 12]
            if seg_m.size < 12:
                continue
            order = np.argsort(seg_m)
            ax.plot(np.arange(1, 13), seg_v[order], color="0.75",
                    linewidth=0.45, alpha=0.55, zorder=1)
            n_years += 1
        ax.plot(np.arange(1, 13), clim, color=colour, linewidth=2.4, zorder=3,
                marker="o", markersize=3.4)
        ax.set_title(f"({'abc'[k]}) {name}\n{elev.reshape(-1)[cell]:.0f} m, "
                     f"{n_years} training years", fontsize=9, loc="left")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_xlim(0.5, 12.5)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.set_ylabel("precipitation (mm)")

    # ---- (b) the prediction on the validation months -----------------------
    ax = axes[3]
    cell = cells[1]
    series = flat[:, cell]
    clim = np.array([np.nanmean(series[:SPLIT][months[:SPLIT] == m])
                     for m in range(1, 13)])
    val = np.arange(SPLIT, T)
    # The models are scored on the last 44 of these months, not on all 104:
    # each forecast origin needs 60 months of input behind it and 12 leads ahead.
    # Showing the whole post-training record without saying so would imply the
    # anchor and the models were compared over four times the evidence they were.
    # The panel title names the shading; an in-image label here would repeat it,
    # and the axis limits are not final until the data is drawn.
    ax.axvspan(SCORED[0], SCORED[1], color="0.88", zorder=0)
    ax.plot(val, series[val], color="0.25", linewidth=1.1, label="observed")
    ax.plot(val, clim[months[val] - 1], color=BAND[1][1], linewidth=1.8,
            label="climatology")
    ax.set_title("(d) what the anchor returns, after the training cutoff\n"
                 f"same cell as (b); shaded: the {SCORED[1]-SCORED[0]+1} months "
                 f"the models are scored on", fontsize=9, loc="left")
    ax.set_xlabel("monthly step")
    ax.set_ylabel("precipitation (mm)")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    for a in axes[:3]:
        a.set_xlabel("calendar month")
    for a in axes:
        a.grid(alpha=0.25, linewidth=0.5)
        a.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, OUT, dpi=OUTPUT_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.name} and its PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
