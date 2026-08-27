"""Draw the two regimes of the evaluation target, and the test they permit.

CHIRPS publishes which gauges it blended each month. Inside this study's domain
that count is 445 to 588 through the training period and between 0 and 191 over
the months the models are scored on. The models are compared across those months
as though the target were one product; it is two.

Panel (a) is the count. Panel (b) is what the split is for: if the anchor's
standing came from CHIRPS falling back on its climatological backbone where
gauges are thin, the anchor should do relatively better in the thin months. It
does not, for the two strongest models.

Reads the two provenance files rather than recomputing, so the figure cannot
disagree with the numbers the manuscript quotes.

Typography: every size comes from the shared paper profile. An earlier version
set six different sizes by hand, from 7.5 to 9.5 pt, so components of one figure
disagreed with each other and with the rest of the paper. Nothing here overrides
the profile, and no summary statistic is drawn inside the axes: counts, medians
and the reading of the 1.0 line belong in the caption, where they can be read
without competing with the data.

Usage: python models/scripts/figures/analysis/gauge_regimes.py
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "models" / "scripts" / "figures"))
from _config import setup_paper_style, save_figure, OUTPUT_DPI  # noqa: E402

OVERLAP = ROOT / "models" / "provenance" / "chirps_station_overlap.csv"
SKILL = ROOT / "models" / "provenance" / "skill_vs_gauge_density.txt"
OUT = ROOT / ".docs" / "papers" / "5" / "figures" / "gauge_regimes.png"
THIN = 20
THIN_COL, THICK_COL = "#D55E00", "#56B4E9"
SERIES = (("ConvLSTM-Bidir", "#0072B2"), ("GNN-TAT-GAT", "#E69F00"),
          ("Late Fusion", "#117733"))


def main():
    for p in (OVERLAP, SKILL):
        if not p.exists():
            print(f"missing {p}")
            return 1

    scored = [r for r in csv.DictReader(OVERLAP.open(encoding="utf-8"))
              if r["period"] == "scored"]
    scored.sort(key=lambda r: (int(r["year"]), int(r["month"])))
    counts = [int(r["in_domain"]) for r in scored]
    labels = [f"{r['year']}-{int(r['month']):02d}" for r in scored]

    body = SKILL.read_text(encoding="utf-8").split("\n", 1)[1]
    skill = list(csv.DictReader(io.StringIO(body)))
    thin = [r for r in skill if int(r["gauges"]) <= THIN]
    thick = [r for r in skill if int(r["gauges"]) > THIN]

    setup_paper_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 3.6),
                                 gridspec_kw={"width_ratios": [1.75, 1]})

    # ---- (a) the count, month by month ---------------------------------
    x = np.arange(len(counts))
    col = [THIN_COL if c <= THIN else THICK_COL for c in counts]
    ax.bar(x, counts, color=col, width=0.82, edgecolor="none")
    ax.set_ylabel("Gauges blended in the domain")
    ax.set_title("(a) The target is two regimes, not one", loc="left")
    step = 6
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=30, ha="right")
    ax.set_ylim(0, 205)
    ax.axhline(THIN, color="0.45", linewidth=0.8, linestyle=":")
    # A month with no gauges draws no bar, and it is the most consequential
    # point in the panel, so it is marked on the axis rather than left as a gap.
    for i, c in enumerate(counts):
        if c == 0:
            ax.plot(i, 0, marker="x", color=THIN_COL, markersize=6,
                    markeredgewidth=1.4, clip_on=False, zorder=5)
    # R1: under the axes, not over the bars.
    ax.legend(handles=[Patch(color=THICK_COL, label=f"more than {THIN} gauges"),
                       Patch(color=THIN_COL, label=f"{THIN} or fewer")],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              frameon=False, handlelength=1.3, columnspacing=1.6)

    # ---- (b) what the split tests --------------------------------------
    width = 0.34
    pos = np.arange(len(SERIES))
    for k, (name, colour) in enumerate(SERIES):
        a = np.mean([float(r[f"ratio:{name}"]) for r in thin])
        b = np.mean([float(r[f"ratio:{name}"]) for r in thick])
        sd = np.std([float(r[f"ratio:{name}"]) for r in thick], ddof=1)
        bx.bar(pos[k] - width / 2, a, width, color=colour, alpha=0.45,
               edgecolor="none")
        bx.bar(pos[k] + width / 2, b, width, color=colour, edgecolor="none")
        bx.errorbar(pos[k] + width / 2, b, yerr=sd, color="0.25", capsize=3,
                    linewidth=1.0)
    bx.set_xticks(pos)
    bx.set_xticklabels([n.replace("-Bidir", "").replace("-TAT-GAT", "")
                        for n, _ in SERIES])
    bx.set_ylabel("RMSE(climatology) / RMSE(model)")
    bx.set_ylim(0, 1.15)
    bx.axhline(1.0, color="0.35", linewidth=0.9)
    bx.set_title("(b) The anchor does not gain where gauges thin", loc="left")
    bx.legend(handles=[Patch(facecolor="0.55", alpha=0.45, label="thin months"),
                       Patch(facecolor="0.55", label="thick months")],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              frameon=False, handlelength=1.3, columnspacing=1.6)

    for a_ in (ax, bx):
        a_.grid(axis="y", alpha=0.25, linewidth=0.5)
        a_.set_axisbelow(True)
        for side in ("top", "right"):
            a_.spines[side].set_visible(False)

    plt.tight_layout()
    save_figure(fig, OUT, dpi=OUTPUT_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.name} and its PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
