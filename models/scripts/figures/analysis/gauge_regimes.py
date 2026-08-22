"""Draw the two regimes of the evaluation target, and the test they permit.

CHIRPS publishes which gauges it blended each month. Inside this study's domain
that count is 445 to 588 through the training period and between 0 and 191 over
the 44 months the models are scored on. The models are compared across those 44
months as though the target were one product; it is two.

Panel (a) is the count. Panel (b) is what the split is for: if the anchor's
standing came from CHIRPS falling back on its climatological backbone where
gauges are thin, the anchor should do relatively better in the thin months. It
does not, for the two strongest models.

Reads the two provenance files rather than recomputing, so the figure cannot
disagree with the numbers the manuscript quotes.

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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "models" / "scripts" / "figures"))
from _config import setup_paper_style, save_figure, OUTPUT_DPI  # noqa: E402

OVERLAP = ROOT / "models" / "provenance" / "chirps_station_overlap.csv"
SKILL = ROOT / "models" / "provenance" / "skill_vs_gauge_density.txt"
OUT = ROOT / ".docs" / "papers" / "5" / "figures" / "gauge_regimes.png"
THIN = 20
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
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 3.3),
                                 gridspec_kw={"width_ratios": [1.75, 1]})

    # ---- (a) the count, month by month ---------------------------------
    x = np.arange(len(counts))
    col = ["#D55E00" if c <= THIN else "#56B4E9" for c in counts]
    ax.bar(x, counts, color=col, width=0.82, edgecolor="none")
    ax.set_ylabel("gauges CHIRPS blended\nin the domain")
    ax.set_title("(a) the evaluation target is two regimes, not one",
                 fontsize=9.5, loc="left")
    step = 4
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7.5)
    ax.tick_params(labelsize=8)
    ax.axhline(THIN, color="0.4", linewidth=0.7, linestyle=":")
    ax.text(len(counts) - 0.5, THIN + 6, f"{THIN} gauges", fontsize=7.5,
            color="0.4", ha="right")
    n_thin = sum(1 for c in counts if c <= THIN)
    ax.text(0.015, 0.95, f"{n_thin} thin months, {len(counts) - n_thin} thick "
                         f"(median {int(np.median([c for c in counts if c > THIN]))})",
            transform=ax.transAxes, fontsize=8, va="top", color="0.25")
    # A month with no gauges draws no bar, and that month is the most striking
    # point in the panel, so it is annotated rather than left as a gap.
    for i, c in enumerate(counts):
        if c == 0:
            ax.annotate(f"{labels[i]}\nno gauges", xy=(i, 0), xytext=(i + 1.5, 62),
                        fontsize=7.5, color="#D55E00", ha="left",
                        arrowprops=dict(arrowstyle="-|>", color="#D55E00",
                                        linewidth=0.8, shrinkB=1))

    # ---- (b) what the split tests --------------------------------------
    width = 0.34
    pos = np.arange(len(SERIES))
    for k, (name, colour) in enumerate(SERIES):
        a = np.mean([float(r[f"ratio:{name}"]) for r in thin])
        b = np.mean([float(r[f"ratio:{name}"]) for r in thick])
        sd = np.std([float(r[f"ratio:{name}"]) for r in thick], ddof=1)
        bx.bar(pos[k] - width / 2, a, width, color=colour, alpha=0.55,
               edgecolor="none")
        bx.bar(pos[k] + width / 2, b, width, color=colour, edgecolor="none")
        bx.errorbar(pos[k] + width / 2, b, yerr=sd, color="0.3", capsize=3,
                    linewidth=0.9)
    bx.set_xticks(pos)
    bx.set_xticklabels([n.replace("-Bidir", "").replace("-TAT-GAT", "")
                        for n, _ in SERIES], fontsize=8.5)
    bx.set_ylabel("RMSE(climatology) / RMSE(model)")
    bx.axhline(1.0, color="0.35", linewidth=0.8)
    bx.set_title("(b) the anchor does not gain where gauges thin",
                 fontsize=9.5, loc="left")
    bx.tick_params(labelsize=8)
    bx.text(0.03, 0.04, "pale: thin months   solid: thick months",
            transform=bx.transAxes, fontsize=8, va="bottom", color="0.25")
    bx.text(0.97, 0.955, "above 1.0: climatology wins", transform=bx.transAxes,
            fontsize=7.5, va="top", ha="right", color="0.4")

    for a_ in (ax, bx):
        a_.grid(axis="y", alpha=0.25, linewidth=0.5)
        a_.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, OUT, dpi=OUTPUT_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.name} and its PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
