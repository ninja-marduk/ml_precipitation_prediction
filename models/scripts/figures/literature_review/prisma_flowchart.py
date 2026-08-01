"""
Study-selection flow diagram (custom design, publication quality)
=================================================================
Reproduces the systematic-review selection flowchart used in the paper
(database columns IEEE/ScienceDirect/Scopus, snowball, Review 1/2/3 stages,
grey chevron stage labels), as vector PDF + 600-DPI PNG.

This matches the original submitted figure design exactly, with the single
correction of the quantitative-synthesis count (34 -> 32), consistent with
Tables 2-5 and the body text.

Output:
  .docs/papers/1/latex/figures/image_prisma.pdf + .png
  mirrored to .docs/papers/1/delivery/figures/

Usage:
  python models/scripts/figures/literature_review/prisma_flowchart.py
"""
import sys
import io
import shutil
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
FIG_DIR = PROJECT_ROOT / ".docs" / "papers" / "1" / "latex" / "figures"
DELIVERY_FIG_DIR = PROJECT_ROOT / ".docs" / "papers" / "1" / "delivery" / "figures"

# --- Palette (matches original grey/white flowchart style) ---
C_STAGE = "#9A9A9A"       # grey chevron stage blocks
C_STAGE_TXT = "white"
C_BOX_EC = "#666666"      # box border
C_BOX_FC = "white"
C_TXT = "#1A1A1A"
C_ARROW = "#777777"
C_REVIEW = "#1A1A1A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Times New Roman", "DejaVu Serif"],
})

# --- Alignment grid ---
# Three evenly spaced column axes; the centre axis (Science Direct) is also
# the vertical spine for the main flow (duplicates -> total -> full-text).
COL_X = [2.6, 5.2, 7.8]          # even gap of 2.6 between column centres
AXIS = COL_X[1]                  # central spine = 5.2
COL_NAMES = ["IEEE Xplore", "Science Direct", "Scopus"]
SEARCH_N = ["[2]", "[9]", "[54]"]
SNOW_N = ["[1]", "[41]", "[31]"]

# --- Row y-positions (even vertical rhythm) ---
Y_HEADER = 11.0
Y_COLHEAD = 10.15
Y_SEARCH = 9.45
Y_SNOW = 8.35
Y_DUP = 7.15
Y_T322 = 5.85
Y_T162 = 4.55
Y_FULL = 3.25
Y_INCL = 1.45

# --- Box widths (uniform per role) ---
W_COLBOX = 2.05
W_FLOW = 3.0
W_DUP = 2.7
W_CRIT = 1.55
W_INCL = 2.75
INCL_X = [AXIS - 1.85, AXIS + 1.85]   # symmetric included boxes
CRIT_X = 1.95
REVIEW_X = 9.05


def _rbox(ax, cx, cy, w, h, text, dashed=False, bold=False, fs=8.0):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        facecolor=C_BOX_FC, edgecolor=C_BOX_EC,
        linewidth=0.9, linestyle=("--" if dashed else "-"), zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=C_TXT, fontweight=("bold" if bold else "normal"),
            zorder=4, linespacing=1.2)


def _arrow(ax, x1, y1, x2, y2, lw=1.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=C_ARROW, zorder=2, shrinkA=1, shrinkB=1,
    ))


def _line(ax, x1, y1, x2, y2, lw=1.0):
    """Plain orthogonal connector segment (no arrowhead)."""
    ax.plot([x1, x2], [y1, y2], color=C_ARROW, linewidth=lw, zorder=2,
            solid_capstyle="round")


def _chevron(ax, cy_top, cy_bot, label):
    """Grey downward chevron/arrow block on the left margin."""
    x0, w = 0.12, 0.70
    tip = 0.30  # triangular tip height
    pts = [
        (x0, cy_top),
        (x0 + w, cy_top),
        (x0 + w, cy_bot + tip),
        (x0 + w / 2, cy_bot),
        (x0, cy_bot + tip),
    ]
    ax.add_patch(Polygon(pts, closed=True, facecolor=C_STAGE,
                         edgecolor="none", zorder=1))
    ax.text(x0 + w / 2, (cy_top + cy_bot) / 2, label, rotation=90,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=C_STAGE_TXT, zorder=2)


def generate_flowchart(out_pdf):
    fig, ax = plt.subplots(figsize=(7.4, 7.7))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 11.7)
    ax.axis("off")

    H_COL, H_SNOW, H_FLOW, H_DUP_B, H_CRIT, H_INCL = 0.62, 0.74, 0.66, 0.60, 0.74, 0.80

    # --- Stage chevrons (left, contiguous, aligned to each stage's rows) ---
    _chevron(ax, 11.45, 7.85, "Identification")
    _chevron(ax, 7.80, 3.95, "Screening")
    _chevron(ax, 3.90, 2.55, "Eligibility")
    _chevron(ax, 2.50, 0.45, "Included")

    # --- Header (dashed), centred on AXIS, spanning the column band ---
    _rbox(ax, AXIS, Y_HEADER, 7.6, 0.60,
          "Preliminary Analysis and Criteria of Inclusion",
          dashed=True, fs=8.5)

    # --- Column headers + Search + Snow Ball (each on its column axis) ---
    for cx, name, sn, snow in zip(COL_X, COL_NAMES, SEARCH_N, SNOW_N):
        ax.text(cx, Y_COLHEAD, name, ha="center", va="center",
                fontsize=9, fontweight="bold", color=C_TXT)
        _rbox(ax, cx, Y_SEARCH, W_COLBOX, H_COL, f"Search results {sn}", fs=7.5)
        _rbox(ax, cx, Y_SNOW, W_COLBOX, H_SNOW,
              f"Snow Ball\nSearch results {snow}", dashed=True, fs=7.0)
        _arrow(ax, cx, Y_SEARCH - H_COL / 2, cx, Y_SNOW + H_SNOW / 2)

    # --- Duplicates excluded (on AXIS), square (orthogonal) connectors ---
    _rbox(ax, AXIS, Y_DUP, W_DUP, H_DUP_B, "Duplicates excluded [130]", fs=8.0)
    dup_top = Y_DUP + H_DUP_B / 2
    dup_l, dup_r = AXIS - W_DUP / 2, AXIS + W_DUP / 2
    snow_bot = Y_SNOW - H_SNOW / 2
    # centre column: straight down into the box top
    _arrow(ax, AXIS, snow_bot, AXIS, dup_top)
    # left/right columns: down, then right-angle turn into the box side
    _line(ax, COL_X[0], snow_bot, COL_X[0], Y_DUP)
    _arrow(ax, COL_X[0], Y_DUP, dup_l, Y_DUP)
    _line(ax, COL_X[2], snow_bot, COL_X[2], Y_DUP)
    _arrow(ax, COL_X[2], Y_DUP, dup_r, Y_DUP)

    # --- Total selected [322] + Review 1 ---
    _rbox(ax, AXIS, Y_T322, W_FLOW, H_FLOW, "Total selected in Databases [322]", fs=8.0)
    _arrow(ax, AXIS, Y_DUP - H_DUP_B / 2, AXIS, Y_T322 + H_FLOW / 2)
    _rbox(ax, CRIT_X, Y_T322, W_CRIT, H_CRIT, "Criteria of\nexclusion", fs=7.0)
    _arrow(ax, CRIT_X + W_CRIT / 2, Y_T322, AXIS - W_FLOW / 2, Y_T322)
    ax.annotate("", xy=(REVIEW_X, Y_T322), xytext=(AXIS + W_FLOW / 2, Y_T322),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=C_REVIEW), zorder=3)
    ax.text(REVIEW_X + 0.15, Y_T322 + 0.26, "Review 1", ha="left", va="center",
            fontsize=10, fontweight="bold", color=C_REVIEW)
    ax.text(REVIEW_X + 0.15, Y_T322 - 0.16, "Title, Abstract and Key Words",
            ha="left", va="center", fontsize=7, color=C_TXT)

    # --- Total selected [162] + Review 2 ---
    _rbox(ax, AXIS, Y_T162, W_FLOW, H_FLOW, "Total selected in Databases [162]", fs=8.0)
    _arrow(ax, AXIS, Y_T322 - H_FLOW / 2, AXIS, Y_T162 + H_FLOW / 2)
    _rbox(ax, CRIT_X, Y_T162, W_CRIT, H_CRIT, "Criteria of\nexclusion", fs=7.0)
    _arrow(ax, CRIT_X + W_CRIT / 2, Y_T162, AXIS - W_FLOW / 2, Y_T162)
    ax.annotate("", xy=(REVIEW_X, Y_T162), xytext=(AXIS + W_FLOW / 2, Y_T162),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=C_REVIEW), zorder=3)
    ax.text(REVIEW_X + 0.15, Y_T162 + 0.26, "Review 2", ha="left", va="center",
            fontsize=10, fontweight="bold", color=C_REVIEW)
    ax.text(REVIEW_X + 0.15, Y_T162 - 0.16, "Introduction and Conclusions",
            ha="left", va="center", fontsize=7, color=C_TXT)

    # --- Full-text documents [162] + Review 3 ---
    _rbox(ax, AXIS, Y_FULL, W_FLOW, H_SNOW, "Full-text documents for\nanalysis [162]", fs=8.0)
    _arrow(ax, AXIS, Y_T162 - H_FLOW / 2, AXIS, Y_FULL + H_SNOW / 2)
    ax.annotate("", xy=(REVIEW_X, Y_FULL), xytext=(AXIS + W_FLOW / 2, Y_FULL),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=C_REVIEW), zorder=3)
    ax.text(REVIEW_X + 0.15, Y_FULL + 0.30, "Review 3", ha="left", va="center",
            fontsize=10, fontweight="bold", color=C_REVIEW)
    ax.text(REVIEW_X + 0.15, Y_FULL - 0.18,
            "Review complete document and\nanalyzing the trends and\nopportunities",
            ha="left", va="center", fontsize=7, color=C_TXT, linespacing=1.2)

    # --- Included row: final analysis [85] + quantitative synthesis [32] ---
    _rbox(ax, INCL_X[0], Y_INCL, W_INCL, H_INCL,
          "Studies included in\nfinal analysis [85]", fs=8.0)
    _rbox(ax, INCL_X[1], Y_INCL, W_INCL, H_INCL,
          "Studies included in\nquantitative synthesis [32]", fs=8.0)
    # square split connector: down to a horizontal bus, then down into each box
    y_bus = (Y_FULL - H_SNOW / 2 + Y_INCL + H_INCL / 2) / 2
    _line(ax, AXIS, Y_FULL - H_SNOW / 2, AXIS, y_bus)
    _line(ax, INCL_X[0], y_bus, INCL_X[1], y_bus)
    _arrow(ax, INCL_X[0], y_bus, INCL_X[0], Y_INCL + H_INCL / 2)
    _arrow(ax, INCL_X[1], y_bus, INCL_X[1], Y_INCL + H_INCL / 2)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DELIVERY_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = Path(out_pdf)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    for p in (out_pdf, out_png):
        shutil.copy2(p, DELIVERY_FIG_DIR / p.name)
        print(f"  {p.name}: saved ({p.stat().st_size / 1024:.0f} KB)")


def main():
    print("=" * 55)
    print("  Study-selection flowchart (custom design, n=32)")
    print("=" * 55)
    generate_flowchart(FIG_DIR / "image_prisma.pdf")
    print("=" * 55)
    print("  Done")
    print("=" * 55)


if __name__ == "__main__":
    main()
