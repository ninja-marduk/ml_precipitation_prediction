"""
Generate Graphical Abstract for Paper 1
========================================
Systematic Review of Hybrid Models for Spatiotemporal Precipitation Prediction.

Output: docs/papers/1/latex/figures/image3.png (1200 DPI)
        docs/papers/1/latex/figures/image3.pdf (vector)

Data from Phase 28 corrections:
- 85 total studies, 19 with R² values
- Global median R² = 0.904
- Kruskal-Wallis p = 0.63
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# -- Configuration --
DPI = 1200
FONT = 'Arial'

# Colors
BLUE = '#0078BA'
BLUE_DK = '#005A8C'
BLUE_LT = '#E6F2FA'
WHITE = '#FFFFFF'
BLACK = '#111111'
GRAY = '#555555'
BORDER = '#C8CDD3'
BG_CHART = '#F5F7FA'

# Taxonomy (Okabe-Ito)
C_PRE = '#0072B2'
C_OPT = '#E69F00'
C_COM = '#009E73'
C_POS = '#CC79A7'
C_DEP = '#56B4E9'

# Layout constants
GAP = 0.015          # gap between section label bottom and content top
INTER_GAP = 0.020    # gap between content bottom and next label top
LABEL_H = 0.016      # approximate height of section label text


def setup():
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': DPI,
        'font.family': 'sans-serif',
        'font.sans-serif': [FONT, 'Helvetica', 'DejaVu Sans'],
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
        'savefig.transparent': False,
        'text.antialiased': True,
        'lines.antialiased': True,
        'patch.antialiased': True,
        'pdf.fonttype': 42,         # TrueType fonts in PDF (scalable)
        'ps.fonttype': 42,
    })


def box(ax, x, y, w, h, fc=WHITE, ec='none', lw=0, r=0.010, z=3, a=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       zorder=z, alpha=a)
    ax.add_patch(p)


def conn(ax, x, y0, y1):
    """Black arrow from (x, y0) down to (x, y1), no shrink."""
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->,head_width=0.06,head_length=0.035',
                                color=BLACK, lw=1.2, shrinkA=0, shrinkB=0),
                zorder=3)


def section(ax, x, y, text):
    """Section label with white background to mask arrows behind it."""
    ax.text(x, y, text, ha='left', va='bottom', fontsize=8,
            fontweight='bold', color=BLACK, zorder=7,
            bbox=dict(facecolor=WHITE, edgecolor='none', pad=2, alpha=0.95))


def draw():
    setup()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.set_facecolor(WHITE)

    L = 0.04
    R = 0.96
    C = 0.50
    W = R - L

    # =================================================================
    #  Computed vertical positions (top to bottom, uniform spacing)
    #  Each section: label_top → label_bottom → GAP → content_top → content_bottom
    #  Between sections: content_bottom → INTER_GAP → next_label_top
    # =================================================================

    # Row 1: Banner
    banner_y, banner_h = 0.940, 0.048       # [0.940, 0.988]

    # Row 2: Data Sources
    ds_label_top = banner_y - INTER_GAP      # 0.920
    ds_label_y = ds_label_top - LABEL_H      # 0.904
    ds_h = 0.038
    ds_top = ds_label_y - GAP               # 0.889
    ds_y = ds_top - ds_h                    # 0.851

    # Row 3: Hybrid Taxonomy
    ht_label_top = ds_y - INTER_GAP          # 0.831
    ht_label_y = ht_label_top - LABEL_H      # 0.815
    ht_h = 0.036
    ht_top = ht_label_y - GAP               # 0.800
    ht_y = ht_top - ht_h                    # 0.764

    # Row 4: Performance Chart
    perf_label_top = ht_y - INTER_GAP        # 0.744
    perf_label_y = perf_label_top - LABEL_H  # 0.728
    ch = 0.235
    cy_top = perf_label_y - GAP             # 0.713
    cy = cy_top - ch                        # 0.478

    # Row 5: Key Finding (no label, just gap)
    kh = 0.090
    ky_top = cy - INTER_GAP                 # 0.458
    ky = ky_top - kh                        # 0.368

    # Row 6: Recommendations
    # Account for KF glow box visual overflow (extends 0.004+0.016=0.020 below ky)
    kf_glow_overflow = 0.020
    rec_label_top = ky - kf_glow_overflow - 0.012  # visible gap after glow
    rec_label_y = rec_label_top - LABEL_H    # label bottom
    ry_top = rec_label_y - GAP
    ry = 0.040                               # bottom margin
    rh = ry_top - ry

    # ── ROW 1: Banner ───────────────────────────────────────────
    box(ax, L + 0.002, banner_y - 0.003, W, banner_h,
        fc='#707070', r=0.012, z=2, a=0.15)
    box(ax, L, banner_y, W, banner_h, fc=BLUE_DK, r=0.012)
    ax.text(C, banner_y + banner_h / 2,
            '85 studies  \u00b7  2020\u20132025  \u00b7  19 with R\u00b2',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=WHITE, zorder=6)

    # ── ROW 2: Data Sources ─────────────────────────────────────
    section(ax, L, ds_label_y, 'DATA SOURCES')

    sw = (W - 0.04) / 3
    for i, s in enumerate(['Station gauges', 'Satellite / gridded', 'Climate indices']):
        sx = L + i * (sw + 0.02)
        box(ax, sx, ds_y, sw, ds_h, fc=BLUE_LT, ec=BLUE, lw=0.6, r=0.007)
        ax.text(sx + sw / 2, ds_y + ds_h / 2, s,
                ha='center', va='center', fontsize=9, color=BLACK, zorder=6)

    # Arrow: DS visual bottom → HT visual top (accounting for border radius)
    conn(ax, C, ds_y - 0.007, ht_y + ht_h + 0.007)

    # ── ROW 3: Hybrid Taxonomy ──────────────────────────────────
    section(ax, L, ht_label_y, 'HYBRID TAXONOMY')

    tw = (W - 0.045) / 4
    for i, (t, c) in enumerate([('Preprocessing', C_PRE), ('Optimization', C_OPT),
                                 ('Combination', C_COM), ('Post-processing', C_POS)]):
        tx = L + i * (tw + 0.015)
        box(ax, tx, ht_y, tw, ht_h, fc=c, r=0.007)
        ax.text(tx + tw / 2, ht_y + ht_h / 2, t,
                ha='center', va='center', fontsize=8.5, fontweight='bold',
                color=WHITE, zorder=6)

    # Arrow: HT visual bottom → chart visual top
    conn(ax, C, ht_y - 0.007, cy + ch + 0.010)

    # ── ROW 4: Performance Chart ────────────────────────────────
    section(ax, L, perf_label_y, 'PERFORMANCE BY CLASS')

    box(ax, L, cy, W, ch, fc=BG_CHART, ec=BORDER, lw=0.5, r=0.010, z=1)

    bars = [('Optimization', 0.975, C_OPT),
            ('Preprocessing', 0.904, C_PRE),
            ('Deep hybrids', 0.870, C_DEP),
            ('Combination / Post-proc.', 0.650, C_COM)]

    # Chart internal positions (relative to cy, ch)
    bl = 0.28
    bw_max = 0.56
    bh = 0.037
    bgap = 0.016
    btop = cy + ch - 0.038            # first bar y-position

    gb = cy + 0.040                   # grid bottom
    gt = btop + bh + 0.005            # grid top

    for v in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rx = bl + bw_max * v
        ax.plot([rx, rx], [gb, gt], color='#DDE0E5', lw=0.4, zorder=1.5)
        ax.text(rx, cy + 0.023, f'{v:.1f}',
                ha='center', va='center', fontsize=7, color=GRAY, zorder=6)

    mx = bl + bw_max * 0.904
    ax.plot([mx, mx], [gb, gt], color=BLUE, lw=0.8, alpha=0.25, zorder=2)

    for i, (name, val, col) in enumerate(bars):
        by = btop - i * (bh + bgap)
        bw = bw_max * val
        ax.text(bl - 0.010, by + bh / 2, name,
                ha='right', va='center', fontsize=8.5, color=BLACK, zorder=6)
        box(ax, bl, by, bw, bh, fc=col, r=0.005)
        ax.text(bl + bw + 0.010, by + bh / 2, f'{val:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold',
                color=BLACK, zorder=6)

    ax.text(bl + bw_max / 2, cy + 0.006,
            'Median R\u00b2  |  Kruskal\u2013Wallis p = 0.63 (n.s. at \u03b1 = 0.05)',
            ha='center', va='center', fontsize=7.5, fontstyle='italic',
            color=GRAY, zorder=6)

    # Arrow: chart visual bottom → KF visual top
    conn(ax, C, cy - 0.010, ky + kh + 0.014)

    # ── ROW 5: Key Finding ──────────────────────────────────────
    km = 0.08
    kw = 0.84
    box(ax, km + 0.003, ky - 0.003, kw, kh, fc='#606060', r=0.014, z=2, a=0.15)
    box(ax, km - 0.004, ky - 0.004, kw + 0.008, kh + 0.008,
        fc=BLUE_LT, r=0.016, z=2, a=0.45)
    box(ax, km, ky, kw, kh, fc=BLUE, r=0.014)

    ax.text(C, ky + kh * 0.60, 'Global median R\u00b2 = 0.904',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=WHITE, zorder=6)
    ax.text(C, ky + kh * 0.22, 'No significant difference across hybrid classes',
            ha='center', va='center', fontsize=9, color='#BFD9EC', zorder=6)

    # Arrow: KF visual bottom → rec visual top
    conn(ax, C, ky - 0.014, ry + rh + 0.010)

    # ── ROW 6: Gaps & Recommendations ───────────────────────────
    section(ax, L, rec_label_y, 'IDENTIFIED GAPS & RECOMMENDATIONS')

    box(ax, L + 0.002, ry - 0.002, W, rh, fc='#808080', r=0.010, z=2, a=0.05)
    box(ax, L, ry, W, rh, fc=WHITE, ec=BLUE, lw=0.8, r=0.010)

    items = [
        ('Cross-validation',
         'Leakage-safe\nspatiotemporal CV\nwith blocked /\nstratified splits'),
        ('Standardized metrics',
         'Mandatory R\u00b2, RMSE\nand MAE reporting\nfor cross-study\ncomparison'),
        ('Reproducibility',
         'Open data, code and\nhyperparameter\ndisclosure across\nstudies'),
    ]

    col_w = (W - 0.06) / 3
    col_gap = 0.03

    # Center content vertically within the rec box
    # Content height ~0.125 (accent lines to desc bottom)
    content_h = 0.125
    top_pad = (rh - content_h) / 2
    col_top = ry + rh - top_pad - 0.005   # accent line is 0.005 above col_top

    for i, (title, desc) in enumerate(items):
        cx = L + 0.015 + i * (col_w + col_gap)
        cc = cx + col_w / 2

        ax.plot([cx + 0.02, cx + col_w - 0.02],
                [col_top + 0.005, col_top + 0.005],
                color=BLUE, lw=2.0, zorder=5, solid_capstyle='round')

        ax.text(cc, col_top - 0.020, title,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=BLUE_DK, zorder=6)

        ax.text(cc, col_top - 0.085, desc,
                ha='center', va='center', fontsize=8.5, color=BLACK,
                zorder=6, linespacing=1.4)

        if i < 2:
            sx = cx + col_w + col_gap / 2
            ax.plot([sx, sx], [ry + 0.020, ry + rh - 0.020],
                    color='#E0E4E8', lw=0.5, zorder=4)

    return fig


def main():
    fig = draw()
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                           'docs', 'papers', '1', 'latex', 'figures')
    out_dir = os.path.abspath(out_dir)

    # PDF vector - primary output for LaTeX (infinite resolution)
    pdf_path = os.path.join(out_dir, 'image3.pdf')
    fig.savefig(pdf_path, format='pdf', facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}  ({os.path.getsize(pdf_path) / 1024:.0f} KB) [VECTOR]")

    # PNG raster - RGB (no alpha), 1200 DPI for fallback / submission
    png_path = os.path.join(out_dir, 'image3.png')
    fig.savefig(png_path, format='png', dpi=DPI,
                facecolor='white', edgecolor='none',
                transparent=False, pil_kwargs={'compress_level': 6})
    # Convert RGBA → RGB (remove unused alpha channel)
    from PIL import Image
    img = Image.open(png_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
        img.save(png_path, dpi=(DPI, DPI), compress_level=6)
    w, h = img.size
    print(f"Saved: {png_path}  ({os.path.getsize(png_path) / 1024:.0f} KB) "
          f"[{w}x{h} px, RGB]")

    plt.close(fig)


if __name__ == '__main__':
    main()
