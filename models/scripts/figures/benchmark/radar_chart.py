"""Regenerate radar_chart.png for Paper 5 with proper label padding.

Fixes review issue 5: the "Efficiency" label was clipping against the
chart polygon. This regen pushes the angular tick labels outward so
they sit clearly outside the radar plot, and updates Late Fusion R^2
to the path-C value (0.672).

Output: .docs/papers/5/figures/radar_chart.png  (also copies to delivery/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIGURES_ROOT = Path(__file__).resolve().parent.parent
REPO = FIGURES_ROOT.parents[2]
sys.path.insert(0, str(FIGURES_ROOT))
from _config import setup_paper_style, save_figure  # noqa: E402

FIG_OUT = REPO / '.docs' / 'papers' / '5' / 'figures' / 'radar_chart.png'
FIG_OUT_DELIVERY = REPO / '.docs' / 'papers' / '5' / 'delivery' / 'figures' / 'radar_chart.png'

# Categories: clockwise from top.
#
# Five axes, every one of them a measured quantity. Two notes on what is here
# and what is not.
#
# 'Efficiency' is back after an interval, and with a different number. It used
# to scale a parameter count in which the convolutional entry was 316K, a figure
# appearing in no table, and Late Fusion was given 1.00 on the grounds that the
# combiner adds only three coefficients. That is the wrong accounting: the
# fusion cannot predict without both branches trained, so it costs their sum,
# 246K, and is the least parameter-efficient design on the chart rather than
# the most. The scale is 1 - params/300K, so a longer spoke is a cheaper model.
#
# 'Stability' is not here. It would need an inter-seed dispersion for every
# model shown, and the stacking ensemble was run once, so one of the four values
# would have to be invented. The seed dispersions that do exist are in the
# multi-seed table.
CATEGORIES = ['R$^2$', '1$-$NRMSE', '1$-$NMAE', '1$-|$Bias$|$', 'Efficiency']

MODELS_DATA = {
    'Enh. ConvLSTM': [0.628 / 0.7, 1 - 81.05 / 120, 1 - 58.91 / 100, 1 - 10.50 / 30, 1 - 148 / 300],
    'GNN-TAT':       [0.628 / 0.7, 1 - 82.29 / 120, 1 - 58.19 / 100, 1 - 28.80 / 30, 1 - 98 / 300],
    'Stacking Ens.': [0.212 / 0.7, 1 - 117.93 / 120, 1 - 92.41 / 100, 0.30,          1 - 200 / 300],
    'Late Fusion':   [0.672 / 0.7, 1 - 76.23 / 120, 1 - 56.12 / 100, 1 - 0.002 / 30, 1 - 246 / 300],
}

# Okabe-Ito-aligned palette (matches the rest of the paper)
COLORS = {
    'Enh. ConvLSTM': '#0072B2',  # blue
    'GNN-TAT':       '#E69F00',  # orange
    'Stacking Ens.': '#CC79A7',  # pink
    'Late Fusion':   '#009E73',  # green
}


def main() -> int:
    n = len(CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    setup_paper_style()
    # Fig 10 (radar) is embedded at 0.60\textwidth → +4 pt source font compensates
    # for the scale-down so on-page sizes match the rest of the figure set.
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 15,
        'xtick.labelsize': 14,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
    })

    fig, ax = plt.subplots(figsize=(11, 7.5), subplot_kw=dict(polar=True))
    ax.set_facecolor('white')
    # Shrink polar plot horizontally so category labels (esp. 'Efficiency' at 180 deg)
    # sit clearly outside the data circle without overlapping the polygon.
    ax.set_position([0.22, 0.10, 0.50, 0.80])

    for label, values in MODELS_DATA.items():
        values_plot = values + values[:1]
        color = COLORS[label]
        ax.plot(angles, values_plot, 'o-', linewidth=1.6, color=color,
                label=label, markersize=5)
        ax.fill(angles, values_plot, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                        color='#555555')

    # Push the angular tick labels well outside the polygon (pad=22 vs the 14 used
    # in earlier versions; previous value still produced visible overlap on the
    # left-hand 'Efficiency' label when the figure was scaled to a paper column).
    ax.tick_params(axis='x', pad=22)
    ax.set_rlabel_position(50)  # rotate the radial tick labels off the data spokes

    ax.grid(color='#cccccc', linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#888888')
        spine.set_linewidth(0.6)

    ax.legend(loc='center left', bbox_to_anchor=(1.20, 0.5),
              framealpha=0.95)
    save_figure(fig, FIG_OUT, dpi=900, mirror=FIG_OUT_DELIVERY,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'Wrote: {FIG_OUT.relative_to(REPO)}')
    print(f'Wrote: {FIG_OUT_DELIVERY.relative_to(REPO)}')
    print(f'Sizes: {FIG_OUT.stat().st_size/1024:.1f} KB / {FIG_OUT_DELIVERY.stat().st_size/1024:.1f} KB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
