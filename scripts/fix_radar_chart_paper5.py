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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'models' / 'scripts'))
from figure_config import setup_paper_style  # noqa: E402

FIG_OUT = REPO / '.docs' / 'papers' / '5' / 'figures' / 'radar_chart.png'
FIG_OUT_DELIVERY = REPO / '.docs' / 'papers' / '5' / 'delivery' / 'figures' / 'radar_chart.png'

# Categories: clockwise from top
CATEGORIES = ['R$^2$', '1$-$NRMSE', '1$-$NMAE', 'Efficiency', 'Stability', '1$-|$Bias$|$']

# Path-C values (R^2 updated 0.668 -> 0.672)
MODELS_DATA = {
    'Enh. ConvLSTM': [0.628 / 0.7, 1 - 81.05 / 120, 1 - 58.91 / 100, 1 - 316 / 500, 0.80, 1 - 10.50 / 30],
    'GNN-TAT':       [0.628 / 0.7, 1 - 82.29 / 120, 1 - 58.19 / 100, 1 -  98 / 500, 0.85, 1 - 28.80 / 30],
    'Stacking Ens.': [0.212 / 0.7, 1 - 117.93 / 120, 1 - 92.41 / 100, 1 - 200 / 500, 0.30, 0.30],
    'Late Fusion':   [0.672 / 0.7, 1 - 76.23 / 120, 1 - 56.12 / 100, 1.00, 0.95, 1 - 0.002 / 30],
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

    fig, ax = plt.subplots(figsize=(8, 7.5), subplot_kw=dict(polar=True))
    ax.set_facecolor('white')

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

    # Push the angular tick labels outward to avoid overlap with the polygon.
    ax.tick_params(axis='x', pad=14)

    ax.grid(color='#cccccc', linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#888888')
        spine.set_linewidth(0.6)

    ax.legend(loc='lower right', bbox_to_anchor=(1.18, -0.02),
              framealpha=0.95)

    plt.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=900, bbox_inches='tight', facecolor='white')

    # Mirror to delivery
    FIG_OUT_DELIVERY.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT_DELIVERY, dpi=900, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'Wrote: {FIG_OUT.relative_to(REPO)}')
    print(f'Wrote: {FIG_OUT_DELIVERY.relative_to(REPO)}')
    print(f'Sizes: {FIG_OUT.stat().st_size/1024:.1f} KB / {FIG_OUT_DELIVERY.stat().st_size/1024:.1f} KB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
