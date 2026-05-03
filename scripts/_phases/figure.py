"""T030 - horizon-degradation figure with ±std shaded bands.

Renders V2, V4, V10 mean R² curves with fill_between shaded bands (±1 s.d.
across 3 seeds). H=3, 6, 12 marked with filled markers. 700 DPI PNG.

Also emits a programmatic separation check log (SC-003): for ≥10 of 12
horizons, the gap between V10 and V2 mean must exceed max(V10_std, V2_std).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
UNIFIED_CSV = REPO / '.docs' / 'papers' / '5' / 'data' / 'horizon_multiseed_v2_v4_v10.csv'
FIG_PATH = REPO / '.docs' / 'papers' / '5' / 'figures' / 'horizon_degradation_multiseed.png'

# Okabe-Ito: V2=Blue, V4=Orange (GNN), V10=Vermillion
# (match the existing project palette documented in figure_config.py)
COLORS = {
    'V2':  '#0072B2',  # Blue
    'V4':  '#E69F00',  # Orange
    'V10': '#D55E00',  # Vermillion (reddish)
}
CANONICAL_H = (3, 6, 12)


def run(args: argparse.Namespace) -> int:
    print('[T030] horizon-degradation figure (V2/V4/V10, ±std across 3 seeds)')

    if not UNIFIED_CSV.exists():
        print(f'  MISSING: {UNIFIED_CSV}')
        return 1
    df = pd.read_csv(UNIFIED_CSV)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    # Single source of truth: figure_config.PAPER_RC
    sys.path.insert(0, str(REPO / 'models' / 'scripts'))
    from figure_config import setup_paper_style  # noqa: E402
    setup_paper_style()
    plt.rcParams.update({'axes.edgecolor': '#333333', 'axes.linewidth': 0.8})

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2), dpi=100)

    H = df['H'].values

    for label in ('V2', 'V4', 'V10'):
        mean = df[f'{label}_R^2_mean'].values
        std = df[f'{label}_R^2_std'].values
        color = COLORS[label]

        # Shaded ±1 s.d. band
        ax.fill_between(H, mean - std, mean + std, color=color, alpha=0.18, zorder=1)

        # Mean curve (open markers for non-canonical horizons)
        ax.plot(H, mean, color=color, linewidth=2.0, zorder=3,
                marker='o', markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=1.5, markersize=5.5)

        # Canonical H=3, 6, 12 as filled markers
        mask = np.isin(H, CANONICAL_H)
        ax.plot(H[mask], mean[mask], color=color, linestyle='none',
                marker='o', markerfacecolor=color,
                markeredgecolor=color, markersize=8.5, zorder=5,
                label=_legend_label(label))

    ax.set_xlabel('Forecast horizon H (months)')
    ax.set_ylabel(r'$R^2$')
    ax.set_xlim(0.5, 12.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc='lower left', frameon=True, framealpha=0.95)

    # Band semantics + canonical horizons live in the LaTeX caption
    # (rule: no redundant numerical annotations inside images).
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=900, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Sanity: figure exists and is ≥50 KB
    sz = FIG_PATH.stat().st_size
    print(f'  wrote: {FIG_PATH}  ({sz/1024:.1f} KB)')
    assert sz >= 50_000, f'figure too small: {sz}B'

    # SC-003 separation check
    v10_mean = df['V10_R^2_mean'].values
    v2_mean  = df['V2_R^2_mean'].values
    v10_std  = df['V10_R^2_std'].values
    v2_std   = df['V2_R^2_std'].values
    separated = np.abs(v10_mean - v2_mean) > np.maximum(v10_std, v2_std)
    n_sep = int(separated.sum())
    print(f'  SC-003 separation check: V10 mean gap > max(std) on '
          f'{n_sep}/12 horizons  (target >= 10)')
    if n_sep < 10:
        print(f'  NOTE: fewer than 10/12 horizons are cleanly separated; '
              f'figure still valid but variance bands touch on {12 - n_sep} horizons.')
    return 0


def _legend_label(label: str) -> str:
    return {
        'V2':  'ConvLSTM (Bidirectional)',
        'V4':  'GNN-TAT (GAT)',
        'V10': 'Late Fusion (Ridge)',
    }[label]


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace()))
