"""Feature 003 T031 - factorial grouped bar plot.

Renders a publication-ready bar chart of R^2 at H=12 across the 3 GNN variants
grouped by feature bundle (BASIC vs PAFC). Error bars are inter-seed s.d.
Okabe-Ito palette. 700 DPI PNG + PDF vector.

ANOVA summary lives in the LaTeX caption (rule: no redundant numerical
annotations inside images; statistics live in the caption).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AGG_CSV = REPO / '.docs' / 'papers' / '5' / 'data' / 'factorial_feat_variant.csv'
FIG_PNG = REPO / '.docs' / 'papers' / '5' / 'figures' / 'factorial_feat_variant_r2.png'
FIG_PDF = REPO / '.docs' / 'papers' / '5' / 'figures' / 'factorial_feat_variant_r2.pdf'

# Okabe-Ito feature-set colours (match project figure_config convention)
FEAT_COLORS = {
    'BASIC': '#009E73',  # Bluish green
    'PAFC':  '#56B4E9',  # Sky blue
}
VARIANTS_ORDER = ('GAT', 'GCN', 'SAGE')
FEATS_ORDER = ('BASIC', 'PAFC')


def run(args: argparse.Namespace) -> int:
    print('[T031] factorial grouped bar plot (feat x variant at H=12)')

    if not AGG_CSV.exists():
        print(f'  MISSING: {AGG_CSV}')
        return 1

    df = pd.read_csv(AGG_CSV)
    assert len(df) == 6, f'expected 6 rows, got {len(df)}'

    # ANOVA values now belong to the figure caption (rule: no redundant
    # numerical text inside images; statistics live in the caption).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Single source of truth: figure_config.PAPER_RC
    sys.path.insert(0, str(REPO / 'models' / 'scripts'))
    from figure_config import setup_paper_style  # noqa: E402
    setup_paper_style()
    plt.rcParams.update({'axes.edgecolor': '#333333', 'axes.linewidth': 0.8})

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0), dpi=100)

    # Positions: 3 variant groups on x-axis, 2 bars per group (BASIC, PAFC)
    x = np.arange(len(VARIANTS_ORDER))
    width = 0.36

    for i, feat in enumerate(FEATS_ORDER):
        means = []
        stds = []
        for variant in VARIANTS_ORDER:
            row = df[(df['feat'] == feat) & (df['variant'] == variant)].iloc[0]
            means.append(row['R^2_mean'])
            stds.append(row['R^2_std'])
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=feat, color=FEAT_COLORS[feat],
                      edgecolor='#222222', linewidth=0.7,
                      error_kw={'ecolor': '#333333', 'elinewidth': 1.0})
        # FIGURE EXCEPTION: bar value labels need fontsize-1 to fit above error bars
        for xc, m, s in zip(x + offset, means, stds):
            ax.text(xc, m + s + 0.012, f'{m:.3f}', ha='center', va='bottom',
                    fontsize=plt.rcParams['font.size'] - 1, color='#222222')

    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS_ORDER)
    ax.set_xlabel('GNN variant')
    ax.set_ylabel(r'$R^2$ at H=12 (mean $\pm$ s.d. across 3 seeds)')

    # Y-axis limits with padding above the highest error bar
    y_top = max(df['R^2_mean'] + df['R^2_std']) + 0.08
    y_bot = max(0.0, min(df['R^2_mean'] - df['R^2_std']) - 0.05)
    ax.set_ylim(y_bot, y_top)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    # Legend: feature bundles
    ax.legend(title='Feature bundle', loc='upper right', frameon=True,
              framealpha=0.95)


    plt.tight_layout()
    FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PNG, dpi=900, bbox_inches='tight', facecolor='white')
    plt.savefig(FIG_PDF, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    sz_png = FIG_PNG.stat().st_size
    sz_pdf = FIG_PDF.stat().st_size
    print(f'  wrote: {FIG_PNG.relative_to(REPO)}  ({sz_png/1024:.1f} KB)')
    print(f'  wrote: {FIG_PDF.relative_to(REPO)}  ({sz_pdf/1024:.1f} KB)')
    assert sz_png >= 30_000, f'PNG too small: {sz_png}B'
    print('[T031] figure complete')
    return 0


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace()))
