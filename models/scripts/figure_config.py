"""
Q1 Journal Figure Standards - Shared Configuration
===================================================
Based on Nature, PNAS, AGU, Elsevier, and MDPI guidelines.
Uses Okabe-Ito colorblind-safe palette (Nature-recommended).

Usage:
    from figure_config import COLORS, setup_style, add_panel_label
    setup_style()
"""

import matplotlib.pyplot as plt
import matplotlib


# Okabe-Ito colorblind-safe palette
COLORS = {
    'v1': '#999999',      # Gray - Baseline (ConvRNN/ConvLSTM/ConvGRU)
    'v2': '#0072B2',      # Blue - Enhanced ConvLSTM
    'v3': '#F0E442',      # Yellow - FNO-Hybrid
    'v4': '#E69F00',      # Orange - GNN-TAT
    'v5': '#D55E00',      # Red-Orange - Stacking Ensemble (failed)
    'v6': '#CC79A7',      # Pink - Stratified Ensemble
    'v9': '#882255',      # Wine - GNN-BiMamba (failed)
    'v10': '#009E73',     # Green - Late Fusion (best)
    'baseline': '#999999',
    'success': '#009E73',
    'failure': '#D55E00',
}

# Output configuration
OUTPUT_DPI = 800


def setup_style():
    """Apply Q1 journal figure standards."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': OUTPUT_DPI,
        'savefig.dpi': OUTPUT_DPI,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def add_panel_label(ax, label, x=-0.08, y=1.05):
    """Add bold panel label (a), (b), etc. to top-left of axis."""
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='right')
