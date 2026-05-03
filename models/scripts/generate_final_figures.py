"""
Generate Final Figures for Thesis Architecture Analysis
=======================================================
Q1 journal standards: Okabe-Ito palette, Arial font, 600 DPI,
panel labels, no in-figure text blocks or equations.

Uses descriptive architecture names (no V-number labels).

Output: models/output/final_figures/
"""

import sys
from pathlib import Path

# Ensure scripts directory is on path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

from figure_config import COLORS, setup_style, add_panel_label, OUTPUT_DPI

# Configuration
OUTPUT_DIR = Path("models/output/final_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Apply Q1 style
setup_style()


def load_v10_summary():
    """Load V10 summary data."""
    summary_path = Path("models/output/V10_Late_Fusion/v10_summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def load_v9_metrics():
    """Load V9 metrics data."""
    metrics_path = Path("models/output/V9_GNN_BiMamba/comparisons/v9_metrics.csv")
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None


def figure_1_v1_v10_r2_evolution():
    """R² evolution across all architectures."""
    print("Generating Figure 1: Architecture R² Evolution...")

    architectures = ['Baseline', 'Enh.\nConvLSTM', 'FNO-\nHybrid', 'GNN-\nTAT',
                     'Stacking\nEns.', 'Stratified\nEns.', 'GNN-\nBiMamba', 'Late\nFusion']
    r2_values = [0.58, 0.628, 0.312, 0.628, 0.212, 0.597, 0.200, 0.668]
    status = ['baseline', 'baseline', 'failed', 'success', 'failed', 'no_improve', 'failed', 'best']

    color_map = {
        'best': COLORS['v10'],
        'failed': COLORS['failure'],
        'success': COLORS['v4'],
        'baseline': COLORS['baseline'],
        'no_improve': COLORS['baseline'],
    }
    colors = [color_map[s] for s in status]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(architectures, r2_values, color=colors, width=0.7)

    for bar, val in zip(bars, r2_values):
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)

    ax.axhline(y=0.628, color=COLORS['v2'], linestyle='--', alpha=0.7,
               linewidth=1, label='ConvLSTM baseline (0.628)')
    ax.axhline(y=0.668, color=COLORS['v10'], linestyle='--', alpha=0.7,
               linewidth=1, label='Late Fusion best (0.668)')

    ax.set_xlabel('Architecture')
    ax.set_ylabel('R² score')
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'v1_v10_r2_evolution.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'v1_v10_r2_evolution.png'}")


def _draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor='#444444',
                      linewidth=0.8, text='', fontsize=7.5, fontcolor='white',
                      fontweight='bold', alpha=1.0, zorder=3):
    """Draw a professional rounded-corner box with centered text."""
    from matplotlib.patches import FancyBboxPatch
    pad = 0.12
    box = FancyBboxPatch((x + pad, y + pad), w - 2 * pad, h - 2 * pad,
                         boxstyle=f"round,pad={pad}",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, color=fontcolor, fontweight=fontweight,
                zorder=zorder + 1)
    return box


def _draw_arrow(ax, x_from, y_from, x_to, y_to, color='#555555',
                linewidth=1.0, style='->', shrink_a=4, shrink_b=4):
    """Draw a clean professional arrow between two points."""
    from matplotlib.patches import FancyArrowPatch
    arrow = FancyArrowPatch(
        (x_from, y_from), (x_to, y_to),
        arrowstyle=f'{style},head_length=4,head_width=2.5',
        color=color, linewidth=linewidth,
        shrinkA=shrink_a, shrinkB=shrink_b,
        connectionstyle='arc3,rad=0',
        zorder=2)
    ax.add_patch(arrow)
    return arrow


def figure_2_early_vs_late_fusion():
    """Early fusion vs late fusion - bar chart only (panel b is TikZ in paper).

    Late Fusion R² uses the canonical multiseed mean (seeds {42,123,456}):
    0.655 ± 0.018, matching Section 4.6.2 / Table 13 of Paper 5.
    """
    print("Generating Figure 2: Early vs Late Fusion (bar chart only)...")

    fig, ax1 = plt.subplots(figsize=(5.5, 3.2))

    methods = ['Stacking\n(Early Fusion)', 'Late Fusion\n(Ridge)']
    r2_vals = [0.212, 0.655]
    r2_errs = [0.000, 0.018]
    bar_colors = ['#BDBDBD', COLORS['v2']]

    bars = ax1.bar(methods, r2_vals, color=bar_colors, width=0.45,
                   edgecolor=['#888888', '#005A8C'], linewidth=0.6,
                   yerr=r2_errs, capsize=4, ecolor='#333333',
                   error_kw={'elinewidth': 0.8})
    ax1.axhline(y=0.628, color='#999999', linestyle='--',
                linewidth=0.8, label='Best single model (R² = 0.628)')

    labels = [f'{r2_vals[0]:.3f}', f'{r2_vals[1]:.3f} ± {r2_errs[1]:.3f}']
    for bar, val, lbl, err in zip(bars, r2_vals, labels, r2_errs):
        ax1.annotate(lbl,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + err),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom')

    ax1.set_ylabel('R²')
    ax1.set_ylim(0, 0.78)
    ax1.legend(loc='upper left', framealpha=0.9)

    plt.savefig(OUTPUT_DIR / 'early_vs_late_fusion_bar.png', dpi=OUTPUT_DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'early_vs_late_fusion_bar.png'}")


def figure_3_v10_fusion_weights():
    """Late Fusion learned weights - clean version without equation."""
    print("Generating Figure 3: Late Fusion Weights...")

    v10_data = load_v10_summary()
    if v10_data is None:
        w_v2, w_v4, bias = 0.446, 0.710, -5.53
    else:
        weights = v10_data.get('learned_weights', {})
        w_v2 = weights.get('w_v2', 0.446)
        w_v4 = weights.get('w_v4', 0.710)
        bias = weights.get('bias', -5.53)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel (a): Relative contributions (horizontal stacked bar) ---
    ax1 = axes[0]
    add_panel_label(ax1, 'a')

    total = abs(w_v2) + abs(w_v4)
    pct_v2 = abs(w_v2) / total * 100
    pct_v4 = abs(w_v4) / total * 100

    ax1.barh(['Model\ncontribution'], [pct_v2], height=0.5,
             color=COLORS['v2'], label=f'ConvLSTM ({pct_v2:.1f}%)')
    ax1.barh(['Model\ncontribution'], [pct_v4], left=[pct_v2], height=0.5,
             color=COLORS['v4'], label=f'GNN-TAT ({pct_v4:.1f}%)')

    ax1.set_xlabel('Relative contribution (%)')
    ax1.set_xlim(0, 100)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, framealpha=0.9)

    # --- Panel (b): Raw weights (bar chart) ---
    ax2 = axes[1]
    add_panel_label(ax2, 'b')

    models = ['ConvLSTM', 'GNN-TAT', 'Bias (mm)']
    weight_vals = [w_v2, w_v4, bias]
    bar_colors = [COLORS['v2'], COLORS['v4'], COLORS['baseline']]

    bars = ax2.bar(models, weight_vals, color=bar_colors, width=0.6)

    for bar, val in zip(bars, weight_vals):
        height = bar.get_height()
        if abs(val) < 1:
            label = f'{val:.3f}'
        else:
            label = f'{val:.2f}'
        # Place labels above bars (positive) or inside bars (large negative)
        if height >= 0:
            y_offset = 3
            va = 'bottom'
        else:
            # Place inside the bar, near the top, to avoid x-label overlap
            y_offset = 12
            va = 'top'
        ax2.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, y_offset), textcoords="offset points",
                    ha='center', va=va, fontsize=8, fontweight='bold')

    ax2.set_ylabel('Weight value')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    # Extra bottom margin so x-labels don't overlap with bar annotations
    ax2.set_ylim(bottom=min(weight_vals) * 1.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'v10_fusion_weights.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'v10_fusion_weights.png'}")


def figure_4_master_comparison():
    """Master comparison horizontal bar chart."""
    print("Generating Figure 4: Master Comparison...")

    data = {
        'Version': ['Baseline', 'Enh. ConvLSTM', 'FNO-Hybrid', 'GNN-TAT',
                     'Stacking Ens.', 'Stratified Ens.', 'GNN-BiMamba', 'Late Fusion'],
        'R2': [0.58, 0.628, 0.312, 0.628, 0.212, 0.597, 0.200, 0.668],
        'RMSE': [85.0, 81.05, 110.0, 82.29, 117.93, 84.1, 111.18, 76.67],
        'Status': ['baseline', 'v2', 'v3', 'v4', 'v5', 'baseline', 'v9', 'v10']
    }
    df = pd.DataFrame(data)
    colors = [COLORS.get(s, COLORS['baseline']) for s in df['Status']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) R² comparison
    ax1 = axes[0]
    add_panel_label(ax1, 'a')
    bars1 = ax1.barh(df['Version'], df['R2'], color=colors)
    ax1.axvline(x=0.628, color=COLORS['v2'], linestyle='--', alpha=0.6, linewidth=1)
    ax1.axvline(x=0.668, color=COLORS['v10'], linestyle='--', alpha=0.6, linewidth=1)
    ax1.set_xlabel('R² score')
    ax1.set_xlim(0, 0.8)

    for bar, val in zip(bars1, df['R2']):
        ax1.annotate(f'{val:.3f}',
                    xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=7)

    # (b) RMSE comparison
    ax2 = axes[1]
    add_panel_label(ax2, 'b')
    bars2 = ax2.barh(df['Version'], df['RMSE'], color=colors)
    ax2.axvline(x=81.05, color=COLORS['v2'], linestyle='--', alpha=0.6, linewidth=1)
    ax2.axvline(x=76.67, color=COLORS['v10'], linestyle='--', alpha=0.6, linewidth=1)
    ax2.set_xlabel('RMSE (mm)')

    for bar, val in zip(bars2, df['RMSE']):
        ax2.annotate(f'{val:.1f}',
                    xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'master_comparison.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'master_comparison.png'}")


def figure_5_v9_failure_analysis():
    """V9 failure analysis: flat R² profile + comparison."""
    print("Generating Figure 5: V9 Failure Analysis...")

    v9_metrics = load_v9_metrics()
    if v9_metrics is None:
        horizons = list(range(1, 13))
        r2_vals = [0.196, 0.192, 0.192, 0.192, 0.195, 0.201,
                   0.207, 0.208, 0.207, 0.205, 0.202, 0.200]
    else:
        horizons = v9_metrics['H'].tolist()
        r2_vals = v9_metrics['R^2'].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) R² by horizon
    ax1 = axes[0]
    add_panel_label(ax1, 'a')
    ax1.plot(horizons, r2_vals, 'o-', color=COLORS['v9'], linewidth=1.5, markersize=5)
    ax1.axhline(y=0.628, color=COLORS['v2'], linestyle='--', linewidth=1,
                label='ConvLSTM baseline (0.628)')
    ax1.axhline(y=0.212, color=COLORS['v5'], linestyle=':', alpha=0.7, linewidth=1,
                label='Stacking Ens. (0.212)')
    ax1.fill_between(horizons, r2_vals, alpha=0.2, color=COLORS['v9'])

    ax1.set_xlabel('Forecast horizon (months)')
    ax1.set_ylabel('R² score')
    ax1.set_ylim(0, 0.8)
    ax1.set_xticks(horizons)
    ax1.legend(loc='upper right', framealpha=0.9)

    # (b) Comparison of failures
    ax2 = axes[1]
    add_panel_label(ax2, 'b')
    failures = ['FNO-Hybrid', 'Stacking Ens.', 'GNN-BiMamba']
    failure_r2 = [0.312, 0.212, 0.200]
    failure_colors = [COLORS['v3'], COLORS['v5'], COLORS['v9']]

    bars = ax2.bar(failures, failure_r2, color=failure_colors, width=0.5)
    ax2.axhline(y=0.628, color=COLORS['v2'], linestyle='--', linewidth=1,
                label='ConvLSTM baseline')

    for bar, val in zip(bars, failure_r2):
        ax2.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', fontsize=8, fontweight='bold')

    ax2.set_ylabel('R² score')
    ax2.set_ylim(0, 0.8)
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'v9_failure_analysis.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'v9_failure_analysis.png'}")


def figure_6_v10_improvement():
    """Late Fusion improvement over ensemble methods."""
    print("Generating Figure 6: Late Fusion Improvement...")

    fig, ax = plt.subplots(figsize=(10, 5))

    methods = ['ConvLSTM', 'GNN-TAT', 'Simple Avg', 'Weighted Avg', 'Late Fusion\n(Ridge)']
    r2_vals = [0.629, 0.597, 0.633, 0.636, 0.668]
    bar_colors = [COLORS['v2'], COLORS['v4'], COLORS['baseline'],
                  COLORS['baseline'], COLORS['v10']]

    bars = ax.bar(methods, r2_vals, color=bar_colors, width=0.6)

    for bar, val in zip(bars, r2_vals):
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, val),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', fontsize=8)

    ax.axhline(y=0.628, color=COLORS['v2'], linestyle='--', alpha=0.6,
               linewidth=1, label='Best single model')
    ax.set_ylabel('R² score')
    ax.set_ylim(0.55, 0.7)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'v10_improvement_chart.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'v10_improvement_chart.png'}")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating Final Figures (Q1 Journal Standards)")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    figure_1_v1_v10_r2_evolution()
    figure_2_early_vs_late_fusion()
    figure_3_v10_fusion_weights()
    figure_4_master_comparison()
    figure_5_v9_failure_analysis()
    figure_6_v10_improvement()

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
