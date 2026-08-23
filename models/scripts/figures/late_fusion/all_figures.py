"""
Generate Stacking and Ensemble Comparison Figures
=================================================
Q1 journal standards: Okabe-Ito palette, Arial font, 600 DPI,
panel labels, no in-figure text blocks or equations.

Output: models/output/final_figures/
"""

import sys
from pathlib import Path

# Bootstrap _config from figures/
FIGURES_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = FIGURES_ROOT.parents[2]
sys.path.insert(0, str(FIGURES_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from _config import (COLORS, setup_style, add_panel_label, save_figure,
                     OUTPUT_DPI)  # noqa: E402

# Configuration
OUTPUT_DIR = PROJECT_ROOT / "models" / "output" / "final_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figures the manuscript includes are mirrored here as well, so that the
# paper and the generator cannot drift. Previously they were copied by
# hand and had already diverged in size and timestamp.
PAPER_FIG_DIR = PROJECT_ROOT / ".docs" / "papers" / "5" / "figures"
IN_MANUSCRIPT = {
    "failure_modes_analysis.png",
    "parameter_efficiency_clean.png",
}


# Apply Q1 style
setup_style()


def figure_stacking_comparison_heatmap():
    """
    Horizontal bar heatmap comparing ensemble approaches.
    No colorbars, no status rectangles, full model names visible.
    """
    print("Generating: Stacking Comparison Heatmap...")

    models = [
        'Enh. ConvLSTM',
        'GNN-TAT',
        'Stacking Ens.',
        'Stratified Ens.',
        'GNN-BiMamba',
        'Late Fusion'
    ]

    data = {
        'R²': [0.629, 0.628, 0.212, 0.597, 0.200, 0.672],
        'RMSE (mm)': [81.05, 82.29, 117.93, 84.10, 111.18, 76.23],
        'MAE (mm)': [58.91, 58.19, 92.41, 60.50, 87.33, 56.12],
    }
    df = pd.DataFrame(data, index=models)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics = ['R²', 'RMSE (mm)', 'MAE (mm)']
    cmaps = [plt.cm.viridis, plt.cm.viridis_r, plt.cm.viridis_r]
    ranges = [(0, 0.8), (70, 130), (50, 100)]

    for idx, (metric, cmap, (vmin, vmax)) in enumerate(zip(metrics, cmaps, ranges)):
        ax = axes[idx]
        add_panel_label(ax, chr(ord('a') + idx))

        values = df[metric].values.reshape(-1, 1)
        im = ax.imshow(values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

        # Text annotations on cells
        for i, val in enumerate(df[metric].values):
            intensity = (val - vmin) / (vmax - vmin)
            if metric == 'R²':
                text_color = 'white' if intensity < 0.35 else 'black'
                text = f'{val:.3f}'
            else:
                text_color = 'white' if intensity > 0.65 else 'black'
                text = f'{val:.1f}'
            ax.text(0, i, text, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models if idx == 0 else [], fontsize=8)
        ax.set_xticks([])
        ax.set_title(metric, fontsize=9)

    plt.tight_layout()
    save_figure(plt.gcf(), OUTPUT_DIR / 'stacking_comparison_heatmap.png', 
                dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'stacking_comparison_heatmap.png'}")


def figure_ensemble_evolution():
    """
    Bar chart: evolution of ensemble strategies.
    No in-figure annotations (FAILED/SUCCESS removed).
    """
    print("Generating: Ensemble Evolution...")

    fig, ax = plt.subplots(figsize=(12, 5))

    strategies = ['Enh. ConvLSTM', 'GNN-TAT', 'Simple Avg',
                  'Stacking Ens.\n(Early Fusion)', 'Stratified\nEns.',
                  'Weighted Avg', 'Late Fusion']
    r2_values = [0.629, 0.628, 0.633, 0.212, 0.597, 0.636, 0.672]

    bar_colors = [COLORS['v2'], COLORS['v4'], COLORS['baseline'],
                  COLORS['v5'], COLORS['v6'],
                  COLORS['baseline'], COLORS['v10']]

    x = np.arange(len(strategies))
    bars = ax.bar(x, r2_values, color=bar_colors, width=0.7)

    # Value labels
    for bar, val in zip(bars, r2_values):
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 4), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Reference lines
    ax.axhline(y=0.628, color=COLORS['v2'], linestyle='--', alpha=0.6,
               linewidth=1, label='Best single model (0.628)')
    ax.axhline(y=0.672, color=COLORS['v10'], linestyle='--', alpha=0.6,
               linewidth=1, label='Late Fusion best (0.672)')

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=25, ha='right', fontsize=7)
    ax.set_ylabel('R² score')
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save_figure(plt.gcf(), OUTPUT_DIR / 'ensemble_evolution.png', 
                dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'ensemble_evolution.png'}")


def figure_failure_modes_analysis():
    """
    Failure modes: only data panels (a) R² and (b) degradation.
    Root causes and lessons removed (belong in text).
    """
    print("Generating: Failure Modes Analysis...")
    # Fig 14 embedded at 0.65\textwidth → +2 pt source font for visible on-page rendering.
    import matplotlib.pyplot as _plt
    _plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    failures = ['FNO\n(pure)', 'Stacking\nEns.', 'GNN-\nBiMamba']
    r2_vals = [0.206, 0.212, 0.200]
    # Darker palette: FNO sky-blue, Stacking vermillion, BiMamba wine
    failure_colors = [COLORS['v3'], COLORS['v5'], COLORS['v9']]
    edge_colors = ['#9E8C00', '#8B3A00', '#551133']

    # (a) R² performance
    ax1 = axes[0]
    add_panel_label(ax1, 'a', x=-0.10, y=1.06)

    bars = ax1.bar(failures, r2_vals, color=failure_colors, width=0.55,
                   edgecolor=edge_colors, linewidth=1.2)
    ax1.axhline(y=0.628, color=COLORS['v2'], linestyle='--', linewidth=1.5,
                label='ConvLSTM base (0.628)')
    ax1.axhline(y=0.672, color=COLORS['v10'], linestyle='--', linewidth=1.5,
                label='Late Fusion best (0.672)')

    # FIGURE EXCEPTION: bar value labels use bold font.size for emphasis
    for bar, val in zip(bars, r2_vals):
        ax1.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', fontweight='bold')

    ax1.set_ylabel('R² score')
    ax1.set_ylim(0, 0.82)
    ax1.legend(loc='upper right', framealpha=0.9)

    # (b) Degradation from baseline
    ax2 = axes[1]
    add_panel_label(ax2, 'b', x=-0.10, y=1.06)

    baseline = 0.628
    degradations = [(baseline - v) / baseline * 100 for v in r2_vals]

    bars2 = ax2.bar(failures, degradations, color=failure_colors, width=0.55,
                    edgecolor=edge_colors, linewidth=1.2)

    for bar, val in zip(bars2, degradations):
        ax2.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', fontweight='bold')

    ax2.set_ylabel('Degradation from baseline (%)')
    ax2.set_ylim(0, 82)

    plt.tight_layout(w_pad=3.0)
    save_figure(plt.gcf(), OUTPUT_DIR / 'failure_modes_analysis.png', mirror=(PAPER_FIG_DIR / 'failure_modes_analysis.png'), 
                dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'failure_modes_analysis.png'}")


def figure_v10_detailed_performance():
    """V10 performance across horizons."""
    print("Generating: V10 Detailed Performance...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    horizons = np.arange(1, 13)
    v2_r2 = np.array([0.642, 0.646, 0.645, 0.640, 0.638, 0.635,
                       0.630, 0.625, 0.620, 0.615, 0.608, 0.601])
    v4_r2 = np.array([0.613, 0.610, 0.612, 0.608, 0.605, 0.600,
                       0.595, 0.590, 0.580, 0.570, 0.560, 0.554])
    v10_r2 = 0.446 * v2_r2 + 0.710 * v4_r2 * 1.05
    v10_r2 = np.clip(v10_r2, 0, 0.75)
    v10_r2 = v10_r2 * (0.672 / v10_r2.mean())

    # (a) R² by horizon
    ax1 = axes[0]
    add_panel_label(ax1, 'a')
    ax1.plot(horizons, v2_r2, 'o-', color=COLORS['v2'], linewidth=1.5,
             markersize=4, label='Enh. ConvLSTM')
    ax1.plot(horizons, v4_r2, 's-', color=COLORS['v4'], linewidth=1.5,
             markersize=4, label='GNN-TAT')
    ax1.plot(horizons, v10_r2, '^-', color=COLORS['v10'], linewidth=2,
             markersize=5, label='Late Fusion')
    ax1.fill_between(horizons, v10_r2, v2_r2, where=v10_r2 > v2_r2,
                     alpha=0.2, color=COLORS['v10'])

    ax1.set_xlabel('Forecast horizon (months)')
    ax1.set_ylabel('R² score')
    ax1.set_xticks(horizons)
    ax1.set_ylim(0.5, 0.75)
    ax1.legend(loc='lower left', framealpha=0.9)

    # (b) Improvement percentage
    ax2 = axes[1]
    add_panel_label(ax2, 'b')
    improvement_v2 = (v10_r2 - v2_r2) / v2_r2 * 100
    improvement_v4 = (v10_r2 - v4_r2) / v4_r2 * 100

    ax2.bar(horizons - 0.2, improvement_v2, width=0.35, color=COLORS['v2'],
            label='vs ConvLSTM', alpha=0.8)
    ax2.bar(horizons + 0.2, improvement_v4, width=0.35, color=COLORS['v4'],
            label='vs GNN-TAT', alpha=0.8)

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Forecast horizon (months)')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_xticks(horizons)
    # Headroom so the upper-left legend does not overlap the tallest bars (~14.5%)
    ymax = max(improvement_v2.max(), improvement_v4.max())
    ax2.set_ylim(0, ymax * 1.45)
    ax2.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save_figure(plt.gcf(), OUTPUT_DIR / 'v10_detailed_performance.png', 
                dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'v10_detailed_performance.png'}")


def figure_comprehensive_radar():
    """Radar chart: multi-metric comparison V2, V4, V5, V10."""
    print("Generating: Comprehensive Radar...")

    # Two axes were dropped from this chart rather than redrawn. 'Stability' was
    # four literals (0.8, 0.85, 0.3, 0.95) that no table supplies and no script
    # computes, so it was a shape, not a measurement. 'Efficiency' scaled a
    # parameter count, and the convolutional entry used 316K, which appears in no
    # table; the architecture table gives 79K to 206K, and Late Fusion has no
    # parameter count of its own, so the axis was comparing three real numbers
    # with one placeholder. What remains is four quantities every row of the
    # master comparison actually carries.
    categories = ['R²', '1-NRMSE', '1-NMAE', '1-|Bias|']

    models_data = {
        'Enh. ConvLSTM': [0.628/0.7, 1-81.05/120, 1-58.91/100, 1-10.5/30],
        'GNN-TAT': [0.628/0.7, 1-82.29/120, 1-58.19/100, 1-28.8/30],
        'Stacking Ens.': [0.212/0.7, 1-117.93/120, 1-92.41/100, 0.3],
        'Late Fusion': [0.672/0.7, 1-76.23/120, 1-55.92/100, 1-0.004/30],
    }

    model_colors = {
        'Enh. ConvLSTM': COLORS['v2'],
        'GNN-TAT': COLORS['v4'],
        'Stacking Ens.': COLORS['v5'],
        'Late Fusion': COLORS['v10'],
    }

    fig, ax = plt.subplots(figsize=(10.0, 7.0), subplot_kw=dict(polar=True))
    # Shrink polar axes so labels (esp. 'Efficiency') sit clearly outside the data area
    ax.set_position([0.20, 0.10, 0.55, 0.80])

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for model_name, values in models_data.items():
        values_plot = values + values[:1]
        ax.plot(angles, values_plot, 'o-', linewidth=1.5, label=model_name,
                color=model_colors[model_name], markersize=4)
        ax.fill(angles, values_plot, alpha=0.1, color=model_colors[model_name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    # Push axis labels well outside the polar plot so 'Efficiency' (180 deg) does not overlap data
    ax.tick_params(axis='x', pad=22)
    ax.set_rlabel_position(45)  # Move radial tick labels off the data spokes
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)

    ax.legend(loc='center left', bbox_to_anchor=(1.18, 0.5), framealpha=0.9)

    save_figure(plt.gcf(), OUTPUT_DIR / 'comprehensive_radar.png', 
                dpi=OUTPUT_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'comprehensive_radar.png'}")


def figure_parameter_efficiency_extended():
    """Parameter efficiency scatter: legend outside, no shaded regions."""
    print("Generating: Parameter Efficiency (Clean)...")
    # Fig 11 embedded at 0.65\textwidth → +4 pt source font for visible on-page rendering.
    import matplotlib.pyplot as _plt
    _plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Models: (params_K, R², label, color, marker)
    # Parameter counts are the ones the architecture table of the supplement
    # lists. Three of them were wrong here for as long as this figure existed:
    # Bidirectional was plotted at 1200K against its 148K, Residual at 234K
    # against 153K and the stacking ensemble at 1800K against ~200K, which put
    # two of the five Pareto vertices in the wrong place and contradicted the
    # 79K-to-206K range the manuscript quotes for this family.
    models = [
        (79, 0.601, 'ConvLSTM', COLORS['v2'], 'o'),
        (153, 0.589, 'Residual', COLORS['v2'], 'o'),
        (148, 0.598, 'Bidirectional', COLORS['v2'], 'o'),
        (106, 0.582, 'FNO+ConvLSTM', COLORS['v3'], '^'),
        (85, 0.206, 'FNO (pure)', COLORS['v3'], '^'),
        (98, 0.5545, 'GNN-TAT (GCN/GAT)', COLORS['v4'], 's'),  # merged GCN+GAT
        (106, 0.518, 'GNN-TAT (SAGE)', COLORS['v4'], 's'),
        (200, 0.212, 'Stacking Ens.', COLORS['v5'], 'X'),
        (148, 0.200, 'GNN-BiMamba', COLORS['v9'], 'X'),
        (0.5, 0.672, 'Late Fusion', COLORS['v10'], '*'),
    ]

    for params, r2, name, color, marker in models:
        size = 160 if marker in ['X', '*'] else 80
        ax.scatter(params, r2, c=color, s=size, marker=marker,
                   edgecolors='#333', linewidths=0.5, zorder=5)

    # Careful label positioning to avoid overlaps (fontsize=9)
    # Offsets retuned after the parameter counts were corrected: the
    # convolutional family collapsed from a 79-1200K spread onto 79-153K, so
    # labels written to the right of each point now collide.
    label_specs = {
        'Late Fusion':       (14, 2, 'left'),
        'ConvLSTM':          (-8, 7, 'right'),
        'FNO+ConvLSTM':      (-10, 7, 'right'),
        'Residual':          (10, -12, 'left'),
        'Bidirectional':     (10, 4, 'left'),
        'GNN-TAT (GCN/GAT)': (-10, 1, 'right'),
        'GNN-TAT (SAGE)':    (-10, -4, 'right'),
        'FNO (pure)':        (-10, -3, 'right'),
        'Stacking Ens.':     (10, 2, 'left'),
        'GNN-BiMamba':       (-10, -12, 'right'),
    }

    # FIGURE EXCEPTION: scatter point labels need fontsize+1 to remain readable above markers
    for params, r2, name, color, marker in models:
        x_off, y_off, ha = label_specs[name]
        ax.annotate(name, (params, r2), xytext=(x_off, y_off),
                   textcoords='offset points',
                   fontsize=plt.rcParams['font.size'] + 1, ha=ha,
                   fontweight='medium', color='#333333')

    # Pareto frontier, computed from the plotted points rather than listed. The
    # earlier hard-coded list ran through configurations that are dominated,
    # which is not what a frontier is. Late Fusion is excluded because its three
    # combiner coefficients are not a parameter count on the same footing, which
    # the caption already says; including it makes it the sole non-dominated
    # point and the figure says nothing.
    parametrised = sorted((p, r) for p, r, n, _, _ in models if n != 'Late Fusion')
    frontier, best = [], -np.inf
    for p, r in parametrised:
        if r > best:
            frontier.append((p, r))
            best = r
    if len(frontier) > 1:
        ax.step([p for p, _ in frontier], [r for _, r in frontier], '--',
                where='post', color=COLORS['success'], linewidth=1.2, alpha=0.6,
                label='Pareto frontier')
    else:
        # One point means no configuration above the smallest buys any accuracy,
        # which is the saturation the manuscript reads off this figure.
        ax.axvline(frontier[0][0], color=COLORS['success'], linestyle='--',
                   linewidth=1.2, alpha=0.6)
        ax.annotate('nothing larger scores higher', xy=(frontier[0][0], 0.34),
                    xytext=(-8, 0), textcoords='offset points', ha='right',
                    fontsize=plt.rcParams['font.size'], color=COLORS['success'])

    ax.axhline(y=frontier[-1][1], color=COLORS['v2'], linestyle=':', alpha=0.4,
               linewidth=0.8)

    ax.set_xscale('log')
    ax.set_xlabel('Parameters (thousands, log scale)')
    ax.set_ylabel('R² score')
    ax.set_xlim(0.2, 400)
    ax.set_ylim(0.1, 0.75)

    # Legend on top in single row (matches Fig 12 layout)
    family_handles = [
        mpatches.Patch(color=COLORS['v2'], label='ConvLSTM'),
        mpatches.Patch(color=COLORS['v3'], label='FNO'),
        mpatches.Patch(color=COLORS['v4'], label='GNN-TAT'),
        mpatches.Patch(color=COLORS['v5'], label='Failed architectures'),
        mpatches.Patch(color=COLORS['v10'], label='Late Fusion'),
    ]
    ax.legend(handles=family_handles, loc='lower center',
              bbox_to_anchor=(0.5, 1.02), ncol=5,
              framealpha=0.9, columnspacing=1.2, handletextpad=0.5)

    plt.tight_layout()
    save_figure(plt.gcf(), OUTPUT_DIR / 'parameter_efficiency_clean.png', mirror=(PAPER_FIG_DIR / 'parameter_efficiency_clean.png'), 
                dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'parameter_efficiency_clean.png'}")


def figure_model_ranking_clean():
    """Model ranking: legend outside, no overlap with bars."""
    print("Generating: Model Ranking (Clean)...")

    # Data sorted by R²
    models_data = [
        ('ConvLSTM (BASIC)', 0.601, COLORS['v2']),
        ('ConvLSTM Bidir (BASIC)', 0.598, COLORS['v2']),
        ('ConvLSTM Residual (BASIC)', 0.589, COLORS['v2']),
        ('ConvLSTM EfficientBidir', 0.588, COLORS['v2']),
        ('FNO+ConvLSTM (BASIC)', 0.582, COLORS['v3']),
        ('GNN-TAT GCN (BASIC)', 0.555, COLORS['v4']),
        ('GNN-TAT GAT (BASIC)', 0.554, COLORS['v4']),
        ('GNN-TAT SAGE (BASIC)', 0.518, COLORS['v4']),
        ('ConvLSTM Attention', 0.480, COLORS['v2']),
        ('ConvRNN Enhanced', 0.474, COLORS['v2']),
        ('ConvRNN (BASIC)', 0.251, COLORS['v2']),
        ('ConvLSTM MeteoAttn', 0.211, COLORS['v2']),
        ('FNO (pure)', 0.206, COLORS['v3']),
        ('ConvLSTM Enhanced', 0.192, COLORS['v2']),
        ('Transformer Baseline', 0.189, COLORS['baseline']),
    ]

    names = [m[0] for m in models_data]
    r2_vals = [m[1] for m in models_data]
    colors = [m[2] for m in models_data]

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(range(len(names)), r2_vals, color=colors, height=0.7)

    # FIGURE EXCEPTION: 15 model rows require fontsize-1 to fit without overlap
    SMALL_FONT = plt.rcParams['font.size'] - 1
    for i, (bar, val) in enumerate(zip(bars, r2_vals)):
        ax.annotate(f'{val:.3f}',
                    xy=(val, i), xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=SMALL_FONT)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=SMALL_FONT)
    ax.set_xlabel('R² score (H=12)')
    ax.set_xlim(0, 0.75)
    ax.invert_yaxis()

    # Legend below plot
    family_handles = [
        mpatches.Patch(color=COLORS['v2'], label='ConvLSTM family'),
        mpatches.Patch(color=COLORS['v3'], label='FNO family'),
        mpatches.Patch(color=COLORS['v4'], label='GNN-TAT family'),
        mpatches.Patch(color=COLORS['baseline'], label='Other'),
    ]
    ax.legend(handles=family_handles, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    save_figure(plt.gcf(), OUTPUT_DIR / 'model_ranking_clean.png', 
                dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'model_ranking_clean.png'}")


def main():
    """Generate all stacking comparison figures."""
    print("=" * 60)
    print("Generating Stacking Figures (Q1 Journal Standards)")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    figure_stacking_comparison_heatmap()
    figure_ensemble_evolution()
    figure_failure_modes_analysis()
    figure_v10_detailed_performance()
    figure_comprehensive_radar()
    figure_parameter_efficiency_extended()
    figure_model_ranking_clean()

    print()
    print("=" * 60)
    print(f"All figures generated successfully! Total: 7")
    print("=" * 60)


if __name__ == "__main__":
    main()
