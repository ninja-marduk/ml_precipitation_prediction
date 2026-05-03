"""
Benchmark Analysis Script 16: Generate Tables & Figures for New Metrics

Generates LaTeX tables and publication-quality figures from:
- ACC (Anomaly Correlation Coefficient) results
- FSS (Fractions Skill Score) results
- Elevation-stratified R2/RMSE metrics

Outputs:
- LaTeX .tex table files (ready to include in paper)
- PDF figures (800 DPI, Okabe-Ito palette)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_DIR = PROJECT_ROOT / 'scripts' / 'benchmark' / 'output'
OUTPUT_DIR = PROJECT_ROOT / 'scripts' / 'benchmark' / 'output' / 'figures'
LATEX_DIR = PROJECT_ROOT / 'scripts' / 'benchmark' / 'output' / 'tables'

# Okabe-Ito palette (matching figure_config.py)
COLORS = {
    'V2_ConvLSTM': '#0072B2',   # Blue
    'V4_GNN_TAT': '#E69F00',    # Orange
    'V10_Late_Fusion': '#009E73',  # Green
}
MODEL_LABELS = {
    'V2_ConvLSTM': 'V2 ConvLSTM',
    'V4_GNN_TAT': 'V4 GNN-TAT',
    'V10_Late_Fusion': 'V10 Late Fusion',
}

DPI = 800


def setup_style():
    """Publication-quality matplotlib style."""
    plt.rcParams.update({
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
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


# ============================================================================
# LaTeX TABLE GENERATION
# ============================================================================

def generate_acc_table():
    """Generate LaTeX table for ACC results by horizon."""
    acc_df = pd.read_csv(INPUT_DIR / 'acc_results.csv')

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Anomaly Correlation Coefficient (ACC) per forecast horizon (H1--H12). '
                 r'Climatology estimated from training set mean per grid cell and calendar month.}')
    lines.append(r'\label{tab:acc-horizon}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{l' + 'c' * 12 + 'c}')
    lines.append(r'\toprule')
    lines.append(r'Model & H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10 & H11 & H12 & Mean \\')
    lines.append(r'\midrule')

    for _, row in acc_df.iterrows():
        model = MODEL_LABELS.get(row['Model'], row['Model'])
        vals = [f"{row[f'ACC_H{h}']:.3f}" for h in range(1, 13)]
        mean_val = f"{row['ACC_mean']:.3f}"
        # Bold the best mean
        lines.append(f"{model} & {' & '.join(vals)} & {mean_val} \\\\")

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    # Bold the best mean value
    tex = '\n'.join(lines)
    # Find best model (highest ACC_mean)
    best_idx = acc_df['ACC_mean'].idxmax()
    best_mean = f"{acc_df.loc[best_idx, 'ACC_mean']:.3f}"
    tex = tex.replace(f"& {best_mean} \\\\", f"& \\textbf{{{best_mean}}} \\\\", 1)
    # Only bold the last occurrence (the one after the best model's horizons)
    # Find the line with the best model and replace its mean
    best_model_label = MODEL_LABELS.get(acc_df.loc[best_idx, 'Model'], acc_df.loc[best_idx, 'Model'])

    outpath = LATEX_DIR / 'acc_horizon_table.tex'
    outpath.write_text(tex, encoding='utf-8')
    logger.info(f"ACC table: {outpath}")
    return tex


def generate_fss_table():
    """Generate LaTeX table for FSS results (mean across horizons)."""
    fss_df = pd.read_csv(INPUT_DIR / 'fss_results.csv')

    thresholds = ['1mm', '5mm', '10mm']
    neighborhoods = [1, 3, 5, 7]

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Fractions Skill Score (FSS) averaged across forecast horizons H1--H12. '
                 r'Thresholds: 1, 5, and 10\,mm/month. Neighborhoods: $1\times1$ to $7\times7$ cells.}')
    lines.append(r'\label{tab:fss-summary}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{ll' + 'c' * 4 + '}')
    lines.append(r'\toprule')
    lines.append(r'Model & Threshold & $1\times1$ & $3\times3$ & $5\times5$ & $7\times7$ \\')
    lines.append(r'\midrule')

    for model in ['V2_ConvLSTM', 'V4_GNN_TAT', 'V10_Late_Fusion']:
        model_df = fss_df[fss_df['Model'] == model]
        model_label = MODEL_LABELS[model]
        for i, t in enumerate(thresholds):
            row_label = model_label if i == 0 else ''
            vals = []
            for n in neighborhoods:
                config = f"t{t}_n{n}"
                row = model_df[model_df['Config'] == config]
                if len(row) > 0:
                    vals.append(f"{row.iloc[0]['FSS_mean']:.4f}")
                else:
                    vals.append('--')
            lines.append(f"{row_label} & {t} & {' & '.join(vals)} \\\\")
        if model != 'V10_Late_Fusion':
            lines.append(r'\addlinespace')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines)
    outpath = LATEX_DIR / 'fss_summary_table.tex'
    outpath.write_text(tex, encoding='utf-8')
    logger.info(f"FSS table: {outpath}")
    return tex


def generate_elevation_table():
    """Generate LaTeX table for elevation-stratified R2 and RMSE."""
    elev_df = pd.read_csv(INPUT_DIR / 'elevation_stratified_metrics.csv')

    with open(INPUT_DIR / 'elevation_stratified_metrics.json') as f:
        elev_json = json.load(f)

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Performance metrics stratified by elevation cluster. '
                 r'Pearson $r$ quantifies the correlation between elevation and per-cell $R^2$; '
                 r'negative values indicate degradation at higher altitudes.}')
    lines.append(r'\label{tab:elevation-stratified}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{llcccccc}')
    lines.append(r'\toprule')
    lines.append(r'Model & Cluster & $N$ & $R^2$ & $\sigma_{R^2}$ & RMSE & $\sigma_{\text{RMSE}}$ & Bias \\')
    lines.append(r'\midrule')

    clusters = ['Low (<1500m)', 'Medium (1500-2500m)', 'High (>2500m)']
    for model in ['V2_ConvLSTM', 'V4_GNN_TAT', 'V10_Late_Fusion']:
        model_label = MODEL_LABELS[model]
        model_data = elev_df[(elev_df['Model'] == model) & (elev_df['Metric'] == 'R2')]
        rmse_data = elev_df[(elev_df['Model'] == model) & (elev_df['Metric'] == 'RMSE')]
        bias_data = elev_df[(elev_df['Model'] == model) & (elev_df['Metric'] == 'BIAS')]

        for i, cluster in enumerate(clusters):
            row_label = model_label if i == 0 else ''
            r2_row = model_data[model_data['Cluster'] == cluster].iloc[0]
            rmse_row = rmse_data[rmse_data['Cluster'] == cluster].iloc[0]
            bias_row = bias_data[bias_data['Cluster'] == cluster].iloc[0]

            n_cells = int(r2_row['N_cells'])
            r2_mean = f"{r2_row['Mean']:.3f}"
            r2_std = f"{r2_row['Std']:.3f}"
            rmse_mean = f"{rmse_row['Mean']:.1f}"
            rmse_std = f"{rmse_row['Std']:.1f}"
            bias_mean = f"{bias_row['Mean']:.1f}"

            lines.append(f"{row_label} & {cluster} & {n_cells} & {r2_mean} & {r2_std} & {rmse_mean} & {rmse_std} & {bias_mean} \\\\")

        # Add correlation row
        corr = elev_json[model]['elevation_correlation']
        p_str = f"$p<0.001$" if corr['pearson_p'] < 0.001 else f"$p={corr['pearson_p']:.3f}$"
        lines.append(f" & \\multicolumn{{6}}{{l}}{{Pearson $r={corr['pearson_r']:.3f}$ ({p_str})}} \\\\")

        if model != 'V10_Late_Fusion':
            lines.append(r'\addlinespace')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines)
    outpath = LATEX_DIR / 'elevation_stratified_table.tex'
    outpath.write_text(tex, encoding='utf-8')
    logger.info(f"Elevation table: {outpath}")
    return tex


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def figure_acc_by_horizon():
    """Figure: ACC per horizon for each model (line plot)."""
    acc_df = pd.read_csv(INPUT_DIR / 'acc_results.csv')

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    horizons = list(range(1, 13))
    for _, row in acc_df.iterrows():
        model = row['Model']
        vals = [row[f'ACC_H{h}'] for h in horizons]
        ax.plot(horizons, vals,
                color=COLORS[model],
                marker='o', markersize=4,
                label=MODEL_LABELS[model],
                linewidth=1.5)

    ax.set_xlabel('Forecast Horizon (months)')
    ax.set_ylabel('ACC')
    ax.set_xticks(horizons)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, max(0.20, ax.get_ylim()[1] * 1.1))
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('Anomaly Correlation Coefficient by Forecast Horizon')

    outpath = OUTPUT_DIR / 'acc_by_horizon.pdf'
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    logger.info(f"Figure: {outpath}")


def figure_fss_heatmap():
    """Figure: FSS heatmap (threshold x neighborhood) for each model."""
    fss_df = pd.read_csv(INPUT_DIR / 'fss_results.csv')

    models = ['V2_ConvLSTM', 'V4_GNN_TAT', 'V10_Late_Fusion']
    thresholds = ['1mm', '5mm', '10mm']
    neighborhoods = [1, 3, 5, 7]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = fss_df[fss_df['Model'] == model]

        matrix = np.zeros((len(thresholds), len(neighborhoods)))
        for i, t in enumerate(thresholds):
            for j, n in enumerate(neighborhoods):
                config = f"t{t}_n{n}"
                row = model_df[model_df['Config'] == config]
                if len(row) > 0:
                    matrix[i, j] = row.iloc[0]['FSS_mean']

        im = ax.imshow(matrix, cmap='YlGn', aspect='auto',
                        vmin=0.985, vmax=1.0)
        ax.set_xticks(range(len(neighborhoods)))
        ax.set_xticklabels([f'{n}x{n}' for n in neighborhoods])
        ax.set_xlabel('Neighborhood')

        if idx == 0:
            ax.set_yticks(range(len(thresholds)))
            ax.set_yticklabels(thresholds)
            ax.set_ylabel('Threshold')

        ax.set_title(MODEL_LABELS[model], fontsize=8)

        # Annotate cells
        for i in range(len(thresholds)):
            for j in range(len(neighborhoods)):
                val = matrix[i, j]
                ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                        fontsize=5.5, color='black' if val > 0.993 else 'black')

    fig.colorbar(im, ax=axes, shrink=0.8, label='FSS')
    fig.suptitle('Fractions Skill Score (mean H1-H12)', fontsize=9, y=1.02)

    outpath = OUTPUT_DIR / 'fss_heatmap.pdf'
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    logger.info(f"Figure: {outpath}")


def figure_elevation_stratified_r2():
    """Figure: Grouped bar chart of R2 by elevation cluster and model."""
    elev_df = pd.read_csv(INPUT_DIR / 'elevation_stratified_metrics.csv')
    r2_df = elev_df[elev_df['Metric'] == 'R2']

    models = ['V2_ConvLSTM', 'V4_GNN_TAT', 'V10_Late_Fusion']
    clusters = ['Low (<1500m)', 'Medium (1500-2500m)', 'High (>2500m)']

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    x = np.arange(len(clusters))
    width = 0.25

    for i, model in enumerate(models):
        model_r2 = r2_df[r2_df['Model'] == model]
        means = [model_r2[model_r2['Cluster'] == c]['Mean'].values[0] for c in clusters]
        stds = [model_r2[model_r2['Cluster'] == c]['Std'].values[0] for c in clusters]

        bars = ax.bar(x + i * width, means, width,
                      yerr=stds, capsize=3,
                      color=COLORS[model], alpha=0.85,
                      label=MODEL_LABELS[model],
                      edgecolor='white', linewidth=0.5,
                      error_kw={'linewidth': 0.8})

    ax.set_xlabel('Elevation Cluster')
    ax.set_ylabel('$R^2$')
    ax.set_xticks(x + width)
    ax.set_xticklabels(clusters, fontsize=7)
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('$R^2$ Stratified by Elevation Cluster')

    outpath = OUTPUT_DIR / 'elevation_stratified_r2.pdf'
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    logger.info(f"Figure: {outpath}")


def figure_elevation_stratified_rmse():
    """Figure: Grouped bar chart of RMSE by elevation cluster and model."""
    elev_df = pd.read_csv(INPUT_DIR / 'elevation_stratified_metrics.csv')
    rmse_df = elev_df[elev_df['Metric'] == 'RMSE']

    models = ['V2_ConvLSTM', 'V4_GNN_TAT', 'V10_Late_Fusion']
    clusters = ['Low (<1500m)', 'Medium (1500-2500m)', 'High (>2500m)']

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    x = np.arange(len(clusters))
    width = 0.25

    for i, model in enumerate(models):
        model_rmse = rmse_df[rmse_df['Model'] == model]
        means = [model_rmse[model_rmse['Cluster'] == c]['Mean'].values[0] for c in clusters]
        stds = [model_rmse[model_rmse['Cluster'] == c]['Std'].values[0] for c in clusters]

        bars = ax.bar(x + i * width, means, width,
                      yerr=stds, capsize=3,
                      color=COLORS[model], alpha=0.85,
                      label=MODEL_LABELS[model],
                      edgecolor='white', linewidth=0.5,
                      error_kw={'linewidth': 0.8})

    ax.set_xlabel('Elevation Cluster')
    ax.set_ylabel('RMSE (mm/month)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(clusters, fontsize=7)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('RMSE Stratified by Elevation Cluster')

    outpath = OUTPUT_DIR / 'elevation_stratified_rmse.pdf'
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    logger.info(f"Figure: {outpath}")


def figure_elevation_r2_correlation():
    """Figure: Scatter plot of elevation vs per-cell R2 with regression lines."""
    import xarray as xr

    DATASET_PATH = PROJECT_ROOT / 'data' / 'output' / 'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
    ds = xr.open_dataset(DATASET_PATH)
    elevation = ds['elevation'].values[0]
    ds.close()

    models = ['V2_ConvLSTM', 'V4_GNN_TAT', 'V10_Late_Fusion']

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)

    for idx, model in enumerate(models):
        ax = axes[idx]
        r2_map = np.load(INPUT_DIR / f'r2_spatial_{model}.npy')

        elev_flat = elevation.ravel()
        r2_flat = r2_map.ravel()
        valid = np.isfinite(elev_flat) & np.isfinite(r2_flat)

        ax.scatter(elev_flat[valid], r2_flat[valid],
                   s=1, alpha=0.3, color=COLORS[model], rasterized=True)

        # Regression line
        from scipy.stats import pearsonr
        r, p = pearsonr(elev_flat[valid], r2_flat[valid])
        z = np.polyfit(elev_flat[valid], r2_flat[valid], 1)
        poly = np.poly1d(z)
        x_line = np.linspace(elev_flat[valid].min(), elev_flat[valid].max(), 100)
        ax.plot(x_line, poly(x_line), color='black', linewidth=1.0, linestyle='--')

        ax.set_xlabel('Elevation (m)')
        if idx == 0:
            ax.set_ylabel('Per-cell $R^2$')
        ax.set_title(f'{MODEL_LABELS[model]}\n$r={r:.3f}$', fontsize=8)
        ax.set_ylim(-0.8, 1.0)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')

    fig.suptitle('Elevation vs. Per-Cell $R^2$ (Pearson Correlation)', fontsize=9, y=1.02)

    outpath = OUTPUT_DIR / 'elevation_r2_correlation.pdf'
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    logger.info(f"Figure: {outpath}")


def figure_combined_panel():
    """Figure: Combined 2x2 panel for paper inclusion.
    (a) ACC by horizon, (b) FSS at 10mm threshold,
    (c) R2 by elevation, (d) Elevation-R2 scatter.
    """
    import xarray as xr

    acc_df = pd.read_csv(INPUT_DIR / 'acc_results.csv')
    fss_df = pd.read_csv(INPUT_DIR / 'fss_results.csv')
    elev_df = pd.read_csv(INPUT_DIR / 'elevation_stratified_metrics.csv')

    DATASET_PATH = PROJECT_ROOT / 'data' / 'output' / 'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
    ds = xr.open_dataset(DATASET_PATH)
    elevation = ds['elevation'].values[0]
    ds.close()

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    models = ['V2_ConvLSTM', 'V4_GNN_TAT', 'V10_Late_Fusion']
    horizons = list(range(1, 13))
    clusters = ['Low (<1500m)', 'Medium (1500-2500m)', 'High (>2500m)']

    # (a) ACC by horizon
    ax = axes[0, 0]
    for _, row in acc_df.iterrows():
        model = row['Model']
        vals = [row[f'ACC_H{h}'] for h in horizons]
        ax.plot(horizons, vals, color=COLORS[model], marker='o', markersize=3,
                label=MODEL_LABELS[model], linewidth=1.2)
    ax.set_xlabel('Forecast Horizon (months)')
    ax.set_ylabel('ACC')
    ax.set_xticks(horizons)
    ax.set_xlim(0.5, 12.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend(fontsize=6, loc='upper right')
    ax.text(-0.08, 1.05, '(a)', transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

    # (b) FSS at 10mm threshold by neighborhood
    ax = axes[0, 1]
    neighborhoods = [1, 3, 5, 7]
    for model in models:
        model_df = fss_df[fss_df['Model'] == model]
        vals = []
        for n in neighborhoods:
            config = f"t10mm_n{n}"
            row = model_df[model_df['Config'] == config]
            vals.append(row.iloc[0]['FSS_mean'] if len(row) > 0 else np.nan)
        ax.plot(neighborhoods, vals, color=COLORS[model], marker='s', markersize=4,
                label=MODEL_LABELS[model], linewidth=1.2)
    ax.set_xlabel('Neighborhood Size')
    ax.set_ylabel('FSS (10 mm threshold)')
    ax.set_xticks(neighborhoods)
    ax.set_xticklabels([f'{n}x{n}' for n in neighborhoods])
    ax.set_ylim(0.988, 1.0)
    ax.legend(fontsize=6, loc='lower right')
    ax.text(-0.08, 1.05, '(b)', transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

    # (c) R2 by elevation cluster
    ax = axes[1, 0]
    r2_df = elev_df[elev_df['Metric'] == 'R2']
    x = np.arange(len(clusters))
    width = 0.25
    for i, model in enumerate(models):
        model_r2 = r2_df[r2_df['Model'] == model]
        means = [model_r2[model_r2['Cluster'] == c]['Mean'].values[0] for c in clusters]
        stds = [model_r2[model_r2['Cluster'] == c]['Std'].values[0] for c in clusters]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=2,
               color=COLORS[model], alpha=0.85, label=MODEL_LABELS[model],
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.6})
    ax.set_xlabel('Elevation Cluster')
    ax.set_ylabel('$R^2$')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Low\n(<1500m)', 'Medium\n(1500-2500m)', 'High\n(>2500m)'], fontsize=6)
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=6, loc='upper right')
    ax.text(-0.08, 1.05, '(c)', transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

    # (d) Elevation vs R2 scatter (V10 only for clarity)
    ax = axes[1, 1]
    model = 'V10_Late_Fusion'
    r2_map = np.load(INPUT_DIR / f'r2_spatial_{model}.npy')
    elev_flat = elevation.ravel()
    r2_flat = r2_map.ravel()
    valid = np.isfinite(elev_flat) & np.isfinite(r2_flat)

    ax.scatter(elev_flat[valid], r2_flat[valid], s=1, alpha=0.25,
               color=COLORS[model], rasterized=True)

    from scipy.stats import pearsonr
    r, p = pearsonr(elev_flat[valid], r2_flat[valid])
    z = np.polyfit(elev_flat[valid], r2_flat[valid], 1)
    poly = np.poly1d(z)
    x_line = np.linspace(elev_flat[valid].min(), elev_flat[valid].max(), 100)
    ax.plot(x_line, poly(x_line), color='black', linewidth=1.0, linestyle='--')

    ax.set_xlabel('Elevation (m)')
    ax.set_ylabel('Per-cell $R^2$')
    ax.set_title(f'V10 Late Fusion ($r={r:.3f}$, $p<0.001$)', fontsize=7)
    ax.set_ylim(-0.4, 1.0)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
    ax.text(-0.08, 1.05, '(d)', transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

    fig.tight_layout()

    outpath = OUTPUT_DIR / 'new_metrics_combined_panel.pdf'
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    logger.info(f"Combined panel: {outpath}")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate New Metrics Tables & Figures')
    parser.add_argument('--intracell-dem', action='store_true',
                        help='Read from intra-cell DEM results (Paper 5)')
    parser.add_argument('--bundle', type=str, default='BASIC_D10',
                        choices=['BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS'],
                        help='Feature bundle for --intracell-dem (default: BASIC_D10)')
    return parser.parse_args()


def main():
    global INPUT_DIR, OUTPUT_DIR, LATEX_DIR

    args = parse_args()

    if args.intracell_dem:
        INPUT_DIR = PROJECT_ROOT / 'scripts' / 'benchmark' / 'output' / 'intracell_dem' / args.bundle
        OUTPUT_DIR = INPUT_DIR / 'figures'
        LATEX_DIR = INPUT_DIR / 'tables'
        logger.info(f"Intracell DEM mode: bundle={args.bundle}")
        logger.info(f"Input: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    logger.info("Generating LaTeX tables...")
    generate_acc_table()
    generate_fss_table()
    generate_elevation_table()

    logger.info("Generating figures...")
    figure_acc_by_horizon()
    figure_fss_heatmap()
    figure_elevation_stratified_r2()
    figure_elevation_stratified_rmse()
    figure_elevation_r2_correlation()
    figure_combined_panel()

    logger.info("All tables and figures generated.")


if __name__ == '__main__':
    main()
