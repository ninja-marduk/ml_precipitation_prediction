"""
Generate Poster-Optimized Figures for EGU26 A0 Poster
=====================================================
Larger fonts, thicker lines, bigger legends for ~1m viewing distance.
Outputs to .docs/conferences/EGU26/poster/figures/

Based on the same data as paper figures but with 2-3x larger text.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
V2_DIR = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models'
V3_DIR = PROJECT_ROOT / 'models' / 'output' / 'V3_FNO_Models'
V4_DIR = PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models'
OUTPUT_DIR = PROJECT_ROOT / '.docs' / 'conferences' / 'EGU26' / 'poster' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito palette (same as paper)
COLORS = {
    'ConvLSTM': '#0072B2',
    'FNO': '#F0E442',
    'GNN-TAT': '#E69F00',
    'BASIC': '#009E73',
    'KCE': '#CC79A7',
    'PAFC': '#56B4E9',
}
VARIANT_COLORS = {
    'ConvLSTM_Bidir': '#56B4E9',
    'ConvLSTM_Residual': '#999999',
    'FNO_Pure': '#CC79A7',
    'GNN_TAT_alt': '#D55E00',
}

# Poster style: 900 DPI for 3x zoom capability on A0
DPI = 900  # High DPI to support 300% zoom at poster sessions
POSTER_RC = {
    'font.size': 20,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 1.0,
    'axes.linewidth': 1.5,
    'lines.linewidth': 3.0,
    'lines.markersize': 10,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.dpi': 150,
}

PARAM_COUNTS = {
    'ConvLSTM': 148_000,
    'FNO': 106_000,
    'GNN-TAT': 98_000,
    'Late Fusion': 247_000,  # V2 (148K) + V4 (98K) + Ridge (1K)
}
RMSE_SD = {
    'ConvLSTM': 27.43,
    'FNO': 23.60,
    'GNN-TAT': 6.94,
    'Late Fusion': 2.31,  # std across H=1-12 (best seed42)
}

# Late Fusion (V10) metrics at H=12 — best seed (seed42)
# Source: models/output/V10_Late_Fusion/v10_metrics.csv + v10_summary.json
LATE_FUSION_H12 = {
    'R2': 0.6368,
    'RMSE': 79.84,
    'MAE': 57.72,
    'Bias': -6.28,
}


def setup_poster_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(POSTER_RC)


def load_all_metrics() -> pd.DataFrame:
    """Load and combine metrics from all three families."""
    frames = []
    for path, family in [
        (V2_DIR / 'metrics_spatial_v2_refactored_h12.csv', 'ConvLSTM'),
        (V3_DIR / 'metrics_spatial_v2_refactored_h12.csv', 'FNO'),
        (V4_DIR / 'metrics_spatial_v4_gnn_tat_h12.csv', 'GNN-TAT'),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            df['Family'] = family
            if 'mean_bias_mm' not in df.columns and 'Mean_Pred_mm' in df.columns:
                df['mean_bias_mm'] = df['Mean_Pred_mm'] - df['Mean_True_mm']
            frames.append(df)
            logger.info(f"Loaded {family}: {len(df)} records")
        else:
            logger.warning(f"Missing: {path}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# FIGURE 1: Early vs Late Fusion Bar Chart
# ─────────────────────────────────────────────────────────────
def poster_early_vs_late_fusion():
    """Bar chart comparing early and late fusion R²."""
    logger.info("Generating poster: Early vs Late Fusion...")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor('white')

    methods = ['Early Fusion\n(feature-level)', 'Late Fusion\n(Ridge)']
    r2_vals = [0.212, 0.655]
    r2_errs = [0.000, 0.018]  # 3-seed std for Late Fusion; Early is single experiment
    bar_colors = ['#BDBDBD', COLORS['ConvLSTM']]
    edge_colors = ['#888888', '#005A8C']

    bars = ax.bar(methods, r2_vals, color=bar_colors, width=0.50,
                  edgecolor=edge_colors, linewidth=2,
                  yerr=r2_errs, capsize=10, ecolor='#333333',
                  error_kw={'elinewidth': 1.6})
    ax.axhline(y=0.628, color='#999999', linestyle='--',
               linewidth=2, label='Best single model (R² = 0.628)')

    labels = ['0.212', '0.655 ± 0.018']
    for bar, val, lbl, err in zip(bars, r2_vals, labels, r2_errs):
        ax.annotate(lbl,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + err),
                    xytext=(0, 10), textcoords="offset points",
                    ha='center', va='bottom', fontsize=22, fontweight='bold')

    ax.set_ylabel('R²', fontsize=22)
    ax.set_ylim(0, 0.80)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=16)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    out = OUTPUT_DIR / 'early_vs_late_fusion_bar.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', transparent=True)
    plt.close()
    logger.info(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# FIGURE 2: Radar Chart
# ─────────────────────────────────────────────────────────────
def poster_radar_chart(combined_df: pd.DataFrame):
    """Multi-metric radar chart with poster-sized text."""
    logger.info("Generating poster: Radar Chart...")

    h12_data = combined_df[combined_df['H'] == 12]

    categories = ['R²', 'RMSE\n(inv)', 'MAE\n(inv)', 'Bias\n(inv)',
                  'Param\nEfficiency', 'Training\nStability']
    N = len(categories)

    def get_family_metrics(df, family):
        family_data = df[df['Family'] == family]
        if family_data.empty:
            return None
        max_rmse = df['RMSE'].max()
        max_mae = df['MAE'].max()
        best_idx = family_data['R^2'].idxmax()
        best = family_data.loc[best_idx]
        r2 = best['R^2']

        min_p, max_p = min(PARAM_COUNTS.values()), max(PARAM_COUNTS.values())
        param_eff = ((max_p - PARAM_COUNTS[family]) / (max_p - min_p)) * r2

        min_sd, max_sd = min(RMSE_SD.values()), max(RMSE_SD.values())
        stability = (max_sd - RMSE_SD[family]) / (max_sd - min_sd)

        return [
            r2,
            1 - (best['RMSE'] / max_rmse),
            1 - (best['MAE'] / max_mae),
            1 - abs(best.get('mean_bias_mm', 0)) / 50,
            param_eff,
            stability,
        ]

    # Late Fusion metrics — injected manually (not in combined_df)
    max_rmse = h12_data['RMSE'].max()
    max_mae = h12_data['MAE'].max()
    min_p = min(PARAM_COUNTS.values())
    max_p = max(PARAM_COUNTS.values())
    min_sd = min(RMSE_SD.values())
    max_sd = max(RMSE_SD.values())

    lf_param_eff = ((max_p - PARAM_COUNTS['Late Fusion']) / (max_p - min_p)) * LATE_FUSION_H12['R2']
    lf_stability = (max_sd - RMSE_SD['Late Fusion']) / (max_sd - min_sd)
    lf_metrics = [
        LATE_FUSION_H12['R2'],
        1 - (LATE_FUSION_H12['RMSE'] / max_rmse),
        1 - (LATE_FUSION_H12['MAE'] / max_mae),
        1 - abs(LATE_FUSION_H12['Bias']) / 50,
        lf_param_eff,
        lf_stability,
    ]

    families = {
        'ConvLSTM': (get_family_metrics(h12_data, 'ConvLSTM'), COLORS['ConvLSTM'], 'o', 'ConvLSTM'),
        'FNO': (get_family_metrics(h12_data, 'FNO'), COLORS['FNO'], 's', 'FNO-Hybrid'),
        'GNN-TAT': (get_family_metrics(h12_data, 'GNN-TAT'), COLORS['GNN-TAT'], 'D', 'GNN-TAT'),
        'Late Fusion': (lf_metrics, '#117733', '*', 'Late Fusion (Ridge)'),
    }

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 7.2), subplot_kw=dict(projection='polar'))
    ax.set_facecolor('white')
    # Legend on top → equal left/right margins, plot truly centered
    fig.subplots_adjust(left=0.12, right=0.88, top=0.82, bottom=0.08)

    for key, (vals, color, marker, label) in families.items():
        if vals is None:
            continue
        values = vals + [vals[0]]
        ax.plot(angles, values, f'{marker}-', linewidth=3.5, color=color,
                label=label, markersize=14)
        ax.fill(angles, values, alpha=0.15, color=color)
        # Add edge color for FNO yellow markers
        if key == 'FNO':
            ax.plot(angles, values, f'{marker}', color=color, markersize=14,
                    markeredgecolor='#666666', markeredgewidth=1.5, linestyle='None')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.tick_params(axis='x', pad=14)  # separate axis labels from outer circle
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_rlabel_position(45)  # move radial labels off the R² axis area
    # Title removed: caption / poster panel header already names the figure
    # Legend on top, single horizontal row → centers the radar plot
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.04), ncol=4,
              fontsize=12, markerscale=1.1, framealpha=0.9,
              borderpad=0.4, columnspacing=1.2, handletextpad=0.4)

    out = OUTPUT_DIR / 'radar_chart.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', transparent=True)
    plt.close()
    logger.info(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# FIGURE 3: Horizon Degradation
# ─────────────────────────────────────────────────────────────
def poster_horizon_degradation(combined_df: pd.DataFrame):
    """Horizon degradation curves with poster-sized text."""
    logger.info("Generating poster: Horizon Degradation...")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('white')
    fig.subplots_adjust(left=0.07, right=0.74, top=0.88, bottom=0.13)

    # Champions per family + fusion strategies (5-line coherent narrative)
    models_to_plot = [
        ('ConvLSTM_Bidirectional', 'BASIC', 'ConvLSTM', COLORS['ConvLSTM'], '-', 'o',
         'ConvLSTM Bidir (BASIC)'),
        ('FNO_ConvLSTM_Hybrid', 'BASIC', 'FNO', COLORS['FNO'], '-', 'D',
         'FNO-ConvLSTM Hybrid (BASIC)'),
        ('GNN_TAT_GAT', 'BASIC', 'GNN-TAT', COLORS['GNN-TAT'], '-', 'p',
         'GNN-TAT-GAT (BASIC)'),
    ]

    horizons = range(1, 13)

    for model, exp, family, color, linestyle, marker, label in models_to_plot:
        model_data = combined_df[(combined_df['Model'] == model) &
                                  (combined_df['Experiment'] == exp) &
                                  (combined_df['Family'] == family)]
        if model_data.empty:
            continue

        r2_by_h = []
        for h in horizons:
            h_data = model_data[model_data['H'] == h]
            r2_by_h.append(h_data['R^2'].mean() if not h_data.empty else np.nan)

        mkw = {}
        if color == COLORS['FNO']:
            mkw = {'markeredgecolor': '#666666', 'markeredgewidth': 1.5}
        ax.plot(horizons, r2_by_h, color=color, linestyle=linestyle,
                marker=marker, markersize=12, linewidth=3.5, label=label, **mkw)

    # Late Fusion (V10, 3-seed mean)
    try:
        v10_csv = PROJECT_ROOT / 'models' / 'output' / 'V10_Late_Fusion' / \
            'metrics_multiseed_consolidated.csv'
        v10 = pd.read_csv(v10_csv).sort_values('H')
        ax.plot(v10['H'], v10['R^2_mean'], color='#117733', linestyle='-',
                marker='*', markersize=18, linewidth=3.8,
                label='Late Fusion (Ridge)')
        ax.fill_between(v10['H'],
                        v10['R^2_mean'] - v10['R^2_std'],
                        v10['R^2_mean'] + v10['R^2_std'],
                        color='#117733', alpha=0.15)
    except Exception as e:
        logger.warning(f"  Late Fusion data missing: {e}")

    # Early Fusion (V5, single-seed BASIC_KCE)
    try:
        v5_csv = PROJECT_ROOT / 'models' / 'output' / 'V5_GNN_ConvLSTM_Stacking' / \
            'metrics_spatial_v5_all_horizons.csv'
        v5 = pd.read_csv(v5_csv).sort_values('H')
        ax.plot(v5['H'], v5['R^2'], color='#999999', linestyle='--',
                marker='X', markersize=11, linewidth=2.8,
                label='Early Fusion (concat)')
    except Exception as e:
        logger.warning(f"  Early Fusion data missing: {e}")

    ax.set_xlabel('Forecast Horizon (months)', fontsize=22)
    ax.set_ylabel('R² Score', fontsize=22)
    ax.set_title('Forecast Horizon Degradation\n(Family Champions vs Fusion Strategies)',
                 fontsize=22, fontweight='bold')
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.0, 0.75)
    ax.set_xticks(list(horizons))
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0.6, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(11.6, 0.605, 'R²=0.6', fontsize=16, va='center', color='gray')

    ax.legend(loc='upper left', bbox_to_anchor=(1.06, 1.0), framealpha=0.9,
              fontsize=13, ncol=1, borderpad=0.7, labelspacing=0.8,
              title='Architecture / Strategy', title_fontsize=14)

    out = OUTPUT_DIR / 'horizon_degradation.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', transparent=True)
    plt.close()
    logger.info(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# FIGURE 4: Spatial R² Map — 3-panel (V2 / V4 / V10 Late Fusion)
# ─────────────────────────────────────────────────────────────
def poster_spatial_r2_3panel():
    """3-panel cell-level R² map: V2 ConvLSTM | V4 GNN-TAT | V10 Late Fusion."""
    logger.info("Generating poster: Spatial R² 3-panel...")

    V2_PRED = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / \
        'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM_Bidirectional'
    V4_PRED = PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models' / \
        'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT'
    V10_DIR = PROJECT_ROOT / 'models' / 'output' / 'V10_Late_Fusion'
    DATA_NC = PROJECT_ROOT / 'data' / 'output' / \
        'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
    SHP = PROJECT_ROOT / 'data' / 'input' / 'MGN_Departamento.shp'

    try:
        import xarray as xr
        ds = xr.open_dataset(DATA_NC)
        lats = ds.latitude.values
        lons = ds.longitude.values
        ds.close()
    except Exception as e:
        logger.warning(f"  NetCDF load failed ({e}); using indices for axes")
        lats = np.arange(61)
        lons = np.arange(65)

    try:
        import geopandas as gpd
        gdf = gpd.read_file(SHP)
    except Exception as e:
        logger.warning(f"  Shapefile load failed: {e}")
        gdf = None

    def _load(d):
        p = np.load(d / 'predictions.npy')
        t = np.load(d / 'targets.npy')
        if p.ndim == 5:
            p = p[..., 0]
            t = t[..., 0]
        return p, t

    def _r2(pred, tgt):
        s, h, nlat, nlon = pred.shape
        p = pred.reshape(s * h, nlat, nlon)
        t = tgt.reshape(s * h, nlat, nlon)
        ss_res = np.nansum((t - p) ** 2, axis=0)
        ss_tot = np.nansum((t - np.nanmean(t, axis=0, keepdims=True)) ** 2, axis=0)
        return 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)

    p2, t2 = _load(V2_PRED)
    p4, t4 = _load(V4_PRED)
    p10, t10 = _load(V10_DIR)

    r2_v2 = _r2(p2, t2)
    r2_v4 = _r2(p4, t4)
    r2_v10 = _r2(p10, t10)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), sharey=True)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=-0.2, vmax=0.8)

    panels = [
        (axes[0], r2_v2,  'ConvLSTM\n(R² = 0.632)',     'a'),
        (axes[1], r2_v4,  'GNN-TAT\n(R² = 0.597)',      'b'),
        (axes[2], r2_v10, 'Late Fusion (Ridge)\n(R² = 0.668)', 'c'),
    ]

    im = None
    for ax, r2, title, label in panels:
        im = ax.pcolormesh(lon_grid, lat_grid, r2, cmap=cmap, norm=norm, shading='auto')
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color='k', linewidth=1.0, zorder=5)
        ax.set_title(title, fontsize=20, fontweight='bold', pad=10)
        ax.set_xlabel('Longitude', fontsize=18)
        ax.set_aspect('equal')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(x):.1f}°W"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}°N"))
        ax.tick_params(labelsize=14)
        ax.text(0.03, 0.97, f'({label})', transform=ax.transAxes, fontsize=20,
                fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    axes[0].set_ylabel('Latitude', fontsize=18)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, aspect=30)
    cbar.set_label('R$^{2}$ (NSE)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    out = OUTPUT_DIR / 'spatial_r2_map_3panel.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# FIGURE 5: Elevation-Stratified — 3-bar (ConvLSTM / GNN-TAT / Late Fusion)
# ─────────────────────────────────────────────────────────────
def poster_elevation_stratified():
    """Elevation-stratified R² and RMSE bar chart with three architectures."""
    logger.info("Generating poster: Elevation-Stratified 3-bar...")

    V2_PRED = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / \
        'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM_Bidirectional'
    V4_PRED = PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models' / \
        'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT'
    V10_DIR = PROJECT_ROOT / 'models' / 'output' / 'V10_Late_Fusion'
    DATA_NC = PROJECT_ROOT / 'data' / 'output' / \
        'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'

    import xarray as xr
    ds = xr.open_dataset(DATA_NC)
    elev_raw = ds['elevation'].values
    if elev_raw.ndim == 3:
        elev = elev_raw[0]
    elif elev_raw.ndim == 4:
        elev = elev_raw[0, :, :, 0]
    else:
        elev = elev_raw
    elev = np.asarray(elev, dtype=np.float64)
    ds.close()

    def _load(d):
        p = np.load(d / 'predictions.npy')
        t = np.load(d / 'targets.npy')
        if p.ndim == 5:
            p = p[..., 0]
            t = t[..., 0]
        return p, t

    bands = [
        (0,    1000, '<1000 m'),
        (1000, 2000, '1000–2000 m'),
        (2000, 3000, '2000–3000 m'),
        (3000, 6000, '>3000 m'),
    ]

    results = {
        'ConvLSTM':    {'r2': [], 'rmse': [], 'n': []},
        'GNN-TAT':     {'r2': [], 'rmse': [], 'n': []},
        'Late Fusion': {'r2': [], 'rmse': [], 'n': []},
    }

    sources = [
        ('ConvLSTM',    *_load(V2_PRED)),
        ('GNN-TAT',     *_load(V4_PRED)),
        ('Late Fusion', *_load(V10_DIR)),
    ]

    for lo, hi, _ in bands:
        mask = (elev >= lo) & (elev < hi)
        n_cells = int(np.sum(mask))
        for name, pred, tgt in sources:
            s, h, nlat, nlon = pred.shape
            p_flat = pred.reshape(s * h, nlat, nlon)[:, mask]
            t_flat = tgt.reshape(s * h, nlat, nlon)[:, mask]
            obs = t_flat.ravel()
            prd = p_flat.ravel()
            valid = np.isfinite(obs) & np.isfinite(prd)
            obs, prd = obs[valid], prd[valid]
            ss_res = np.sum((obs - prd) ** 2)
            ss_tot = np.sum((obs - obs.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = float(np.sqrt(np.mean((obs - prd) ** 2)))
            results[name]['r2'].append(r2)
            results[name]['rmse'].append(rmse)
            results[name]['n'].append(n_cells)

    band_labels = [b[2] for b in bands]
    n_per_band = results['ConvLSTM']['n']
    x = np.arange(len(band_labels))
    w = 0.27

    bar_colors = {
        'ConvLSTM':    COLORS['ConvLSTM'],
        'GNN-TAT':     COLORS['GNN-TAT'],
        'Late Fusion': '#117733',
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    # (a) R²
    ax = axes[0]
    for i, name in enumerate(['ConvLSTM', 'GNN-TAT', 'Late Fusion']):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, results[name]['r2'], w,
                      color=bar_colors[name], label=name,
                      edgecolor='#333', linewidth=0.8)
        for j, bar in enumerate(bars):
            v = results[name]['r2'][j]
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=16)
    ax.set_ylabel('R² (NSE)', fontsize=20)
    # X-axis label pushed down via labelpad to leave room for the n=... line.
    ax.set_xlabel('Elevation band', fontsize=20, labelpad=22)
    ax.set_ylim(0, max(0.78, max(max(results[k]['r2']) for k in results) + 0.12))
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(loc='upper right', framealpha=0.92, fontsize=11,
              borderpad=0.4, labelspacing=0.4, handlelength=1.6,
              handletextpad=0.5, columnspacing=0.6)
    ax.text(-0.08, 1.02, '(a)', transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='bottom')
    # n=... annotations: lower (y=-0.10) and bigger font (15) so they sit
    # cleanly between the band labels and the "Elevation band" axis title.
    for i, n in enumerate(n_per_band):
        ax.text(i, -0.10, f'n={n}', ha='center', va='top',
                fontsize=15, fontweight='bold',
                transform=ax.get_xaxis_transform(), color='#444')

    # (b) RMSE
    ax = axes[1]
    for i, name in enumerate(['ConvLSTM', 'GNN-TAT', 'Late Fusion']):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, results[name]['rmse'], w,
                      color=bar_colors[name], label=name,
                      edgecolor='#333', linewidth=0.8)
        for j, bar in enumerate(bars):
            v = results[name]['rmse'][j]
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                        f'{v:.1f}', ha='center', va='bottom',
                        fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=16)
    ax.set_ylabel('RMSE (mm)', fontsize=20)
    ax.set_xlabel('Elevation band', fontsize=20, labelpad=22)
    rmse_max = max(max(results[k]['rmse']) for k in results)
    ax.set_ylim(0, rmse_max * 1.18)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(loc='upper right', framealpha=0.92, fontsize=11,
              borderpad=0.4, labelspacing=0.4, handlelength=1.6,
              handletextpad=0.5, columnspacing=0.6)
    ax.text(-0.08, 1.02, '(b)', transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='bottom')
    # Mirror the n=... annotations on panel (b) for visual symmetry.
    for i, n in enumerate(n_per_band):
        ax.text(i, -0.10, f'n={n}', ha='center', va='top',
                fontsize=15, fontweight='bold',
                transform=ax.get_xaxis_transform(), color='#444')

    plt.tight_layout()
    out = OUTPUT_DIR / 'elevation_stratified_3bar.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {out}")
    logger.info(f"  Cell counts: {dict(zip(band_labels, n_per_band))}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    setup_poster_style()

    # Figure 1: standalone (hardcoded values)
    poster_early_vs_late_fusion()

    # Figures 2 & 3: need combined metrics
    combined_df = load_all_metrics()
    if combined_df.empty:
        logger.error("No metrics data found!")
        sys.exit(1)

    poster_radar_chart(combined_df)
    poster_horizon_degradation(combined_df)

    # Figure 4: spatial R² 3-panel (V2 / V4 / V10)
    try:
        poster_spatial_r2_3panel()
    except Exception as e:
        logger.error(f"Spatial 3-panel failed: {e}")

    # Figure 5: elevation-stratified 3-bar
    try:
        poster_elevation_stratified()
    except Exception as e:
        logger.error(f"Elevation-stratified failed: {e}")

    logger.info(f"\nAll poster figures saved to: {OUTPUT_DIR}")
