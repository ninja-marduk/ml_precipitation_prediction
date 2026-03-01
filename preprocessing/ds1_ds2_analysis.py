"""
DS1 vs DS2 Bidirectional Analysis (Recommendation 1)

DS1: CHIRPS -> DEM   (How does precipitation vary with elevation?)
DS2: DEM -> CHIRPS   (How does topography predict precipitation patterns?)

This script analyzes the bidirectional relationship between elevation
and precipitation in the Boyaca study area using the existing dataset.

Output:
    - Correlation analysis (Pearson, Spearman) by elevation cluster
    - Scatter plots: precipitation vs elevation by month/season
    - Boxplots: precipitation distribution by elevation band
    - Spatial patterns: precipitation anomalies vs topographic features
    - Summary statistics for Paper 5 Methods/Results sections

Usage:
    python preprocessing/ds1_ds2_analysis.py
    python preprocessing/ds1_ds2_analysis.py --dataset data/output/complete_dataset_*.nc
    python preprocessing/ds1_ds2_analysis.py --output-dir scripts/benchmark/output/ds1_ds2
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Okabe-Ito palette
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'sky': '#56B4E9',
}

ELEVATION_CLUSTERS = {
    'Low (<1500m)': (0, 1500),
    'Medium (1500-2500m)': (1500, 2500),
    'High (>2500m)': (2500, 9999),
}

SEASONS = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}


def load_dataset(dataset_path):
    """Load the main precipitation dataset."""
    logger.info(f"Loading dataset: {dataset_path}")
    ds = xr.open_dataset(dataset_path)
    logger.info(f"  Time: {ds.sizes['time']}, Lat: {ds.sizes['latitude']}, Lon: {ds.sizes['longitude']}")
    return ds


def ds1_precipitation_vs_elevation(ds, output_dir):
    """DS1: How does precipitation vary with elevation?

    Analyzes the statistical relationship from CHIRPS precipitation to DEM elevation.
    """
    from scipy import stats as scipy_stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("  DS1: CHIRPS -> DEM (Precipitation vs Elevation)")
    logger.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    precip = ds['total_precipitation'].values  # (time, lat, lon)
    elev = ds['elevation'].isel(time=0).values  # (lat, lon)
    months = ds['month'].isel(latitude=0, longitude=0).values if 'month' in ds else None

    n_time = precip.shape[0]

    # Mean annual precipitation per cell
    mean_precip = np.nanmean(precip, axis=0)  # (lat, lon)

    results = {}

    # --- 1. Overall correlation ---
    flat_elev = elev.flatten()
    flat_precip = mean_precip.flatten()
    valid = ~np.isnan(flat_elev) & ~np.isnan(flat_precip)

    r_pearson, p_pearson = scipy_stats.pearsonr(flat_elev[valid], flat_precip[valid])
    r_spearman, p_spearman = scipy_stats.spearmanr(flat_elev[valid], flat_precip[valid])
    logger.info(f"Overall Pearson r = {r_pearson:.3f} (p={p_pearson:.2e})")
    logger.info(f"Overall Spearman rho = {r_spearman:.3f} (p={p_spearman:.2e})")

    results['overall'] = {
        'pearson_r': r_pearson, 'pearson_p': p_pearson,
        'spearman_rho': r_spearman, 'spearman_p': p_spearman,
        'n_cells': int(valid.sum()),
    }

    # --- 2. Correlation by elevation cluster ---
    logger.info("\nCorrelation by elevation cluster:")
    cluster_results = {}
    for name, (lo, hi) in ELEVATION_CLUSTERS.items():
        mask = (elev >= lo) & (elev < hi)
        if mask.sum() == 0:
            continue
        e = elev[mask]
        p = mean_precip[mask]
        valid_m = ~np.isnan(e) & ~np.isnan(p)
        if valid_m.sum() < 10:
            continue
        r, pval = scipy_stats.pearsonr(e[valid_m], p[valid_m])
        rho, pval_s = scipy_stats.spearmanr(e[valid_m], p[valid_m])
        logger.info(f"  {name}: r={r:.3f} (p={pval:.2e}), rho={rho:.3f}, n={valid_m.sum()}")
        cluster_results[name] = {
            'pearson_r': r, 'pearson_p': pval,
            'spearman_rho': rho, 'spearman_p': pval_s,
            'n_cells': int(valid_m.sum()),
            'mean_precip': float(np.nanmean(p)),
            'std_precip': float(np.nanstd(p)),
            'mean_elev': float(np.nanmean(e)),
        }
    results['by_cluster'] = cluster_results

    # --- 3. Scatter: mean precipitation vs elevation ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_colors = [COLORS['green'], COLORS['orange'], COLORS['red']]

    for idx, (name, (lo, hi)) in enumerate(ELEVATION_CLUSTERS.items()):
        mask = (elev >= lo) & (elev < hi)
        ax.scatter(elev[mask], mean_precip[mask],
                   c=cluster_colors[idx], alpha=0.5, s=15, label=name,
                   edgecolors='none')

    ax.set_xlabel('Elevation (m)', fontfamily='Arial', fontsize=11)
    ax.set_ylabel('Mean Monthly Precipitation (mm)', fontfamily='Arial', fontsize=11)
    ax.set_title('DS1: Precipitation vs Elevation', fontfamily='Arial', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add regression line
    z = np.polyfit(flat_elev[valid], flat_precip[valid], 1)
    x_line = np.linspace(flat_elev[valid].min(), flat_elev[valid].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), '--', color='gray', linewidth=1.5,
            label=f'Linear: {z[0]:.3f}x + {z[1]:.1f}')
    ax.legend(fontsize=9)

    fig_path = output_dir / 'ds1_precip_vs_elevation.pdf'
    plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # --- 4. Boxplot: precipitation by elevation band ---
    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = []
    box_labels = []
    for name, (lo, hi) in ELEVATION_CLUSTERS.items():
        mask = (elev >= lo) & (elev < hi)
        box_data.append(mean_precip[mask][~np.isnan(mean_precip[mask])])
        box_labels.append(name)

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], cluster_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Mean Monthly Precipitation (mm)', fontfamily='Arial', fontsize=11)
    ax.set_title('DS1: Precipitation Distribution by Elevation Band', fontfamily='Arial', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    fig_path = output_dir / 'ds1_precip_boxplot_by_elevation.pdf'
    plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # --- 5. Seasonal analysis ---
    if months is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, (season, month_list) in zip(axes.flat, SEASONS.items()):
            season_mask = np.isin(months, month_list)
            if season_mask.sum() == 0:
                continue
            season_precip = np.nanmean(precip[season_mask], axis=0)

            for idx, (name, (lo, hi)) in enumerate(ELEVATION_CLUSTERS.items()):
                mask = (elev >= lo) & (elev < hi)
                ax.scatter(elev[mask], season_precip[mask],
                           c=cluster_colors[idx], alpha=0.4, s=10, label=name,
                           edgecolors='none')

            ax.set_xlabel('Elevation (m)', fontsize=9)
            ax.set_ylabel('Mean Precip (mm)', fontsize=9)
            ax.set_title(f'{season}', fontfamily='Arial', fontsize=11)
            ax.grid(True, alpha=0.3)

        axes[0, 0].legend(fontsize=8)
        plt.suptitle('DS1: Seasonal Precipitation vs Elevation', fontfamily='Arial', fontsize=13)
        plt.tight_layout()

        fig_path = output_dir / 'ds1_seasonal_precip_vs_elevation.pdf'
        plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {fig_path}")

    return results


def ds2_topography_predicts_precipitation(ds, output_dir):
    """DS2: How does topography predict precipitation patterns?

    Uses elevation features (if available) to explain spatial variance in precipitation.
    """
    from scipy import stats as scipy_stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("  DS2: DEM -> CHIRPS (Topography predicts Precipitation)")
    logger.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    precip = ds['total_precipitation'].values
    elev = ds['elevation'].isel(time=0).values
    mean_precip = np.nanmean(precip, axis=0)
    std_precip = np.nanstd(precip, axis=0)

    results = {}

    # --- 1. Spatial variance explained by elevation ---
    flat_e = elev.flatten()
    flat_p = mean_precip.flatten()
    valid = ~np.isnan(flat_e) & ~np.isnan(flat_p)

    # Linear regression: precipitation = a * elevation + b
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        flat_e[valid], flat_p[valid])
    r2 = r_value ** 2

    logger.info(f"Linear model: precip = {slope:.4f} * elev + {intercept:.1f}")
    logger.info(f"R-squared = {r2:.3f} (spatial variance explained by elevation alone)")

    results['linear_model'] = {
        'slope': slope, 'intercept': intercept,
        'r_squared': r2, 'p_value': p_value, 'std_err': std_err,
    }

    # --- 2. Polynomial fit (quadratic) ---
    coeffs = np.polyfit(flat_e[valid], flat_p[valid], 2)
    p_pred = np.polyval(coeffs, flat_e[valid])
    ss_res = np.sum((flat_p[valid] - p_pred) ** 2)
    ss_tot = np.sum((flat_p[valid] - np.mean(flat_p[valid])) ** 2)
    r2_quad = 1 - ss_res / ss_tot
    logger.info(f"Quadratic R-squared = {r2_quad:.3f}")

    results['quadratic_model'] = {
        'coefficients': coeffs.tolist(),
        'r_squared': r2_quad,
    }

    # --- 3. Slope/aspect influence (if available) ---
    topo_vars = {}
    for var in ['slope', 'aspect']:
        if var in ds:
            topo_vars[var] = ds[var].isel(time=0).values.flatten()

    if topo_vars:
        logger.info("\nTopographic variable correlations with mean precipitation:")
        for var, values in topo_vars.items():
            valid_v = valid & ~np.isnan(values)
            if valid_v.sum() > 10:
                r, p = scipy_stats.pearsonr(values[valid_v], flat_p[valid_v])
                logger.info(f"  {var}: r={r:.3f} (p={p:.2e})")
                results[f'{var}_correlation'] = {'pearson_r': r, 'p_value': p}

    # --- 4. Precipitation variability vs elevation ---
    # CV (coefficient of variation) as a measure of temporal variability
    cv_precip = std_precip / (mean_precip + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals map
    predicted = slope * elev + intercept
    residuals = mean_precip - predicted

    im1 = axes[0].pcolormesh(ds['longitude'].values, ds['latitude'].values,
                              residuals, cmap='RdBu_r', shading='auto')
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Residual (mm)')
    axes[0].set_title('DS2: Precipitation Residuals\n(observed - linear elevation model)',
                       fontfamily='Arial', fontsize=11)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')

    # CV vs elevation
    flat_cv = cv_precip.flatten()
    valid_cv = valid & ~np.isnan(flat_cv) & (flat_cv < 10)
    axes[1].scatter(flat_e[valid_cv], flat_cv[valid_cv], alpha=0.3, s=10,
                    c=COLORS['blue'], edgecolors='none')
    axes[1].set_xlabel('Elevation (m)', fontfamily='Arial', fontsize=11)
    axes[1].set_ylabel('CV of Precipitation', fontfamily='Arial', fontsize=11)
    axes[1].set_title('DS2: Precipitation Variability vs Elevation',
                       fontfamily='Arial', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / 'ds2_topography_predicts_precip.pdf'
    plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # --- 5. Spatial map of mean precip + elevation contours ---
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(ds['longitude'].values, ds['latitude'].values,
                        mean_precip, cmap='YlGnBu', shading='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Mean Precip (mm)')

    # Elevation contours
    ct = ax.contour(ds['longitude'].values, ds['latitude'].values,
                     elev, levels=[1000, 1500, 2000, 2500, 3000, 3500],
                     colors='black', linewidths=0.8, alpha=0.7)
    ax.clabel(ct, fontsize=7, fmt='%d m')

    ax.set_title('DS2: Mean Precipitation + Elevation Contours', fontfamily='Arial', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    fig_path = output_dir / 'ds2_precip_elevation_overlay.pdf'
    plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {fig_path}")

    return results


def save_summary(ds1_results, ds2_results, output_dir):
    """Save analysis summary as CSV and LaTeX table."""
    import csv

    output_dir = Path(output_dir)

    # CSV summary
    csv_path = output_dir / 'ds1_ds2_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Analysis', 'Metric', 'Value'])

        # DS1
        if 'overall' in ds1_results:
            r = ds1_results['overall']
            writer.writerow(['DS1_Overall', 'Pearson_r', f"{r['pearson_r']:.3f}"])
            writer.writerow(['DS1_Overall', 'Spearman_rho', f"{r['spearman_rho']:.3f}"])
            writer.writerow(['DS1_Overall', 'N_cells', r['n_cells']])

        if 'by_cluster' in ds1_results:
            for name, r in ds1_results['by_cluster'].items():
                writer.writerow([f'DS1_{name}', 'Pearson_r', f"{r['pearson_r']:.3f}"])
                writer.writerow([f'DS1_{name}', 'Mean_precip_mm', f"{r['mean_precip']:.1f}"])
                writer.writerow([f'DS1_{name}', 'Mean_elev_m', f"{r['mean_elev']:.0f}"])
                writer.writerow([f'DS1_{name}', 'N_cells', r['n_cells']])

        # DS2
        if 'linear_model' in ds2_results:
            r = ds2_results['linear_model']
            writer.writerow(['DS2_Linear', 'R_squared', f"{r['r_squared']:.3f}"])
            writer.writerow(['DS2_Linear', 'Slope', f"{r['slope']:.4f}"])
            writer.writerow(['DS2_Linear', 'Intercept', f"{r['intercept']:.1f}"])

        if 'quadratic_model' in ds2_results:
            writer.writerow(['DS2_Quadratic', 'R_squared', f"{ds2_results['quadratic_model']['r_squared']:.3f}"])

    logger.info(f"Summary saved: {csv_path}")

    # LaTeX table
    tex_path = output_dir / 'ds1_ds2_table.tex'
    with open(tex_path, 'w') as f:
        f.write("% DS1 vs DS2 Bidirectional Analysis Summary\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Bidirectional analysis: precipitation-elevation relationship.}\n")
        f.write("\\label{tab:ds1-ds2}\n")
        f.write("\\begin{tabular}{llrr}\n")
        f.write("\\toprule\n")
        f.write("Direction & Cluster & Pearson $r$ & $n$ \\\\\n")
        f.write("\\midrule\n")

        if 'overall' in ds1_results:
            r = ds1_results['overall']
            f.write(f"DS1 (CHIRPS$\\to$DEM) & Overall & {r['pearson_r']:.3f} & {r['n_cells']} \\\\\n")

        if 'by_cluster' in ds1_results:
            for name, r in ds1_results['by_cluster'].items():
                f.write(f" & {name} & {r['pearson_r']:.3f} & {r['n_cells']} \\\\\n")

        f.write("\\midrule\n")
        if 'linear_model' in ds2_results:
            r2 = ds2_results['linear_model']['r_squared']
            f.write(f"DS2 (DEM$\\to$CHIRPS) & Linear $R^2$ & {r2:.3f} & -- \\\\\n")
        if 'quadratic_model' in ds2_results:
            r2q = ds2_results['quadratic_model']['r_squared']
            f.write(f" & Quadratic $R^2$ & {r2q:.3f} & -- \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    logger.info(f"LaTeX table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description='DS1 vs DS2 Bidirectional Analysis')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset .nc')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    dataset_path = args.dataset or str(
        PROJECT_ROOT / 'data' / 'output' /
        'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
    )
    output_dir = args.output_dir or str(PROJECT_ROOT / 'scripts' / 'benchmark' / 'output' / 'ds1_ds2')

    ds = load_dataset(dataset_path)

    ds1_results = ds1_precipitation_vs_elevation(ds, output_dir)
    ds2_results = ds2_topography_predicts_precipitation(ds, output_dir)
    save_summary(ds1_results, ds2_results, output_dir)

    ds.close()

    logger.info("=" * 60)
    logger.info("  DS1 vs DS2 ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
