"""
Benchmark Analysis Script 15: Elevation-Stratified Metrics

Computes R2, RMSE, MAE, and Bias stratified by elevation clusters:
- Low: < 1500m
- Medium: 1500-2500m
- High: > 2500m

Also computes Pearson correlation between elevation and per-cell R2
to quantify altitudinal bias.

Uses existing predictions.npy and elevation data from the main dataset.
"""

import argparse
import numpy as np
import pandas as pd
import json
import xarray as xr
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / 'data' / 'output' / 'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
OUTPUT_DIR = PROJECT_ROOT / 'scripts' / 'benchmark' / 'output'

MODELS = {
    'V2_ConvLSTM': {
        'predictions': PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / 'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM' / 'predictions.npy',
        'targets': PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / 'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM' / 'targets.npy',
    },
    'V4_GNN_TAT': {
        'predictions': PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT' / 'predictions.npy',
        'targets': PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / 'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM' / 'targets.npy',
    },
    'V10_Late_Fusion': {
        'predictions': PROJECT_ROOT / 'models' / 'output' / 'V10_Late_Fusion' / 'predictions.npy',
        'targets': PROJECT_ROOT / 'models' / 'output' / 'V10_Late_Fusion' / 'targets.npy',
    },
}


def latest_run(base_dir):
    """Return path to most recent run folder under base_dir."""
    base = Path(base_dir)
    if not base.exists():
        return None
    runs = sorted(base.glob('[0-9]*_*'))
    return runs[-1] if runs else None


def get_intracell_dem_models(bundle='BASIC_D10'):
    """Build MODELS dict for intracell DEM predictions (Paper 5).

    Uses latest_run() to auto-discover the most recent run folder under
    models/output/intracell_dem/{Model}/{Bundle}/{YYYYMMDD_N}/.
    """
    idem = PROJECT_ROOT / 'models' / 'output' / 'intracell_dem'
    v2_run = latest_run(idem / 'ConvLSTM' / bundle)
    v4_run = latest_run(idem / 'GNN_TAT_GAT' / bundle)
    v10_run = latest_run(idem / 'Late_Fusion' / bundle)

    models = {}
    if v2_run:
        models['V2_ConvLSTM'] = {
            'predictions': v2_run / 'predictions.npy',
            'targets': v2_run / 'targets.npy',
        }
        logger.info(f'V2_ConvLSTM: {v2_run.name}')
    if v4_run:
        models['V4_GNN_TAT'] = {
            'predictions': v4_run / 'predictions.npy',
            'targets': v2_run / 'targets.npy' if v2_run else v4_run / 'targets.npy',
        }
        logger.info(f'V4_GNN_TAT: {v4_run.name}')
    if v10_run:
        models['V10_Late_Fusion'] = {
            'predictions': v10_run / 'predictions.npy',
            'targets': v10_run / 'targets.npy',
        }
        logger.info(f'V10_Late_Fusion: {v10_run.name}')

    if not models:
        logger.warning(f'No intracell DEM runs found for bundle={bundle}')
    return models


ELEVATION_CLUSTERS = {
    'Low (<1500m)': (0, 1500),
    'Medium (1500-2500m)': (1500, 2500),
    'High (>2500m)': (2500, 5000),
}


def load_predictions(model_name: str):
    """Load predictions and targets, remove channel dim."""
    cfg = MODELS[model_name]
    pred = np.load(cfg['predictions'])
    targ = np.load(cfg['targets'])
    if pred.ndim == 5:
        pred = pred[..., 0]
    if targ.ndim == 5:
        targ = targ[..., 0]
    return pred, targ


def load_elevation_grid():
    """Load elevation grid from the dataset (static, take first timestep)."""
    ds = xr.open_dataset(DATASET_PATH)
    elevation = ds['elevation'].values[0]  # (61, 65) - static across time
    ds.close()
    logger.info(f"Elevation grid: shape={elevation.shape}, range=[{elevation.min():.0f}, {elevation.max():.0f}]m")
    return elevation


def compute_per_cell_metrics(pred, targ):
    """Compute R2, RMSE, MAE, Bias per grid cell.

    Args:
        pred: (N, H, lat, lon)
        targ: (N, H, lat, lon)

    Returns:
        dict of (lat, lon) arrays: r2, rmse, mae, bias
    """
    N, H, nlat, nlon = pred.shape

    # Flatten time dimensions: (N*H, lat, lon)
    pred_flat = pred.reshape(-1, nlat, nlon)
    targ_flat = targ.reshape(-1, nlat, nlon)

    n_samples = pred_flat.shape[0]

    # Per-cell metrics
    r2_map = np.zeros((nlat, nlon))
    rmse_map = np.zeros((nlat, nlon))
    mae_map = np.zeros((nlat, nlon))
    bias_map = np.zeros((nlat, nlon))

    for i in range(nlat):
        for j in range(nlon):
            p = pred_flat[:, i, j]
            t = targ_flat[:, i, j]

            # R2
            ss_res = np.sum((t - p) ** 2)
            ss_tot = np.sum((t - t.mean()) ** 2)
            r2_map[i, j] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # RMSE
            rmse_map[i, j] = np.sqrt(np.mean((p - t) ** 2))

            # MAE
            mae_map[i, j] = np.mean(np.abs(p - t))

            # Bias
            bias_map[i, j] = np.mean(p - t)

    return {'r2': r2_map, 'rmse': rmse_map, 'mae': mae_map, 'bias': bias_map}


def stratify_by_elevation(metrics_maps, elevation, clusters=ELEVATION_CLUSTERS):
    """Stratify per-cell metrics by elevation clusters.

    Returns DataFrame with one row per cluster per metric.
    """
    rows = []
    for cluster_name, (lo, hi) in clusters.items():
        mask = (elevation >= lo) & (elevation < hi)
        n_cells = mask.sum()

        for metric_name, metric_map in metrics_maps.items():
            values = metric_map[mask]
            rows.append({
                'Cluster': cluster_name,
                'N_cells': int(n_cells),
                'Metric': metric_name.upper(),
                'Mean': float(values.mean()),
                'Std': float(values.std()),
                'Median': float(np.median(values)),
                'Min': float(values.min()),
                'Max': float(values.max()),
                'Q25': float(np.percentile(values, 25)),
                'Q75': float(np.percentile(values, 75)),
            })

    return pd.DataFrame(rows)


def compute_elevation_correlation(r2_map, elevation):
    """Compute Pearson correlation between elevation and per-cell R2.

    A negative correlation indicates altitude bias (model degrades at high elevation).
    """
    elev_flat = elevation.ravel()
    r2_flat = r2_map.ravel()

    # Remove NaN/inf
    valid = np.isfinite(elev_flat) & np.isfinite(r2_flat)
    from scipy.stats import pearsonr, spearmanr

    r_pearson, p_pearson = pearsonr(elev_flat[valid], r2_flat[valid])
    r_spearman, p_spearman = spearmanr(elev_flat[valid], r2_flat[valid])

    return {
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Elevation-Stratified Metrics')
    parser.add_argument('--intracell-dem', action='store_true',
                        help='Use intra-cell DEM predictions (Paper 5)')
    parser.add_argument('--bundle', type=str, default='BASIC_D10',
                        choices=['BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS'],
                        help='Feature bundle for --intracell-dem (default: BASIC_D10)')
    return parser.parse_args()


def main():
    global MODELS, OUTPUT_DIR

    args = parse_args()

    if args.intracell_dem:
        MODELS = get_intracell_dem_models(args.bundle)
        OUTPUT_DIR = PROJECT_ROOT / 'scripts' / 'benchmark' / 'output' / 'intracell_dem' / args.bundle
        logger.info(f"Intracell DEM mode: bundle={args.bundle}")
        logger.info(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elevation = load_elevation_grid()

    all_results = {}
    all_stratified = []

    for model_name in MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {model_name}")
        logger.info(f"{'='*60}")

        pred, targ = load_predictions(model_name)
        metrics_maps = compute_per_cell_metrics(pred, targ)

        # Save spatial maps
        for metric_name, metric_map in metrics_maps.items():
            np.save(OUTPUT_DIR / f'{metric_name}_spatial_{model_name}.npy', metric_map)

        # Stratify by elevation
        strat_df = stratify_by_elevation(metrics_maps, elevation)
        strat_df['Model'] = model_name
        all_stratified.append(strat_df)

        # Elevation-R2 correlation
        corr = compute_elevation_correlation(metrics_maps['r2'], elevation)

        all_results[model_name] = {
            'overall': {
                'R2_mean': float(metrics_maps['r2'].mean()),
                'RMSE_mean': float(metrics_maps['rmse'].mean()),
                'MAE_mean': float(metrics_maps['mae'].mean()),
                'Bias_mean': float(metrics_maps['bias'].mean()),
            },
            'elevation_correlation': corr,
            'per_cluster': strat_df.to_dict(orient='records'),
        }

        # Print summary
        logger.info(f"\n  Overall: R2={metrics_maps['r2'].mean():.4f}, RMSE={metrics_maps['rmse'].mean():.2f}mm")
        logger.info(f"  Elevation-R2 correlation: Pearson r={corr['pearson_r']:.4f} (p={corr['pearson_p']:.4f})")
        for cluster_name, (lo, hi) in ELEVATION_CLUSTERS.items():
            mask = (elevation >= lo) & (elevation < hi)
            r2_vals = metrics_maps['r2'][mask]
            rmse_vals = metrics_maps['rmse'][mask]
            logger.info(f"  {cluster_name} ({mask.sum()} cells): R2={r2_vals.mean():.4f}, RMSE={rmse_vals.mean():.2f}mm")

    # Save combined results
    combined_df = pd.concat(all_stratified, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / 'elevation_stratified_metrics.csv', index=False)
    logger.info(f"\nStratified metrics saved to {OUTPUT_DIR / 'elevation_stratified_metrics.csv'}")

    with open(OUTPUT_DIR / 'elevation_stratified_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Full results saved to {OUTPUT_DIR / 'elevation_stratified_metrics.json'}")

    # Print formatted summary
    print("\n" + "=" * 70)
    print("  ELEVATION-STRATIFIED METRICS SUMMARY")
    print("=" * 70)

    for model_name, data in all_results.items():
        print(f"\n  {model_name}")
        print(f"  {'-'*40}")
        print(f"  Overall R2: {data['overall']['R2_mean']:.4f}")
        corr = data['elevation_correlation']
        sign = "negative (degrades at altitude)" if corr['pearson_r'] < 0 else "positive (improves at altitude)"
        sig = "significant" if corr['pearson_p'] < 0.05 else "not significant"
        print(f"  Elevation-R2 corr: r={corr['pearson_r']:.4f} ({sign}, {sig})")

        r2_rows = [r for r in data['per_cluster'] if r['Metric'] == 'R2']
        for r in r2_rows:
            print(f"    {r['Cluster']}: R2={r['Mean']:.4f} +/- {r['Std']:.4f} ({r['N_cells']} cells)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
