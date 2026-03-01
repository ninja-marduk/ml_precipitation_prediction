"""
Benchmark Analysis Script 14: Spatiotemporal Metrics (ACC + FSS)

Computes advanced spatiotemporal metrics from existing predictions:
1. ACC (Anomaly Correlation Coefficient) - Temporal structure validation
   Standard: ECMWF/NMME for monthly forecasts
2. FSS (Fractions Skill Score) - Spatial structure validation
   Standard: WeatherBench2. Thresholds: 1mm, 5mm, 10mm

Uses existing predictions.npy (no retraining required).

References:
- ACC: Murphy & Epstein (1989), WMO verification guidelines
- FSS: Roberts & Lean (2008), doi:10.1175/2007MWR2123.1
"""

import argparse
import numpy as np
import pandas as pd
import json
import xarray as xr
from pathlib import Path
from scipy.ndimage import uniform_filter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / 'data' / 'output' / 'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
OUTPUT_DIR = PROJECT_ROOT / 'scripts' / 'benchmark' / 'output'

# Paper 4 default prediction paths
MODELS = {
    'V2_ConvLSTM': {
        'predictions': PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / 'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM' / 'predictions.npy',
        'targets': PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / 'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM' / 'targets.npy',
        'metadata': PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / 'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM' / 'metadata.json',
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


def get_intracell_dem_models(bundle='BASIC_D10'):
    """Build MODELS dict for intracell DEM predictions (Paper 5)."""
    return {
        'V2_ConvLSTM': {
            'predictions': PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models_intracell_dem' / 'map_exports' / 'H12' / bundle / 'ConvLSTM' / 'predictions.npy',
            'targets': PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models_intracell_dem' / 'map_exports' / 'H12' / bundle / 'ConvLSTM' / 'targets.npy',
            'metadata': PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models_intracell_dem' / 'map_exports' / 'H12' / bundle / 'ConvLSTM' / 'metadata.json',
        },
        'V4_GNN_TAT': {
            'predictions': PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models_intracell_dem' / 'map_exports' / 'H12' / bundle / 'GNN_TAT_GAT' / 'predictions.npy',
            'targets': PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models_intracell_dem' / 'map_exports' / 'H12' / bundle / 'ConvLSTM' / 'targets.npy',
        },
        'V10_Late_Fusion': {
            'predictions': PROJECT_ROOT / 'models' / 'output' / 'V10_Late_Fusion_intracell_dem' / bundle / 'predictions.npy',
            'targets': PROJECT_ROOT / 'models' / 'output' / 'V10_Late_Fusion_intracell_dem' / bundle / 'targets.npy',
        },
    }

# FSS configuration
FSS_THRESHOLDS_MM = [1.0, 5.0, 10.0]
FSS_NEIGHBORHOODS = [1, 3, 5, 7]  # 1=pixel, 3=3x3, 5=5x5, 7=7x7


def load_predictions(model_name: str):
    """Load predictions and targets for a model."""
    cfg = MODELS[model_name]
    pred = np.load(cfg['predictions'])
    targ = np.load(cfg['targets'])
    # Remove channel dimension if present: (N, H, lat, lon, 1) -> (N, H, lat, lon)
    if pred.ndim == 5:
        pred = pred[..., 0]
    if targ.ndim == 5:
        targ = targ[..., 0]
    logger.info(f"{model_name}: pred={pred.shape}, targ={targ.shape}")
    return pred, targ


def compute_climatology():
    """Compute monthly climatology from training data.

    Climatology = mean precipitation per grid cell per calendar month,
    computed from the training portion of the dataset (first 80%).
    """
    ds = xr.open_dataset(DATASET_PATH)
    precip = ds['total_precipitation'].values  # (518, 61, 65)
    months = ds['month'].values  # (518, 61, 65) - same value across grid

    # Training set = first 80% of timesteps (indices 0 to ~413)
    n_train = int(0.8 * precip.shape[0])
    train_precip = precip[:n_train]
    train_months = months[:n_train, 0, 0]  # month is same across grid

    # Compute mean for each calendar month (1-12)
    climatology = np.zeros((12, precip.shape[1], precip.shape[2]))
    for m in range(1, 13):
        mask = train_months == m
        if mask.sum() > 0:
            climatology[m - 1] = train_precip[mask].mean(axis=0)

    ds.close()
    logger.info(f"Climatology computed from {n_train} training months, shape={climatology.shape}")
    return climatology


def get_forecast_months(model_name: str):
    """Get calendar months for each forecast sample and horizon.

    Returns array of shape (N_samples, H) with calendar month indices (0-11).
    """
    cfg = MODELS[model_name]
    meta_path = cfg.get('metadata', MODELS['V2_ConvLSTM']['metadata'])
    with open(meta_path) as f:
        meta = json.load(f)

    forecast_dates = meta['forecast_dates']  # list of lists of "YYYY-MM" strings
    month_indices = []
    for sample_dates in forecast_dates:
        sample_months = [int(d.split('-')[1]) - 1 for d in sample_dates]  # 0-indexed
        month_indices.append(sample_months)

    return np.array(month_indices)  # (N_samples, H)


def compute_acc(pred, targ, climatology, forecast_months):
    """Compute Anomaly Correlation Coefficient.

    ACC = sum((F' * O')) / sqrt(sum(F'^2) * sum(O'^2))

    where F' = pred - clim, O' = obs - clim
    Computed per horizon, averaged over samples and grid cells.

    Args:
        pred: (N, H, lat, lon)
        targ: (N, H, lat, lon)
        climatology: (12, lat, lon)
        forecast_months: (N, H) calendar month indices 0-11

    Returns:
        dict with per-horizon ACC values
    """
    N, H, nlat, nlon = pred.shape
    acc_per_horizon = {}

    for h in range(H):
        # Build climatology field for each sample at this horizon
        clim_field = np.zeros((N, nlat, nlon))
        for i in range(N):
            m = forecast_months[i, h]
            clim_field[i] = climatology[m]

        # Anomalies
        pred_anom = pred[:, h] - clim_field  # (N, lat, lon)
        obs_anom = targ[:, h] - clim_field

        # ACC per sample, then average
        acc_samples = []
        for i in range(N):
            pa = pred_anom[i].ravel()
            oa = obs_anom[i].ravel()
            num = np.sum(pa * oa)
            den = np.sqrt(np.sum(pa ** 2) * np.sum(oa ** 2))
            if den > 0:
                acc_samples.append(num / den)
        acc_per_horizon[f'H{h + 1}'] = float(np.mean(acc_samples))

    return acc_per_horizon


def compute_acc_spatial(pred, targ, climatology, forecast_months):
    """Compute ACC per grid cell (spatial map).

    Returns: (lat, lon) array of ACC values averaged over samples and horizons.
    """
    N, H, nlat, nlon = pred.shape
    acc_map = np.zeros((nlat, nlon))

    for i_lat in range(nlat):
        for i_lon in range(nlon):
            numerator = 0.0
            denominator_p = 0.0
            denominator_o = 0.0
            for i in range(N):
                for h in range(H):
                    m = forecast_months[i, h]
                    clim = climatology[m, i_lat, i_lon]
                    pa = pred[i, h, i_lat, i_lon] - clim
                    oa = targ[i, h, i_lat, i_lon] - clim
                    numerator += pa * oa
                    denominator_p += pa ** 2
                    denominator_o += oa ** 2
            den = np.sqrt(denominator_p * denominator_o)
            acc_map[i_lat, i_lon] = numerator / den if den > 0 else 0.0

    return acc_map


def compute_fss(pred, targ, threshold_mm, neighborhood_size):
    """Compute Fractions Skill Score for a given threshold and neighborhood.

    FSS = 1 - MSE_frac / (MSE_frac_ref)
    where fractions are computed using uniform_filter on binary fields.

    Args:
        pred: (N, H, lat, lon)
        targ: (N, H, lat, lon)
        threshold_mm: precipitation threshold in mm
        neighborhood_size: size of neighborhood window (must be odd)

    Returns:
        dict with per-horizon FSS values
    """
    N, H, nlat, nlon = pred.shape
    fss_per_horizon = {}

    for h in range(H):
        fss_samples = []
        for i in range(N):
            # Binary fields: 1 where precip > threshold
            pred_bin = (pred[i, h] > threshold_mm).astype(float)
            obs_bin = (targ[i, h] > threshold_mm).astype(float)

            # Compute fractions using uniform filter
            if neighborhood_size == 1:
                pred_frac = pred_bin
                obs_frac = obs_bin
            else:
                pred_frac = uniform_filter(pred_bin, size=neighborhood_size, mode='constant')
                obs_frac = uniform_filter(obs_bin, size=neighborhood_size, mode='constant')

            # MSE of fractions
            mse_frac = np.mean((pred_frac - obs_frac) ** 2)
            # Reference MSE (worst case: no overlap)
            mse_ref = np.mean(pred_frac ** 2) + np.mean(obs_frac ** 2)

            if mse_ref > 0:
                fss_samples.append(1.0 - mse_frac / mse_ref)
            else:
                fss_samples.append(1.0)  # Both fields are zero

        fss_per_horizon[f'H{h + 1}'] = float(np.mean(fss_samples))

    return fss_per_horizon


def run_acc_analysis():
    """Run ACC analysis for all models."""
    logger.info("=" * 60)
    logger.info("ACC (Anomaly Correlation Coefficient) Analysis")
    logger.info("=" * 60)

    climatology = compute_climatology()
    results = {}

    for model_name in MODELS:
        pred, targ = load_predictions(model_name)
        forecast_months = get_forecast_months(model_name)

        # Per-horizon ACC
        acc = compute_acc(pred, targ, climatology, forecast_months)
        acc_mean = np.mean(list(acc.values()))
        results[model_name] = {'per_horizon': acc, 'mean': float(acc_mean)}

        logger.info(f"{model_name}: ACC mean={acc_mean:.4f}")
        for h, v in acc.items():
            logger.info(f"  {h}: {v:.4f}")

        # Spatial ACC map
        acc_map = compute_acc_spatial(pred, targ, climatology, forecast_months)
        np.save(OUTPUT_DIR / f'acc_spatial_{model_name}.npy', acc_map)
        logger.info(f"  Spatial ACC map saved: mean={acc_map.mean():.4f}, std={acc_map.std():.4f}")

    return results


def run_fss_analysis():
    """Run FSS analysis for all models."""
    logger.info("=" * 60)
    logger.info("FSS (Fractions Skill Score) Analysis")
    logger.info("=" * 60)

    results = {}

    for model_name in MODELS:
        pred, targ = load_predictions(model_name)
        model_results = {}

        for thresh in FSS_THRESHOLDS_MM:
            for neigh in FSS_NEIGHBORHOODS:
                key = f"t{thresh:.0f}mm_n{neigh}"
                fss = compute_fss(pred, targ, thresh, neigh)
                fss_mean = np.mean(list(fss.values()))
                model_results[key] = {'per_horizon': fss, 'mean': float(fss_mean)}
                logger.info(f"{model_name} [{key}]: FSS mean={fss_mean:.4f}")

        results[model_name] = model_results

    return results


def save_results(acc_results, fss_results):
    """Save results to CSV and JSON."""
    # ACC CSV
    acc_rows = []
    for model, data in acc_results.items():
        row = {'Model': model, 'ACC_mean': data['mean']}
        row.update({f'ACC_{h}': v for h, v in data['per_horizon'].items()})
        acc_rows.append(row)
    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(OUTPUT_DIR / 'acc_results.csv', index=False)
    logger.info(f"ACC results saved to {OUTPUT_DIR / 'acc_results.csv'}")

    # FSS CSV
    fss_rows = []
    for model, configs in fss_results.items():
        for config_key, data in configs.items():
            row = {'Model': model, 'Config': config_key, 'FSS_mean': data['mean']}
            row.update({f'FSS_{h}': v for h, v in data['per_horizon'].items()})
            fss_rows.append(row)
    fss_df = pd.DataFrame(fss_rows)
    fss_df.to_csv(OUTPUT_DIR / 'fss_results.csv', index=False)
    logger.info(f"FSS results saved to {OUTPUT_DIR / 'fss_results.csv'}")

    # Combined JSON
    combined = {'acc': acc_results, 'fss': fss_results}
    with open(OUTPUT_DIR / 'spatiotemporal_metrics.json', 'w') as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Combined results saved to {OUTPUT_DIR / 'spatiotemporal_metrics.json'}")

    return acc_df, fss_df


def print_summary(acc_df, fss_df):
    """Print formatted summary table."""
    print("\n" + "=" * 70)
    print("  SPATIOTEMPORAL METRICS SUMMARY")
    print("=" * 70)

    print("\n--- ACC (Anomaly Correlation Coefficient) ---")
    print(f"  Standard: ECMWF/NMME for monthly precipitation forecasts")
    print(f"  Interpretation: >0.6 = useful, >0.8 = good, >0.9 = excellent\n")
    print(acc_df.to_string(index=False, float_format='%.4f'))

    print("\n--- FSS (Fractions Skill Score) ---")
    print(f"  Standard: WeatherBench2, Roberts & Lean (2008)")
    print(f"  Interpretation: >0.5 = useful at this scale\n")

    # Pivot for readability: one row per model, columns per config
    for model in fss_df['Model'].unique():
        model_data = fss_df[fss_df['Model'] == model][['Config', 'FSS_mean']]
        print(f"  {model}:")
        for _, row in model_data.iterrows():
            print(f"    {row['Config']}: {row['FSS_mean']:.4f}")

    print("\n" + "=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description='Spatiotemporal Metrics (ACC + FSS)')
    parser.add_argument('--intracell-dem', action='store_true',
                        help='Use intra-cell DEM predictions (Paper 5)')
    parser.add_argument('--bundle', type=str, default='BASIC_D10',
                        choices=['BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS'],
                        help='Feature bundle for --intracell-dem (default: BASIC_D10)')
    return parser.parse_args()


def main():
    global MODELS, OUTPUT_DIR

    args = parse_args()

    # Switch to intracell DEM paths if requested
    if args.intracell_dem:
        MODELS = get_intracell_dem_models(args.bundle)
        OUTPUT_DIR = PROJECT_ROOT / 'scripts' / 'benchmark' / 'output' / 'intracell_dem' / args.bundle
        logger.info(f"Intracell DEM mode: bundle={args.bundle}")
        logger.info(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    acc_results = run_acc_analysis()
    fss_results = run_fss_analysis()
    acc_df, fss_df = save_results(acc_results, fss_results)
    print_summary(acc_df, fss_df)

    logger.info("Spatiotemporal metrics analysis complete.")


if __name__ == '__main__':
    main()
