"""
Benchmark Analysis Script 17: Variogram Analysis

Computes empirical variograms from model predictions and observations to
evaluate spatial coherence - whether models preserve the spatial autocorrelation
structure of precipitation fields.

Metrics:
1. Empirical variogram of prediction errors (spatial error structure)
2. Comparison of observed vs predicted variograms (spatial fidelity)
3. Fitted model parameters: nugget, sill, range

Uses existing predictions.npy (no retraining required).

References:
- Matheron (1963), Principles of Geostatistics
- Cressie (1993), Statistics for Spatial Data
"""

import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'scripts' / 'benchmark' / 'output'

# Paper 4 default prediction paths (same as scripts 14-16)
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

# Variogram configuration
GRID_RESOLUTION_DEG = 0.05  # CHIRPS resolution in degrees
DEG_TO_KM = 111.0  # approximate km per degree at equator (Boyaca ~5-7N latitude)
MAX_DISTANCE_KM = 200.0  # max lag distance
N_BINS = 30  # number of distance bins
GRID_SHAPE = (61, 65)


def latest_run(base_dir):
    """Return path to most recent run folder under base_dir."""
    base = Path(base_dir)
    if not base.exists():
        return None
    runs = sorted(base.glob('[0-9]*_*'))
    return runs[-1] if runs else None


def get_intracell_dem_models(bundle='BASIC_D10'):
    """Build MODELS dict for intracell DEM predictions (Paper 5)."""
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


def load_predictions(model_name):
    """Load predictions and targets for a model."""
    cfg = MODELS[model_name]
    pred = np.load(cfg['predictions'])
    targ = np.load(cfg['targets'])
    if pred.ndim == 5:
        pred = pred[..., 0]
    if targ.ndim == 5:
        targ = targ[..., 0]
    logger.info(f"{model_name}: pred={pred.shape}, targ={targ.shape}")
    return pred, targ


def build_grid_coordinates():
    """Build (lat, lon) coordinates in km for the 61x65 grid.

    Returns array of shape (n_cells, 2) with (y_km, x_km).
    """
    coords = []
    for i in range(GRID_SHAPE[0]):
        for j in range(GRID_SHAPE[1]):
            y_km = i * GRID_RESOLUTION_DEG * DEG_TO_KM
            x_km = j * GRID_RESOLUTION_DEG * DEG_TO_KM
            coords.append((y_km, x_km))
    return np.array(coords)


def compute_pairwise_distances(coords):
    """Compute pairwise distances between all grid cells (km).

    Uses condensed distance matrix to save memory.
    Returns condensed form (for pdist compatibility).
    """
    logger.info(f"Computing pairwise distances for {len(coords)} cells...")
    distances = pdist(coords, metric='euclidean')
    logger.info(f"Distance pairs: {len(distances)}, range: {distances.min():.1f}-{distances.max():.1f} km")
    return distances


def compute_empirical_variogram(values, distances, max_dist=MAX_DISTANCE_KM, n_bins=N_BINS):
    """Compute empirical semivariogram from spatial data.

    Args:
        values: 1D array of values at each grid cell (n_cells,)
        distances: condensed pairwise distance matrix
        max_dist: maximum lag distance in km
        n_bins: number of distance bins

    Returns:
        bin_centers: array of bin center distances (km)
        semivariance: array of semivariance per bin
        n_pairs: array of number of pairs per bin
    """
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Compute squared differences for all pairs (condensed form)
    n = len(values)
    sq_diffs = pdist(values.reshape(-1, 1), metric='sqeuclidean')

    semivariance = np.zeros(n_bins)
    n_pairs = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (distances >= bin_edges[b]) & (distances < bin_edges[b + 1])
        count = mask.sum()
        if count > 0:
            semivariance[b] = 0.5 * sq_diffs[mask].mean()
            n_pairs[b] = count

    return bin_centers, semivariance, n_pairs


def spherical_model(h, nugget, sill, range_param):
    """Spherical variogram model.

    gamma(h) = nugget + (sill - nugget) * [1.5*(h/range) - 0.5*(h/range)^3]  for h <= range
    gamma(h) = sill                                                             for h > range
    """
    hr = np.minimum(h / range_param, 1.0)
    return nugget + (sill - nugget) * (1.5 * hr - 0.5 * hr ** 3)


def fit_variogram_model(bin_centers, semivariance, n_pairs):
    """Fit spherical variogram model to empirical data.

    Returns dict with nugget, sill, range_km, r_squared.
    """
    # Filter out empty bins
    valid = n_pairs > 10
    if valid.sum() < 3:
        logger.warning("Not enough valid bins for variogram fitting")
        return None

    h = bin_centers[valid]
    gamma = semivariance[valid]

    # Initial guesses
    nugget0 = gamma[0]
    sill0 = gamma.max()
    range0 = h[np.argmax(gamma > 0.95 * sill0)] if np.any(gamma > 0.95 * sill0) else h[-1] / 2

    try:
        popt, _ = curve_fit(
            spherical_model, h, gamma,
            p0=[nugget0, sill0, range0],
            bounds=([0, 0, 1], [sill0 * 2, sill0 * 5, MAX_DISTANCE_KM]),
            maxfev=5000
        )
        nugget, sill, range_km = popt

        # R-squared of fit
        gamma_pred = spherical_model(h, *popt)
        ss_res = np.sum((gamma - gamma_pred) ** 2)
        ss_tot = np.sum((gamma - gamma.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'nugget': float(nugget),
            'sill': float(sill),
            'range_km': float(range_km),
            'r_squared': float(r2),
        }
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Variogram fit failed: {e}")
        return None


def run_variogram_analysis():
    """Run variogram analysis for all models."""
    logger.info("=" * 60)
    logger.info("VARIOGRAM ANALYSIS - Spatial Coherence Evaluation")
    logger.info("=" * 60)

    coords = build_grid_coordinates()
    distances = compute_pairwise_distances(coords)

    results = {}
    variogram_data = {}

    for model_name in MODELS:
        logger.info(f"\n--- {model_name} ---")
        pred, targ = load_predictions(model_name)

        # Flatten spatial dimensions: mean over (windows, horizons) → (lat, lon)
        pred_mean = pred.mean(axis=(0, 1)).ravel()  # (61*65,)
        targ_mean = targ.mean(axis=(0, 1)).ravel()
        error_mean = (pred_mean - targ_mean)  # mean error per cell

        # 1. Error variogram: spatial structure of prediction errors
        h, gamma_err, n_err = compute_empirical_variogram(error_mean, distances)
        fit_err = fit_variogram_model(h, gamma_err, n_err)

        # 2. Prediction variogram: spatial structure of predictions
        h, gamma_pred, n_pred = compute_empirical_variogram(pred_mean, distances)
        fit_pred = fit_variogram_model(h, gamma_pred, n_pred)

        # 3. Observation variogram (same for all models using same targets)
        h, gamma_obs, n_obs = compute_empirical_variogram(targ_mean, distances)
        fit_obs = fit_variogram_model(h, gamma_obs, n_obs)

        results[model_name] = {
            'error_variogram': fit_err,
            'prediction_variogram': fit_pred,
            'observation_variogram': fit_obs,
        }

        variogram_data[model_name] = {
            'bin_centers_km': h.tolist(),
            'error_semivariance': gamma_err.tolist(),
            'prediction_semivariance': gamma_pred.tolist(),
            'observation_semivariance': gamma_obs.tolist(),
            'n_pairs': n_err.tolist(),
        }

        # Log key results
        if fit_err:
            logger.info(f"  Error variogram: nugget={fit_err['nugget']:.1f}, "
                        f"sill={fit_err['sill']:.1f}, range={fit_err['range_km']:.1f}km, "
                        f"R²={fit_err['r_squared']:.3f}")
        if fit_pred and fit_obs:
            logger.info(f"  Prediction range: {fit_pred['range_km']:.1f}km vs "
                        f"Observation range: {fit_obs['range_km']:.1f}km")

    return results, variogram_data


def save_results(results, variogram_data):
    """Save results to CSV and JSON."""
    # Summary CSV
    rows = []
    for model, data in results.items():
        row = {'Model': model}
        for vtype in ['error_variogram', 'prediction_variogram', 'observation_variogram']:
            prefix = vtype.split('_')[0]
            fit = data[vtype]
            if fit:
                row[f'{prefix}_nugget'] = fit['nugget']
                row[f'{prefix}_sill'] = fit['sill']
                row[f'{prefix}_range_km'] = fit['range_km']
                row[f'{prefix}_r2_fit'] = fit['r_squared']
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / 'variogram_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Variogram summary saved to {csv_path}")

    # Full JSON with empirical variogram curves
    json_path = OUTPUT_DIR / 'variogram_data.json'
    with open(json_path, 'w') as f:
        json.dump({
            'results': {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
            'variograms': variogram_data,
            'config': {
                'grid_shape': list(GRID_SHAPE),
                'resolution_deg': GRID_RESOLUTION_DEG,
                'max_distance_km': MAX_DISTANCE_KM,
                'n_bins': N_BINS,
            }
        }, f, indent=2, default=str)
    logger.info(f"Full variogram data saved to {json_path}")

    return df


def print_summary(df):
    """Print summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("VARIOGRAM ANALYSIS SUMMARY")
    logger.info("=" * 80)

    print(f"\n{'Model':<20} {'Err Nugget':>10} {'Err Sill':>10} {'Err Range':>10} "
          f"{'Pred Range':>11} {'Obs Range':>10}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['Model']:<20} "
              f"{row.get('error_nugget', 0):>10.1f} "
              f"{row.get('error_sill', 0):>10.1f} "
              f"{row.get('error_range_km', 0):>9.1f}km "
              f"{row.get('prediction_range_km', 0):>10.1f}km "
              f"{row.get('observation_range_km', 0):>9.1f}km")

    print()
    logger.info("Interpretation:")
    logger.info("  - Error range: distance at which prediction errors become uncorrelated")
    logger.info("  - Pred vs Obs range: how well the model preserves spatial structure")
    logger.info("  - Nugget: micro-scale variability (noise floor)")


def parse_args():
    parser = argparse.ArgumentParser(description='Variogram Analysis (Script 17)')
    parser.add_argument('--intracell-dem', action='store_true',
                        help='Use intracell DEM predictions instead of Paper 4 defaults')
    parser.add_argument('--bundle', type=str, default='BASIC_D10',
                        choices=['BASIC', 'BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS'],
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

    results, variogram_data = run_variogram_analysis()
    df = save_results(results, variogram_data)
    print_summary(df)

    logger.info("Variogram analysis complete.")


if __name__ == '__main__':
    main()
