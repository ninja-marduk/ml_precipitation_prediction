"""
Pipeline Stage 04: Feature Engineering

Performs elevation and precipitation clustering on the merged dataset.
Extracts K-means clusters for elevation bands and precipitation patterns,
then creates cross-cluster features for model training.

Source: notebooks/data_clustering.ipynb

Usage:
    python workflows/04_feature_engineering.py
    python workflows/04_feature_engineering.py --config workflows/config.yaml
    python workflows/04_feature_engineering.py --n-elev-clusters 3 --n-precip-clusters 4
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _versions import log_environment, log_script_version


def load_config(config_path=None):
    """Load pipeline configuration."""
    if config_path is None:
        config_path = PROJECT_ROOT / 'workflows' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_datasets(merged_nc_path, monthly_nc_path):
    """Load the merged DEM+precipitation and monthly datasets."""
    logger.info(f"Loading merged dataset: {merged_nc_path}")
    ds_merged = xr.open_dataset(merged_nc_path)

    logger.info(f"Loading monthly dataset: {monthly_nc_path}")
    ds_monthly = xr.open_dataset(monthly_nc_path)

    return ds_merged, ds_monthly


def cluster_elevation(ds_merged, n_clusters=3, seed=42):
    """Perform K-means clustering on elevation data.

    Returns cluster labels as a 2D array (lat, lon).
    """
    logger.info(f"Clustering elevation into {n_clusters} clusters...")

    elevation = ds_merged['DEM'].isel(month_index=0).values
    elev_flat = elevation.ravel().reshape(-1, 1)

    valid_mask = np.isfinite(elev_flat.ravel())
    elev_valid = elev_flat[valid_mask]

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = np.full(elev_flat.shape[0], -1, dtype=int)
    labels[valid_mask] = kmeans.fit_predict(elev_valid)

    labels_2d = labels.reshape(elevation.shape)
    centers = kmeans.cluster_centers_.ravel()
    sil_score = silhouette_score(elev_valid, kmeans.labels_)

    # Sort clusters by elevation (low to high)
    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    labels_2d_sorted = np.vectorize(lambda x: remap.get(x, -1))(labels_2d)

    logger.info(f"Elevation clusters: centers={np.sort(centers).astype(int).tolist()}, "
                f"silhouette={sil_score:.4f}")

    return labels_2d_sorted, np.sort(centers), sil_score


def cluster_precipitation_monthly(ds_monthly, n_clusters=4, seed=42):
    """Cluster grid cells by their 12-month precipitation profile.

    Returns cluster labels as a 2D array (lat, lon).
    """
    logger.info(f"Clustering precipitation patterns into {n_clusters} clusters...")

    precip_var = None
    for var in ['mean_precipitation', 'total_precipitation']:
        if var in ds_monthly.data_vars:
            precip_var = var
            break

    if precip_var is None:
        raise ValueError("No precipitation variable found in monthly dataset")

    precip = ds_monthly[precip_var].values  # (12, lat, lon)
    n_months, n_lat, n_lon = precip.shape

    # Flatten to (n_cells, 12)
    precip_flat = precip.reshape(n_months, -1).T
    valid_mask = np.all(np.isfinite(precip_flat), axis=1)
    precip_valid = precip_flat[valid_mask]

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = np.full(precip_flat.shape[0], -1, dtype=int)
    labels[valid_mask] = kmeans.fit_predict(precip_valid)

    labels_2d = labels.reshape(n_lat, n_lon)
    sil_score = silhouette_score(precip_valid, kmeans.labels_)

    logger.info(f"Precipitation clusters: {n_clusters} clusters, silhouette={sil_score:.4f}")

    return labels_2d, kmeans.cluster_centers_, sil_score


def cross_cluster_analysis(elev_labels, precip_labels):
    """Compute cross-tabulation and chi-square test between clusters."""
    elev_flat = elev_labels.ravel()
    precip_flat = precip_labels.ravel()

    valid = (elev_flat >= 0) & (precip_flat >= 0)
    contingency = pd.crosstab(
        pd.Series(elev_flat[valid], name='Elevation'),
        pd.Series(precip_flat[valid], name='Precipitation')
    )

    chi2, p_value, dof, expected = chi2_contingency(contingency)
    logger.info(f"Chi-square test: chi2={chi2:.2f}, p={p_value:.6f}, dof={dof}")

    return contingency, chi2, p_value


def save_clustering_results(elev_labels, precip_labels, output_dir):
    """Save cluster label arrays as .npy files."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'elevation_cluster_labels.npy'), elev_labels)
    np.save(os.path.join(output_dir, 'precipitation_cluster_labels.npy'), precip_labels)

    # Cross-cluster encoding
    cross_labels = elev_labels * 10 + precip_labels
    np.save(os.path.join(output_dir, 'cross_cluster_labels.npy'), cross_labels)

    logger.info(f"Cluster labels saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Feature engineering: clustering')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--n-elev-clusters', type=int, default=None,
                        help='Number of elevation clusters')
    parser.add_argument('--n-precip-clusters', type=int, default=None,
                        help='Number of precipitation clusters')
    args = parser.parse_args()

    log_environment(logger, ['numpy', 'pandas', 'xarray', 'netCDF4', 'h5netcdf', 'sklearn', 'scipy', 'pyyaml'])
    log_script_version(logger, __file__)

    cfg = load_config(args.config)
    features_cfg = cfg.get('features', {})
    env_cfg = cfg.get('environment', {})

    n_elev = args.n_elev_clusters or features_cfg.get('elevation_clusters', {}).get('n_clusters', 3)
    n_precip = args.n_precip_clusters or features_cfg.get('precipitation_clusters', {}).get('n_clusters', 4)
    seed = env_cfg.get('random_seed', 42)

    merged_path = str(PROJECT_ROOT / cfg['data']['merged_nc'])
    monthly_path = str(PROJECT_ROOT / cfg['data']['annual_avg_nc'])
    output_dir = str(PROJECT_ROOT / 'output' / 'clustering')

    ds_merged, ds_monthly = load_datasets(merged_path, monthly_path)

    # Elevation clustering
    elev_labels, elev_centers, elev_sil = cluster_elevation(ds_merged, n_elev, seed)

    # Precipitation clustering
    precip_labels, precip_centers, precip_sil = cluster_precipitation_monthly(
        ds_monthly, n_precip, seed)

    # Cross-cluster analysis
    contingency, chi2, p_value = cross_cluster_analysis(elev_labels, precip_labels)
    print(f"\nContingency table:\n{contingency}")
    print(f"\nChi-square: {chi2:.2f}, p-value: {p_value:.6f}")

    # Save results
    save_clustering_results(elev_labels, precip_labels, output_dir)

    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    main()
