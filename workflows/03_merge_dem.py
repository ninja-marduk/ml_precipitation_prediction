"""
Pipeline Stage 03: Merge DEM Elevation Data with Precipitation

Aligns DEM (Digital Elevation Model) elevation data to CHIRPS grid coordinates
using nearest-neighbor interpolation, then merges into the precipitation dataset.

Source: data/transformation/merge/chirps-2.0-monthly-with-dem-in-standard-resolution.py

Usage:
    python workflows/03_merge_dem.py
    python workflows/03_merge_dem.py --config workflows/config.yaml
    python workflows/03_merge_dem.py --dem-path /path/to/dem.nc
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import os
from pathlib import Path
import logging

import numpy as np
import xarray as xr
import yaml

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


def find_nearest(array, value):
    """Find index of nearest value in array."""
    return int(np.abs(array - value).argmin())


def merge_datasets(chirps_path, dem_path, output_path):
    """Merge CHIRPS and DEM datasets by aligning DEM to CHIRPS coordinates.

    Uses nearest-neighbor interpolation to map high-resolution DEM
    elevation values to the coarser CHIRPS grid.
    """
    logger.info(f"Loading CHIRPS: {chirps_path}")
    chirps_ds = xr.open_dataset(chirps_path)

    logger.info(f"Loading DEM: {dem_path}")
    dem_ds = xr.open_dataset(dem_path)

    chirps_lon = chirps_ds["longitude"].values
    chirps_lat = chirps_ds["latitude"].values
    month_index = chirps_ds["month_index"].values

    dem_data = dem_ds["DEM"].values
    dem_lon = dem_ds["longitude"].values
    dem_lat = dem_ds["latitude"].values

    logger.info(f"CHIRPS grid: {len(chirps_lat)}x{len(chirps_lon)}, months={len(month_index)}")
    logger.info(f"DEM grid: {dem_data.shape}")

    elevation_data = np.empty((len(month_index), len(chirps_lat), len(chirps_lon)))

    logger.info("Aligning DEM to CHIRPS coordinates (nearest-neighbor)...")
    for t, month in enumerate(month_index):
        for i, lat in enumerate(chirps_lat):
            for j, lon in enumerate(chirps_lon):
                nearest_lat_idx = find_nearest(dem_lat, lat)
                nearest_lon_idx = find_nearest(dem_lon, lon)
                elevation_data[t, i, j] = dem_data[nearest_lat_idx, nearest_lon_idx]

    chirps_ds["DEM"] = (("month_index", "latitude", "longitude"), elevation_data)
    chirps_ds["DEM"].attrs["units"] = "meters"
    chirps_ds["DEM"].attrs["description"] = "Elevation aligned to CHIRPS grid (nearest-neighbor)"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chirps_ds.to_netcdf(output_path)
    logger.info(f"Merged dataset saved: {output_path}")
    logger.info(f"Elevation range: [{elevation_data.min():.0f}, {elevation_data.max():.0f}] m")


def main():
    parser = argparse.ArgumentParser(description='Merge DEM with precipitation data')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--dem-path', type=str, default=None,
                        help='Override path to DEM NetCDF')
    parser.add_argument('--chirps-path', type=str, default=None,
                        help='Override path to CHIRPS climatology NetCDF')
    args = parser.parse_args()

    log_environment(logger, ['numpy', 'pandas', 'xarray', 'netCDF4', 'h5netcdf', 'rioxarray', 'pyyaml'])
    log_script_version(logger, __file__)

    cfg = load_config(args.config)

    chirps_path = args.chirps_path or str(PROJECT_ROOT / cfg['data']['annual_avg_nc'])
    dem_path = args.dem_path or str(PROJECT_ROOT / 'data' / 'output' / 'dem_boyaca_90.nc')
    output_path = str(PROJECT_ROOT / cfg['data']['merged_nc'])

    merge_datasets(chirps_path, dem_path, output_path)
    logger.info("DEM merge complete.")


if __name__ == "__main__":
    main()
