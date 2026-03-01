"""
Pipeline Stage 01: Download and Crop CHIRPS Daily Precipitation Data

Downloads CHIRPS 2.0 daily NetCDF files and crops them to the region of interest
(Boyaca, Colombia). Combines all cropped files into a single dataset.

Source: data/load/chirps-2.0_daily.py

Usage:
    python workflows/01_download_chirps.py
    python workflows/01_download_chirps.py --config workflows/config.yaml
    python workflows/01_download_chirps.py --chirps-dir /path/to/chirps/data
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import os
from pathlib import Path
import logging

import xarray as xr
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    """Load pipeline configuration."""
    if config_path is None:
        config_path = PROJECT_ROOT / 'workflows' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_nc_files(path):
    """List all NetCDF files in the given directory."""
    logger.info(f"Listing NetCDF files in: {path}")
    files = sorted([f for f in os.listdir(path) if f.endswith('.nc')])
    logger.info(f"Found {len(files)} NetCDF files.")
    return files


def load_and_crop_file(file_path, lon_min, lon_max, lat_min, lat_max):
    """Load a NetCDF file, crop to the region of interest."""
    logger.info(f"Loading and cropping: {file_path}")
    ds = xr.open_dataset(file_path, chunks={"time": 100})
    ds_cropped = ds.sel(
        longitude=slice(lon_min, lon_max),
        latitude=slice(lat_min, lat_max)
    )
    return ds_cropped


def save_cropped_file(ds, output_path, filename):
    """Save a cropped dataset to NetCDF."""
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, filename)
    ds.to_netcdf(file_path)
    logger.info(f"Saved: {file_path}")


def process_and_crop_files(chirps_dir, temp_dir, lon_min, lon_max, lat_min, lat_max):
    """Process all CHIRPS NetCDF files: crop to region and save to temp."""
    files = list_nc_files(chirps_dir)
    for f in files:
        file_path = os.path.join(chirps_dir, f)
        ds_cropped = load_and_crop_file(file_path, lon_min, lon_max, lat_min, lat_max)
        save_cropped_file(ds_cropped, temp_dir, f"cropped_{f}")


def combine_cropped_files(temp_dir):
    """Combine all cropped NetCDF files into a single dataset."""
    logger.info("Combining cropped files...")
    files = list_nc_files(temp_dir)
    datasets = [xr.open_dataset(os.path.join(temp_dir, f)) for f in files]
    combined = xr.concat(datasets, dim="time")
    logger.info(f"Combined dataset: {combined.dims}")
    return combined


def save_to_netcdf(df, output_path, filename):
    """Save DataFrame as NetCDF."""
    ds = df.set_index(['time', 'latitude', 'longitude']).to_xarray()
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, filename)
    ds.to_netcdf(file_path)
    logger.info(f"Saved: {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Download and crop CHIRPS data')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--chirps-dir', type=str, default=None,
                        help='Directory containing CHIRPS NetCDF files')
    args = parser.parse_args()

    cfg = load_config(args.config)
    region = cfg['region']

    chirps_dir = args.chirps_dir or str(PROJECT_ROOT / cfg['data']['chirps_raw_dir'])
    output_dir = str(PROJECT_ROOT / 'data' / 'output')
    temp_dir = str(PROJECT_ROOT / cfg['data']['temp_dir'])

    lon_min = region['lon_min']
    lon_max = region['lon_max']
    lat_min = region['lat_min']
    lat_max = region['lat_max']

    logger.info(f"CHIRPS source: {chirps_dir}")
    logger.info(f"Region: lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]")

    # Step 1: Crop all files
    process_and_crop_files(chirps_dir, temp_dir, lon_min, lon_max, lat_min, lat_max)

    # Step 2: Combine
    combined_ds = combine_cropped_files(temp_dir)

    # Step 3: Save
    save_to_netcdf(
        combined_ds.to_dataframe().reset_index(),
        output_dir,
        "boyaca_region_daily.nc"
    )
    logger.info("CHIRPS daily data processing complete.")


if __name__ == "__main__":
    main()
