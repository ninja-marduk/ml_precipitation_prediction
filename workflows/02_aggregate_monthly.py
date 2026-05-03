"""
Pipeline Stage 02: Aggregate Daily Precipitation to Monthly

Aggregates CHIRPS daily precipitation to monthly totals per grid cell.
Also computes monthly climatology (mean across years).

Source: data/transformation/aggregation/chirps_2.0_daily_to_monthly_coordinates_sum.py
        data/transformation/aggregation/chirps_2.0_monthly_to_months_coordinates_aggregation_avg.py

Usage:
    python workflows/02_aggregate_monthly.py
    python workflows/02_aggregate_monthly.py --config workflows/config.yaml
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import os
import calendar
from pathlib import Path
import logging

import xarray as xr
import pandas as pd
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


def load_dataset(file_path):
    """Load a NetCDF dataset."""
    logger.info(f"Loading dataset: {file_path}")
    ds = xr.open_dataset(file_path)
    logger.info(f"Loaded: {ds.dims}")
    return ds


def aggregate_daily_to_monthly(ds):
    """Aggregate daily precipitation to monthly totals by coordinate.

    Computes: total, max, min precipitation per month per grid cell.
    """
    logger.info("Aggregating daily to monthly...")
    df = ds.to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])

    monthly_stats = df.groupby(
        [df['time'].dt.to_period('M'), 'latitude', 'longitude']
    ).agg(
        total_precipitation=('precip', 'sum'),
        max_daily_precipitation=('precip', 'max'),
        min_daily_precipitation=('precip', 'min')
    ).reset_index()

    monthly_stats['time'] = monthly_stats['time'].dt.to_timestamp()
    monthly_stats['YYYY-MM'] = monthly_stats['time'].dt.strftime('%Y-%m')
    monthly_stats['YYYY'] = monthly_stats['time'].dt.year.astype(str)
    monthly_stats['MM'] = monthly_stats['time'].dt.month.apply(
        lambda x: calendar.month_name[x]
    ).str.capitalize()

    logger.info(f"Monthly aggregation: {len(monthly_stats)} rows")
    return monthly_stats


def aggregate_monthly_to_climatology(ds):
    """Compute climatological mean per month across all years.

    Groups by calendar month and computes mean.
    """
    logger.info("Computing monthly climatology...")
    ds["time"] = pd.to_datetime(ds["time"].values)
    months_avg = ds.groupby("time.month").mean(dim="time")

    if "total_precipitation" in months_avg.data_vars:
        months_avg = months_avg.rename({"total_precipitation": "mean_precipitation"})

    months_avg = months_avg.rename({"month": "month_index"})
    months_avg.attrs["description"] = "Monthly climatology averaged across all years"
    logger.info("Climatology computed.")
    return months_avg


def save_dataframe_as_netcdf(df, output_path, filename):
    """Save DataFrame as NetCDF."""
    ds = df.set_index(['time', 'latitude', 'longitude']).to_xarray()
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, filename)
    ds.to_netcdf(file_path)
    logger.info(f"Saved: {file_path}")


def save_dataset(ds, output_path):
    """Save xarray Dataset to NetCDF."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_netcdf(output_path)
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate daily precipitation to monthly')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--skip-climatology', action='store_true',
                        help='Skip monthly climatology computation')
    args = parser.parse_args()

    log_environment(logger, ['numpy', 'pandas', 'xarray', 'netCDF4', 'h5netcdf', 'pyyaml'])
    log_script_version(logger, __file__)

    cfg = load_config(args.config)
    output_dir = str(PROJECT_ROOT / 'data' / 'output')

    # Step 1: Daily to monthly
    daily_path = str(PROJECT_ROOT / cfg['data']['daily_nc'])
    logger.info(f"Input: {daily_path}")

    ds_daily = load_dataset(daily_path)
    monthly_df = aggregate_daily_to_monthly(ds_daily)
    save_dataframe_as_netcdf(monthly_df, output_dir, "boyaca_region_monthly_coordinates_sum.nc")

    # Step 2: Monthly climatology (optional)
    if not args.skip_climatology:
        monthly_path = str(PROJECT_ROOT / cfg['data']['monthly_nc'])
        ds_monthly = load_dataset(monthly_path)
        climatology = aggregate_monthly_to_climatology(ds_monthly)
        save_dataset(climatology, str(PROJECT_ROOT / cfg['data']['annual_avg_nc']))

    logger.info("Monthly aggregation complete.")


if __name__ == "__main__":
    main()
