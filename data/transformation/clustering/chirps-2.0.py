import xarray as xr
import numpy as np
import os

# Ruta del dataset original
INPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_combined_downscaled_with_monthly_moving_avg.nc"

# Rutas de salida para los archivos clusterizados
OUTPUT_LOW = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_low_elevation.nc"
OUTPUT_MEDIUM = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_medium_elevation.nc"
OUTPUT_HIGH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_high_elevation.nc"

def cluster_by_elevation(ds, low_threshold, high_threshold):
    """
    Cluster the dataset into low, medium, and high elevation levels.

    Parameters:
        ds (xarray.Dataset): The input dataset.
        low_threshold (float): The maximum elevation for the low cluster.
        high_threshold (float): The minimum elevation for the high cluster.

    Returns:
        tuple: Three xarray.Datasets for low, medium, and high elevation clusters.
    """
    # Cluster for low elevation
    ds_low = ds.where(ds["elevation"] <= low_threshold, drop=True)

    # Cluster for medium elevation
    ds_medium = ds.where(
        (ds["elevation"] > low_threshold) & (ds["elevation"] <= high_threshold), drop=True
    )

    # Cluster for high elevation
    ds_high = ds.where(ds["elevation"] > high_threshold, drop=True)

    return ds_low, ds_medium, ds_high

def save_clustered_datasets(ds_low, ds_medium, ds_high, output_low, output_medium, output_high):
    """
    Save the clustered datasets to NetCDF files.

    Parameters:
        ds_low (xarray.Dataset): Dataset for low elevation.
        ds_medium (xarray.Dataset): Dataset for medium elevation.
        ds_high (xarray.Dataset): Dataset for high elevation.
        output_low (str): Path to save the low elevation dataset.
        output_medium (str): Path to save the medium elevation dataset.
        output_high (str): Path to save the high elevation dataset.
    """
    print("Saving low elevation dataset...")
    ds_low.to_netcdf(output_low)
    print(f"Low elevation dataset saved to: {output_low}")

    print("Saving medium elevation dataset...")
    ds_medium.to_netcdf(output_medium)
    print(f"Medium elevation dataset saved to: {output_medium}")

    print("Saving high elevation dataset...")
    ds_high.to_netcdf(output_high)
    print(f"High elevation dataset saved to: {output_high}")

def main():
    """
    Main function to perform clustering by elevation and save the results.
    """
    print("Loading dataset...")
    ds = xr.open_dataset(INPUT_PATH)
    print("Dataset loaded successfully!")

    # Define elevation thresholds
    low_threshold = 1500  # Example: Elevation <= 1500m is considered low
    high_threshold = 2500  # Example: Elevation > 2500m is considered high

    print("Clustering dataset by elevation levels...")
    ds_low, ds_medium, ds_high = cluster_by_elevation(ds, low_threshold, high_threshold)
    print("Clustering completed!")

    print("Saving clustered datasets...")
    save_clustered_datasets(ds_low, ds_medium, ds_high, OUTPUT_LOW, OUTPUT_MEDIUM, OUTPUT_HIGH)
    print("All clustered datasets saved successfully!")

if __name__ == "__main__":
    main()
