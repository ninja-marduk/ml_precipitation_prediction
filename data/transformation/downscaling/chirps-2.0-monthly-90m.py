import xarray as xr
import numpy as np
from rasterio.warp import reproject, Resampling
import os

# Constants
CHIRPS_MONTHLY_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_monthly.nc"
DEM_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/qgis_output/dem_boyaca_90.nc"
OUTPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_combined_downscaled.nc"

def load_nc_file(file_path):
    """
    Load a NetCDF file as an xarray Dataset.

    Parameters:
        file_path (str): Path to the NetCDF file.

    Returns:
        xarray.Dataset: Loaded dataset.
    """
    return xr.open_dataset(file_path)


def downscale_variable_to_dem(variable, chirps_ds, dem_ds, dem_transform, dem_crs, dem_width, dem_height):
    """
    Downscale a specific CHIRPS variable to match the resolution of the DEM.

    Parameters:
        variable (str): Name of the variable to downscale.
        chirps_ds (xarray.Dataset): CHIRPS dataset.
        dem_ds (xarray.Dataset): DEM dataset.
        dem_transform (Affine): Transform of the DEM.
        dem_crs (CRS): CRS of the DEM.
        dem_width (int): Width of the DEM.
        dem_height (int): Height of the DEM.

    Returns:
        np.ndarray: Downscaled data for the variable.
    """
    downscaled_data = np.empty((chirps_ds.time.size, dem_height, dem_width), dtype=np.float32)

    for i, time_step in enumerate(chirps_ds.time):
        reproject(
            source=chirps_ds[variable].isel(time=i).values,
            destination=downscaled_data[i],
            src_transform=chirps_ds.rio.transform(),
            src_crs=chirps_ds.rio.crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )
    return downscaled_data


def downscale_chirps_to_dem(chirps_ds, dem_ds):
    """
    Downscale CHIRPS data (mean, max, min precipitation) to match the resolution of the DEM.
    """
    # Ensure CHIRPS dataset has a CRS
    if not chirps_ds.rio.crs:
        chirps_ds = chirps_ds.rio.write_crs("EPSG:4326")

    # Ensure DEM dataset has a CRS
    if not dem_ds.rio.crs:
        dem_ds = dem_ds.rio.write_crs("EPSG:4326")

    # Extract DEM metadata
    dem_transform = dem_ds.rio.transform()
    dem_crs = dem_ds.rio.crs
    dem_width = dem_ds.sizes["longitude"]
    dem_height = dem_ds.sizes["latitude"]

    # Downscale mean, max, and min precipitation
    downscaled_mean = downscale_variable_to_dem("mean_precipitation", chirps_ds, dem_ds, dem_transform, dem_crs, dem_width, dem_height)
    downscaled_max = downscale_variable_to_dem("max_precipitation", chirps_ds, dem_ds, dem_transform, dem_crs, dem_width, dem_height)
    downscaled_min = downscale_variable_to_dem("min_precipitation", chirps_ds, dem_ds, dem_transform, dem_crs, dem_width, dem_height)

    # Create DataArrays for each variable
    downscaled_mean_da = xr.DataArray(
        downscaled_mean,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": chirps_ds.time,
            "latitude": dem_ds.latitude,
            "longitude": dem_ds.longitude
        },
        attrs={"units": "mm/month", "description": "Mean precipitation downscaled to DEM resolution"}
    )

    downscaled_max_da = xr.DataArray(
        downscaled_max,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": chirps_ds.time,
            "latitude": dem_ds.latitude,
            "longitude": dem_ds.longitude
        },
        attrs={"units": "mm/month", "description": "Max precipitation downscaled to DEM resolution"}
    )

    downscaled_min_da = xr.DataArray(
        downscaled_min,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": chirps_ds.time,
            "latitude": dem_ds.latitude,
            "longitude": dem_ds.longitude
        },
        attrs={"units": "mm/month", "description": "Min precipitation downscaled to DEM resolution"}
    )

    return downscaled_mean_da, downscaled_max_da, downscaled_min_da


def save_combined_dataset(downscaled_mean, downscaled_max, downscaled_min, dem_ds, output_path):
    """
    Save the combined dataset (downscaled CHIRPS and DEM) to a NetCDF file.

    Parameters:
        downscaled_mean (xarray.DataArray): Downscaled mean precipitation.
        downscaled_max (xarray.DataArray): Downscaled max precipitation.
        downscaled_min (xarray.DataArray): Downscaled min precipitation.
        dem_ds (xarray.Dataset): DEM dataset.
        output_path (str): Path to save the NetCDF file.
    """
    # Combine the datasets
    combined_ds = xr.Dataset(
        {
            "downscaled_mean_precipitation": downscaled_mean,
            "downscaled_max_precipitation": downscaled_max,
            "downscaled_min_precipitation": downscaled_min,
            "elevation": dem_ds["DEM"]
        },
        coords={
            "latitude": dem_ds["latitude"],
            "longitude": dem_ds["longitude"],
            "time": downscaled_mean["time"]
        },
        attrs={
            "description": "Combined dataset with downscaled precipitation and elevation",
            "source": "CHIRPS and DEM",
            "author": "Your Name"
        }
    )

    # Save to NetCDF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_ds.to_netcdf(output_path)
    print(f"Combined dataset saved to: {output_path}")


def main():
    """
    Main function to perform downscaling and save the combined dataset.
    """
    print("Loading CHIRPS and DEM datasets...")
    chirps_ds = load_nc_file(CHIRPS_MONTHLY_PATH)
    dem_ds = load_nc_file(DEM_PATH)
    print("Datasets loaded successfully!")

    print("Performing downscaling...")
    downscaled_mean, downscaled_max, downscaled_min = downscale_chirps_to_dem(chirps_ds, dem_ds)
    print("Downscaling completed!")

    print("Saving combined dataset...")
    save_combined_dataset(downscaled_mean, downscaled_max, downscaled_min, dem_ds, OUTPUT_PATH)
    print("Process completed!")


if __name__ == "__main__":
    main()
