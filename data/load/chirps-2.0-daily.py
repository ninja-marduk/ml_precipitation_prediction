import os
import xarray as xr
import pandas as pd

# Constants
PATH_CHIRPS = '/Users/riperez/Conda/anaconda3/doc/precipitation/CHIRPS-2.0/daily/'
OUTPUT_PATH = '/Users/riperez/Conda/anaconda3/doc/precipitation/output/'
TEMP_PATH = os.path.join(OUTPUT_PATH, "temp/")
LON_BOYACA_MIN = -74.8
LON_BOYACA_MAX = -71.9
LAT_BOYACA_MIN = 4.5
LAT_BOYACA_MAX = 7.3

def list_nc_files(path):
    """List all NetCDF files in the given directory."""
    return [file for file in os.listdir(path) if file.endswith('.nc')]

def load_and_crop_file(file_path, lon_min, lon_max, lat_min, lat_max):
    """
    Load a NetCDF file, crop it to the specified region, and return the cropped dataset.

    Parameters:
        file_path (str): Path to the NetCDF file.
        lon_min, lon_max, lat_min, lat_max (float): Bounding box for cropping.

    Returns:
        xarray.Dataset: Cropped dataset.
    """
    ds = xr.open_dataset(file_path, chunks={"time": 100})  # Use dask for efficient loading
    ds_cropped = ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
    return ds_cropped

def save_cropped_file(ds, output_path, filename):
    """
    Save a cropped dataset to a NetCDF file.

    Parameters:
        ds (xarray.Dataset): Cropped dataset.
        output_path (str): Directory to save the NetCDF file.
        filename (str): Name of the NetCDF file.
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, filename)
    ds.to_netcdf(file_path)
    print(f"Cropped data saved to: {file_path}")

def process_and_crop_files(path, output_path, lon_min, lon_max, lat_min, lat_max):
    """
    Process all NetCDF files in the directory, crop them to the specified region, and save them.

    Parameters:
        path (str): Directory containing the NetCDF files.
        output_path (str): Directory to save the cropped files.
        lon_min, lon_max, lat_min, lat_max (float): Bounding box for cropping.
    """
    files = list_nc_files(path)
    for file in files:
        file_path = os.path.join(path, file)
        print(f"Processing file: {file}")
        ds_cropped = load_and_crop_file(file_path, lon_min, lon_max, lat_min, lat_max)
        save_cropped_file(ds_cropped, output_path, f"cropped_{file}")

def combine_cropped_files(temp_path):
    """
    Combine all cropped NetCDF files into a single xarray.Dataset.

    Parameters:
        temp_path (str): Directory containing the cropped NetCDF files.

    Returns:
        xarray.Dataset: Combined dataset.
    """
    files = list_nc_files(temp_path)
    datasets = [xr.open_dataset(os.path.join(temp_path, file)) for file in files]
    combined_ds = xr.concat(datasets, dim="time")
    return combined_ds

def aggregate_monthly_coordinates_precipitation(ds):
    """
    Aggregate precipitation data by month and coordinates, calculating mean, max, and min.

    Parameters:
        ds (xarray.Dataset): Dataset containing precipitation data.

    Returns:
        pd.DataFrame: Aggregated DataFrame with monthly and coordinate-based statistics.
    """
    df = ds.to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    monthly_coords_stats = df.groupby(
        [df['time'].dt.to_period('M'), 'latitude', 'longitude']
    ).agg(
        mean_precipitation=('precip', 'mean'),
        max_precipitation=('precip', 'max'),
        min_precipitation=('precip', 'min')
    ).reset_index()
    monthly_coords_stats['time'] = monthly_coords_stats['time'].dt.to_timestamp()
    return monthly_coords_stats

def save_to_netcdf(df, output_path, filename):
    """
    Save a DataFrame to a NetCDF file.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Directory to save the NetCDF file.
        filename (str): Name of the NetCDF file.
    """
    ds = df.set_index(['time', 'latitude', 'longitude']).to_xarray()
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, filename)
    ds.to_netcdf(file_path)
    print(f"Data saved to NetCDF file: {file_path}")

def main():
    """Main function to process CHIRPS data."""
    print("Processing and cropping CHIRPS daily data for Boyacá region...")
    process_and_crop_files(PATH_CHIRPS, TEMP_PATH, LON_BOYACA_MIN, LON_BOYACA_MAX, LAT_BOYACA_MIN, LAT_BOYACA_MAX)
    print("Cropping completed!")

    print("Combining cropped files...")
    combined_ds = combine_cropped_files(TEMP_PATH)
    print("Files combined successfully!")

    print("Aggregating Boyacá data by month and coordinates...")
    monthly_boyaca = aggregate_monthly_coordinates_precipitation(combined_ds)
    print("Boyacá monthly and coordinates aggregation completed!")

    print("Saving processed data to NetCDF files...")
    save_to_netcdf(combined_ds.to_dataframe().reset_index(), OUTPUT_PATH, "boyaca_region_daily.nc")
    save_to_netcdf(monthly_boyaca, OUTPUT_PATH, "boyaca_region_monthly.nc")
    print("Data saved to NetCDF files.")

if __name__ == "__main__":
    main()
