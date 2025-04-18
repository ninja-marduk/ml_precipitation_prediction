import os
import xarray as xr
import pandas as pd
import logging
from datetime import datetime

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../logs")
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists

# Generate log file name with the required format
log_filename = datetime.now().strftime("log-%Y-%m-%d.log")
LOG_FILE = os.path.join(LOG_DIR, log_filename)

# Custom log format
log_format = "%(asctime)s - %(levelname)s - [%(pathname)s-%(funcName)s-%(lineno)d] - %(message)s"

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format=log_format,  # Log message format
    handlers=[
        logging.StreamHandler(),  # Output logs to the terminal
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")  # Save logs to a daily file
    ]
)

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
    try:
        logging.info(f"Listing NetCDF files in directory: {path}")
        files = [file for file in os.listdir(path) if file.endswith('.nc')]
        logging.info(f"Found {len(files)} NetCDF files.")

        # Log each file found
        for file in files:
            logging.info(f"Found file: {file}")

        return files
    except Exception as e:
        logging.error(f"Error listing NetCDF files in directory {path}: {e}")
        raise

def load_and_crop_file(file_path, lon_min, lon_max, lat_min, lat_max):
    """
    Load a NetCDF file, crop it to the specified region, and return the cropped dataset.
    """
    try:
        logging.info(f"Loading and cropping file: {file_path}")
        ds = xr.open_dataset(file_path, chunks={"time": 100})  # Use dask for efficient loading
        ds_cropped = ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
        logging.info(f"File {file_path} cropped successfully.")
        return ds_cropped
    except Exception as e:
        logging.error(f"Error loading or cropping file {file_path}: {e}")
        raise

def save_cropped_file(ds, output_path, filename):
    """
    Save a cropped dataset to a NetCDF file.
    """
    try:
        logging.info(f"Saving cropped dataset to file: {filename}")
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, filename)
        ds.to_netcdf(file_path)
        logging.info(f"Cropped data saved to: {file_path}")
    except Exception as e:
        logging.error(f"Error saving cropped dataset to file {filename}: {e}")
        raise

def process_and_crop_files(path, output_path, lon_min, lon_max, lat_min, lat_max):
    """
    Process all NetCDF files in the directory, crop them to the specified region, and save them.
    """
    try:
        logging.info("Starting processing and cropping of NetCDF files...")
        files = list_nc_files(path)
        for file in files:
            file_path = os.path.join(path, file)
            logging.info(f"Processing file: {file}")
            ds_cropped = load_and_crop_file(file_path, lon_min, lon_max, lat_min, lat_max)
            save_cropped_file(ds_cropped, output_path, f"cropped_{file}")
        logging.info("Processing and cropping completed successfully.")
    except Exception as e:
        logging.error(f"Error during processing and cropping of files: {e}")
        raise

def combine_cropped_files(temp_path):
    """
    Combine all cropped NetCDF files into a single xarray.Dataset.
    """
    try:
        logging.info("Combining cropped NetCDF files...")
        files = list_nc_files(temp_path)
        datasets = [xr.open_dataset(os.path.join(temp_path, file)) for file in files]
        combined_ds = xr.concat(datasets, dim="time")
        logging.info("Cropped files combined successfully.")
        return combined_ds
    except Exception as e:
        logging.error(f"Error combining cropped files: {e}")
        raise

def save_to_netcdf(df, output_path, filename):
    """
    Save a DataFrame to a NetCDF file.
    """
    try:
        logging.info(f"Saving DataFrame to NetCDF file: {filename}")
        ds = df.set_index(['time', 'latitude', 'longitude']).to_xarray()
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, filename)
        ds.to_netcdf(file_path)
        logging.info(f"Data saved to NetCDF file: {file_path}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to NetCDF file {filename}: {e}")
        raise

def main():
    """Main function to process CHIRPS data."""
    try:
        logging.info("Starting CHIRPS daily data processing for Boyacá region...")
        process_and_crop_files(PATH_CHIRPS, TEMP_PATH, LON_BOYACA_MIN, LON_BOYACA_MAX, LAT_BOYACA_MIN, LAT_BOYACA_MAX)
        logging.info("Cropping completed!")

        logging.info("Combining cropped files...")
        combined_ds = combine_cropped_files(TEMP_PATH)
        logging.info("Files combined successfully!")

        logging.info("Saving processed data to NetCDF file...")
        save_to_netcdf(combined_ds.to_dataframe().reset_index(), OUTPUT_PATH, "boyaca_region_daily.nc")
        logging.info("Data saved to NetCDF file successfully!")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main process: {e}")
        raise

if __name__ == "__main__":
    main()
