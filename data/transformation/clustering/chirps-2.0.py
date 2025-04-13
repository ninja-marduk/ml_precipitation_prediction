import xarray as xr
import os
import logging
from datetime import datetime

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../../logs/")
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
    try:
        logging.info("Clustering dataset by elevation levels...")
        # Cluster for low elevation
        ds_low = ds.where(ds["elevation"] <= low_threshold, drop=True)
        logging.info(f"Low elevation cluster created with threshold <= {low_threshold}m.")

        # Cluster for medium elevation
        ds_medium = ds.where(
            (ds["elevation"] > low_threshold) & (ds["elevation"] <= high_threshold), drop=True
        )
        logging.info(f"Medium elevation cluster created with thresholds > {low_threshold}m and <= {high_threshold}m.")

        # Cluster for high elevation
        ds_high = ds.where(ds["elevation"] > high_threshold, drop=True)
        logging.info(f"High elevation cluster created with threshold > {high_threshold}m.")

        return ds_low, ds_medium, ds_high
    except Exception as e:
        logging.error(f"Error clustering dataset by elevation: {e}")
        raise

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
    try:
        logging.info("Saving clustered datasets to NetCDF files...")

        logging.info(f"Saving low elevation dataset to {output_low}...")
        ds_low.to_netcdf(output_low)
        logging.info("Low elevation dataset saved successfully.")

        logging.info(f"Saving medium elevation dataset to {output_medium}...")
        ds_medium.to_netcdf(output_medium)
        logging.info("Medium elevation dataset saved successfully.")

        logging.info(f"Saving high elevation dataset to {output_high}...")
        ds_high.to_netcdf(output_high)
        logging.info("High elevation dataset saved successfully.")
    except Exception as e:
        logging.error(f"Error saving clustered datasets: {e}")
        raise

def main():
    """
    Main function to perform clustering by elevation and save the results.
    """
    try:
        logging.info("Starting clustering process for CHIRPS dataset...")
        logging.info(f"Loading dataset from {INPUT_PATH}...")
        ds = xr.open_dataset(INPUT_PATH)
        logging.info("Dataset loaded successfully!")

        # Define elevation thresholds
        low_threshold = 1500  # Example: Elevation <= 1500m is considered low
        high_threshold = 2500  # Example: Elevation > 2500m is considered high

        logging.info("Clustering dataset by elevation levels...")
        ds_low, ds_medium, ds_high = cluster_by_elevation(ds, low_threshold, high_threshold)
        logging.info("Clustering completed successfully!")

        logging.info("Saving clustered datasets...")
        save_clustered_datasets(ds_low, ds_medium, ds_high, OUTPUT_LOW, OUTPUT_MEDIUM, OUTPUT_HIGH)
        logging.info("All clustered datasets saved successfully!")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main process: {e}")
        raise

if __name__ == "__main__":
    main()
