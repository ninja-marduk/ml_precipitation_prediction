import xarray as xr
import numpy as np
import os
import logging
from datetime import datetime

# Logging configuration
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../../logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("log-%Y-%m-%d.log")
LOG_FILE = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    ]
)

# Constants for file paths
CHIRPS_PATH = os.path.join(os.path.dirname(__file__), "../../../data/output/boyaca_region_months_aggregate_avg.nc")
DEM_PATH = os.path.join(os.path.dirname(__file__), "../../../data/output/dem_boyaca_90.nc")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../../../data/output/boyaca_region_months_aggregated_avg_merged_dem.nc")

def find_nearest(array, value):
    """
    Find the index of the nearest value in an array.

    Parameters:
        array (numpy.ndarray): Array to search.
        value (float): Value to find the nearest to.

    Returns:
        int: Index of the nearest value.
    """
    idx = (np.abs(array - value)).argmin()
    return idx

def merge_datasets(chirps_path, dem_path, output_path):
    """
    Merge CHIRPS and DEM datasets by aligning DEM elevation data to CHIRPS coordinates.

    Parameters:
        chirps_path (str): Path to the CHIRPS dataset.
        dem_path (str): Path to the DEM dataset.
        output_path (str): Path to save the merged dataset.
    """
    try:
        logging.info("Loading CHIRPS dataset...")
        chirps_ds = xr.open_dataset(chirps_path)

        logging.info("Loading DEM dataset...")
        dem_ds = xr.open_dataset(dem_path)

        # Extract coordinates from CHIRPS
        chirps_lon = chirps_ds["longitude"].values
        chirps_lat = chirps_ds["latitude"].values
        month_index = chirps_ds["month_index"].values

        # Extract DEM data and coordinates
        dem_data = dem_ds["DEM"].values
        dem_lon = dem_ds["longitude"].values
        dem_lat = dem_ds["latitude"].values

        # Create an empty array for elevation data aligned to CHIRPS
        elevation_data = np.empty((len(month_index), len(chirps_lat), len(chirps_lon)))

        logging.info("Aligning DEM data to CHIRPS coordinates...")
        for t, month in enumerate(month_index):
            for i, lat in enumerate(chirps_lat):
                for j, lon in enumerate(chirps_lon):
                    # Find the nearest DEM point for each CHIRPS coordinate
                    nearest_lat_idx = find_nearest(dem_lat, lat)
                    nearest_lon_idx = find_nearest(dem_lon, lon)
                    elevation_data[t, i, j] = dem_data[nearest_lat_idx, nearest_lon_idx]

        # Add the elevation data as a new variable to the CHIRPS dataset
        chirps_ds["DEM"] = (("month_index", "latitude", "longitude"), elevation_data)
        chirps_ds["DEM"].attrs["units"] = "meters"
        chirps_ds["DEM"].attrs["description"] = "Elevation data aligned to CHIRPS coordinates with month_index"

        # Save the merged dataset
        logging.info(f"Saving merged dataset to {output_path}...")
        chirps_ds.to_netcdf(output_path)
        logging.info("Merged dataset saved successfully!")

    except Exception as e:
        logging.error(f"An error occurred while merging datasets: {e}")
        raise

if __name__ == "__main__":
    # Run the merge process
    merge_datasets(CHIRPS_PATH, DEM_PATH, OUTPUT_PATH)
