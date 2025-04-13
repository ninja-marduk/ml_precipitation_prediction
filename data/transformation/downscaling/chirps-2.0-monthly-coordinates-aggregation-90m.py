import xarray as xr
import numpy as np
import logging
import os
import scipy.interpolate as interp1d
from datetime import datetime

# Configuración de logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../../logs")
os.makedirs(LOG_DIR, exist_ok=True)  # Asegurar que el directorio de logs exista

log_filename = datetime.now().strftime("log-%Y-%m-%d.log")
LOG_FILE = os.path.join(LOG_DIR, log_filename)

log_format = "%(asctime)s - %(levelname)s - [%(pathname)s | function: %(funcName)s | line: %(lineno)d] - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # Mostrar logs en la terminal
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")  # Guardar logs en un archivo
    ]
)

logger = logging.getLogger(__name__)

# Rutas de los datasets
AGGREGATED_DATASET_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_monthly_aggregated.nc"
DEM_90M_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/dem_boyaca_90.nc"
OUTPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_monthly_coordinates_aggregation_downscaled_90m.nc"

def load_dataset(file_path):
    """
    Load a NetCDF dataset as an xarray Dataset.

    Parameters:
        file_path (str): Path to the NetCDF file.

    Returns:
        xarray.Dataset: Loaded dataset.
    """
    try:
        logger.info(f"Loading dataset from: {file_path}")
        ds = xr.open_dataset(file_path)
        logger.info("Dataset loaded successfully!")
        return ds
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        raise

def perform_downscaling(aggregated_ds, dem_ds):
    """
    Perform downscaling of the aggregated dataset to match the resolution of the DEM dataset.

    Parameters:
        aggregated_ds (xarray.Dataset): The aggregated dataset to downscale.
        dem_ds (xarray.Dataset): The DEM dataset to use as a reference for resolution.

    Returns:
        xarray.Dataset: Downscaled dataset.
    """
    try:
        logger.info("Starting downscaling process...")

        # Extract the first variable from the aggregated dataset
        first_var = list(aggregated_ds.data_vars.keys())[0]

        # Reproject the aggregated dataset to match the DEM resolution
        downscaled_data = aggregated_ds[first_var].interp(
            longitude=dem_ds["longitude"],
            latitude=dem_ds["latitude"],
            method="linear"
        )

        # Create a new dataset with the downscaled data
        downscaled_ds = downscaled_data.to_dataset(name=f"{first_var}_downscaled")
        downscaled_ds.attrs["description"] = "Downscaled dataset to match DEM-90m resolution"
        logger.info("Downscaling process completed successfully!")
        return downscaled_ds
    except Exception as e:
        logger.error(f"Error during downscaling: {e}")
        raise

def merge_datasets(downscaled_ds, dem_ds):
    """
    Merge the downscaled dataset with the DEM dataset.

    Parameters:
        downscaled_ds (xarray.Dataset): The downscaled dataset.
        dem_ds (xarray.Dataset): The DEM dataset.

    Returns:
        xarray.Dataset: Merged dataset.
    """
    try:
        logger.info("Merging downscaled dataset with DEM dataset...")
        merged_ds = xr.merge([downscaled_ds, dem_ds])
        merged_ds.attrs["description"] = "Merged dataset with downscaled precipitation and DEM data"
        logger.info("Merging process completed successfully!")
        return merged_ds
    except Exception as e:
        logger.error(f"Error during merging: {e}")
        raise

def save_dataset(ds, output_path):
    """
    Save the dataset to a NetCDF file.

    Parameters:
        ds (xarray.Dataset): The dataset to save.
        output_path (str): Path to save the NetCDF file.
    """
    try:
        logger.info(f"Saving dataset to: {output_path}")
        ds.to_netcdf(output_path)
        logger.info(f"Dataset saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving dataset to {output_path}: {e}")
        raise

def main():
    """
    Main function to perform downscaling and merging of datasets.
    """
    try:
        logger.info("Starting the downscaling and merging process...")

        # Load datasets
        aggregated_ds = load_dataset(AGGREGATED_DATASET_PATH)
        dem_ds = load_dataset(DEM_90M_PATH)

        # Perform downscaling
        downscaled_ds = perform_downscaling(aggregated_ds, dem_ds)

        # Merge datasets
        merged_ds = merge_datasets(downscaled_ds, dem_ds)

        # Save the merged dataset
        save_dataset(merged_ds, OUTPUT_PATH)

        logger.info("Downscaling and merging process completed successfully!")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
