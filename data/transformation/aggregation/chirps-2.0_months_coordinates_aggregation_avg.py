import pandas as pd  # Agregar esta línea para importar pandas
import xarray as xr
import matplotlib.pyplot as plt
import logging
import os
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

# Ruta del dataset
DATASET_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_monthly_sum.nc"
OUTPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_months_aggregate_avg.nc"
MAPS_OUTPUT_DIR = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/maps/"

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

def aggregate_months_average(ds):
    """
    Aggregate the dataset by calculating the months average across all years.

    Parameters:
        ds (xarray.Dataset): The dataset to aggregate.

    Returns:
        xarray.Dataset: Aggregated dataset with months averages.
    """
    try:
        logger.info("Starting months aggregation...")
        # Ensure the time dimension is in datetime format
        ds["time"] = pd.to_datetime(ds["time"].values)

        # Group by month and calculate the mean
        months_avg = ds.groupby("time.month").mean(dim="time")

        # Rename the variable from total_precipitation to mean_precipitation
        if "total_precipitation" in months_avg.data_vars:
            months_avg = months_avg.rename({"total_precipitation": "mean_precipitation"})

        # Add metadata to the result
        months_avg = months_avg.rename({"month": "month_index"})
        months_avg.attrs["description"] = "Months average aggregated across all years"
        logger.info("Months aggregation completed successfully!")
        return months_avg
    except Exception as e:
        logger.error(f"Error during monthly aggregation: {e}")
        raise

def save_dataset(ds, output_path):
    """
    Save the aggregated dataset to a NetCDF file.

    Parameters:
        ds (xarray.Dataset): The dataset to save.
        output_path (str): Path to save the NetCDF file.
    """
    try:
        logger.info(f"Saving aggregated dataset to: {output_path}")
        ds.to_netcdf(output_path)
        logger.info(f"Aggregated dataset saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving dataset to {output_path}: {e}")
        raise

def generate_months_maps(ds, output_dir):
    """
    Generate a map for each month showing the precipitation values for each coordinate.

    Parameters:
        ds (xarray.Dataset): The aggregated dataset with months averages.
        output_dir (str): Directory to save the maps.
    """
    try:
        logger.info("Generating maps for each month...")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

        # Extract the first variable from the dataset
        first_var = list(ds.data_vars.keys())[0]

        # Loop through each month and generate a map
        for month in range(1, 13):
            logger.info(f"Generating map for month: {month}")
            data = ds[first_var].sel(month_index=month)

            plt.figure(figsize=(10, 6))
            plt.pcolormesh(
                ds["longitude"], ds["latitude"], data,
                cmap="Blues", shading="auto"
            )
            plt.colorbar(label=f"{first_var} (units: {ds[first_var].attrs.get('units', 'AVG [mm/month]')})")
            plt.title(f"Precipitation - Month {month}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.tight_layout()

            # Save the map
            map_path = os.path.join(output_dir, f"map_month_{month:02d}.png")
            plt.savefig(map_path)
            plt.close()
            logger.info(f"Map for month {month} saved to: {map_path}")

        logger.info("All maps generated successfully!")
    except Exception as e:
        logger.error(f"Error generating maps: {e}")
        raise

def generate_combined_months_map(ds, output_path):
    """
    Generate a single image containing subplots for each month's precipitation values.

    Parameters:
        ds (xarray.Dataset): The aggregated dataset with months averages.
        output_path (str): Path to save the combined map image.
    """
    try:
        logger.info("Generating combined map for all months...")

        # Extract the first variable from the dataset
        first_var = list(ds.data_vars.keys())[0]

        # Create a figure with 12 subplots (3 rows x 4 columns)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
        axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

        # Loop through each month and plot in a subplot
        for month in range(1, 13):
            logger.info(f"Adding map for month: {month} to the combined image...")
            data = ds[first_var].sel(month_index=month)

            ax = axes[month - 1]
            im = ax.pcolormesh(
                ds["longitude"], ds["latitude"], data,
                cmap="Blues", shading="auto"
            )
            ax.set_title(f"Month {month}", fontsize=12)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        # Add a colorbar to the figure
        cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
        cbar.set_label(f"{first_var} (units: {ds[first_var].attrs.get('units', 'N/A')})")

        # Save the combined map
        plt.suptitle("Months Precipitation Averages", fontsize=16)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Combined map saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Error generating combined map: {e}")
        raise

def main():
    """
    Main function to load, aggregate, save, and generate maps for the dataset.
    """
    try:
        logger.info("Starting the months aggregation process...")
        logger.info("Loading dataset...")
        ds = load_dataset(DATASET_PATH)

        logger.info("Aggregating dataset by months average...")
        months_avg_ds = aggregate_months_average(ds)

        logger.info("Saving aggregated dataset...")
        save_dataset(months_avg_ds, OUTPUT_PATH)

        logger.info("Generating combined map for all months...")
        combined_map_path = os.path.join(MAPS_OUTPUT_DIR, "combined_months_map.png")
        generate_combined_months_map(months_avg_ds, combined_map_path)

        logger.info("Months aggregation process completed successfully!")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
