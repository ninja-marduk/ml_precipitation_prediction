import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import numpy as np
import os
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
DEM_PATH_90 = "/Users/riperez/Conda/anaconda3/doc/precipitation/qgis_output/dem_boyaca_90.tif"
SHAPEFILE_BOYACA = "/Users/riperez/Conda/anaconda3/doc/precipitation/shapes/MGN_Departamento.shp"
OUTPUT_NETCDF_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/dem_boyaca_90.nc"

def load_dem(dem_path):
    """
    Load a DEM GeoTIFF file and return its data, metadata, and coordinates.

    Parameters:
        dem_path (str): Path to the DEM GeoTIFF file.

    Returns:
        tuple: A tuple containing the DEM data (numpy array), metadata (dict), and coordinates (longitude, latitude).
    """
    try:
        logging.info(f"Loading DEM from: {dem_path}")
        with rasterio.open(dem_path) as dem_dataset:
            dem_data = dem_dataset.read(1)  # Read the first band
            dem_meta = dem_dataset.meta  # Get metadata

            # Extract coordinates
            transform = dem_dataset.transform
            width = dem_dataset.width
            height = dem_dataset.height
            lon = np.arange(transform[2], transform[2] + width * transform[0], transform[0])
            lat = np.arange(transform[5], transform[5] + height * transform[4], transform[4])

        logging.info("DEM loaded successfully.")
        return dem_data, dem_meta, lon, lat
    except Exception as e:
        logging.error(f"Error loading DEM from {dem_path}: {e}")
        raise

def plot_dem_with_boundary(dem_data, dem_meta, shapefile_path, title, output_path=None):
    """
    Plot a DEM with a shapefile boundary overlay.

    Parameters:
        dem_data (numpy array): DEM data.
        dem_meta (dict): Metadata of the DEM.
        shapefile_path (str): Path to the shapefile for boundary overlay.
        title (str): Title of the plot.
        output_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    try:
        logging.info(f"Plotting DEM with boundary from shapefile: {shapefile_path}")
        # Load the shapefile
        gdf_boundary = gpd.read_file(shapefile_path)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        elevation_plot = ax.imshow(
            dem_data,
            cmap='terrain',
            extent=[
                dem_meta['transform'][2],
                dem_meta['transform'][2] + dem_meta['transform'][0] * dem_meta['width'],
                dem_meta['transform'][5] + dem_meta['transform'][4] * dem_meta['height'],
                dem_meta['transform'][5]
            ]
        )
        plt.colorbar(elevation_plot, label='Elevation (meters)', ax=ax)

        # Add the boundary overlay
        gdf_boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)

        # Configure title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)

        # Save or show the plot
        if output_path:
            plt.savefig(output_path, dpi=300)
            logging.info(f"Plot saved to {output_path}")
        else:
            plt.show()
            logging.info("Plot displayed successfully.")
    except Exception as e:
        logging.error(f"Error plotting DEM with boundary: {e}")
        raise

def save_dem_to_netcdf(dem_data, lon, lat, output_path):
    """
    Save DEM data to a NetCDF file.

    Parameters:
        dem_data (numpy array): DEM data.
        lon (numpy array): Longitude values.
        lat (numpy array): Latitude values.
        output_path (str): Path to save the NetCDF file.
    """
    try:
        logging.info(f"Saving DEM data to NetCDF file: {output_path}")
        # Create an xarray DataArray
        dem_da = xr.DataArray(
            dem_data,
            coords={"latitude": lat, "longitude": lon},
            dims=["latitude", "longitude"],
            name="DEM"
        )

        # Add metadata
        dem_da.attrs["units"] = "meters"
        dem_da.attrs["description"] = "Digital Elevation Model (90m resolution)"

        # Save to NetCDF
        dem_da.to_netcdf(output_path)
        logging.info(f"DEM data saved to NetCDF file: {output_path}")
    except Exception as e:
        logging.error(f"Error saving DEM data to NetCDF file {output_path}: {e}")
        raise

def main():
    """
    Main function to load, plot, and export the DEM with the Boyacá boundary.
    """
    try:
        logging.info("Starting DEM processing for Boyacá region...")
        dem_data, dem_meta, lon, lat = load_dem(DEM_PATH_90)
        logging.info(f"DEM Metadata: {dem_meta}")
        logging.info(f"DEM Shape: {dem_data.shape}")

        logging.info("Plotting DEM with Boyacá boundary...")
        plot_dem_with_boundary(
            dem_data,
            dem_meta,
            SHAPEFILE_BOYACA,
            title="Elevation Map of Boyacá (90m Resolution)"
        )
        logging.info("Plotting completed successfully.")

        logging.info("Saving DEM data to NetCDF...")
        save_dem_to_netcdf(dem_data, lon, lat, OUTPUT_NETCDF_PATH)
        logging.info("DEM data saved successfully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main process: {e}")
        raise

if __name__ == "__main__":
    main()
