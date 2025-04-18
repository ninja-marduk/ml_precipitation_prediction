import xarray as xr
import pandas as pd
import numpy as np
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
DATASET_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_months_aggregated_avg_merged_dem.nc"
OUTPUT_DIR = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/correlation/"

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

def calculate_correlations(ds):
    """
    Calculate the correlation between precipitation and elevation for each month.

    Parameters:
        ds (xarray.Dataset): The dataset containing precipitation and elevation data.

    Returns:
        pd.DataFrame: A DataFrame containing the correlation coefficients for each month.
    """
    try:
        logger.info("Calculating correlations between precipitation and elevation...")

        # Extract variables
        elevation = ds["DEM"].values.flatten()  # Cambiado de 'elevation' a 'DEM'
        correlations = {}

        # Loop through each month and calculate correlation
        for month in range(1, 13):
            precipitation = ds["mean_precipitation"].sel(month_index=month).values.flatten()

            # Remove NaN values for correlation calculation
            valid_mask = ~np.isnan(precipitation) & ~np.isnan(elevation)
            if valid_mask.sum() > 0:
                corr = np.corrcoef(precipitation[valid_mask], elevation[valid_mask])[0, 1]
            else:
                corr = np.nan

            correlations[month] = corr

        logger.info("Correlation calculation completed successfully!")
        return pd.DataFrame.from_dict(correlations, orient="index", columns=["Correlation"]).rename_axis("Month")
    except Exception as e:
        logger.error(f"Error calculating correlations: {e}")
        raise

def plot_correlations(correlations, output_path):
    """
    Plot the correlation coefficients for each month.

    Parameters:
        correlations (pd.DataFrame): DataFrame containing the correlation coefficients.
        output_path (str): Path to save the plot.
    """
    try:
        logger.info("Generating correlation plot...")

        plt.figure(figsize=(10, 6))
        plt.bar(correlations.index, correlations["Correlation"], color="skyblue", edgecolor="black")
        plt.axhline(0, color="red", linestyle="--", linewidth=1)
        plt.title("Correlation Between Precipitation and Elevation by Month")
        plt.xlabel("Month")
        plt.ylabel("Correlation Coefficient")
        plt.xticks(range(1, 13), [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ])
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Correlation plot saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Error generating correlation plot: {e}")
        raise

def calculate_correlation_by_elevation_levels(ds, low_threshold, high_threshold):
    """
    Calculate the correlation between elevation and monthly aggregated precipitation
    for three elevation levels: low, medium, and high.

    Parameters:
        ds (xarray.Dataset): The dataset containing precipitation and elevation data.
        low_threshold (float): The maximum elevation for the low level.
        high_threshold (float): The minimum elevation for the high level.

    Returns:
        pd.DataFrame: A DataFrame containing correlation coefficients for each elevation level.
    """
    try:
        logger.info("Calculating correlation between precipitation and elevation by elevation levels...")

        # Extract elevation and precipitation variables
        correlations = {"low": [], "medium": [], "high": []}

        # Loop through each month and calculate correlation for each elevation level
        for month in range(1, 13):
            # Select the current month for both precipitation and elevation
            precipitation = ds["mean_precipitation"].sel(month_index=month).values  # Precipitation as 2D array
            elevation = ds["DEM"].sel(month_index=month).values  # Elevation as 2D array

            # Ensure elevation and precipitation have the same shape
            if elevation.shape != precipitation.shape:
                raise ValueError(f"Shape mismatch: elevation {elevation.shape}, precipitation {precipitation.shape}")

            # Low elevation
            low_mask = elevation <= low_threshold
            low_precipitation = precipitation[low_mask]
            low_elevation = elevation[low_mask]
            valid_low = ~np.isnan(low_precipitation) & ~np.isnan(low_elevation)
            low_corr = np.corrcoef(low_precipitation[valid_low], low_elevation[valid_low])[0, 1] if valid_low.sum() > 0 else np.nan
            correlations["low"].append(low_corr)

            # Medium elevation
            medium_mask = (elevation > low_threshold) & (elevation <= high_threshold)
            medium_precipitation = precipitation[medium_mask]
            medium_elevation = elevation[medium_mask]
            valid_medium = ~np.isnan(medium_precipitation) & ~np.isnan(medium_elevation)
            medium_corr = np.corrcoef(medium_precipitation[valid_medium], medium_elevation[valid_medium])[0, 1] if valid_medium.sum() > 0 else np.nan
            correlations["medium"].append(medium_corr)

            # High elevation
            high_mask = elevation > high_threshold
            high_precipitation = precipitation[high_mask]
            high_elevation = elevation[high_mask]
            valid_high = ~np.isnan(high_precipitation) & ~np.isnan(high_elevation)
            high_corr = np.corrcoef(high_precipitation[valid_high], high_elevation[valid_high])[0, 1] if valid_high.sum() > 0 else np.nan
            correlations["high"].append(high_corr)

        logger.info("Correlation calculation by elevation levels completed successfully!")
        return pd.DataFrame(correlations, index=range(1, 13)).rename_axis("Month")
    except Exception as e:
        logger.error(f"Error calculating correlation by elevation levels: {e}")
        raise

def plot_correlation_by_levels(correlations, output_path):
    """
    Plot the correlation coefficients for each elevation level by month.

    Parameters:
        correlations (pd.DataFrame): DataFrame containing the correlation coefficients.
        output_path (str): Path to save the plot.
    """
    try:
        logger.info("Generating correlation plot by elevation levels...")

        plt.figure(figsize=(12, 8))
        for level in correlations.columns:
            plt.plot(correlations.index, correlations[level], marker="o", label=f"{level.capitalize()} Elevation")

        plt.axhline(0, color="red", linestyle="--", linewidth=1)
        plt.title("Correlation Between Precipitation and Elevation by Elevation Levels")
        plt.xlabel("Month")
        plt.ylabel("Correlation Coefficient")
        plt.xticks(range(1, 13), [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ])
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Correlation plot by elevation levels saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Error generating correlation plot by elevation levels: {e}")
        raise

def plot_all_correlations(correlations, output_dir):
    """
    Generate 3 plots with subplots for each month, showing low, medium, and high elevation correlations.

    Parameters:
        correlations (pd.DataFrame): DataFrame containing the correlation coefficients for each elevation level.
        output_dir (str): Directory to save the plots.
    """
    try:
        logger.info("Generating plots for all correlations by elevation levels...")

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define correlation types
        correlation_types = ["low", "medium", "high"]

        # Loop through each correlation type and generate a plot
        for corr_type in correlation_types:
            logger.info(f"Generating plot for {corr_type} correlations...")

            # Create a figure with 12 subplots (3 rows x 4 columns)
            fig, axes = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
            axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

            # Loop through each month and plot in a subplot
            for month in range(1, 13):
                ax = axes[month - 1]
                data = correlations[corr_type].loc[month]
                title = f"{corr_type.capitalize()} Correlation - Month {month}"

                # Plot the data
                ax.bar([1], [data], color="skyblue", edgecolor="black")
                ax.set_title(title, fontsize=10)
                ax.set_ylim(-1, 1)  # Correlation coefficients range from -1 to 1
                ax.set_xticks([])
                ax.set_ylabel("Correlation")

            # Save the plot
            plot_path = os.path.join(output_dir, f"{corr_type}_correlations.png")
            plt.suptitle(f"{corr_type.capitalize()} Correlations by Month", fontsize=16)
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Plot for {corr_type} correlations saved to: {plot_path}")

        logger.info("All correlation plots generated successfully!")
    except Exception as e:
        logger.error(f"Error generating plots for correlations: {e}")
        raise

def calculate_correlations_by_month(ds):
    """
    Calculate the correlation between precipitation and elevation for each month.

    Parameters:
        ds (xarray.Dataset): The dataset containing precipitation and elevation data.

    Returns:
        pd.DataFrame: A DataFrame containing the correlation coefficients for each month.
    """
    try:
        logger.info("Calculating correlations between precipitation and elevation by month...")

        # Extract variables
        correlations = {}

        # Loop through each month and calculate correlation
        for month in range(1, 13):
            # Select the current month for both precipitation and elevation
            precipitation = ds["mean_precipitation"].sel(month_index=month).values.flatten()
            elevation = ds["DEM"].sel(month_index=month).values.flatten()

            # Remove NaN values for correlation calculation
            valid_mask = ~np.isnan(precipitation) & ~np.isnan(elevation)
            if valid_mask.sum() > 0:
                corr = np.corrcoef(precipitation[valid_mask], elevation[valid_mask])[0, 1]
            else:
                corr = np.nan

            correlations[month] = corr

        logger.info("Correlation calculation by month completed successfully!")
        return pd.DataFrame.from_dict(correlations, orient="index", columns=["Correlation"]).rename_axis("Month")
    except Exception as e:
        logger.error(f"Error calculating correlations by month: {e}")
        raise

def calculate_correlation_map(ds):
    """
    Calculate the correlation between precipitation and elevation for each grid point.

    Parameters:
        ds (xarray.Dataset): The dataset containing precipitation and elevation data.

    Returns:
        xarray.DataArray: A DataArray containing the correlation coefficients for each grid point.
    """
    try:
        logger.info("Calculating correlation map between precipitation and elevation...")

        # Extract variables
        elevation = ds["DEM"].values
        precipitation = ds["mean_precipitation"].values

        # Depuración: Verificar estadísticas básicas de las variables
        logger.info(f"Elevation stats: mean={np.nanmean(elevation)}, min={np.nanmin(elevation)}, max={np.nanmax(elevation)}")
        logger.info(f"Precipitation stats: mean={np.nanmean(precipitation)}, min={np.nanmin(precipitation)}, max={np.nanmax(precipitation)}")

        # Inicializar el mapa de correlación con NaN
        correlation_map = np.full(elevation.shape[1:], np.nan)  # Excluir la dimensión de tiempo

        # Loop through each grid point and calculate correlation
        for i in range(elevation.shape[1]):  # Latitude
            for j in range(elevation.shape[2]):  # Longitude
                # Extraer los datos de precipitación para el punto de la cuadrícula
                precip_values = precipitation[:, i, j]

                # Crear una máscara de valores válidos
                elevation_broadcasted = np.full(precip_values.shape, elevation[0, i, j])  # Broadcast elevation
                valid_mask = ~np.isnan(precip_values) & ~np.isnan(elevation_broadcasted)

                # Depuración: Verificar si hay datos válidos
                if valid_mask.sum() > 0:
                    logger.debug(f"Valid data found at grid point ({i}, {j})")
                    corr = np.corrcoef(precip_values[valid_mask], elevation_broadcasted[valid_mask])[0, 1]
                    correlation_map[i, j] = corr
                else:
                    logger.debug(f"No valid data at grid point ({i}, {j})")

        logger.info("Correlation map calculation completed successfully!")
        return xr.DataArray(
            correlation_map,
            dims=["latitude", "longitude"],
            coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
            name="correlation"
        )
    except Exception as e:
        logger.error(f"Error calculating correlation map: {e}")
        raise

def plot_correlation_map(correlation_map, output_path):
    """
    Plot the correlation map between precipitation and elevation.

    Parameters:
        correlation_map (xarray.DataArray): The correlation map.
        output_path (str): Path to save the plot.
    """
    try:
        logger.info("Generating correlation map plot...")

        plt.figure(figsize=(12, 8))
        correlation_map.plot(
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            cbar_kwargs={"label": "Correlation Coefficient"}
        )
        plt.title("Correlation Between Precipitation and Elevation")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Correlation map plot saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Error generating correlation map plot: {e}")
        raise

def calculate_statistics(data_array, variable_name):
    """
    Calculate basic statistics for a DataArray.

    Parameters:
        data_array (xarray.DataArray): The DataArray to analyze.
        variable_name (str): Name of the variable being analyzed.

    Returns:
        None: Logs the statistics.
    """
    try:
        stats = {
            "mean": float(data_array.mean().values),
            "std_dev": float(data_array.std().values),
            "min": float(data_array.min().values),
            "max": float(data_array.max().values),
            "count": int(data_array.count().values)
        }
        logger.info(f"Statistics for {variable_name}: {stats}")
    except Exception as e:
        logger.error(f"Error calculating statistics for {variable_name}: {e}")
        raise

def main():
    """
    Main function to calculate and visualize correlations between precipitation and elevation.
    """
    try:
        logger.info("Starting correlation analysis process...")

        # Load dataset
        ds = load_dataset(DATASET_PATH)

        # Generate statistics for input variables
        if "mean_precipitation" in ds:
            calculate_statistics(ds["mean_precipitation"], "mean_precipitation")
        else:
            logger.warning("Variable 'mean_precipitation' not found in the dataset.")

        if "DEM" in ds:
            calculate_statistics(ds["DEM"], "DEM")
        else:
            logger.warning("Variable 'DEM' not found in the dataset.")

        # Define elevation thresholds
        low_threshold = 1500  # Elevation <= 1500m is considered low
        high_threshold = 2500  # Elevation > 2500m is considered high

        # Calculate correlations by elevation levels
        correlations_by_levels = calculate_correlation_by_elevation_levels(ds, low_threshold, high_threshold)

        # Save correlations by elevation levels to CSV
        correlation_levels_csv_path = os.path.join(OUTPUT_DIR, "correlations_by_elevation_levels.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        correlations_by_levels.to_csv(correlation_levels_csv_path)
        logger.info(f"Correlations by elevation levels saved to: {correlation_levels_csv_path}")

        # Generate plots for correlations by elevation levels
        plot_all_correlations(correlations_by_levels, OUTPUT_DIR)

        # Calculate correlations by month
        correlations_by_month = calculate_correlations_by_month(ds)

        # Save correlations by month to CSV
        correlation_month_csv_path = os.path.join(OUTPUT_DIR, "correlations_by_month.csv")
        correlations_by_month.to_csv(correlation_month_csv_path)
        logger.info(f"Correlations by month saved to: {correlation_month_csv_path}")

        # Generate plot for correlations by month
        correlation_month_plot_path = os.path.join(OUTPUT_DIR, "correlations_by_month.png")
        plot_correlations(correlations_by_month, correlation_month_plot_path)

        # Calculate correlation map
        correlation_map = calculate_correlation_map(ds)

        # Save correlation map to NetCDF
        correlation_map_path = os.path.join(OUTPUT_DIR, "correlation_map.nc")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        correlation_map.to_netcdf(correlation_map_path)
        logger.info(f"Correlation map saved to: {correlation_map_path}")

        # Generate plot for correlation map
        correlation_map_plot_path = os.path.join(OUTPUT_DIR, "correlation_map.png")
        plot_correlation_map(correlation_map, correlation_map_plot_path)

        logger.info("Correlation analysis process completed successfully!")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
