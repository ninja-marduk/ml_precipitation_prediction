import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# Ruta del archivo NetCDF
DATASET_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_monthly_aggregated_merged_dem.nc"

def load_dataset(file_path):
    """
    Load a NetCDF dataset as an xarray Dataset.

    Parameters:
        file_path (str): Path to the NetCDF file.

    Returns:
        xarray.Dataset: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    print(f"Loading dataset from: {file_path}")
    ds = xr.open_dataset(file_path)
    print("Dataset loaded successfully!")
    return ds

def calculate_statistics(data_array):
    """
    Calculate basic statistics for a DataArray.

    Parameters:
        data_array (xarray.DataArray): The DataArray to analyze.

    Returns:
        dict: A dictionary containing basic statistics.
    """
    stats = {
        "mean": float(data_array.mean().values),
        "std_dev": float(data_array.std().values),
        "min": float(data_array.min().values),
        "max": float(data_array.max().values),
        "count": int(data_array.count().values)
    }
    return stats

def plot_data_map(data_array, title, output_path=None):
    """
    Plot a data map.

    Parameters:
        data_array (xarray.DataArray): The DataArray to plot.
        title (str): Title of the plot.
        output_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    # Verificar si la variable tiene dimensiones adecuadas para un mapa
    if "latitude" in data_array.dims and "longitude" in data_array.dims:
        plt.figure(figsize=(12, 8))
        data_array.plot(
            cmap="viridis",
            cbar_kwargs={"label": data_array.attrs.get("units", "Value")}
        )
        plt.title(title)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            print(f"Map saved to: {output_path}")
        else:
            plt.show()
    else:
        print(f"Cannot plot {title}: DataArray does not have 'latitude' and 'longitude' dimensions.")

def main():
    # Cargar el dataset
    ds = load_dataset(DATASET_PATH)

    # Calcular estadísticas para precipitación media
    if "mean_precipitation" in ds:
        mean_precipitation = ds["mean_precipitation"]

        # Seleccionar el primer mes si tiene una dimensión adicional (por ejemplo, 'time' o 'month_index')
        if "month_index" in mean_precipitation.dims:
            mean_precipitation = mean_precipitation.sel(month_index=1)

        print("Statistics for Mean Precipitation:")
        stats = calculate_statistics(mean_precipitation)
        for key, value in stats.items():
            print(f"{key.capitalize()}: {value}")

        # Graficar el mapa de precipitación media
        plot_output_path = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/mean_precipitation_map.png"
        plot_data_map(mean_precipitation, "Mean Precipitation Map", output_path=plot_output_path)

    # Calcular estadísticas para elevación (DEM)
    if "DEM" in ds:
        elevation = ds["DEM"]

        print("\nStatistics for Elevation (DEM):")
        stats = calculate_statistics(elevation)
        for key, value in stats.items():
            print(f"{key.capitalize()}: {value}")

        # Graficar el mapa de elevación (DEM)
        plot_output_path = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/elevation_map.png"
        plot_data_map(elevation, "Elevation Map (DEM)", output_path=plot_output_path)

if __name__ == "__main__":
    main()
