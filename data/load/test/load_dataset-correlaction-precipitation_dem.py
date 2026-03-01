import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# Ruta del archivo NetCDF
DATASET_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/correlation/correlation_map.nc"

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

def plot_correlation_map(data_array, output_path=None):
    """
    Plot the correlation map.

    Parameters:
        data_array (xarray.DataArray): The DataArray containing correlation data.
        output_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    plt.figure(figsize=(12, 8))
    data_array.plot(
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar_kwargs={"label": "Correlation Coefficient"}
    )
    plt.title("Correlation Between Precipitation and Elevation")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Correlation map saved to: {output_path}")
    else:
        plt.show()

def main():
    # Cargar el dataset
    ds = load_dataset(DATASET_PATH)

    # Extraer el mapa de correlación
    correlation_map = ds["correlation"]

    # Calcular estadísticas básicas
    stats = calculate_statistics(correlation_map)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value}")

    # Graficar el mapa de correlación
    plot_output_path = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/correlation/correlation_map_visualization.png"
    plot_correlation_map(correlation_map, output_path=plot_output_path)

if __name__ == "__main__":
    main()
