import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
import argparse
import psutil
import time
from tqdm import tqdm

# Rutas de los datasets clusterizados
DATASET_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/"
HIGH_ELEVATION_PATH = os.path.join(DATASET_PATH, "ds_high_elevation.nc")
MEDIUM_ELEVATION_PATH = os.path.join(DATASET_PATH, "ds_medium_elevation.nc")
LOW_ELEVATION_PATH = os.path.join(DATASET_PATH, "ds_low_elevation.nc")

# Rutas de los datasets de agregaciones
AGGREGATED_DATASETS = {
    "monthly": os.path.join(DATASET_PATH, "aggregated_monthly.nc"),
    "annual": os.path.join(DATASET_PATH, "aggregated_annual.nc"),
    "seasonal": os.path.join(DATASET_PATH, "aggregated_seasonal.nc"),
    "bimodal": os.path.join(DATASET_PATH, "aggregated_bimodal.nc"),
}

def monitor_resources():
    """
    Monitor and print resource usage (CPU, memory).
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"Resource usage - Memory: {memory_info:.2f} MB, CPU: {cpu_percent:.2f}%")

def load_combined_data():
    """
    Load data from the elevation datasets and combine them into a single DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with columns 'elevation_category', 'time', and 'precipitation'.
    """
    print("Loading datasets...")
    print("High DS")
    high_ds = xr.open_dataset(HIGH_ELEVATION_PATH, chunks={"time": 100})
    print("Medium DS")
    medium_ds = xr.open_dataset(MEDIUM_ELEVATION_PATH, chunks={"time": 100})
    print("Low DS")
    low_ds = xr.open_dataset(LOW_ELEVATION_PATH, chunks={"time": 100})

    print("High DS to DataFrame")
    high_df = high_ds["downscaled_mean_precipitation"].to_dataframe().reset_index()
    high_df["elevation_category"] = "high"

    print("Medium DS to DataFrame")
    medium_df = medium_ds["downscaled_mean_precipitation"].to_dataframe().reset_index()
    medium_df["elevation_category"] = "medium"

    print("Low DS to DataFrame")
    low_df = low_ds["downscaled_mean_precipitation"].to_dataframe().reset_index()
    low_df["elevation_category"] = "low"

    print("Combining datasets...")
    combined_df = pd.concat([high_df, medium_df, low_df], ignore_index=True)

    print("Filling NaN values with 0...")
    combined_df["downscaled_mean_precipitation"] = combined_df["downscaled_mean_precipitation"].fillna(0)

    print("Datasets loaded and combined successfully!")
    monitor_resources()
    return combined_df

def print_statistics(data, label):
    """
    Print statistics of the dataset.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        label (str): Label for the dataset (e.g., "monthly", "annual").
    """
    print(f"\nStatistics for {label} aggregation:")
    print(f"Number of records: {len(data)}")
    print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    print(f"Elevation categories: {data['elevation_category'].unique()}")
    print(f"Precipitation range: {data['downscaled_mean_precipitation'].min()} to {data['downscaled_mean_precipitation'].max()}")

def aggregate_data_in_chunks(data, freq, output_path, chunk_size=100000):
    """
    Aggregate precipitation data in chunks to avoid memory issues.

    Parameters:
        data (pd.DataFrame): The input DataFrame with 'time' and 'precipitation'.
        freq (str): The aggregation frequency ('ME', 'YE', 'Q', etc.).
        output_path (str): Path to save the aggregated dataset.
        chunk_size (int): Number of rows to process in each chunk.
    """
    data["time"] = pd.to_datetime(data["time"])
    aggregated_chunks = []

    for i in tqdm(range(0, len(data), chunk_size), desc="Processing chunks"):
        chunk = data.iloc[i:i + chunk_size]
        aggregated_chunk = chunk.groupby(["elevation_category", pd.Grouper(key="time", freq=freq)]).mean().reset_index()
        aggregated_chunks.append(aggregated_chunk)

    aggregated_data = pd.concat(aggregated_chunks, ignore_index=True)

    # Asegurarse de que el índice sea único
    aggregated_data = aggregated_data.drop_duplicates(subset=["time", "elevation_category"])

    # Verificar unicidad del índice
    if not aggregated_data.set_index(["time", "elevation_category"]).index.is_unique:
        raise ValueError("The DataFrame index is not unique. Check for duplicates in the data.")

    # Convertir a xarray.Dataset
    ds = aggregated_data.set_index(["time", "elevation_category"]).to_xarray()
    ds.to_netcdf(output_path)
    print(f"Aggregated dataset saved to: {output_path}")
    monitor_resources()

def load_or_create_aggregated_datasets(data, recreate):
    """
    Load or create aggregated datasets based on the recreate flag.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        recreate (bool): Whether to recreate the datasets or use existing ones.
    """
    for label, path in AGGREGATED_DATASETS.items():
        if recreate or not os.path.exists(path):
            print(f"Creating {label} aggregation...")
            freq = {
                "monthly": "ME",
                "annual": "YE",
                "seasonal": "QE",
                "bimodal": "6M",
            }[label]
            aggregate_data_in_chunks(data, freq, path)
        else:
            print(f"Using existing {label} aggregation at {path}.")

def calculate_correlation(data):
    """
    Calculate correlation statistics for the given data.

    Parameters:
        data (pd.DataFrame): The input DataFrame with 'time', 'elevation_category', and 'downscaled_mean_precipitation'.

    Returns:
        pd.DataFrame: A DataFrame containing correlation statistics for each elevation category over time.
    """
    correlation_results = []
    for category in data['elevation_category'].unique():
        category_data = data[data['elevation_category'] == category]
        correlation = category_data['downscaled_mean_precipitation'].corr(category_data['time'].apply(lambda x: x.toordinal()))
        correlation_results.append({
            'elevation_category': category,
            'time': category_data['time'].iloc[-1],  # Use the last time point for simplicity
            'correlation': correlation
        })
    return pd.DataFrame(correlation_results)

def plot_correlation(correlation_stats, label):
    """
    Plot the correlation statistics and save the plot to a file.

    Parameters:
        correlation_stats (pd.DataFrame): DataFrame containing correlation statistics.
        label (str): Label for the dataset (e.g., "monthly", "annual").
    """
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, f'correlation_plot_{label}.png')
    plt.figure(figsize=(10, 6))
    for category in correlation_stats['elevation_category'].unique():
        category_data = correlation_stats[correlation_stats['elevation_category'] == category]
        plt.plot(category_data['time'], category_data['correlation'], label=category)

    plt.title("Correlation Over Time")
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()

    print(f"Correlation plot saved to: {output_path}")

def main():
    """
    Main function to load data, process aggregations, and generate plots.
    """
    parser = argparse.ArgumentParser(description="Process CHIRPS data and generate correlations.")
    parser.add_argument("--recreate", action="store_true", help="Recreate aggregated datasets.")
    args = parser.parse_args()

    print("Loading data...")
    data = load_combined_data()
    print("Data loaded successfully!")

    # Imprimir estadísticas generales del dataset combinado
    print_statistics(data, "combined")

    # Crear o cargar datasets agregados
    load_or_create_aggregated_datasets(data, args.recreate)

    # Generar estadísticas y correlaciones para cada agregación
    for label, path in AGGREGATED_DATASETS.items():
        print(f"\nLoading {label} dataset...")
        aggregated_ds = xr.open_dataset(path).to_dataframe().reset_index()
        print_statistics(aggregated_ds, label)

        # Generar correlaciones y gráficos
        correlation_stats = calculate_correlation(aggregated_ds)
        print("Correlation stats before plotting:")
        print(correlation_stats.head())
        print(correlation_stats.describe())
        plot_correlation(correlation_stats, label)
        print(f"Completed plotting for {label} dataset.")

    print("All processes completed!")

if __name__ == "__main__":
    main()

"""
# Cargar el dataset combinado

@startuml
start

:Parse command-line arguments (--recreate);
if (Recreate datasets?) then (Yes)
    :Load combined data from elevation datasets;
    :Print statistics for combined data;
    :Create aggregated datasets (monthly, annual, seasonal, bimodal);
else (No)
    :Use existing aggregated datasets;
endif

repeat
    :Load aggregated dataset;
    :Print statistics for the dataset;
    :Calculate correlation statistics;
    :Plot correlation over time;
    :Save correlation plot;
repeat while (More datasets?)

:Print "All processes completed!";
stop
@enduml

"""
