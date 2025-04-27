import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
import argparse
import psutil
import time
from tqdm import tqdm
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

# Paths to clustered datasets
DATASET_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/"
HIGH_ELEVATION_PATH = os.path.join(DATASET_PATH, "ds_high_elevation.nc")
MEDIUM_ELEVATION_PATH = os.path.join(DATASET_PATH, "ds_medium_elevation.nc")
LOW_ELEVATION_PATH = os.path.join(DATASET_PATH, "ds_low_elevation.nc")

# Paths to aggregated datasets
AGGREGATED_DATASETS = {
    "monthly": os.path.join(DATASET_PATH, "aggregated_monthly.nc"),
    "annual": os.path.join(DATASET_PATH, "aggregated_annual.nc"),
    "seasonal": os.path.join(DATASET_PATH, "aggregated_seasonal.nc"),
    "bimodal": os.path.join(DATASET_PATH, "aggregated_bimodal.nc"),
}

def monitor_resources():
    """
    Monitor and log resource usage (CPU, memory).
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        cpu_percent = psutil.cpu_percent(interval=1)
        logging.info(f"Resource usage - Memory: {memory_info:.2f} MB, CPU: {cpu_percent:.2f}%")
    except Exception as e:
        logging.error(f"Error monitoring resources: {e}")

def load_combined_data():
    """
    Load data from the elevation datasets and combine them into a single DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with columns 'elevation_category', 'time', and 'precipitation'.
    """
    try:
        logging.info("Loading datasets...")
        high_ds = xr.open_dataset(HIGH_ELEVATION_PATH, chunks={"time": 100})
        medium_ds = xr.open_dataset(MEDIUM_ELEVATION_PATH, chunks={"time": 100})
        low_ds = xr.open_dataset(LOW_ELEVATION_PATH, chunks={"time": 100})

        logging.info("Converting datasets to DataFrames...")
        high_df = high_ds["downscaled_mean_precipitation"].to_dataframe().reset_index()
        high_df["elevation_category"] = "high"

        medium_df = medium_ds["downscaled_mean_precipitation"].to_dataframe().reset_index()
        medium_df["elevation_category"] = "medium"

        low_df = low_ds["downscaled_mean_precipitation"].to_dataframe().reset_index()
        low_df["elevation_category"] = "low"

        logging.info("Combining datasets...")
        combined_df = pd.concat([high_df, medium_df, low_df], ignore_index=True)

        logging.info("Filling NaN values with 0...")
        combined_df["downscaled_mean_precipitation"] = combined_df["downscaled_mean_precipitation"].fillna(0)

        logging.info("Datasets loaded and combined successfully!")
        monitor_resources()
        return combined_df
    except Exception as e:
        logging.error(f"Error loading and combining datasets: {e}")
        raise

def aggregate_data_in_chunks(data, freq, output_path, chunk_size=100000):
    """
    Aggregate precipitation data in chunks to avoid memory issues.

    Parameters:
        data (pd.DataFrame): The input DataFrame with 'time' and 'precipitation'.
        freq (str): The aggregation frequency ('ME', 'YE', 'Q', etc.).
        output_path (str): Path to save the aggregated dataset.
        chunk_size (int): Number of rows to process in each chunk.
    """
    try:
        logging.info(f"Aggregating data with frequency '{freq}' in chunks...")
        data["time"] = pd.to_datetime(data["time"])
        aggregated_chunks = []

        for i in tqdm(range(0, len(data), chunk_size), desc="Processing chunks"):
            chunk = data.iloc[i:i + chunk_size]
            aggregated_chunk = chunk.groupby(["elevation_category", pd.Grouper(key="time", freq=freq)]).mean().reset_index()
            aggregated_chunks.append(aggregated_chunk)

        aggregated_data = pd.concat(aggregated_chunks, ignore_index=True)

        # Ensure unique index
        aggregated_data = aggregated_data.drop_duplicates(subset=["time", "elevation_category"])

        # Verify index uniqueness
        if not aggregated_data.set_index(["time", "elevation_category"]).index.is_unique:
            raise ValueError("The DataFrame index is not unique. Check for duplicates in the data.")

        # Convert to xarray.Dataset
        ds = aggregated_data.set_index(["time", "elevation_category"]).to_xarray()
        ds.to_netcdf(output_path)
        logging.info(f"Aggregated dataset saved to: {output_path}")
        monitor_resources()
    except Exception as e:
        logging.error(f"Error aggregating data in chunks: {e}")
        raise

def calculate_correlation(data):
    """
    Calculate correlation statistics for the given data.

    Parameters:
        data (pd.DataFrame): The input DataFrame with 'time', 'elevation_category', and 'downscaled_mean_precipitation'.

    Returns:
        pd.DataFrame: A DataFrame containing correlation statistics for each elevation category over time.
    """
    try:
        logging.info("Calculating correlation statistics...")
        correlation_results = []
        for category in data['elevation_category'].unique():
            category_data = data[data['elevation_category'] == category]
            correlation = category_data['downscaled_mean_precipitation'].corr(category_data['time'].apply(lambda x: x.toordinal()))
            correlation_results.append({
                'elevation_category': category,
                'time': category_data['time'].iloc[-1],  # Use the last time point for simplicity
                'correlation': correlation
            })
        logging.info("Correlation statistics calculated successfully.")
        return pd.DataFrame(correlation_results)
    except Exception as e:
        logging.error(f"Error calculating correlation statistics: {e}")
        raise

def plot_correlation(correlation_stats, label):
    """
    Plot the correlation statistics and save the plot to a file.

    Parameters:
        correlation_stats (pd.DataFrame): DataFrame containing correlation statistics.
        label (str): Label for the dataset (e.g., "monthly", "annual").
    """
    try:
        logging.info(f"Plotting correlation statistics for {label} dataset...")
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

        logging.info(f"Correlation plot saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error plotting correlation statistics: {e}")
        raise

def main():
    """
    Main function to load data, process aggregations, and generate plots.
    """
    try:
        logging.info("Starting CHIRPS data processing and correlation analysis...")
        parser = argparse.ArgumentParser(description="Process CHIRPS data and generate correlations.")
        parser.add_argument("--recreate", action="store_true", help="Recreate aggregated datasets.")
        args = parser.parse_args()

        logging.info("Loading combined data...")
        data = load_combined_data()
        logging.info("Data loaded successfully!")

        # Create or load aggregated datasets
        for label, path in AGGREGATED_DATASETS.items():
            if args.recreate or not os.path.exists(path):
                logging.info(f"Creating {label} aggregation...")
                freq = {
                    "monthly": "ME",
                    "annual": "YE",
                    "seasonal": "QE",
                    "bimodal": "6M",
                }[label]
                aggregate_data_in_chunks(data, freq, path)
            else:
                logging.info(f"Using existing {label} aggregation at {path}.")

            # Load aggregated dataset
            logging.info(f"Loading {label} dataset...")
            aggregated_ds = xr.open_dataset(path).to_dataframe().reset_index()

            # Calculate correlation and plot
            correlation_stats = calculate_correlation(aggregated_ds)
            plot_correlation(correlation_stats, label)

        logging.info("All processes completed successfully!")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main process: {e}")
        raise

if __name__ == "__main__":
    main()
