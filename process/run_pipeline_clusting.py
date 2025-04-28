import subprocess
import time
import psutil
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
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

# Add the project's root directory to PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Relative paths based on the project's structure
DATA_OUTPUT_DIR = os.path.join(project_root, "data", "output")

# Dataset paths
INPUT_DATASET = os.path.join(DATA_OUTPUT_DIR, "ds_combined_downscaled_with_monthly_moving_avg.nc")
OUTPUT_LOW = os.path.join(DATA_OUTPUT_DIR, "ds_low_elevation.nc")
OUTPUT_MEDIUM = os.path.join(DATA_OUTPUT_DIR, "ds_medium_elevation.nc")
OUTPUT_HIGH = os.path.join(DATA_OUTPUT_DIR, "ds_high_elevation.nc")

# Elevation thresholds
LOW_ELEVATION_THRESHOLD = 1500
HIGH_ELEVATION_THRESHOLD = 2500

def run_script(script_path):
    """
    Run a Python script using subprocess and monitor its performance.

    Parameters:
        script_path (str): Path to the Python script to execute.
    """
    if not os.path.isfile(script_path):
        logging.error(f"The script {script_path} does not exist.")
        sys.exit(1)

    logging.info(f"Running script: {script_path}")

    try:
        # Start monitoring
        start_time = time.time()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        cpu_cores = psutil.cpu_count(logical=True)

        # Run the script
        result = subprocess.run(["python", script_path], capture_output=True, text=True)

        # End monitoring
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_after = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        memory_used = memory_after - memory_before
        cpu_percent = process.cpu_percent(interval=None)
        threads_used = process.num_threads()

        # Log performance metrics
        logging.info(f"Script {script_path} executed in {elapsed_time:.2f} seconds.")
        logging.info(f"Memory used: {memory_used:.2f} MB")
        logging.info(f"CPU cores available: {cpu_cores}")
        logging.info(f"CPU usage: {cpu_percent:.2f}%")
        logging.info(f"Threads used: {threads_used}")

        # Handle errors
        if result.returncode != 0:
            logging.error(f"Error executing {script_path}:\n{result.stderr}")
            sys.exit(1)
        else:
            logging.info(f"Script {script_path} executed successfully!")

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess error while executing {script_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while executing {script_path}: {e}")
        sys.exit(1)

def main():
    """
    Main function to execute the clustering pipeline.
    """
    # Verify that the input file exists
    if not os.path.exists(INPUT_DATASET):
        logging.error(f"The input file does not exist: {INPUT_DATASET}")
        sys.exit(1)

    logging.info("Starting elevation-based clustering pipeline...")
    logging.info(f"Input file: {INPUT_DATASET}")

    # Import the clustering transformation module
    sys.path.append(os.path.join(project_root, "data", "transformation", "clustering"))
    from data.transformation.clustering.chirps_2_0 import cluster_by_elevation, save_clustered_datasets

    try:
        # Import xarray
        import xarray as xr

        # Load the dataset
        logging.info("Loading dataset...")
        ds = xr.open_dataset(INPUT_DATASET)

        # Perform clustering
        logging.info(f"Performing clustering with thresholds: low <= {LOW_ELEVATION_THRESHOLD}, high > {HIGH_ELEVATION_THRESHOLD}...")
        ds_low, ds_medium, ds_high = cluster_by_elevation(ds, LOW_ELEVATION_THRESHOLD, HIGH_ELEVATION_THRESHOLD)

        # Save results
        logging.info("Saving clustered datasets...")
        save_clustered_datasets(ds_low, ds_medium, ds_high, OUTPUT_LOW, OUTPUT_MEDIUM, OUTPUT_HIGH)

        logging.info("Clustering pipeline completed successfully!")
        logging.info(f"Generated files:")
        logging.info(f"- Low elevation: {OUTPUT_LOW}")
        logging.info(f"- Medium elevation: {OUTPUT_MEDIUM}")
        logging.info(f"- High elevation: {OUTPUT_HIGH}")

    except Exception as e:
        logging.error(f"An unexpected error occurred in the clustering pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
