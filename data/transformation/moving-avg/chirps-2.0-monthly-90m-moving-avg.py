import xarray as xr
import matplotlib.pyplot as plt
import os
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

# Ruta del dataset
OUTPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_combined_downscaled.nc"
UPDATED_OUTPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_combined_downscaled_with_monthly_moving_avg.nc"

def calculate_monthly_moving_average(file_path, updated_file_path, window_size=3):
    """
    Calculate a moving average for monthly precipitation and save it to a new NetCDF file.

    Parameters:
        file_path (str): Path to the input NetCDF file.
        updated_file_path (str): Path to save the updated NetCDF file.
        window_size (int): Size of the moving average window (default is 3 months).
    """
    try:
        logging.info("Loading dataset...")
        ds = xr.open_dataset(file_path)
        logging.info("Dataset loaded successfully!")

        # Ensure the variable 'downscaled_mean_precipitation' exists
        if "downscaled_mean_precipitation" not in ds:
            raise KeyError("The variable 'downscaled_mean_precipitation' is not found in the dataset.")

        logging.info(f"Calculating monthly moving average with window size {window_size} months...")
        # Calculate the moving average along the time dimension
        ds["mean_precipitation_monthly_moving_avg"] = (
            ds["downscaled_mean_precipitation"]
            .rolling(time=window_size, center=True)
            .mean()
        )

        # Add metadata to the new variable
        ds["mean_precipitation_monthly_moving_avg"].attrs = {
            "description": f"Monthly moving average of mean precipitation (window size: {window_size} months)",
            "units": "mm/month",
            "window_size": f"{window_size} months"
        }

        logging.info("Saving updated dataset...")
        ds.to_netcdf(updated_file_path)
        logging.info(f"Updated dataset saved to: {updated_file_path}")

        # Plot the moving average for visualization
        logging.info("Visualizing the moving average...")
        ds["mean_precipitation_monthly_moving_avg"].isel(latitude=0, longitude=0).plot(
            figsize=(10, 6)
        )
        plt.title("Monthly Moving Average of Mean Precipitation (First Latitude/Longitude)")
        plt.ylabel("Precipitation (mm/month)")
        plt.xlabel("Time")
        plt.grid()
        plt.show()
        logging.info("Visualization completed successfully.")
    except KeyError as e:
        logging.error(f"KeyError: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise

def main():
    """
    Main function to calculate the monthly moving average and save the updated dataset.
    """
    try:
        logging.info("Starting monthly moving average calculation...")
        calculate_monthly_moving_average(OUTPUT_PATH, UPDATED_OUTPUT_PATH, window_size=3)
        logging.info("Monthly moving average calculation completed successfully!")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main process: {e}")
        raise

if __name__ == "__main__":
    main()
