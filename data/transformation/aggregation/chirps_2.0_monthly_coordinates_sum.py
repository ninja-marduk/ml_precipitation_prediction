import os
import xarray as xr
import pandas as pd
import logging
from datetime import datetime
import calendar

# Logging configuration
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../logs")
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists

log_filename = datetime.now().strftime("log-%Y-%m-%d.log")
LOG_FILE = os.path.join(LOG_DIR, log_filename)

log_format = "%(asctime)s - %(levelname)s - [%(pathname)s-%(funcName)s-%(lineno)d] - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # Display logs in the terminal
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")  # Save logs to a file
    ]
)

# Constants
INPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_daily.nc"
OUTPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/"
OUTPUT_FILENAME = "boyaca_region_monthly_sum.nc"

def load_dataset(file_path):
    """
    Load a NetCDF file and return the dataset.
    """
    try:
        logging.info(f"Loading dataset from: {file_path}")
        ds = xr.open_dataset(file_path)
        logging.info("Dataset successfully loaded.")
        return ds
    except Exception as e:
        logging.error(f"Error loading the dataset: {e}")
        raise

def aggregate_monthly_precipitation(ds):
    """
    Aggregate precipitation data by month, calculating total (sum), max, and min values.
    Latitude and longitude are included as variables in the output dataset.

    Parameters:
    ds (xarray.Dataset): Input dataset containing precipitation data with 'time', 'latitude',
                         'longitude', and 'precip' variables.

    Returns:
    pandas.DataFrame: A DataFrame containing aggregated precipitation statistics by month,
                      including total precipitation, maximum daily precipitation,
                      minimum daily precipitation, latitude, longitude, and additional
                      date-related columns ('YYYY-MM', 'YYYY', 'MM').

    Raises:
    Exception: If an error occurs during the aggregation process.
    """
    """
    Aggregate precipitation data by month, calculating total (sum), max, and min.
    """
    try:
        logging.info("Starting precipitation data aggregation by month...")
        df = ds.to_dataframe().reset_index()
        df['time'] = pd.to_datetime(df['time'])

        # Monthly aggregation by coordinates
        monthly_stats = df.groupby(
            [df['time'].dt.to_period('M'), 'latitude', 'longitude']
        ).agg(
            total_precipitation=('precip', 'sum'),
            max_daily_precipitation=('precip', 'max'),
            min_daily_precipitation=('precip', 'min')
        ).reset_index()

        # Convert the time period to timestamp
        monthly_stats['time'] = monthly_stats['time'].dt.to_timestamp()

        # Add additional date columns
        monthly_stats['YYYY-MM'] = monthly_stats['time'].dt.strftime('%Y-%m')
        monthly_stats['YYYY'] = monthly_stats['time'].dt.year.astype(str)
        monthly_stats['MM'] = monthly_stats['time'].dt.month.apply(
            lambda x: calendar.month_name[x]
        ).str.capitalize()

        logging.info("Aggregation successfully completed with additional date columns.")
        return monthly_stats
    except Exception as e:
        logging.error(f"Error aggregating precipitation data: {e}")
        raise

def save_to_netcdf(df, output_path, filename):
    """
    Save a DataFrame to a NetCDF file.
    """
    try:
        logging.info(f"Saving DataFrame to NetCDF file: {filename}")
        ds = df.set_index(['time', 'latitude', 'longitude']).to_xarray()
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, filename)
        ds.to_netcdf(file_path)
        logging.info(f"Data saved to NetCDF file: {file_path}")
    except Exception as e:
        logging.error(f"Error saving the DataFrame to NetCDF file: {e}")
        raise

def main():
    """
    Main function to process daily precipitation data and generate monthly aggregations.
    """
    try:
        logging.info("Starting processing of daily precipitation data for the Boyacá region...")

        # Load the dataset
        ds = load_dataset(INPUT_PATH)

        # Aggregate data by month
        monthly_boyaca = aggregate_monthly_precipitation(ds)

        # Save the aggregated data to a NetCDF file
        save_to_netcdf(monthly_boyaca, OUTPUT_PATH, OUTPUT_FILENAME)

        logging.info("Processing successfully completed.")
    except Exception as e:
        logging.error(f"Unexpected error in the main process: {e}")
        raise

if __name__ == "__main__":
    main()
