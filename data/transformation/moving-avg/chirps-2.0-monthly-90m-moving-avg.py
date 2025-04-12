import xarray as xr
import matplotlib.pyplot as plt

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
    print("Loading dataset...")
    ds = xr.open_dataset(file_path)
    print("Dataset loaded successfully!")

    # Ensure the variable 'downscaled_mean_precipitation' exists
    if "downscaled_mean_precipitation" not in ds:
        raise KeyError("The variable 'downscaled_mean_precipitation' is not found in the dataset.")

    print(f"Calculating monthly moving average with window size {window_size} months...")
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

    print("Saving updated dataset...")
    ds.to_netcdf(updated_file_path)
    print(f"Updated dataset saved to: {updated_file_path}")

    # Plot the moving average for visualization
    print("Visualizing the moving average...")
    ds["mean_precipitation_monthly_moving_avg"].isel(latitude=0, longitude=0).plot(
        figsize=(10, 6)
    )
    plt.title("Monthly Moving Average of Mean Precipitation (First Latitude/Longitude)")
    plt.ylabel("Precipitation (mm/month)")
    plt.xlabel("Time")
    plt.grid()
    plt.show()

def main():
    """
    Main function to calculate the monthly moving average and save the updated dataset.
    """
    calculate_monthly_moving_average(OUTPUT_PATH, UPDATED_OUTPUT_PATH, window_size=3)

if __name__ == "__main__":
    main()
