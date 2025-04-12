import xarray as xr
import pandas as pd

# Ruta del dataset
OUTPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_combined_downscaled.nc"

def load_dataset(file_path):
    """
    Load a NetCDF dataset as an xarray Dataset.

    Parameters:
        file_path (str): Path to the NetCDF file.

    Returns:
        xarray.Dataset: Loaded dataset.
    """
    return xr.open_dataset(file_path)

def analyze_dataset(ds):
    """
    Analyze the dataset and generate statistics for all variables.

    Parameters:
        ds (xarray.Dataset): The dataset to analyze.

    Returns:
        dict: A dictionary containing statistics for each variable.
    """
    stats = {}
    for var in ds.data_vars:
        data = ds[var].values.flatten()
        data = data[~pd.isnull(data)]  # Remove NaN values for analysis

        stats[var] = {
            "Description": ds[var].attrs.get("description", "No description available"),
            "Units": ds[var].attrs.get("units", "No units available"),
            "Number of Records": len(data),
            "Mean": float(data.mean()) if len(data) > 0 else None,
            "Median": float(pd.Series(data).median()) if len(data) > 0 else None,
            "Standard Deviation": float(data.std()) if len(data) > 0 else None,
            "Min": float(data.min()) if len(data) > 0 else None,
            "Max": float(data.max()) if len(data) > 0 else None,
            "25th Percentile": float(pd.Series(data).quantile(0.25)) if len(data) > 0 else None,
            "50th Percentile (Median)": float(pd.Series(data).quantile(0.5)) if len(data) > 0 else None,
            "75th Percentile": float(pd.Series(data).quantile(0.75)) if len(data) > 0 else None,
            "90th Percentile (P90)": float(pd.Series(data).quantile(0.9)) if len(data) > 0 else None,
            "95th Percentile (P95)": float(pd.Series(data).quantile(0.95)) if len(data) > 0 else None,
            "Number of Nulls": int(ds[var].isnull().sum().values)
        }
    return stats

def print_statistics(stats):
    """
    Print the statistics in a readable format.

    Parameters:
        stats (dict): The statistics dictionary to print.
    """
    for var, var_stats in stats.items():
        print(f"\nVariable: {var}")
        print(f"  Description: {var_stats['Description']}")
        print(f"  Units: {var_stats['Units']}")
        print(f"  Number of Records: {var_stats['Number of Records']}")
        print(f"  Mean: {var_stats['Mean']}")
        print(f"  Median: {var_stats['Median']}")
        print(f"  Standard Deviation: {var_stats['Standard Deviation']}")
        print(f"  Min: {var_stats['Min']}")
        print(f"  Max: {var_stats['Max']}")
        print(f"  25th Percentile: {var_stats['25th Percentile']}")
        print(f"  50th Percentile (Median): {var_stats['50th Percentile (Median)']}")
        print(f"  75th Percentile: {var_stats['75th Percentile']}")
        print(f"  90th Percentile (P90): {var_stats['90th Percentile (P90)']}")
        print(f"  95th Percentile (P95): {var_stats['95th Percentile (P95)']}")
        print(f"  Number of Nulls: {var_stats['Number of Nulls']}")

def main():
    """
    Main function to load and analyze the dataset.
    """
    print("Loading dataset...")
    ds = load_dataset(OUTPUT_PATH)
    print("Dataset loaded successfully!")

    print("Analyzing dataset...")
    stats = analyze_dataset(ds)
    print_statistics(stats)

if __name__ == "__main__":
    main()
