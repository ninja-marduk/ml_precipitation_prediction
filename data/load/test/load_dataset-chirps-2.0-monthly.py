import xarray as xr
import pandas as pd
import os

# Ruta del dataset mensual dinámico (última fecha _yyyymm.nc)
output_dir = os.path.join(os.path.dirname(__file__), "../../output")
files = [f for f in os.listdir(output_dir) if f.endswith(".nc") and "complete_dataset_with_features_with_clusters_elevation_" in f]
latest_file = max(files, key=lambda x: x.split("_")[-1].split(".")[0]) if files else None

if latest_file:
    DATASET_PATH = os.path.join(output_dir, latest_file)
else:
    raise FileNotFoundError("No se encontró ningún archivo con el formato 'complete_dataset_with_features_with_clusters_elevation_yyyymm.nc' en el directorio de salida.")

def load_dataset_as_dataframe(file_path):
    """
    Load a NetCDF dataset and convert it to a pandas DataFrame.

    Parameters:
        file_path (str): Path to the NetCDF file.

    Returns:
        pandas.DataFrame: DataFrame with all variables and coordinates.
    """
    ds = xr.open_dataset(file_path)
    df = ds.to_dataframe().reset_index()
    print("✅ Dataset convertido a DataFrame con columnas:")
    print(df.columns.tolist())
    return df

def analyze_dataset(df):
    """
    Analyze numeric columns in the DataFrame and generate descriptive statistics.

    Parameters:
        df (pandas.DataFrame): The dataset in tabular format.

    Returns:
        dict: A dictionary containing statistics for each numeric column.
    """
    stats = {}
    for var in df.columns:
        if pd.api.types.is_numeric_dtype(df[var]):
            data = df[var].dropna()
            stats[var] = {
                "Number of Records": len(data),
                "Mean": float(data.mean()),
                "Median": float(data.median()),
                "Standard Deviation": float(data.std()),
                "Min": float(data.min()),
                "Max": float(data.max()),
                "25th Percentile": float(data.quantile(0.25)),
                "50th Percentile (Median)": float(data.quantile(0.5)),
                "75th Percentile": float(data.quantile(0.75)),
                "90th Percentile (P90)": float(data.quantile(0.9)),
                "95th Percentile (P95)": float(data.quantile(0.95)),
                "Number of Nulls": int(df[var].isnull().sum())
            }
    return stats

def print_statistics(stats):
    """
    Print the statistics in a readable format.

    Parameters:
        stats (dict): The statistics dictionary to print.
    """
    for var, var_stats in stats.items():
        print(f"\n📊 Variable: {var}")
        for key, value in var_stats.items():
            print(f"  {key}: {value}")

def main():
    """
    Main function to load and analyze the dataset.
    """
    print("📥 Cargando dataset mensual...")
    df = load_dataset_as_dataframe(DATASET_PATH)

    print("\n📈 Analizando variables numéricas del dataset...")
    stats = analyze_dataset(df)

    print("\n📤 Resultados del análisis estadístico:")
    print_statistics(stats)

if __name__ == "__main__":
    main()
