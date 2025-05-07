import os
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import warnings
from dask.diagnostics import ProgressBar
from dask.array.core import PerformanceWarning
import matplotlib.pyplot as plt

# Paths
INPUT_PATH = "/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/input/CHIRPS-2.0/daily/"
OUTPUT_PATH = "/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/output/"
CURRENT_DATASET = "complete_dataset_with_features_with_clusters_elevation.nc"
OUTPUT_TEMPLATE = "complete_dataset_with_features_with_clusters_elevation_{}.nc"

# Actualizar los límites de Boyacá teniendo en cuenta el centroide del shapefile
LON_BOYACA_MIN = -74.975
LON_BOYACA_MAX = -71.725
LAT_BOYACA_MIN = 4.325
LAT_BOYACA_MAX = 7.375

def download_missing_data():
    # URL base de los archivos
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"

    # Realizar la solicitud HTTP y obtener el contenido HTML de la página
    response = requests.get(base_url)
    html_content = response.content

    # Crear un objeto BeautifulSoup para analizar el contenido HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Encontrar todos los enlaces en la página
    links = soup.find_all("a")

    # Crear una carpeta para almacenar los archivos descargados
    os.makedirs("../data/input/CHIRPS-2.0/", exist_ok=True)

    # Iterar sobre los enlaces y descargar los archivos
    for link in tqdm(links, desc="Descargando archivos"):
        file_url = link.get("href")

        # Ignorar enlaces a directorios o archivos no deseados
        if not file_url.endswith(".nc"):
            continue

        # Nombre del archivo para guardar localmente
        file_name = file_url.split("/")[-1]

        # Verificar si el archivo ya existe
        file_path = os.path.join("../data/input/CHIRPS-2.0/", file_name)
        if os.path.exists(file_path):
            continue

        # Descargar el archivo
        file_response = requests.get(base_url + file_url, stream=True)
        with open(file_path, "wb") as file:
            for chunk in file_response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

    # Mostrar el mensaje de descarga completa
    print("Descarga completa.")

import os
import requests
import pandas as pd
import xarray as xr

def update_chirps_daily():
    """
    Updates the CHIRPS daily dataset dynamically based on the current date.
    Si el desfase con la última fecha disponible es > 2 meses, fuerza una nueva descarga.
    """
    # Parámetros base
    current_date = pd.Timestamp.now()
    current_year = current_date.year
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
    daily_file_name = f"chirps-v2.0.{current_year}.days_p05.nc"
    daily_file_path = os.path.join(INPUT_PATH, daily_file_name)

    def download_file():
        url = base_url + daily_file_name
        resp = requests.get(url, stream=True)
        if resp.status_code == 200:
            with open(daily_file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"✔️ Archivo {daily_file_name} descargado/actualizado.")
        else:
            print(f"❌ Error al descargar {daily_file_name}: HTTP {resp.status_code}")

    # 1) Si no existe, descargar
    if not os.path.exists(daily_file_path):
        print(f"⏬ {daily_file_name} no existe. Descargando...")
        download_file()
    else:
        print(f"🔍 {daily_file_name} ya existe. Comprobando fecha...")

    # 2) Leer la última fecha del dataset
    ds = xr.open_dataset(daily_file_path)
    last_date = pd.to_datetime(ds.time.values[-1])
    print(f"📅 Última fecha disponible: {last_date.date()}")

    # 3) Si el desfase es > 2 meses, volver a descargar
    threshold = current_date - pd.DateOffset(months=2)
    if last_date < threshold:
        print(f"⚠️ Dataset desactualizado (>2 meses). Intentando actualizar...")
        download_file()
        # Releer para verificar
        ds = xr.open_dataset(daily_file_path)
        new_last = pd.to_datetime(ds.time.values[-1])
        print(f"📅 Nueva última fecha: {new_last.date()}")
    else:
        print("✅ El dataset está actualizado (≤ 2 meses de desfase).") 

def process_new_data():
    """
    Procesa dinámicamente los meses faltantes extrayendo datos reales de CHIRPS,
    filtra espacialmente a Boyacá, genera lags correctamente hacia atrás usando
    un historial de 48+36 meses para garantizar que no queden NaNs, crea variables
    cíclicas y guarda solo los últimos 48 meses.
    """
    import warnings
    from dask.diagnostics import ProgressBar
    from dask.array.core import PerformanceWarning

    # Silenciar warnings innecesarios
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="The specified chunks separate")
    warnings.filterwarnings("ignore", category=FutureWarning,
                            message=".*'M' is deprecated.*")
    warnings.filterwarnings("ignore", category=PerformanceWarning)

    # 1) Cargar el dataset histórico procesado
    current_path = os.path.join(OUTPUT_PATH, CURRENT_DATASET)
    ds = xr.open_dataset(current_path, chunks={})
    # Filtrar espacialmente a Boyacá
    ds = ds.sel(
        latitude=slice(LAT_BOYACA_MIN, LAT_BOYACA_MAX),
        longitude=slice(LON_BOYACA_MIN, LON_BOYACA_MAX)
    )
    # Rechunk para un mes por bloque, todo el espacio junto
    ds = ds.chunk({'time': 1, 'latitude': -1, 'longitude': -1})

    # 2) Añadir dinámicamente cualquier mes faltante
    latest, now = pd.to_datetime(ds.time.values[-1]), pd.Timestamp.now()
    if latest < now - pd.DateOffset(months=1):
        missing = pd.date_range(start=latest + pd.DateOffset(months=1),
                                end=now - pd.DateOffset(months=1),
                                freq="ME")
        for m in missing:
            daily_file = f"chirps-v2.0.{m.year}.days_p05.nc"
            daily_path = os.path.join(INPUT_PATH, daily_file)
            print(f"Adding month {m.strftime('%Y-%m')}…")
            with xr.open_dataset(daily_path, chunks={'time': -1, 'latitude': -1, 'longitude': -1}) as dailyds:
                dailyds = dailyds.sel(
                    latitude=slice(LAT_BOYACA_MIN, LAT_BOYACA_MAX),
                    longitude=slice(LON_BOYACA_MIN, LON_BOYACA_MAX)
                )
                sel = dailyds['precip'].sel(time=slice(m.replace(day=1), m))
                total = sel.resample(time='ME').sum(dim='time')
                max_d = sel.resample(time='ME').max(dim='time')
                min_d = sel.resample(time='ME').min(dim='time')
                std_d = sel.resample(time='ME').std(dim='time')

            month_ds = xr.Dataset({
                "total_precipitation":     total,
                "max_daily_precipitation": max_d,
                "min_daily_precipitation": min_d,
                "daily_precipitation_std": std_d,
            })
            stat = ds[['elevation','slope','aspect','cluster_elevation']].expand_dims(time=[m])
            month_ds = month_ds.merge(stat)
            ds = xr.concat([ds, month_ds], dim="time")

    # 1) Asegurar orden
    ds = ds.sortby("time")

    # 2) Definir lags y ventana
    lags   = [1, 2, 3, 4, 12, 24, 36]
    WINDOW = 48
    MAX_LAG = max(lags)
    HISTORY = WINDOW + MAX_LAG

    # 3) Recortar historia para tener suficiente para todos los lags
    if len(ds.time) < HISTORY:
        raise RuntimeError(f"Se necesitan al menos {HISTORY} meses; hay {len(ds.time)}")
    ds = ds.isel(time=slice(-HISTORY, None))

    # 4) Construir lags **hacia atrás**
    #    En fecha t, lag_n = valor de total_precipitation en t - n
    for lag in lags:
        ds[f"lag_{lag}"] = ds["total_precipitation"].shift(time=lag)

    # 5) Eliminar las filas donde alguno de los lags es NaN:
    #    esto sólo afecta a los primeros MAX_LAG de ds, que estamos descartando ahora
    ds = ds.dropna(dim="time", subset=[f"lag_{lag}" for lag in lags])

    # 6) Crear variables cíclicas
    ds["month_sin"] = np.sin(2*np.pi*ds.time.dt.month/12)
    ds["month_cos"] = np.cos(2*np.pi*ds.time.dt.month/12)
    ds["doy_sin"]   = np.sin(2*np.pi*ds.time.dt.dayofyear/365)
    ds["doy_cos"]   = np.cos(2*np.pi*ds.time.dt.dayofyear/365)

    # 7) Recortar a los últimos 48 meses, ya con lags completos
    ds = ds.isel(time=slice(-WINDOW, None))

    # 8) Guardar con chunking
    ds = ds.chunk({'time': 1, 'latitude': -1, 'longitude': -1})
    final_date = pd.to_datetime(ds.time.values[-1]).strftime("%Y%m")
    outp = os.path.join(OUTPUT_PATH, OUTPUT_TEMPLATE.format(final_date))
    with ProgressBar():
        ds.to_netcdf(outp, compute=True)
    ds.close()

    print(f"✅ Dataset final listo: {os.path.basename(outp)}")

def validate_output_dataset(output_path=None, expected_time_length=48, required_columns=None):
    """
    Validates the output dataset and compares original values against their lags
    with both tabular output for the final month and time-series overlay plots.
    """
    # Required columns
    if required_columns is None:
        required_columns = [
            "time", "latitude", "longitude",
            "total_precipitation", "max_daily_precipitation", "min_daily_precipitation",
            "daily_precipitation_std", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "elevation", "slope", "aspect", "cluster_elevation"
        ] + [f"lag_{lag}" for lag in [1, 2, 3, 4, 12, 24, 36]]

    # Determine latest file if not provided
    if output_path is None:
        files = [f for f in os.listdir(OUTPUT_PATH)
                 if f.startswith("complete_dataset_with_features_with_clusters_elevation_") and f.endswith(".nc")]
        if not files:
            print("Error: No output datasets found.")
            return False
        files.sort(key=lambda x: pd.to_datetime(x.split("_")[-1].split(".")[0], format="%Y%m"))
        output_path = os.path.join(OUTPUT_PATH, files[-1])

    # Load dataset
    ds = xr.open_dataset(output_path)

    # 1) Validate time length
    if len(ds.time) != expected_time_length:
        print(f"Error: time length {len(ds.time)} != {expected_time_length}")
        return False

    # 2) Validate required columns
    missing = [c for c in required_columns if c not in ds.variables and c not in ds.coords]
    if missing:
        print(f"Error: missing columns {missing}")
        return False

    # 3) Build DataFrame of original vs lags and compute lag dates
    lags = [1, 2, 3, 4, 12, 24, 36]
    df = ds[['total_precipitation'] + [f'lag_{lag}' for lag in lags]] \
           .to_dataframe().reset_index()
    for lag in lags:
        df[f'lag_{lag}_date'] = df['time'] - pd.DateOffset(months=lag)

    # 4) Tabular comparison for the final month
    final_time = df['time'].max()
    comp_cols = ['time', 'total_precipitation'] + sum([[f'lag_{lag}_date', f'lag_{lag}'] for lag in lags], [])
    print("\nComparison for final month:")
    print(df[df.time == final_time][comp_cols].to_string(index=False))

    # 5) Time-series overlay at a representative location (center of grid)
    lat0 = float(ds.latitude[len(ds.latitude)//2])
    lon0 = float(ds.longitude[len(ds.longitude)//2])
    ts = ds['total_precipitation'].sel(latitude=lat0, longitude=lon0)

    plt.figure(figsize=(12,6))
    ds['total_precipitation'].sel(latitude=lat0, longitude=lon0).plot(
        label='Original', marker='o'
    )
    for lag in lags:
        ds[f'lag_{lag}'] \
        .sel(latitude=lat0, longitude=lon0) \
        .plot(label=f'Lag-{lag}', marker='x')
    plt.title(f"Original and Lags at ({lat0:.3f}, {lon0:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nValidation complete: table and overlay plots generated.")
    return True


def verify_updated_dataset(new_filename=None, last_date_last_year=None):
    # Determine the latest dataset by date if new_filename is not provided
    if new_filename is None:
        output_files = [
            f for f in os.listdir(OUTPUT_PATH) if f.startswith("complete_dataset_with_features_with_clusters_elevation_") and f.endswith(".nc")
        ]
        if not output_files:
            print("Error: No output datasets found.")
            return

        # Sort files by date (yyyymm in the filename) and select the latest
        output_files.sort(key=lambda x: pd.to_datetime(x.split("_")[-1].split(".")[0], format="%Y%m"))
        new_filename = output_files[-1]

    # Load the updated dataset
    updated_dataset_path = os.path.join(OUTPUT_PATH, new_filename)
    updated_ds = xr.open_dataset(updated_dataset_path)

    # Get the last date in the updated dataset
    updated_last_date = pd.to_datetime(updated_ds.time.values[-1])
    print(f"Last date in the updated dataset: {updated_last_date}")

    # Compare with the last date of the last year in the new data if provided
    if last_date_last_year is not None:
        if updated_last_date == last_date_last_year:
            print("The updated dataset includes all necessary changes.")
        else:
            print("Warning: The updated dataset does not include all necessary changes.")
    else:
        print("No reference date provided for comparison.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline de construcción del dataset con lags y opcional actualización CHIRPS"
    )
    parser.add_argument(
        "--enable-chirps-update",
        action="store_true",
        help="Si se indica, hará update_chirps_daily() cuando el desfase > 2 meses"
    )
    args = parser.parse_args()

    download_missing_data()  # siempre bajas el histórico local

    if args.enable_chirps_update:
        update_chirps_daily()
    else:
        print("⚙️ CHIRPS-update desactivado (usa --enable-chirps-update para activarlo)")

    process_new_data()
    verify_updated_dataset()
    validate_output_dataset()