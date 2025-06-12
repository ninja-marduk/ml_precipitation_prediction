import xarray as xr
import numpy as np
import pandas as pd

# Paths a datasets
main_ds_path = '/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/output/complete_dataset_with_features_with_clusters_elevation_with_windows.nc'
ceemdan_path = '/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/models/output/features_CEEMDAN.nc'
tvfemd_path = '/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/models/output/features_TVFEMD.nc'

# Cargar datasets
print("Cargando datasets...")
ds = xr.open_dataset(main_ds_path)
try:
    ds_ceemdan = xr.open_dataset(ceemdan_path)
    ceemdan_loaded = True
except Exception as e:
    print(f"Error al cargar CEEMDAN: {e}")
    ceemdan_loaded = False
    
try:
    ds_tvfemd = xr.open_dataset(tvfemd_path)
    tvfemd_loaded = True
except Exception as e:
    print(f"Error al cargar TVFEMD: {e}")
    tvfemd_loaded = False

# Información básica del dataset principal
print(f"\n==== DATASET PRINCIPAL ====")
print(f"Dimensiones: {ds.dims}")
print(f"Total de variables: {len(ds.data_vars)}")
time_values = pd.to_datetime(ds.time.values)
print(f"Rango temporal: {time_values.min()} - {time_values.max()}")
print(f"Total de pasos temporales: {len(time_values)}")

# Analizar NaNs en la precipitación
precip_nans = np.isnan(ds.total_precipitation.values).sum()
total_precip_points = ds.total_precipitation.size
print(f"\nVariable total_precipitation:")
print(f"- NaNs: {precip_nans} de {total_precip_points} ({100*precip_nans/total_precip_points:.2f}%)")

# Analizar NaNs en las variables de lag
lag_vars = [v for v in ds.data_vars if 'lag' in v]
print(f"\nVariables de lag encontradas: {lag_vars}")

for var in lag_vars:
    var_nans = np.isnan(ds[var].values).sum()
    total_var_points = ds[var].size
    print(f"\nVariable {var}:")
    print(f"- NaNs: {var_nans} de {total_var_points} ({100*var_nans/total_var_points:.2f}%)")
    
    # Analizar NaNs por tiempo en los primeros y últimos meses
    if var_nans > 0:
        nan_counts = np.isnan(ds[var].values).sum(axis=(1, 2))  # Sumar por lat/lon
        print("NaNs por paso temporal (primeros 12 pasos):")
        for i, (date, count) in enumerate(zip(time_values[:12], nan_counts[:12])):
            print(f"  {i}: {date.strftime('%Y-%m-%d')} - {count} NaNs")
            
        print("\nNaNs por paso temporal (últimos 6 pasos):")
        for i, (date, count) in enumerate(zip(time_values[-6:], nan_counts[-6:]), len(time_values)-6):
            print(f"  {i}: {date.strftime('%Y-%m-%d')} - {count} NaNs")

# Analizar variables IMF CEEMDAN
if ceemdan_loaded:
    print(f"\n==== DATASET CEEMDAN ====")
    print(f"Variables: {list(ds_ceemdan.data_vars)}")
    print(f"Dimensiones: {ds_ceemdan.dims}")
    
    # Analizar NaNs en variables IMF
    for var in list(ds_ceemdan.data_vars)[:3]:  # Primeras 3 variables
        var_nans = np.isnan(ds_ceemdan[var].values).sum()
        total_var_points = ds_ceemdan[var].size
        print(f"\nVariable {var}:")
        print(f"- NaNs: {var_nans} de {total_var_points} ({100*var_nans/total_var_points:.2f}%)")
        
        # Si hay NaNs, ver distribución
        if var_nans > 0:
            nan_counts = np.isnan(ds_ceemdan[var].values).sum(axis=(1, 2))
            nan_months = [i for i, count in enumerate(nan_counts) if count > 0]
            print(f"- Meses con NaNs: {nan_months[:10]}..." if len(nan_months) > 10 else f"- Meses con NaNs: {nan_months}")

# Analizar variables IMF TVFEMD
if tvfemd_loaded:
    print(f"\n==== DATASET TVFEMD ====")
    print(f"Variables: {list(ds_tvfemd.data_vars)}")
    print(f"Dimensiones: {ds_tvfemd.dims}")
    
    # Analizar NaNs en variables IMF
    for var in list(ds_tvfemd.data_vars)[:3]:  # Primeras 3 variables
        var_nans = np.isnan(ds_tvfemd[var].values).sum()
        total_var_points = ds_tvfemd[var].size
        print(f"\nVariable {var}:")
        print(f"- NaNs: {var_nans} de {total_var_points} ({100*var_nans/total_var_points:.2f}%)")
        
        # Si hay NaNs, ver distribución
        if var_nans > 0:
            nan_counts = np.isnan(ds_tvfemd[var].values).sum(axis=(1, 2))
            nan_months = [i for i, count in enumerate(nan_counts) if count > 0]
            print(f"- Meses con NaNs: {nan_months[:10]}..." if len(nan_months) > 10 else f"- Meses con NaNs: {nan_months}")

print("\n==== RESUMEN ====")
print(f"- Dataset principal: {len(ds.data_vars)} variables")
if ceemdan_loaded:
    print(f"- Dataset CEEMDAN: {len(ds_ceemdan.data_vars)} variables")
if tvfemd_loaded:
    print(f"- Dataset TVFEMD: {len(ds_tvfemd.data_vars)} variables")

print("\nPeríodos con mayor proporción de NaNs:")
print("1. Primeros meses (1981): variables de lag con retrasos de 1-36 meses")
print("2. Variables IMF: depende de la descomposición") 