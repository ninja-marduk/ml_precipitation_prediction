import xarray as xr
import numpy as np
import pandas as pd

# Cargar dataset
ds_path = '/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/output/complete_dataset_with_features_with_clusters_elevation_with_windows.nc'
ds = xr.open_dataset(ds_path)

# Verificar si cluster_elevation está en las variables
print("Variables en el dataset:")
for i, var in enumerate(ds.data_vars):
    print(f"{i+1}. {var}")

# Analizar cluster_elevation si existe
if 'cluster_elevation' in ds.data_vars:
    cluster_data = ds['cluster_elevation'].values
    
    # Verificar tipo de datos
    print(f"\ncluster_elevation info:")
    print(f"- Tipo de datos: {cluster_data.dtype}")
    print(f"- Forma: {cluster_data.shape}")
    
    # Valores únicos
    unique_values = np.unique(cluster_data)
    print(f"- Valores únicos: {unique_values}")
    print(f"- Cantidad de valores únicos: {len(unique_values)}")
    
    # Verificar si se puede aplicar isnan
    try:
        has_nans = np.isnan(cluster_data).any()
        print(f"- ¿Tiene NaNs?: {has_nans}")
    except Exception as e:
        print(f"- Error al verificar NaNs: {e}")
else:
    print("\nNo se encontró 'cluster_elevation' en el dataset")

# Comprobar si hay otras variables categóricas
print("\nBuscando otras posibles variables categóricas...")
for var_name in ds.data_vars:
    var_data = ds[var_name].values
    dtype = var_data.dtype
    
    # Si es objeto o cadena, probablemente sea categórica
    if dtype.kind in ['O', 'S', 'U']:
        print(f"- {var_name}: tipo {dtype}")
    
    # Si es entero pero tiene pocos valores únicos, podría ser categórica
    elif dtype.kind in ['i', 'u']:
        unique_vals = np.unique(var_data)
        if len(unique_vals) < 20:  # Umbral arbitrario para considerar como posible categórica
            print(f"- {var_name}: tipo {dtype}, valores únicos: {len(unique_vals)} - {unique_vals}")
            
            # Intentar aplicar isnan
            try:
                if np.isnan(var_data).any():
                    print(f"  - Contiene NaNs")
            except Exception as e:
                print(f"  - Error al verificar NaNs: {e}") 