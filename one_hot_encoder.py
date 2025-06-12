import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Cargar dataset
ds_path = '/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/output/complete_dataset_with_features_with_clusters_elevation_with_windows.nc'
ds = xr.open_dataset(ds_path)

# Extraer cluster_elevation
cluster_data = ds['cluster_elevation'].values
print(f"Forma original de cluster_elevation: {cluster_data.shape}")
print(f"Tipo de datos: {cluster_data.dtype}")
print(f"Valores únicos: {np.unique(cluster_data)}")

# Definir función para one-hot encoding espacial
def spatial_one_hot_encoding(data, categories=None):
    """
    Convierte una matriz categórica espacial en matrices one-hot manteniendo dimensiones espaciales.
    
    Args:
        data: Array numpy con datos categóricos (height, width)
        categories: Lista de categorías. Si es None, se extraen automáticamente
        
    Returns:
        Dictionary con una matriz por categoría, cada una con forma (height, width)
    """
    if categories is None:
        categories = np.unique(data)
    
    height, width = data.shape
    encoded = {}
    
    for category in categories:
        # Crear máscara binaria donde 1=coincide con la categoría, 0=resto
        mask = (data == category).astype(np.float32)
        encoded[f"cluster_{category}"] = mask
    
    return encoded, categories

# Aplicar encoding espacial
encoded_clusters, categories = spatial_one_hot_encoding(cluster_data)

# Mostrar forma e información
print("\nOne-hot encoding espacial:")
for category, encoded_data in encoded_clusters.items():
    print(f"- {category}: forma {encoded_data.shape}, valores {np.unique(encoded_data)}")

# Visualizar las máscaras para cada categoría
fig, axes = plt.subplots(1, len(categories), figsize=(15, 5))

for i, category in enumerate(categories):
    cat_key = f"cluster_{category}"
    im = axes[i].imshow(encoded_clusters[cat_key], cmap='viridis')
    axes[i].set_title(f'Cluster: {category}')
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.savefig('cluster_one_hot_encoding.png')
print("\nImagen guardada como 'cluster_one_hot_encoding.png'")

# Demostrar cómo integrar estas matrices en el proceso de build_dataloaders
print("\nEjemplo de integración en build_dataloaders:")
print("""
# En build_dataloaders:
if 'cluster_elevation' in dataset and 'cluster_elevation' in features:
    # Obtener datos categóricos
    cluster_data = dataset['cluster_elevation'].values
    
    # Eliminar la característica original de la lista
    available_features.remove('cluster_elevation')
    
    # Aplicar one-hot encoding espacial
    encoded_clusters, _ = spatial_one_hot_encoding(cluster_data)
    
    # Añadir cada categoría como una característica separada
    for category, encoded_data in encoded_clusters.items():
        # Expandir a dimensión temporal
        expanded = np.broadcast_to(
            encoded_data[np.newaxis, :, :],
            (time_steps, spatial_height, spatial_width)
        )
        
        # Guardar como nueva característica
        feature_data[category] = expanded
        feature_shapes[category] = expanded.shape
        available_features.append(category)
        
    info_print(f"✅ Variable categórica 'cluster_elevation' convertida a {len(encoded_clusters)} matrices one-hot")
""")

# Mostrar cómo verificar NaNs de forma segura
print("\nManejo seguro de NaNs (implementado):")
print("""
def safe_has_nan(arr):
    try:
        if hasattr(arr, 'dtype') and np.issubdtype(arr.dtype, np.number):
            return np.isnan(arr).any()
        return False
    except (TypeError, ValueError):
        return False
        
# Uso:
if safe_has_nan(feature_slice):
    feature_slice = np.nan_to_num(feature_slice, nan=0.0)
""") 