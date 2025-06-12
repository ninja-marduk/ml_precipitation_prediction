import os
import sys
import numpy as np
import pandas as pd
import logging
import xarray as xr
from typing import List, Dict, Tuple, Union, Optional, Any

# Importar feature_validator para usar sus definiciones y funciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.feature_validator import (
        BASE_FEATURES, ELEVATION_CLUSTER_FEATURES, 
        validate_features_for_model, spatial_one_hot_encoding,
        safe_is_nan, safe_has_nan
    )
except ImportError:
    # Definiciones fallback si no se puede importar
    BASE_FEATURES = [
        'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
        'elevation', 'slope', 'aspect'
    ]
    ELEVATION_CLUSTER_FEATURES = [
        'cluster_high', 'cluster_medium', 'cluster_low'
    ]
    
    def safe_is_nan(x):
        """Verifica si un valor es NaN de manera segura."""
        try:
            return np.isnan(x)
        except:
            return False
            
    def safe_has_nan(arr):
        """Verifica si un array contiene NaN de manera segura."""
        try:
            return np.isnan(arr).any()
        except:
            return False
            
    def spatial_one_hot_encoding(data, categories=None):
        """Codificación one-hot para datos espaciales."""
        if categories is None:
            categories = np.unique(data)
        
        height, width = data.shape
        encoded = {}
        
        for category in categories:
            # Crear máscara binaria donde 1=coincide con la categoría, 0=resto
            mask = (data == category).astype(np.float32)
            encoded[f"cluster_{category}"] = mask
        
        return encoded, categories

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transform_cluster_elevation(dataset):
    """
    Transforma la variable categórica 'cluster_elevation' en variables one-hot.
    
    Args:
        dataset: Dataset de xarray con la variable 'cluster_elevation'
        
    Returns:
        dataset: Dataset con variables cluster_high, cluster_medium y cluster_low añadidas
    """
    if 'cluster_elevation' not in dataset:
        logger.warning("⚠️ No se encontró 'cluster_elevation' en el dataset")
        return dataset
    
    # Obtener datos categóricos
    cluster_data = dataset['cluster_elevation'].values
    
    # Verificar el tipo de dato
    logger.info(f"📊 Tipo de datos cluster_elevation: {cluster_data.dtype}")
    
    try:
        # Aplicar one-hot encoding espacial
        encoded_clusters, categories = spatial_one_hot_encoding(cluster_data)
        
        # Verificar si se generaron todas las categorías esperadas
        expected_categories = ['high', 'medium', 'low']
        missing_categories = [cat for cat in expected_categories 
                            if f"cluster_{cat}" not in encoded_clusters]
        
        if missing_categories:
            logger.warning(f"⚠️ Categorías faltantes en cluster_elevation: {missing_categories}")
            # Crear categorías vacías para las faltantes
            for cat in missing_categories:
                encoded_clusters[f"cluster_{cat}"] = np.zeros_like(cluster_data, dtype=np.float32)
        
        # Añadir variables one-hot al dataset
        new_dataset = dataset.copy()
        for category, encoded_data in encoded_clusters.items():
            new_dataset[category] = (('latitude', 'longitude'), encoded_data)
            
        logger.info(f"✅ Variable 'cluster_elevation' transformada a {len(encoded_clusters)} variables one-hot")
        return new_dataset
    
    except Exception as e:
        logger.error(f"❌ Error al transformar cluster_elevation: {e}")
        return dataset

def improved_build_dataloaders(
    dataset: xr.Dataset, 
    features: List[str],
    time_window: int = 12,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    require_full_grid: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, tuple], List[str]]:
    """
    Versión mejorada de build_dataloaders que maneja correctamente la transformación
    de cluster_elevation a características one-hot (cluster_high, cluster_medium, cluster_low).
    
    Args:
        dataset: Dataset de xarray con las variables
        features: Lista de características solicitadas
        time_window: Tamaño de la ventana temporal
        batch_size: Tamaño del batch
        shuffle: Si se debe barajar los datos
        drop_last: Si se debe descartar el último batch si es incompleto
        require_full_grid: Si se requiere que todas las características tengan la estructura espacial completa
        
    Returns:
        feature_data: Diccionario con los datos de cada característica
        feature_shapes: Diccionario con la forma de cada característica
        available_features: Lista de características disponibles
    """
    logger.info(f"📊 Construyendo dataloaders con {len(features)} características solicitadas")
    
    # Inicializar contenedores
    feature_data = {}
    feature_shapes = {}
    available_features = []
    
    # Verificar si se requieren características de cluster (one-hot)
    needs_cluster_transform = any(cf in features for cf in ELEVATION_CLUSTER_FEATURES)
    
    # Si se requieren características de cluster, transformar cluster_elevation
    if needs_cluster_transform:
        # Verificar si ya existen las características one-hot en el dataset
        cluster_features_in_dataset = [f for f in ELEVATION_CLUSTER_FEATURES if f in dataset]
        
        # Si no todas las características están en el dataset pero está cluster_elevation
        if len(cluster_features_in_dataset) < len(ELEVATION_CLUSTER_FEATURES) and 'cluster_elevation' in dataset:
            logger.info("⚠️ Transformando 'cluster_elevation' a variables one-hot")
            dataset = transform_cluster_elevation(dataset)
        elif 'cluster_elevation' not in dataset and len(cluster_features_in_dataset) == 0:
            logger.warning("❌ No se encontró 'cluster_elevation' ni características one-hot en el dataset")
    
    # Procesar todas las características
    for feature in features:
        if feature in dataset:
            # Manejo seguro de NaN en las características
            feature_array = dataset[feature].values
            
            # Verificar NaN de manera segura
            if safe_has_nan(feature_array):
                logger.warning(f"⚠️ La característica '{feature}' contiene valores NaN")
                
            # Procesamiento normal para características numéricas
            feature_data[feature] = feature_array
            feature_shapes[feature] = feature_array.shape
            available_features.append(feature)
        else:
            logger.warning(f"⚠️ Característica '{feature}' no encontrada en el dataset")
    
    # Verificar si todas las características tienen la misma estructura espacial
    if require_full_grid:
        shape_check = {}
        for feature, shape in feature_shapes.items():
            shape_key = f"{shape[0]}x{shape[1]}" if len(shape) >= 2 else "scalar"  # Formato "filas x columnas"
            if shape_key not in shape_check:
                shape_check[shape_key] = []
            shape_check[shape_key].append(feature)
        
        if len(shape_check) > 1:
            logger.warning("⚠️ No todas las características tienen la misma estructura espacial:")
            for shape_key, features_list in shape_check.items():
                logger.warning(f"   - Estructura {shape_key}: {len(features_list)} características")
    
    # Verificar que características solicitadas estén disponibles
    missing_features = [f for f in features if f not in available_features]
    if missing_features:
        logger.warning(f"⚠️ Características solicitadas no disponibles: {missing_features}")
    
    logger.info(f"✅ Proceso completado. {len(available_features)} características disponibles.")
    return feature_data, feature_shapes, available_features


def test_improved_dataloader():
    """
    Función para probar la mejora del dataloader con un dataset sintético
    """
    # Crear un dataset sintético
    time = pd.date_range('2000-01-01', periods=100, freq='M')
    lat = np.linspace(0, 10, 20)
    lon = np.linspace(0, 10, 20)
    
    # Variables base (todas numéricas)
    data_vars = {}
    for feature in BASE_FEATURES:
        if feature in ['month_sin', 'month_cos', 'doy_sin', 'doy_cos']:
            # Variables cíclicas
            data_vars[feature] = (('time', 'lat', 'lon'), np.random.random((len(time), len(lat), len(lon))))
        else:
            # Variables normales
            data_vars[feature] = (('time', 'lat', 'lon'), np.random.random((len(time), len(lat), len(lon))))
    
    # Variable categórica cluster_elevation
    # Crear matriz donde cada celda tiene valor 'high', 'medium' o 'low'
    cluster_data = np.random.choice(['high', 'medium', 'low'], size=(len(lat), len(lon)))
    data_vars['cluster_elevation'] = (('lat', 'lon'), cluster_data)
    
    # Crear el dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': time,
            'lat': lat,
            'lon': lon
        }
    )
    
    # Probar el dataloader con diferentes conjuntos de características
    test_cases = [
        # Caso 1: Sólo características base
        BASE_FEATURES,
        # Caso 2: Características base + cluster_elevation (se debe transformar)
        BASE_FEATURES + ['cluster_elevation'],
        # Caso 3: Características base + características de cluster ya transformadas
        BASE_FEATURES + ELEVATION_CLUSTER_FEATURES,
        # Caso 4: Todo junto (debería eliminar duplicados)
        BASE_FEATURES + ['cluster_elevation'] + ELEVATION_CLUSTER_FEATURES
    ]
    
    for i, features in enumerate(test_cases):
        print(f"\n🧪 Caso de prueba {i+1}: {len(features)} características")
        feature_data, feature_shapes, available_features = improved_build_dataloaders(
            ds, features, time_window=12, batch_size=32
        )
        
        # Verificar resultados
        print(f"   ✅ Características disponibles: {len(available_features)}")
        cluster_features_found = [f for f in available_features if f.startswith('cluster_')]
        print(f"   ✅ Características de cluster: {cluster_features_found}")
        
        # Verificar dimensiones
        spatial_shapes = set(str(shape) for shape in feature_shapes.values())
        print(f"   ✅ Estructuras encontradas: {spatial_shapes}")
    
    print("\n✅ Pruebas completadas")
    
    # Probar la transformación directa
    print("\n🧪 Probando transformación directa de cluster_elevation")
    ds_transformed = transform_cluster_elevation(ds)
    for feature in ELEVATION_CLUSTER_FEATURES:
        if feature in ds_transformed:
            print(f"   ✅ Variable {feature} creada correctamente")
        else:
            print(f"   ❌ Error: Variable {feature} no creada")
    
    return ds_transformed

if __name__ == "__main__":
    test_improved_dataloader() 