import os
import sys
import numpy as np
import pandas as pd
import logging

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Definir constantes para las características
BASE_FEATURES = [
    'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
    'elevation', 'slope', 'aspect'
]

# Ahora usamos directamente los nombres de las características one-hot encoding
# que se generarán en build_dataloaders para cluster_elevation
ELEVATION_CLUSTER_FEATURES = [
    'cluster_high',      # Cluster de elevación alta (one-hot)
    'cluster_medium',    # Cluster de elevación media (one-hot)
    'cluster_low'        # Cluster de elevación baja (one-hot)
]

# Características de lag
LAG_FEATURES = [
    'total_precipitation_lag1', 'total_precipitation_lag2', 'total_precipitation_lag12'
]

# Características de IMF
IMF_FEATURES = [
    # CEEMDAN IMFs (8)
    'CEEMDAN_imf_1', 'CEEMDAN_imf_2', 'CEEMDAN_imf_3', 'CEEMDAN_imf_4', 
    'CEEMDAN_imf_5', 'CEEMDAN_imf_6', 'CEEMDAN_imf_7', 'CEEMDAN_imf_8',
    # TVFEMD IMFs (8)
    'TVFEMD_imf_1', 'TVFEMD_imf_2', 'TVFEMD_imf_3', 'TVFEMD_imf_4',
    'TVFEMD_imf_5', 'TVFEMD_imf_6', 'TVFEMD_imf_7', 'TVFEMD_imf_8'
]

# Definición de modelos y sus requisitos de características
MODEL_FEATURE_REQUIREMENTS = {
    'ConvGRU-ED': {
        'required': BASE_FEATURES,
        'uses_cluster': False
    },
    'ConvGRU-ED-KCE': {
        'required': BASE_FEATURES + ELEVATION_CLUSTER_FEATURES,
        'uses_cluster': True
    },
    'ConvGRU-ED-KCE-PAFC': {
        'required': BASE_FEATURES + ELEVATION_CLUSTER_FEATURES + LAG_FEATURES,
        'uses_cluster': True
    },
    'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA': {
        'required': BASE_FEATURES + ELEVATION_CLUSTER_FEATURES + LAG_FEATURES + IMF_FEATURES,
        'uses_cluster': True
    },
    'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA-TopoMask': {
        'required': BASE_FEATURES + ELEVATION_CLUSTER_FEATURES + LAG_FEATURES + IMF_FEATURES,
        'uses_cluster': True
    }
}

def validate_features_for_model(model_key, available_features):
    """
    Valida que todas las características necesarias para un modelo estén disponibles.
    
    Args:
        model_key: Clave del modelo a validar
        available_features: Lista de características disponibles
        
    Returns:
        bool: True si todas las características requeridas están disponibles, False en caso contrario
        missing_features: Lista de características faltantes
    """
    if model_key not in MODEL_FEATURE_REQUIREMENTS:
        logger.warning(f"⚠️ Modelo '{model_key}' no definido en los requisitos de características")
        return False, ["modelo_desconocido"]
    
    required_features = MODEL_FEATURE_REQUIREMENTS[model_key]['required']
    missing_features = [f for f in required_features if f not in available_features]
    
    if missing_features:
        logger.warning(f"⚠️ Faltan características requeridas para {model_key}: {missing_features}")
        return False, missing_features
        
    return True, []

def check_cluster_feature_transformation(dataset, features):
    """
    Verifica si se necesita transformar la característica 'cluster_elevation' en características one-hot.
    
    Args:
        dataset: Dataset con las características
        features: Lista de características solicitadas
        
    Returns:
        bool: True si se necesita transformación, False en caso contrario
        needs_cluster: True si alguna característica de cluster está en features
        has_cluster_elevation: True si 'cluster_elevation' está en el dataset
    """
    # Verificar si alguna característica one-hot de cluster está en features
    needs_cluster = any(cf in features for cf in ELEVATION_CLUSTER_FEATURES)
    
    # Verificar si 'cluster_elevation' está en el dataset
    has_cluster_elevation = 'cluster_elevation' in dataset
    
    if needs_cluster and not has_cluster_elevation:
        logger.warning("⚠️ Se requieren características de cluster, pero 'cluster_elevation' no está en el dataset")
        return True, needs_cluster, has_cluster_elevation
    
    if needs_cluster and has_cluster_elevation:
        logger.info("✅ Se transformará 'cluster_elevation' a características one-hot")
        return True, needs_cluster, has_cluster_elevation
    
    return False, needs_cluster, has_cluster_elevation

def spatial_one_hot_encoding(data, categories=None):
    """
    Aplica codificación one-hot a datos categóricos espaciales.
    
    Args:
        data: Array 2D con valores categóricos
        categories: Lista opcional de categorías a considerar
        
    Returns:
        dict: Diccionario con una matriz por categoría, cada una con forma (height, width)
        categories: Lista de categorías utilizadas
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

def safe_is_nan(x):
    """
    Verifica si un valor es NaN de manera segura, comprobando primero el tipo de dato.
    
    Args:
        x: Valor a verificar
        
    Returns:
        bool: True si es NaN, False en caso contrario o si el tipo no admite NaN
    """
    # Verificar si el tipo de dato puede contener NaN
    if hasattr(x, 'dtype'):
        if np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.datetime64):
            return np.isnan(x)
    elif isinstance(x, (int, float)):
        return np.isnan(x)
    
    # Para tipos que no pueden contener NaN (strings, bool, etc.)
    return False

def safe_has_nan(arr):
    """
    Verifica de manera segura si un array contiene valores NaN.
    
    Args:
        arr: Array a verificar
        
    Returns:
        bool: True si contiene algún NaN, False en caso contrario
    """
    # Para arrays NumPy
    if isinstance(arr, np.ndarray):
        # Solo verificar NaN en tipos que pueden contenerlos
        if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.datetime64):
            return np.isnan(arr).any()
        return False
    
    # Para pandas Series o DataFrame
    elif isinstance(arr, (pd.Series, pd.DataFrame)):
        return arr.isna().any().any()
    
    # Para listas u otros iterables
    elif hasattr(arr, '__iter__') and not isinstance(arr, (str, bytes)):
        return any(safe_is_nan(x) for x in arr)
    
    # Para valores individuales
    else:
        return safe_is_nan(arr)

def test_feature_validation():
    """
    Funcion de prueba para validar la deteccion de caracteristicas
    """
    # Crear un dataset de prueba
    test_features = BASE_FEATURES + ['cluster_elevation']
    print(f"[OK] Test dataset con {len(test_features)} caracteristicas")

    # Probar cada modelo
    for model_key, req in MODEL_FEATURE_REQUIREMENTS.items():
        # Verificar modelo base
        if model_key == 'ConvGRU-ED':
            is_valid, missing = validate_features_for_model(model_key, test_features)
            print(f"  [INFO] {model_key}: {'[OK] Valido' if is_valid else '[ERROR] Invalido'}")
            assert is_valid, f"El modelo base deberia ser valido con las caracteristicas base"

        # Verificar modelos que usan cluster
        elif req['uses_cluster']:
            # Primero sin transformar cluster_elevation
            is_valid, missing = validate_features_for_model(model_key, test_features)
            print(f"  [INFO] {model_key} (sin transformar): {'[OK] Valido' if is_valid else '[ERROR] Invalido - Faltan: ' + str(missing)}")
            assert not is_valid, f"El modelo {model_key} deberia ser invalido sin transformar cluster_elevation"

            # Ahora simulando transformacion cluster_elevation -> [cluster_high, cluster_medium, cluster_low]
            transformed_features = test_features + ELEVATION_CLUSTER_FEATURES
            transformed_features.remove('cluster_elevation')  # Eliminar feature original
            is_valid, missing = validate_features_for_model(model_key, transformed_features)
            print(f"  [INFO] {model_key} (transformado): {'[OK] Valido' if is_valid else '[ERROR] Invalido - Faltan: ' + str(missing)}")

            # Todavia faltaran caracteristicas para modelos avanzados
            if model_key in ['ConvGRU-ED-KCE-PAFC', 'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA', 'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA-TopoMask']:
                assert not is_valid, f"El modelo {model_key} deberia requerir mas caracteristicas"

    print("\n[OK] Validacion de caracteristicas completada")

if __name__ == "__main__":
    test_feature_validation() 