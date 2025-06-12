import os
import sys
import numpy as np
import logging

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar utilidades de validación
from utils.feature_validator import (
    validate_features_for_model, 
    MODEL_FEATURE_REQUIREMENTS
)

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model(model_type, available_features, input_shape, output_shape, config=None):
    """
    Factory para crear diferentes tipos de modelos con validación de características.
    
    Args:
        model_type: Tipo de modelo a crear (ConvGRU-ED, ConvGRU-ED-KCE, etc.)
        available_features: Lista de características disponibles
        input_shape: Forma de entrada al modelo
        output_shape: Forma de salida del modelo
        config: Configuración adicional del modelo
        
    Returns:
        Modelo creado o None si no se pudo crear
    """
    # Validar características necesarias para el modelo
    is_valid, missing_features = validate_features_for_model(model_type, available_features)
    
    if not is_valid:
        logger.error(f"❌ No se puede crear el modelo {model_type}. Faltan características: {missing_features}")
        return None
    
    # Verificar si el modelo requiere características de cluster
    if model_type in MODEL_FEATURE_REQUIREMENTS and MODEL_FEATURE_REQUIREMENTS[model_type].get('uses_cluster', False):
        # Verificar que las características de cluster estén presentes
        cluster_features = ['cluster_high', 'cluster_medium', 'cluster_low']
        missing_clusters = [f for f in cluster_features if f not in available_features]
        
        if missing_clusters:
            logger.error(f"❌ Modelo {model_type} requiere las variables de cluster one-hot: {missing_clusters}")
            return None
        else:
            logger.info(f"✅ Características de cluster validadas para {model_type}")
    
    # Proceder con la creación del modelo según su tipo
    logger.info(f"🔧 Creando modelo: {model_type}")
    
    if model_type == 'ConvGRU-ED':
        # Implementación específica para ConvGRU-ED
        logger.info("✅ Características validadas para modelo base ConvGRU-ED")
        # return create_conv_gru_ed(input_shape, output_shape, config)
        return f"Modelo {model_type} creado con éxito (simulación)"
    
    elif model_type == 'ConvGRU-ED-KCE':
        # Implementación específica para ConvGRU-ED-KCE con validación de cluster
        logger.info("✅ Características validadas para modelo ConvGRU-ED-KCE con K-means Cluster Elevation")
        # return create_conv_gru_ed_kce(input_shape, output_shape, config)
        return f"Modelo {model_type} creado con éxito (simulación)"
    
    elif model_type == 'ConvGRU-ED-KCE-PAFC':
        # Implementación específica para ConvGRU-ED-KCE-PAFC
        logger.info("✅ Características validadas para modelo ConvGRU-ED-KCE-PAFC con Position-Aware Feature Calibration")
        # return create_conv_gru_ed_kce_pafc(input_shape, output_shape, config)
        return f"Modelo {model_type} creado con éxito (simulación)"
    
    elif model_type == 'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA':
        # Implementación específica para AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA
        logger.info("✅ Características validadas para modelo AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA con Multi-Head Attention")
        # return create_ae_fusion_convgru_ed_kce_pafc_mha(input_shape, output_shape, config)
        return f"Modelo {model_type} creado con éxito (simulación)"
    
    elif model_type == 'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA-TopoMask':
        # Implementación específica para AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA-TopoMask
        logger.info("✅ Características validadas para modelo AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA-TopoMask")
        # return create_ae_fusion_convgru_ed_kce_pafc_mha_topomask(input_shape, output_shape, config)
        return f"Modelo {model_type} creado con éxito (simulación)"
    
    else:
        logger.error(f"❌ Tipo de modelo desconocido: {model_type}")
        return None

def test_model_creation():
    """
    Función de prueba para validar la creación de modelos con diferentes conjuntos de características
    """
    # Test case 1: Modelo base con características mínimas
    print("\n🧪 Test case 1: Modelo base con características base")
    basic_features = [
        'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
        'elevation', 'slope', 'aspect'
    ]
    model = create_model('ConvGRU-ED', basic_features, (12, 20, 20, 12), (20, 20, 1))
    print(f"   Resultado: {'✅ Éxito' if model else '❌ Fallo'}")
    
    # Test case 2: Modelo KCE sin clusters
    print("\n🧪 Test case 2: Modelo KCE sin variables de cluster")
    model = create_model('ConvGRU-ED-KCE', basic_features, (12, 20, 20, 12), (20, 20, 1))
    print(f"   Resultado: {'✅ Éxito' if model else '❌ Fallo'}")
    
    # Test case 3: Modelo KCE con clusters
    print("\n🧪 Test case 3: Modelo KCE con variables de cluster")
    cluster_features = basic_features + ['cluster_high', 'cluster_medium', 'cluster_low']
    model = create_model('ConvGRU-ED-KCE', cluster_features, (12, 20, 20, 15), (20, 20, 1))
    print(f"   Resultado: {'✅ Éxito' if model else '❌ Fallo'}")
    
    # Test case 4: Modelo avanzado sin todas las características
    print("\n🧪 Test case 4: Modelo avanzado sin todas las características")
    model = create_model('AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA', cluster_features, (12, 20, 20, 15), (20, 20, 1))
    print(f"   Resultado: {'✅ Éxito' if model else '❌ Fallo'}")
    
    # Test case 5: Modelo avanzado con todas las características
    print("\n🧪 Test case 5: Modelo avanzado con todas las características")
    full_features = cluster_features + [
        'total_precipitation_lag1', 'total_precipitation_lag2', 'total_precipitation_lag12',
        'CEEMDAN_imf_1', 'CEEMDAN_imf_2', 'CEEMDAN_imf_3', 'CEEMDAN_imf_4', 
        'CEEMDAN_imf_5', 'CEEMDAN_imf_6', 'CEEMDAN_imf_7', 'CEEMDAN_imf_8',
        'TVFEMD_imf_1', 'TVFEMD_imf_2', 'TVFEMD_imf_3', 'TVFEMD_imf_4',
        'TVFEMD_imf_5', 'TVFEMD_imf_6', 'TVFEMD_imf_7', 'TVFEMD_imf_8'
    ]
    model = create_model('AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA', full_features, (12, 20, 20, 34), (20, 20, 1))
    print(f"   Resultado: {'✅ Éxito' if model else '❌ Fallo'}")
    
    print("\n✅ Pruebas de creación de modelos completadas")

if __name__ == "__main__":
    test_model_creation() 