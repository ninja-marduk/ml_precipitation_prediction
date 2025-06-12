import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import logging
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar utilidades
from utils.feature_validator import (
    BASE_FEATURES, ELEVATION_CLUSTER_FEATURES, LAG_FEATURES, IMF_FEATURES,
    validate_features_for_model, MODEL_FEATURE_REQUIREMENTS
)
from utils.dataloader_utils import improved_build_dataloaders
from models.model_factory import create_model

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/models_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_experiment(model_type, dataset_path, output_dir, config=None):
    """
    Ejecuta un experimento completo con un modelo específico.
    
    Args:
        model_type: Tipo de modelo a utilizar
        dataset_path: Ruta al dataset
        output_dir: Directorio de salida
        config: Configuración adicional
        
    Returns:
        Resultados del experimento
    """
    logger.info(f"🚀 Iniciando experimento con modelo: {model_type}")
    logger.info(f"   Dataset: {dataset_path}")
    logger.info(f"   Salida: {output_dir}")
    
    # Cargar dataset
    logger.info("📊 Cargando dataset...")
    try:
        ds = xr.open_dataset(dataset_path)
        logger.info(f"   ✅ Dataset cargado con éxito. Dimensiones: {ds.dims}")
    except Exception as e:
        logger.error(f"   ❌ Error al cargar dataset: {e}")
        return None
    
    # Determinar características necesarias según el modelo
    if model_type not in MODEL_FEATURE_REQUIREMENTS:
        logger.error(f"❌ Modelo desconocido: {model_type}")
        return None
    
    required_features = MODEL_FEATURE_REQUIREMENTS[model_type]['required']
    logger.info(f"📋 El modelo requiere {len(required_features)} características")
    
    # Construir dataloaders con la versión mejorada
    logger.info("🔄 Construyendo dataloaders...")
    feature_data, feature_shapes, available_features = improved_build_dataloaders(
        ds, required_features, time_window=12, batch_size=32
    )
    
    # Verificar que todas las características requeridas estén disponibles
    is_valid, missing_features = validate_features_for_model(model_type, available_features)
    if not is_valid:
        logger.error(f"❌ No se puede continuar. Faltan características: {missing_features}")
        return None
    
    # Definir formas de entrada/salida según las características disponibles
    n_features = len(available_features)
    time_window = 12  # Se puede configurar
    spatial_shape = feature_shapes[available_features[0]][1:]  # Tomar forma espacial de la primera característica
    
    input_shape = (time_window,) + spatial_shape + (n_features,)
    output_shape = spatial_shape + (1,)  # Forma de salida para predicción de precipitación
    
    logger.info(f"📐 Forma de entrada: {input_shape}, Forma de salida: {output_shape}")
    
    # Crear modelo
    logger.info("🏗️ Creando modelo...")
    model = create_model(model_type, available_features, input_shape, output_shape, config)
    
    if model is None:
        logger.error("❌ No se pudo crear el modelo")
        return None
    
    logger.info("✅ Modelo creado correctamente")
    
    # Aquí iría el entrenamiento y evaluación del modelo
    # ...
    
    logger.info("✅ Experimento completado con éxito")
    return {
        "model_type": model_type,
        "features_used": available_features,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "model": model
    }

def run_all_experiments():
    """
    Ejecuta experimentos con todos los modelos disponibles.
    """
    # Dataset sintético para simulación
    dataset_path = "data/output/synthetic_dataset.nc"
    output_dir = "models/output/experiments"
    
    # Crear dataset sintético para pruebas
    create_synthetic_dataset(dataset_path)
    
    # Definir modelos a probar
    models_to_test = [
        'ConvGRU-ED',
        'ConvGRU-ED-KCE',
        'ConvGRU-ED-KCE-PAFC',
        'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA',
        'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA-TopoMask'
    ]
    
    results = {}
    
    for model_type in models_to_test:
        logger.info(f"\n{'='*80}\n🔍 EJECUTANDO EXPERIMENTO CON MODELO: {model_type}\n{'='*80}")
        result = run_experiment(model_type, dataset_path, output_dir)
        results[model_type] = result
        
        # Imprimir resultado
        if result:
            logger.info(f"✅ Experimento con {model_type} completado con éxito")
        else:
            logger.error(f"❌ Experimento con {model_type} falló")
    
    # Resumen final
    logger.info("\n🏁 RESUMEN DE EXPERIMENTOS")
    logger.info(f"{'='*80}")
    for model_type, result in results.items():
        status = "✅ ÉXITO" if result else "❌ FALLO"
        logger.info(f"{status} - {model_type}")
    
    return results

def create_synthetic_dataset(output_path):
    """
    Crea un dataset sintético para pruebas con todas las características necesarias.
    """
    logger.info(f"🔨 Creando dataset sintético en: {output_path}")
    
    # Crear directorios si no existen
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Dimensiones
    time = pd.date_range('2000-01-01', periods=100, freq='M')
    lat = np.linspace(0, 10, 20)
    lon = np.linspace(0, 10, 20)
    
    # Variables
    data_vars = {}
    
    # 1. Variables base
    for feature in BASE_FEATURES:
        if feature in ['month_sin', 'month_cos', 'doy_sin', 'doy_cos']:
            # Variables cíclicas
            data_vars[feature] = (('time', 'lat', 'lon'), np.random.random((len(time), len(lat), len(lon))))
        else:
            # Variables normales
            data_vars[feature] = (('time', 'lat', 'lon'), np.random.random((len(time), len(lat), len(lon))))
    
    # 2. Variable categórica cluster_elevation (high, medium, low)
    cluster_data = np.random.choice(['high', 'medium', 'low'], size=(len(lat), len(lon)))
    data_vars['cluster_elevation'] = (('lat', 'lon'), cluster_data)
    
    # 3. Variables de lag
    for feature in LAG_FEATURES:
        data_vars[feature] = (('time', 'lat', 'lon'), np.random.random((len(time), len(lat), len(lon))))
    
    # 4. Variables IMF
    for feature in IMF_FEATURES:
        data_vars[feature] = (('time', 'lat', 'lon'), np.random.random((len(time), len(lat), len(lon))))
    
    # 5. Variable objetivo (precipitación total)
    data_vars['total_precipitation'] = (('time', 'lat', 'lon'), np.random.random((len(time), len(lat), len(lon))))
    
    # Crear dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': time,
            'lat': lat,
            'lon': lon
        }
    )
    
    # Guardar dataset
    try:
        ds.to_netcdf(output_path)
        logger.info(f"✅ Dataset guardado en: {output_path}")
    except Exception as e:
        logger.error(f"❌ Error al guardar dataset: {e}")

if __name__ == "__main__":
    # Crear directorio de logs si no existe
    os.makedirs("logs", exist_ok=True)
    
    logger.info("🚀 Iniciando script de ejecución de modelos")
    results = run_all_experiments()
    logger.info("✅ Script completado") 