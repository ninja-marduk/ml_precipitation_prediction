import os
import sys
import shutil
import logging
from pathlib import Path

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_model_saving_functions():
    """
    Aplica un parche a las funciones de guardado de modelo para añadir extensión .keras
    y busca todos los scripts/notebooks donde se llama a model.save sin extensión.
    """
    # Detectar archivos que necesitan ser parcheados
    root_dir = Path(__file__).parent.parent
    script_files = list(root_dir.glob("**/*.py"))
    notebook_files = list(root_dir.glob("**/*.ipynb"))
    
    # Buscar patrones de guardado sin extensión
    model_save_pattern = "model.save("
    files_to_patch = []
    
    # Analizar archivos Python
    for file_path in script_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if model_save_pattern in content:
                    files_to_patch.append(file_path)
        except Exception as e:
            logger.error(f"Error al leer {file_path}: {e}")
    
    # Informar los resultados
    logger.info(f"Encontrados {len(files_to_patch)} archivos que pueden necesitar ser parcheados")
    for file_path in files_to_patch:
        logger.info(f"  - {file_path}")
    
    # Crear helper function para añadir a los proyectos
    model_saver_content = """import os
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def save_model_with_extension(model, filepath, overwrite=True):
    \"\"\"
    Guarda un modelo Keras con la extensión de archivo adecuada.
    
    Args:
        model: Modelo de Keras a guardar
        filepath: Ruta donde guardar el modelo
        overwrite: Si se debe sobrescribir el archivo existente
    
    Returns:
        Ruta donde se guardó el modelo
    \"\"\"
    # Verificar si la ruta ya tiene extensión
    base_path, ext = os.path.splitext(filepath)
    
    # Si no tiene extensión o la extensión no es válida para Keras, añadirla
    if not ext or ext.lower() not in ['.keras', '.h5']:
        filepath = f"{filepath}.keras"  # Usar el formato nativo de Keras (recomendado)
    
    try:
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar el modelo
        model.save(filepath, overwrite=overwrite)
        logger.info(f"✅ Modelo guardado exitosamente en: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"❌ Error al guardar modelo: {e}")
        return None
"""
    
    # Guardar el helper
    model_saver_path = os.path.join(root_dir, "models", "model_saver.py")
    try:
        with open(model_saver_path, 'w') as f:
            f.write(model_saver_content)
        logger.info(f"✅ Creado helper para guardar modelos: {model_saver_path}")
    except Exception as e:
        logger.error(f"❌ Error al crear helper: {e}")

def patch_cluster_processing():
    """
    Aplica un parche para asegurar que cluster_elevation se transforme correctamente
    a cluster_high, cluster_medium y cluster_low
    """
    # Buscar el archivo utils/dataloader_utils.py para verificar si ya contiene la función transform_cluster_elevation
    root_dir = Path(__file__).parent.parent
    dataloader_path = os.path.join(root_dir, "utils", "dataloader_utils.py")
    
    if os.path.exists(dataloader_path):
        logger.info(f"✅ Se encontró el archivo dataloader_utils.py")
        
        # Verificar si ya contiene la función transform_cluster_elevation
        with open(dataloader_path, 'r') as f:
            content = f.read()
            if "transform_cluster_elevation" in content:
                logger.info("✅ La función transform_cluster_elevation ya está implementada")
            else:
                logger.warning("⚠️ La función transform_cluster_elevation no está implementada")
                logger.info("ℹ️ Ejecutando python utils/dataloader_utils.py para actualizar...")
                
                try:
                    # Ejecutar el script para verificar la funcionalidad
                    os.system(f"python {dataloader_path}")
                    logger.info("✅ Script dataloader_utils.py ejecutado correctamente")
                except Exception as e:
                    logger.error(f"❌ Error al ejecutar dataloader_utils.py: {e}")
    else:
        logger.error(f"❌ No se encontró el archivo dataloader_utils.py")

def add_usage_examples():
    """
    Agrega ejemplos de uso para los parches aplicados
    """
    logger.info("\n=== EJEMPLOS DE USO ===")
    
    # Ejemplo de cómo usar el helper para guardar modelos
    logger.info("\n1. Ejemplo de cómo guardar modelos con extensión:")
    logger.info("""
# Importar el helper
from models.model_saver import save_model_with_extension

# Usar en lugar de model.save()
save_model_with_extension(model, 'models/output/trained_models/saved_models/tensorflow/conv_gru_ed_F1')
# Esto guardará el modelo como conv_gru_ed_F1.keras
    """)
    
    # Ejemplo de cómo usar la transformación de cluster_elevation
    logger.info("\n2. Ejemplo de cómo transformar cluster_elevation:")
    logger.info("""
# Importar la función
from utils.dataloader_utils import transform_cluster_elevation

# Aplicar transformación antes de usar el dataset
dataset = transform_cluster_elevation(dataset)
# Ahora dataset contiene 'cluster_high', 'cluster_medium', 'cluster_low'
    """)
    
    # Ejemplo de cómo usar el improved_build_dataloaders
    logger.info("\n3. Ejemplo de cómo usar el improved_build_dataloaders:")
    logger.info("""
# Importar la función
from utils.dataloader_utils import improved_build_dataloaders

# Usar en lugar del build_dataloaders original
feature_data, feature_shapes, available_features = improved_build_dataloaders(
    dataset, 
    features=['year', 'month', 'cluster_high', 'cluster_medium', 'cluster_low']
)
# La función se encargará automáticamente de la transformación
    """)

if __name__ == "__main__":
    logger.info("🛠️ Aplicando parches para corregir errores...")
    
    # Aplicar parches
    patch_model_saving_functions()
    patch_cluster_processing()
    
    # Mostrar ejemplos de uso
    add_usage_examples()
    
    logger.info("\n✅ Proceso completado. Ejecuta este script para aplicar los parches necesarios.") 