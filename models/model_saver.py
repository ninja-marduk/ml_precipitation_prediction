import os
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def save_model_with_extension(model, filepath, overwrite=True):
    """
    Guarda un modelo Keras con la extensión de archivo adecuada.
    
    Args:
        model: Modelo de Keras a guardar
        filepath: Ruta donde guardar el modelo
        overwrite: Si se debe sobrescribir el archivo existente
    
    Returns:
        Ruta donde se guardó el modelo
    """
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
