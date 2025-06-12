# Corrección de Errores: Guardado de Modelos y Clusters de Elevación

Este archivo explica cómo corregir dos errores comunes en el sistema de predicción de precipitación:

1. **Error al guardar modelos**: Falta de extensión `.keras` en las rutas de guardado
2. **Error con variables cluster**: `cluster_high`, `cluster_medium`, `cluster_low` no encontradas en el dataset

## Solución Rápida

Para aplicar automáticamente las correcciones, simplemente ejecuta:

```bash
python models/patch_save_models.py
```

Este script detectará archivos que necesitan ser parcheados y creará los helpers necesarios.

## Solución 1: Agregar extensión `.keras` al guardar modelos

El error ocurre porque TensorFlow requiere extensiones específicas al guardar modelos:

```
⚠️ Error al guardar modelo: Invalid filepath extension for saving. Please add either a `.keras` extension 
for the native Keras format (recommended) or a `.h5` extension.
```

### Solución manual

Reemplaza las llamadas a `model.save()` con el helper `save_model_with_extension()`:

```python
# Antes (produce error)
model.save(str(tf_path))

# Después (corregido)
from models.model_saver import save_model_with_extension
save_model_with_extension(model, str(tf_path))
```

## Solución 2: Transformar `cluster_elevation` a variables one-hot

El error ocurre porque los modelos esperan variables one-hot (`cluster_high`, `cluster_medium`, `cluster_low`), pero el dataset contiene solo la variable categórica `cluster_elevation`:

```
⚠️ Característica 'cluster_high' no encontrada en el dataset
⚠️ Característica 'cluster_medium' no encontrada en el dataset
⚠️ Característica 'cluster_low' no encontrada en el dataset
```

### Solución manual

Antes de usar el dataset, transforma `cluster_elevation` a variables one-hot:

```python
# Importar la función de transformación
from utils.dataloader_utils import transform_cluster_elevation

# Transformar el dataset
dataset = transform_cluster_elevation(dataset)
```

O usa directamente el dataloader mejorado:

```python
from utils.dataloader_utils import improved_build_dataloaders

feature_data, feature_shapes, available_features = improved_build_dataloaders(
    dataset, 
    features=['year', 'month', 'cluster_high', 'cluster_medium', 'cluster_low']
)
```

## Detalles Técnicos

### 1. Helper para guardar modelos

El archivo `models/model_saver.py` contiene una función que automáticamente:
- Verifica si la ruta ya tiene una extensión válida (`.keras` o `.h5`)
- Añade la extensión `.keras` si es necesario
- Crea directorios intermedios si no existen
- Maneja errores de guardado

### 2. Transformación de `cluster_elevation`

El archivo `utils/dataloader_utils.py` ahora incluye:
- Función `transform_cluster_elevation()` para convertir la variable categórica en one-hot
- Función `improved_build_dataloaders()` que maneja automáticamente esta transformación
- Detección de errores y manejo seguro de NaN

## Notas de Implementación

- Las soluciones son compatibles con versiones anteriores del código
- No se requieren cambios adicionales en los modelos existentes
- Ambas soluciones son robustas frente a errores y casos extremos 