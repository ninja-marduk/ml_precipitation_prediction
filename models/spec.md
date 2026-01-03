# Especificaciones del Framework Data-Driven para Prediccion de Precipitaciones
## ML Precipitation Prediction - Documento de Especificaciones Tecnicas

**Version:** 1.0
**Fecha:** Enero 2026
**Contexto:** Tesis Doctoral - Prediccion de Precipitaciones Mensuales en Zonas Montanosas

---

## 1. VISION Y OBJETIVOS DEL FRAMEWORK

### 1.1 Proposito

Este framework implementa un sistema **Data-Driven** para la prediccion de precipitaciones mensuales en zonas montanosas de los Andes colombianos, utilizando tecnicas avanzadas de Deep Learning espaciotemporal.

### 1.2 Hipotesis de Investigacion

Las hipotesis que el framework debe validar son:

| ID | Hipotesis | Metrica de Validacion | Estado |
|----|-----------|----------------------|--------|
| H1 | Los modelos hibridos GNN-Temporal superan a ConvLSTM en prediccion espacial | R² > 0.60, RMSE < 70mm | **Validada (V4)** |
| H2 | La incorporacion de features topograficas mejora la prediccion | Comparar BASIC vs PAFC | **Validada** |
| H3 | Las relaciones espaciales no-euclidianas capturan mejor la dinamica orografica | GNN vs CNN metrics | **En validacion** |
| H4 | La atencion temporal multi-escala mejora horizontes largos (H6-H12) | R² degradacion < 20% | **Parcialmente validada** |

### 1.3 Dominio del Problema

```
CONTEXTO GEOGRAFICO:
+------------------------------------------+
|  ZONA DE ESTUDIO: Andes Colombianos      |
|  - Altitud: 500m - 4000m                 |
|  - Variabilidad: Alta (orografia)        |
|  - Regimen: Bimodal (2 temporadas)       |
|  - Resolucion: 0.05° (~5km)              |
+------------------------------------------+

DESAFIOS ESPECIFICOS:
1. Gradientes altitudinales pronunciados
2. Efecto orografico complejo
3. Variabilidad interanual (ENSO)
4. Escasez de datos in-situ
5. Patrones no-lineales precipitacion-elevacion
```

---

## 2. ARQUITECTURA DEL FRAMEWORK

### 2.1 Estructura de Directorios

```
ml_precipitation_prediction/
|
+-- data/                          # Datos de entrada
|   +-- raw/                       # Datos originales (NetCDF, GeoTIFF)
|   +-- processed/                 # Datos preprocesados
|   +-- dataset_monthly_*.nc       # Datasets consolidados
|
+-- models/                        # Implementaciones de modelos
|   +-- base_models_*.ipynb        # Notebooks principales (V1-V4)
|   +-- custom_layers/             # Capas personalizadas
|   +-- output/                    # Salidas de entrenamiento
|   |   +-- V2_Enhanced_Models/    # Resultados V2
|   |   +-- V4_GNN_TAT_Models/     # Resultados V4 (actual)
|   +-- metrics.py                 # Funciones de metricas
|   +-- feature_selection.py       # Seleccion de features
|
+-- notebooks/                     # Analisis exploratorio
+-- scripts/                       # Scripts de automatizacion
|   +-- benchmark/                 # Analisis comparativo
|
+-- docs/                          # Documentacion
|   +-- framework/                 # Especificaciones tecnicas
|   +-- models/                    # Documentacion de modelos
|   +-- tesis/                     # Documentos de tesis
|
+-- preprocessing/                 # Preprocesamiento de datos
+-- utils/                         # Utilidades comunes
```

### 2.2 Nomenclatura de Versiones

| Version | Nombre | Arquitectura Base | Estado |
|---------|--------|-------------------|--------|
| V1 | Baseline | ConvLSTM, ConvGRU, ConvRNN | Completado |
| V2 | Enhanced | V1 + Attention + Bidirectional + Residual | Completado |
| V3 | FNO | Fourier Neural Operators | Completado (Underperformed) |
| **V4** | **GNN-TAT** | **Graph Neural Networks + Temporal Attention** | **Actual** |
| V5 | Multi-Modal | V4 + ERA5 + Satellite | Planificado |
| V6 | Ensemble | Best of V2-V5 + Meta-learning | Planificado |

---

## 3. ESTANDARES DE NOTEBOOKS

### 3.1 Estructura Obligatoria de Celdas

Todo notebook de modelo debe seguir esta estructura:

```
CELL 1: HEADER & METADATA
--------------------------
- Titulo del notebook
- Version y fecha
- Autor
- Descripcion del experimento

CELL 2: ENVIRONMENT SETUP
--------------------------
- Deteccion Colab/Local
- Instalacion de dependencias
- Configuracion de paths
- Version matching (netCDF4, h5py, PyTorch)

CELL 3-4: IMPORTS
--------------------------
- Librerias standard (numpy, pandas, xarray)
- Deep Learning (torch, torch_geometric)
- Visualizacion (matplotlib, seaborn)

CELL 5: CONFIGURATION (CONFIG dict)
--------------------------
CONFIG = {
    'input_window': 60,      # Meses de entrada
    'horizon': 12,           # Meses a predecir
    'epochs': 150,
    'batch_size': 4,
    'learning_rate': 0.001,
    'patience': 50,
    'train_val_split': 0.8,
    'light_mode': False,     # True para pruebas rapidas
    'enabled_horizons': [1, 3, 6, 12],
    'gnn_config': {...},     # Config especifica del modelo
}

CELL 6-8: DATA LOADING & VALIDATION
--------------------------
- Carga de NetCDF
- Validacion de features
- Verificacion de dimensiones

CELL 9-11: PREPROCESSING
--------------------------
- Construccion de grafo (para GNN)
- Feature engineering
- Windowing temporal
- Train/Val split

CELL 12-15: MODEL DEFINITION
--------------------------
- Clases de capas custom
- Arquitectura del modelo
- Forward pass

CELL 16-18: TRAINING LOOP
--------------------------
- Funcion de entrenamiento
- Early stopping
- Checkpointing

CELL 19: MAIN EXPERIMENT LOOP
--------------------------
- Iteracion por feature sets (BASIC, KCE, PAFC)
- Iteracion por horizontes
- Guardado de metricas

CELL 20: RESULTS AGGREGATION
--------------------------
- Consolidacion de metricas
- Guardado de CSV final
- Limpieza de memoria
```

### 3.2 Convencion de Nombres de Archivos

```
Notebooks:
  base_models_{ARQUITECTURA}_{VERSION}.ipynb
  Ejemplo: base_models_GNN_TAT_V4.ipynb

Modelos guardados:
  {MODEL_NAME}_best_h{HORIZON}.pt
  Ejemplo: GNN_TAT_GAT_best_h12.pt

Metricas:
  metrics_spatial_{version}_{model}_h{horizon}.csv
  Ejemplo: metrics_spatial_v4_gnn_tat_h12.csv

Training logs:
  {MODEL_NAME}_training_log_h{HORIZON}.csv
  {MODEL_NAME}_history.json
```

---

## 4. ESTANDARES DE SALIDAS

### 4.1 Estructura de Directorio de Salidas

```
models/output/{VERSION}_{MODEL_NAME}/
|
+-- experiment_state_{version}.json    # Estado completo del experimento
|
+-- h{HORIZON}/                        # Por cada horizonte
|   +-- {EXPERIMENT}/                  # BASIC, KCE, PAFC
|   |   +-- training_metrics/
|   |   |   +-- {MODEL}_best_h{H}.pt        # Mejor modelo
|   |   |   +-- {MODEL}_history.json        # Historial de entrenamiento
|   |   |   +-- {MODEL}_training_log_h{H}.csv
|   |   +-- predictions/               # (opcional)
|   |   +-- visualizations/            # (opcional)
|
+-- metrics_spatial_{version}_h{H}.csv  # Metricas consolidadas
+-- graph_visualization.png             # Visualizacion del grafo
```

### 4.2 Formato de Metricas Espaciales (CSV)

```csv
TotalHorizon,Experiment,Model,H,RMSE,MAE,R^2,Mean_True_mm,Mean_Pred_mm,TotalPrecipitation,TotalPrecipitation_Pred,mean_bias_mm,mean_bias_pct
12,BASIC,GNN_TAT_GAT,1,55.28,38.41,0.6642,154.84,150.90,127743.43,124495.39,-3.94,-2.54
...
```

**Columnas obligatorias:**
| Columna | Tipo | Descripcion |
|---------|------|-------------|
| TotalHorizon | int | Horizonte total del experimento |
| Experiment | str | Conjunto de features (BASIC/KCE/PAFC) |
| Model | str | Nombre del modelo |
| H | int | Horizonte especifico (1-12) |
| RMSE | float | Root Mean Squared Error (mm) |
| MAE | float | Mean Absolute Error (mm) |
| R^2 | float | Coeficiente de determinacion |
| Mean_True_mm | float | Precipitacion real promedio |
| Mean_Pred_mm | float | Precipitacion predicha promedio |
| mean_bias_mm | float | Sesgo medio (mm) |
| mean_bias_pct | float | Sesgo porcentual |

### 4.3 Formato de History JSON

```json
{
  "model_name": "GNN_TAT_GAT",
  "experiment": "BASIC",
  "horizon": 12,
  "best_epoch": 23,
  "best_val_loss": 0.6059,
  "final_train_loss": 0.0862,
  "final_val_loss": 0.7986,
  "total_epochs": 73,
  "parameters": 97932
}
```

### 4.4 Formato de Experiment State JSON

```json
{
  "config": {
    "input_window": 60,
    "horizon": 12,
    "epochs": 150,
    "gnn_config": {...}
  },
  "feature_sets": {
    "BASIC": ["year", "month", ...],
    "KCE": [...],
    "PAFC": [...]
  },
  "grid_info": {
    "lat": 5,
    "lon": 5,
    "n_nodes": 25,
    "n_edges": 600
  },
  "training_summaries": {
    "{EXP}_{MODEL}_H{H}": {...}
  },
  "timestamp": "2026-01-03T13:22:06"
}
```

---

## 5. CONJUNTOS DE FEATURES

### 5.1 Definicion de Feature Sets

| Set | Features | Proposito |
|-----|----------|-----------|
| **BASIC** | Temporales + Precipitacion + Topografia basica | Baseline minimo |
| **KCE** | BASIC + Clusters de elevacion (K-means) | Capturar bandas altitudinales |
| **PAFC** | KCE + Lags temporales | Autocorrelacion temporal |

### 5.2 Features Detallados

```python
FEATURE_SETS = {
    'BASIC': [
        # Temporales
        'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        # Precipitacion
        'max_daily_precipitation', 'min_daily_precipitation',
        'daily_precipitation_std',
        # Topografia
        'elevation', 'slope', 'aspect'
    ],

    'KCE': [
        *BASIC,
        # Clusters de elevacion (one-hot)
        'elev_high', 'elev_med', 'elev_low'
    ],

    'PAFC': [
        *KCE,
        # Lags temporales
        'total_precipitation_lag1',   # Mes anterior
        'total_precipitation_lag2',   # 2 meses antes
        'total_precipitation_lag12'   # Mismo mes ano anterior
    ]
}
```

---

## 6. ARQUITECTURA DE MODELOS

### 6.1 Arquitectura GNN-TAT (V4)

```
INPUT: (batch, seq_len, n_nodes, n_features)
        |
        v
+------------------+
| SPATIAL ENCODER  |  GCN/GAT/SAGE layers
| (GNN per timestep)|  - Message passing
+------------------+  - Edge weights: distance + elevation + correlation
        |
        v
+------------------+
| TEMPORAL ENCODER |  Multi-head Self-Attention
| (Attention)      |  - 4 heads
+------------------+  - Residual connection + LayerNorm
        |
        v
+------------------+
| SEQUENCE MODEL   |  Bidirectional LSTM
| (LSTM)           |  - 2 layers
+------------------+  - Dropout 0.1
        |
        v
+------------------+
| OUTPUT PROJECTION|  Linear layers
+------------------+  - Per-node, per-horizon output
        |
        v
OUTPUT: (batch, n_nodes, horizon)
```

### 6.2 Configuracion de Hiperparametros

```python
GNN_CONFIG = {
    # Dimensiones
    'hidden_dim': 64,
    'temporal_hidden_dim': 64,
    'lstm_hidden_dim': 64,

    # Capas
    'num_gnn_layers': 2,
    'num_temporal_heads': 4,
    'num_lstm_layers': 2,

    # Regularizacion
    'dropout': 0.1,
    'temporal_dropout': 0.1,

    # Construccion del grafo
    'edge_threshold': 0.3,
    'max_neighbors': 8,
    'use_distance_edges': True,
    'use_elevation_edges': True,
    'use_correlation_edges': True,

    # Pesos de similaridad
    'distance_scale_km': 10.0,
    'elevation_scale': 0.2,
    'elevation_weight': 0.3,
    'correlation_weight': 0.5,
    'min_edge_weight': 0.01
}
```

### 6.3 Variantes de GNN Soportadas

| Variante | Descripcion | Pros | Contras |
|----------|-------------|------|---------|
| **GCN** | Graph Convolutional Network | Estable, usa edge_weight | Receptivo global limitado |
| **GAT** | Graph Attention Network | Atencion adaptativa | Sin edge_weight nativo |
| **SAGE** | GraphSAGE | Eficiente, sampling | Sin edge_weight nativo |

---

## 7. PIPELINE DE ENTRENAMIENTO

### 7.1 Flujo de Entrenamiento

```
1. PREPARACION
   +-- Cargar datos NetCDF
   +-- Validar features disponibles
   +-- Construir grafo espacial
   +-- Crear ventanas temporales (input_window -> horizon)
   +-- Split train/val (80/20 temporal)

2. LOOP DE EXPERIMENTOS
   FOR each feature_set in [BASIC, KCE, PAFC]:
       FOR each model_type in [GAT, SAGE, GCN]:
           FOR each horizon in enabled_horizons:
               +-- Instanciar modelo
               +-- Entrenar con early stopping
               +-- Guardar mejor checkpoint
               +-- Evaluar en validacion
               +-- Guardar metricas

3. EVALUACION
   +-- Calcular metricas espaciales por celda
   +-- Agregar en CSV consolidado
   +-- Generar visualizaciones
   +-- Guardar experiment_state.json
```

### 7.2 Metricas de Evaluacion

```python
def evaluate_model(y_true, y_pred):
    """
    Metricas estandar del framework.

    Args:
        y_true: (N, horizon, lat, lon) - Valores reales
        y_pred: (N, horizon, lat, lon) - Predicciones

    Returns:
        dict con metricas por horizonte
    """
    metrics = {}
    for h in range(horizon):
        metrics[f'H{h+1}'] = {
            'RMSE': sqrt(mean((y_true[:, h] - y_pred[:, h])**2)),
            'MAE': mean(abs(y_true[:, h] - y_pred[:, h])),
            'R2': 1 - SS_res / SS_tot,
            'Bias': mean(y_pred[:, h] - y_true[:, h]),
            'Bias_pct': 100 * Bias / mean(y_true[:, h])
        }
    return metrics
```

### 7.3 Criterios de Early Stopping

```python
EARLY_STOPPING_CONFIG = {
    'monitor': 'val_loss',
    'patience': 50,
    'min_delta': 1e-4,
    'mode': 'min',
    'restore_best_weights': True
}

LEARNING_RATE_SCHEDULER = {
    'type': 'ReduceLROnPlateau',
    'factor': 0.5,
    'patience': 20,
    'min_lr': 1e-6
}
```

---

## 8. COMPATIBILIDAD COLAB/LOCAL

### 8.1 Deteccion de Entorno

```python
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    # Montar Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = '/content/drive/MyDrive/ml_precipitation_prediction'

    # Instalar dependencias
    !pip install torch-geometric netCDF4==1.7.2 h5py==3.14.0
else:
    BASE_PATH = r'd:\github.com\ninja-marduk\ml_precipitation_prediction'
```

### 8.2 Version Matching

```python
LOCAL_VERSIONS = {
    'netCDF4': '1.7.2',
    'h5py': '3.14.0',
    'HDF5_lib': '1.14.4',
    'netCDF_lib': '4.9.2'
}

# En Colab, instalar versiones matching para evitar errores HDF5
if IN_COLAB:
    !pip install netCDF4=={LOCAL_VERSIONS['netCDF4']}
    !pip install h5py=={LOCAL_VERSIONS['h5py']}
```

### 8.3 Light Mode para Pruebas Rapidas

```python
if CONFIG['light_mode']:
    # Reducir grid para pruebas
    LIGHT_SIZE = CONFIG.get('light_grid_size', 5)
    lat_slice = slice(center_lat - LIGHT_SIZE//2, center_lat + LIGHT_SIZE//2 + 1)
    lon_slice = slice(center_lon - LIGHT_SIZE//2, center_lon + LIGHT_SIZE//2 + 1)

    # Aplicar a datos
    ds = ds.isel(lat=lat_slice, lon=lon_slice)
```

---

## 9. BUENAS PRACTICAS

### 9.1 Reproducibilidad

```python
# Siempre fijar seeds
import torch
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
```

### 9.2 Gestion de Memoria

```python
# Limpiar memoria entre experimentos
import gc

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Llamar despues de cada modelo
cleanup()
```

### 9.3 Logging y Checkpointing

```python
# Guardar checkpoint solo si mejora
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': CONFIG
    }, checkpoint_path)

# Cargar con weights_only=False (PyTorch 2.6+)
checkpoint = torch.load(path, weights_only=False)
```

### 9.4 Manejo de Errores

```python
# Validar features antes de entrenar
def validate_features(ds, required_features):
    available = set(ds.data_vars)
    missing = set(required_features) - available

    if missing:
        raise ValueError(f"Missing features: {missing}")

    return True
```

---

## 10. METRICAS DE EXITO

### 10.1 Criterios de Aceptacion

| Metrica | Baseline (V2) | Target (V4+) | Excelente |
|---------|---------------|--------------|-----------|
| R² (H1-H6) | 0.44 | > 0.60 | > 0.70 |
| R² (H7-H12) | 0.30 | > 0.50 | > 0.60 |
| RMSE (mm) | 98.17 | < 70 | < 55 |
| MAE (mm) | 44.19 | < 50 | < 40 |
| Parametros | 2M+ | < 150K | < 100K |
| Train ratio (val/train loss) | - | < 10x | < 5x |

### 10.2 Resultados V4 GNN-TAT (Actuales)

```
BEST CONFIGURATION: GNN_TAT_SAGE + KCE @ H=3
- R²: 0.707 (vs baseline 0.437) -> +62% mejora
- RMSE: 52.45mm (vs baseline 98.17mm) -> -47% mejora
- Parametros: ~98K (vs 2M+) -> -95% reduccion

RANKING POR MODELO (promedio todos horizontes):
1. GNN_TAT_GAT + PAFC:  R²=0.628, RMSE=58.4mm
2. GNN_TAT_GCN + PAFC:  R²=0.625, RMSE=58.7mm
3. GNN_TAT_SAGE + KCE:  R²=0.618, RMSE=59.1mm
```

---

## 11. GLOSARIO

| Termino | Definicion |
|---------|------------|
| **GNN** | Graph Neural Network - Red que opera sobre grafos |
| **TAT** | Temporal Attention - Mecanismo de atencion temporal |
| **ConvLSTM** | Convolutional LSTM - LSTM con convoluciones espaciales |
| **GAT** | Graph Attention Network - GNN con atencion |
| **SAGE** | GraphSAGE - GNN con sampling de vecinos |
| **GCN** | Graph Convolutional Network - GNN basico |
| **H** | Horizonte de prediccion (meses adelante) |
| **BASIC** | Feature set minimo |
| **KCE** | K-means Cluster Elevation features |
| **PAFC** | Precipitation Auto-correlation Features Complete |
| **R²** | Coeficiente de determinacion |
| **RMSE** | Root Mean Square Error |
| **MAE** | Mean Absolute Error |
| **Orografico** | Relacionado con efectos de elevacion en precipitacion |

---

## 12. REFERENCIAS

### 12.1 Papers Fundamentales

1. Kipf & Welling (2017) - "Semi-Supervised Classification with Graph Convolutional Networks"
2. Velickovic et al. (2018) - "Graph Attention Networks"
3. Hamilton et al. (2017) - "Inductive Representation Learning on Large Graphs" (GraphSAGE)
4. Shi et al. (2015) - "Convolutional LSTM Network" (ConvLSTM)
5. Vaswani et al. (2017) - "Attention Is All You Need"

### 12.2 Datasets

- **CHIRPS 2.0**: Climate Hazards Group InfraRed Precipitation with Station data
- **SRTM DEM**: Shuttle Radar Topography Mission Digital Elevation Model
- **ERA5**: ECMWF Reanalysis v5 (planificado para V5)

---

*Documento generado como parte del Framework ML Precipitation Prediction*
*Ultima actualizacion: Enero 2026*
