# Meta Modelo Neural para Precipitación Multihorizonte

## 🔧 Arquitectura del Modelo

**Nombre:** DeepMetaModel

**Tipo:** Red neuronal fully-connected (Feedforward Neural Network)

**Entrada (in_dim):** Vector de características que incluye:
- Predicciones base (low, medium, high)
- Estadísticas topográficas (elevación: media, desviación estándar, asimetría)
- Pendiente (slope)
- Orientación (aspect)

**Capas del modelo:**

```
Input (in_dim)
  ↓
Linear(in_dim → 128)
  ↓
ReLU
  ↓
Dropout(p=0.3)
  ↓
Linear(128 → 64)
  ↓
ReLU
  ↓
Linear(64 → 1)
```

**Output:** Predicción para un solo horizonte

## ⚙️ Hiperparámetros de Entrenamiento

| Parámetro                | Valor                                      |
|--------------------------|--------------------------------------------|
| Dispositivo (device)    | 'cuda' si disponible, si no 'cpu'         |
| Tasa de aprendizaje (lr) | 1e-3                                       |
| Optimizador              | Adam                                       |
| Función de pérdida       | MSELoss()                                  |
| Épocas (epochs)         | 50                                         |
| Tamaño del batch         | 64                                         |
| Validación               | División 90/10 por horizonte              |

## 📊 Evaluación y Métricas Calculadas

**Métricas Globales por Horizonte:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coeficiente de determinación)
- Porcentaje de datos válidos (valid_data_pct)

**Métricas por Rango de Elevación:**
- Bajo (<200 m), Medio (200–1000 m), Alto (>1000 m)
- RMSE, MAPE y R² por cada rango
- Conteo de datos válidos por rango

**Métricas por Percentiles de la variable objetivo (y_true):**
- Divisiones: [0–25%], [25–50%], [50–75%], [75–100%]
- RMSE, MAPE y R² por grupo

## 🧪 Procedimiento de Entrenamiento por Horizonte

Se entrena un modelo independiente por cada horizonte H ∈ {1, 2, 3}

Para cada h:
- Se construye X_meta con las features y y_true con las observaciones reales
- Se realiza split en conjunto de entrenamiento y validación
- Se entrena DeepMetaModel con validación interna
- Se guarda el modelo entrenado en:
```
MODEL_DIR/deepmeta_H{h}_{ref}.pt
```

## 📈 Visualizaciones Generadas
- Historial de entrenamiento: curva de pérdida (entrenamiento y validación) por horizonte.
- Diagrama de dispersión: True vs Pred por horizonte.
- Mapas geográficos:
  - Mapa de predicción (mm)
  - Mapa de error MAPE (%) con límite superior ajustado al percentil 99
  - Visualización con Cartopy y sobreposición del límite departamental (Boyacá)

## 📁 Archivos de salida generados

| Archivo                              | Contenido                                      |
|--------------------------------------|------------------------------------------------|
| deepmeta_H{h}_{ref}.pt              | Modelo entrenado para horizonte h              |
| deepmeta_training_h{h}.png           | Gráfica de pérdida por época                   |
| deepmeta_scatter_h{h}.png            | Gráfico de dispersión True vs Pred             |
| deepmeta_maps_h{h}.png              | Mapa de predicción + mapa de MAPE (%)         |
| deepmeta_global_metrics_ref{ref}.csv | Tabla de métricas globales por horizonte      |
| deepmeta_elevation_metrics_ref{ref}.csv | Métricas por rango de elevación              |
| deepmeta_percentile_metrics_ref{ref}.csv | Métricas por percentil del objetivo          |
| deepmeta_models_info.txt             | Registro de modelos entrenados y features usados |

## ✅ Resumen de características

| Componente                | Descripción                                    |
|---------------------------|------------------------------------------------|
| Predicción                | 1 horizonte por modelo (3 modelos en total)   |
| Robustez ante NaNs        | Estrategias activas de reemplazo (interpolación, media) |
| Diseño modular            | Entrenamiento individual por horizonte         |
| Validación                | Interna (split 90/10), mejores pesos guardados |
| Evaluación espacial        | Mapas de predicción y error (MAPE) usando grillas y máscaras |
| Análisis por subgrupos    | Por altitud y por percentiles del objetivo   |
| Implementación            | Pytorch, Pandas, Matplotlib, Cartopy         |