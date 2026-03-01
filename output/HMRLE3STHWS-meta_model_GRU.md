# Meta-Modelo Temporal para Predicción Multi-Horizonte

## 🔧 Arquitectura del Modelo (TemporalMetaModel)

**Entrada (in_dim):** Número de características combinadas (predicciones previas, estadísticas topográficas, etc.)

**Número de horizontes (num_horizons):** 3 (predicción para los siguientes 3 meses)

### Capas principales:

#### Capa de codificación de características:
- Linear(in_dim → 128)
- LayerNorm
- ReLU
- Dropout(p=0.2)

#### Capa GRU:
- 2 capas (num_layers=2)
- Tamaño oculto: 128
- dropout=0.1
- Procesamiento batch_first=True

#### Mecanismo de atención temporal:
- MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1)

#### Proyecciones por horizonte:
- Lista de Sequential por cada horizonte:
  - Linear(256 → 128) (por concatenación de GRU + atención)
  - ReLU
  - Linear(128 → 1)

#### Módulo adicional de calibración:
- Linear(256 → 1) (no utilizado en el forward principal)

## 📉 Función de pérdida personalizada (TemporalCoherenceLoss)

Combina:
- MSELoss() entre predicción y objetivo
- Penalización por incoherencia temporal (diferencias abruptas entre horizontes consecutivos)
- Parámetro: lambda_coherence = 0.3

## 🏋️‍♂️ Entrenamiento con Curriculum Learning

### Etapas:
1. **Fase 1:** Entrenamiento solo en horizonte H=1
2. **Fase 2:** Entrenamiento en horizontes H=1 y H=2
3. **Fase 3:** Entrenamiento en todos los horizontes (H=1,2,3)

### Hiperparámetros de entrenamiento:

| Parámetro | Valor |
|-----------|-------|
| Optimización | Adam |
| Tasa de aprendizaje inicial | 5e-4 |
| Esquema de reducción de LR | ReduceLROnPlateau (factor=0.5, paciencia=5) |
| Épocas totales | 60 (20 por fase) |
| Tamaño de lote (batch_size) | 64 |
| Dispositivo | 'cuda' si disponible, si no 'cpu' |

## 🧪 Evaluación y Métricas Implementadas

Para cada horizonte H:
- RMSE (Raíz del error cuadrático medio)
- MAE (Error absoluto medio)
- MAPE (Error porcentual absoluto medio)
- R² (Coeficiente de determinación)
- % de datos válidos (sin NaNs)

## 🗺️ Datos de Entrada y Enriquecimiento de Features

- **Predicciones base:** Modelos fusionados por rama (low, medium, high)
- **Estadísticas topográficas:**
  - Elevación: media, desviación estándar, asimetría (skewness)
  - Pendiente (slope)
  - Orientación (aspect)
- Todos los vectores son verificados y corregidos ante presencia de NaNs

## 💾 Persistencia y Visualización

- **El modelo entrenado se guarda en:** `temporal_meta_model_{ref}.pt`

### Resultados visuales:
- Curvas de entrenamiento (pérdida por época)
- Diagramas de dispersión y vs ŷ por horizonte
- Mapas de comparación de R² y RMSE por modelo

### Comparación con:
- DeepMeta
- XGBoost
- TemporalMeta (modelo actual)

## 📊 Comparación entre Modelos

Se genera una comparación cruzada por horizonte para:
- RMSE
- R²
- Visualización con gráficas de barras por modelo y horizonte

## ✅ Características distintivas del modelo

| Componente | Detalles Técnicos |
|------------|-------------------|
| Tipo de modelo | Red neuronal híbrida secuencial |
| Enfoque temporal | GRU + Atención por horizonte |
| Regularización | Dropout, LayerNorm y penalización por incoherencia temporal |
| Modo de entrenamiento | Curriculum Learning progresivo (H=1 → H=1,2 → H=1,2,3) |
| Entrada | Predicciones de modelos base + características topográficas |
| Predicción | Multi-output simultáneo (1 valor por cada horizonte futuro) |
| Evaluación | Métricas por horizonte y comparativas entre modelos |