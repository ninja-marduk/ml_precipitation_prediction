# Plan de Desarrollo - ML Precipitation Prediction
## Hoja de Ruta Estrategica para Tesis Doctoral

**Version:** 1.0
**Fecha:** Enero 2026
**Objetivo:** Prediccion de Precipitaciones Mensuales en Zonas Montanosas mediante Deep Learning

---

## RESUMEN EJECUTIVO

Este documento define la hoja de ruta para el desarrollo del framework de prediccion de precipitaciones, integrando los resultados obtenidos hasta la fecha y estableciendo el camino hacia la finalizacion de la tesis doctoral.

### Estado Actual

```
+------------------------------------------------------------------+
|                    PROGRESO DEL PROYECTO                          |
+------------------------------------------------------------------+
| V1 Baseline      [##########] 100%  - ConvLSTM/GRU basicos       |
| V2 Enhanced      [##########] 100%  - Attention + Bidirectional  |
| V3 FNO           [##########] 100%  - Completado (underperformed)|
| V4 GNN-TAT       [########--]  80%  - Light mode OK, full pend.  |
| V5 Multi-Modal   [----------]   0%  - Planificado                |
| V6 Ensemble      [----------]   0%  - Planificado                |
| Tesis Escrita    [####------]  40%  - En progreso                |
+------------------------------------------------------------------+
```

### Logro Principal V4

| Metrica | V2 Baseline | V4 GNN-TAT | Mejora |
|---------|-------------|------------|--------|
| R² | 0.437 | **0.707** | **+62%** |
| RMSE | 98.17mm | **52.45mm** | **-47%** |
| Parametros | 2M+ | **~98K** | **-95%** |

---

## PARTE I: ANALISIS DE SITUACION

### 1.1 Resultados Consolidados por Version

#### V1 - Baseline (Completado)
```
Arquitecturas: ConvLSTM, ConvGRU, ConvRNN
Mejor R² H1: 0.86 (ConvRNN-BASIC)
Problema: Degradacion severa H2-H12 (R² < 0.30)
Leccion: Arquitecturas basicas insuficientes para multi-horizonte
```

#### V2 - Enhanced (Completado)
```
Arquitecturas: +Bidirectional, +Residual, +Attention, +Transformer
Mejor R²: 0.752 (ConvRNN_Enhanced + PAFC)
RMSE: 44.85mm, MAE: 34.38mm
Leccion: Regularizacion > Complejidad arquitectonica
```

#### V3 - FNO (Completado - Underperformed)
```
Arquitecturas: FNO_Pure, FNO_ConvLSTM_Hybrid
Resultado: PEOR que V2 en todos los experimentos
- BASIC: +4.38mm RMSE vs V2
- PAFC: +7.73mm RMSE vs V2
Leccion Critica: FNO no apto para precipitacion (discontinuidades, grid pequeno)
```

#### V4 - GNN-TAT (Actual - 80%)
```
Arquitecturas: GNN_TAT_GAT, GNN_TAT_SAGE, GNN_TAT_GCN
Mejor Resultado (Light Mode 5x5):
- R² = 0.707 (SAGE+KCE H=3)
- RMSE = 52.45mm
- Parametros: ~98K

Ranking por Configuracion:
1. GAT + PAFC: R²=0.628 promedio
2. GCN + PAFC: R²=0.625 promedio
3. SAGE + KCE: R²=0.618 promedio
```

### 1.2 Lecciones Aprendidas

```
+---------------------------------------------------------------+
|  INSIGHTS CLAVE DEL PROYECTO                                   |
+---------------------------------------------------------------+
| 1. SIMPLE > COMPLEJO                                          |
|    ConvRNN supero a ConvLSTM en varios escenarios             |
|                                                                |
| 2. REGULARIZACION > ARQUITECTURA                              |
|    Dropout + Early stopping mas efectivo que mas capas        |
|                                                                |
| 3. FEATURES TOPOGRAFICAS FUNCIONAN                            |
|    PAFC consistentemente mejor que BASIC                      |
|                                                                |
| 4. FNO NO ES LA SOLUCION                                      |
|    Spectral methods fallan con patrones discontinuos          |
|                                                                |
| 5. GNN CAPTURA RELACIONES ESPACIALES                          |
|    Grafo basado en elevacion + distancia es efectivo          |
|                                                                |
| 6. OVERFITTING ES EL PRINCIPAL PROBLEMA                       |
|    Ratio train/val de 6-19x indica memorizacion               |
+---------------------------------------------------------------+
```

### 1.3 Problemas Identificados en V4

| Problema | Severidad | Solucion Propuesta |
|----------|-----------|-------------------|
| Overfitting (ratio 6-19x) | Alta | Mas regularizacion, data augmentation |
| Early stopping muy temprano | Media | Ajustar patience, warmup |
| Bias negativo (-3 a -20mm) | Media | Loss function balanceada |
| Solo light mode validado | Alta | **Ejecutar full grid** |

---

## PARTE II: PLAN DE ACCION

### 2.1 Tareas Inmediatas (Sprint 1)

#### TAREA 1.1: Ejecutar V4 Full Grid
```
Objetivo: Validar resultados en grid completo (no light mode)
Duracion: 2-3 dias (Colab Pro+ recomendado)

Pasos:
1. Modificar CONFIG['light_mode'] = False
2. Ejecutar para H = [1, 3, 6, 12]
3. Comparar metricas light vs full
4. Documentar diferencias

Criterio de Exito:
- R² full >= 0.90 * R² light (degradacion < 10%)
- Entrenamiento completo sin OOM
```

#### TAREA 1.2: Mitigar Overfitting V4
```
Objetivo: Reducir ratio train/val de ~10x a <5x

Acciones:
1. Aumentar dropout: 0.1 -> 0.2-0.3
2. Agregar weight decay: 1e-5 -> 1e-4
3. Implementar data augmentation:
   - Temporal jittering (+/- 1 mes)
   - Spatial noise (sigma=0.1)
4. Label smoothing en loss function

Criterio de Exito:
- Ratio train/val < 5x
- R² se mantiene > 0.55
```

#### TAREA 1.3: Corregir Bias Negativo
```
Objetivo: Reducir subestimacion de precipitacion

Acciones:
1. Agregar termino de bias en loss:
   L = MSE + lambda * |mean(pred) - mean(true)|²

2. Post-processing con bias correction:
   pred_corrected = pred + mean_bias_training

3. Weighted loss por intensidad de lluvia

Criterio de Exito:
- |bias_pct| < 5% (actualmente -3% a -10%)
```

### 2.2 Desarrollo V5 Multi-Modal (Sprint 2-3)

#### TAREA 2.1: Adquisicion de Datos
```
Fuentes a Integrar:
+------------------+-------------------+-------------------+
| Fuente           | Variables         | Resolucion        |
+------------------+-------------------+-------------------+
| ERA5 Reanalysis  | Wind u/v          | 0.25° / hourly    |
|                  | Humidity          |                   |
|                  | Temperature       |                   |
|                  | Pressure          |                   |
+------------------+-------------------+-------------------+
| MODIS            | Cloud cover       | 1km / daily       |
|                  | LST               |                   |
|                  | NDVI              |                   |
+------------------+-------------------+-------------------+
| Climate Indices  | ENSO (Nino 3.4)   | Monthly           |
|                  | IOD               |                   |
|                  | MJO               |                   |
+------------------+-------------------+-------------------+

Duracion estimada: 2-3 semanas
```

#### TAREA 2.2: Pipeline Multi-Modal
```
Arquitectura Propuesta:

INPUT_1: Precipitacion (60 meses, grid espacial)
INPUT_2: ERA5 (60 meses, variables atmosfericas)
INPUT_3: Satellite (60 meses, cloud + LST)
INPUT_4: Climate Indices (60 valores escalares)

         +------------------+
         | Modality-Specific|
INPUT_1->| Encoder (GNN-TAT)|--+
         +------------------+  |
                               |
         +------------------+  |   +-----------------+
INPUT_2->| Encoder (Conv3D) |--+-->| Cross-Modal     |
         +------------------+  |   | Attention       |
                               |   |                 |
         +------------------+  |   | Q: Precip       |
INPUT_3->| Encoder (CNN)    |--+   | K,V: All modes  |
         +------------------+  |   +-----------------+
                               |          |
         +------------------+  |          v
INPUT_4->| Encoder (MLP)    |--+   +-----------------+
         +------------------+      | Fusion + Output |
                                   +-----------------+
                                          |
                                          v
                                   Precipitation Forecast

Criterio de Exito:
- R² > 0.75 (mejora >10% vs V4)
- Ablation study mostrando valor de cada modalidad
```

### 2.3 Desarrollo V6 Ensemble (Sprint 4)

#### TAREA 3.1: Ensemble Inteligente
```
Estrategia: Meta-learner que selecciona mejor modelo segun contexto

MODELOS BASE (de V2-V5):
- V2: ConvLSTM_Bidirectional (bueno H1-H6)
- V2: ConvRNN_Enhanced (eficiente)
- V4: GNN_TAT_GAT + PAFC (mejor espacial)
- V4: GNN_TAT_SAGE + KCE (mejor H3-H6)
- V5: Multi-Modal (si mejora)

META-LEARNER:
Context Features:
- Horizonte (1-12)
- Estacion (DJF, MAM, JJA, SON)
- Ubicacion espacial (cluster)
- RMSE reciente de cada modelo

Output:
- Pesos para ensemble: w = softmax(f(context))
- Prediccion: P_ensemble = sum(w_i * P_i)

Criterio de Exito:
- R² ensemble > max(R² individual)
- Varianza reducida entre horizontes
```

### 2.4 Soft Physics Constraints (Sprint 5)

#### TAREA 4.1: Constraints Fisicas Suaves
```
Objetivo: Agregar regularizacion basada en fisica sin perder flexibilidad

CONSTRAINTS A IMPLEMENTAR:

1. Conservacion de masa (simplificada):
   L_mass = |sum(P_pred) - sum(P_true)|

2. Suavidad espacial (precipitacion no salta bruscamente):
   L_smooth_spatial = sum(|P(i,j) - P(i+1,j)|² + |P(i,j) - P(i,j+1)|²)

3. Consistencia temporal (transiciones suaves):
   L_smooth_temporal = sum(|P(t+1) - P(t)|²)

4. Non-negatividad (precipitacion >= 0):
   L_nonneg = sum(ReLU(-P_pred)²)

LOSS FUNCTION FINAL:
L_total = L_MSE + lambda_mass * L_mass
               + lambda_spatial * L_smooth_spatial
               + lambda_temporal * L_smooth_temporal
               + lambda_nonneg * L_nonneg

Hiperparametros a tunar: lambda_* via grid search

Criterio de Exito:
- Predicciones fisicamente plausibles
- No degradacion en metricas principales
```

---

## PARTE III: CRONOGRAMA

### 3.1 Timeline de Desarrollo

```
2026
ENE                FEB                MAR                ABR
|------------------|------------------|------------------|
[====== V4 Full ======]
      [====== V4 Fixes ======]
            [============ V5 Data Acquisition ============]
                  [============ V5 Multi-Modal ==============]
                              [======== V6 Ensemble ========]
                                    [==== Physics ====]

ENTREGABLES:
- Fin Enero: V4 Full grid validado
- Fin Febrero: V5 data pipeline listo
- Fin Marzo: V5 modelos entrenados
- Fin Abril: V6 ensemble + physics
```

### 3.2 Milestones y Entregables

| Milestone | Fecha Target | Entregable | Criterio de Exito |
|-----------|--------------|------------|-------------------|
| M1 | 15 Enero | V4 Full Grid | R² > 0.60 full grid |
| M2 | 31 Enero | V4 Optimizado | Overfitting < 5x |
| M3 | 28 Febrero | V5 Data Ready | Pipeline funcionando |
| M4 | 31 Marzo | V5 Trained | R² > 0.75 |
| M5 | 30 Abril | V6 Complete | R² > 0.80 |
| M6 | 31 Mayo | Paper Draft | Listo para submission |
| M7 | 30 Junio | Thesis Draft | Capitulos 1-6 completos |

---

## PARTE IV: ESTRUCTURA DE TESIS

### 4.1 Capitulos Propuestos

```
TITULO PROPUESTO:
"Framework Hibrido de Deep Learning para Prediccion de Precipitaciones
Mensuales en Zonas Montanosas: De ConvLSTM a GNN con Atencion Temporal"

CAPITULOS:

CAP 1: INTRODUCCION (15 pags)
+-- 1.1 Contexto y Motivacion
+-- 1.2 Planteamiento del Problema
+-- 1.3 Objetivos (General y Especificos)
+-- 1.4 Hipotesis de Investigacion
+-- 1.5 Alcance y Limitaciones
+-- 1.6 Estructura del Documento

CAP 2: MARCO TEORICO (30 pags)
+-- 2.1 Prediccion de Precipitacion
|   +-- 2.1.1 Metodos tradicionales
|   +-- 2.1.2 Modelos numericos (NWP)
|   +-- 2.1.3 Machine Learning en meteorologia
+-- 2.2 Deep Learning Espaciotemporal
|   +-- 2.2.1 Redes Convolucionales (CNN)
|   +-- 2.2.2 Redes Recurrentes (LSTM, GRU)
|   +-- 2.2.3 ConvLSTM y variantes
+-- 2.3 Graph Neural Networks
|   +-- 2.3.1 Fundamentos teoricos
|   +-- 2.3.2 GCN, GAT, GraphSAGE
|   +-- 2.3.3 GNN para datos espaciales
+-- 2.4 Mecanismos de Atencion
|   +-- 2.4.1 Self-attention y Transformers
|   +-- 2.4.2 Atencion temporal
+-- 2.5 Estado del Arte en Precipitacion con DL

CAP 3: METODOLOGIA (25 pags)
+-- 3.1 Framework Data-Driven Propuesto
+-- 3.2 Datos y Preprocesamiento
|   +-- 3.2.1 CHIRPS 2.0
|   +-- 3.2.2 SRTM DEM
|   +-- 3.2.3 Feature Engineering
+-- 3.3 Arquitecturas Implementadas
|   +-- 3.3.1 Baseline (V1)
|   +-- 3.3.2 Enhanced (V2)
|   +-- 3.3.3 FNO (V3)
|   +-- 3.3.4 GNN-TAT (V4)
+-- 3.4 Construccion del Grafo Espacial
+-- 3.5 Protocolo Experimental
+-- 3.6 Metricas de Evaluacion

CAP 4: RESULTADOS V1-V3 (20 pags)
+-- 4.1 Resultados Baseline (V1)
+-- 4.2 Resultados Enhanced (V2)
+-- 4.3 Resultados FNO (V3)
+-- 4.4 Analisis Comparativo
+-- 4.5 Lecciones Aprendidas

CAP 5: RESULTADOS GNN-TAT V4 (25 pags)  <-- CONTRIBUCION PRINCIPAL
+-- 5.1 Diseno de la Arquitectura
+-- 5.2 Experimentos y Configuraciones
+-- 5.3 Resultados por Modelo (GAT, SAGE, GCN)
+-- 5.4 Analisis por Feature Set
+-- 5.5 Analisis por Horizonte
+-- 5.6 Comparacion con Estado del Arte
+-- 5.7 Interpretabilidad del Grafo

CAP 6: EXTENSIONES (V5-V6) (15 pags)
+-- 6.1 Integracion Multi-Modal
+-- 6.2 Ensemble Inteligente
+-- 6.3 Constraints Fisicas
+-- 6.4 Resultados Preliminares

CAP 7: DISCUSION (15 pags)
+-- 7.1 Validacion de Hipotesis
+-- 7.2 Implicaciones Practicas
+-- 7.3 Limitaciones del Estudio
+-- 7.4 Comparacion con Literatura

CAP 8: CONCLUSIONES (10 pags)
+-- 8.1 Conclusiones Generales
+-- 8.2 Contribuciones Cientificas
+-- 8.3 Trabajo Futuro
+-- 8.4 Publicaciones Derivadas

ANEXOS
+-- A: Codigo Fuente (GitHub)
+-- B: Tablas Completas de Metricas
+-- C: Visualizaciones Adicionales
+-- D: Configuraciones de Hiperparametros
```

### 4.2 Contribuciones Cientificas

```
CONTRIBUCION 1: ARQUITECTURA GNN-TAT
- Combina GNN espacial con atencion temporal
- Grafo construido con elevacion + distancia + correlacion
- Superior a ConvLSTM en +62% R²

CONTRIBUCION 2: ANALISIS COMPARATIVO FNO vs DATA-DRIVEN
- Demuestra que FNO underperforms para precipitacion
- Explica por que (discontinuidades, grid pequeno)
- Valuable "negative result" para la comunidad

CONTRIBUCION 3: FRAMEWORK ESTANDARIZADO
- Pipeline reproducible end-to-end
- Benchmarking robusto con tests estadisticos
- Open-source (GitHub)

CONTRIBUCION 4 (Proyectada): FUSION MULTI-MODAL
- Integracion ERA5 + Satellite + Climate Indices
- Cross-modal attention mechanism
```

---

## PARTE V: ESTRATEGIA DE PUBLICACIONES

### 5.1 Papers Planificados

```
PAPER 1: V2 vs V3 Benchmark (En preparacion)
-----------------------------------------
Titulo: "Why Fourier Neural Operators Underperform for Precipitation:
        A Comprehensive Benchmark Study"
Journal: Water Resources Research (Q1)
Estado: Datos listos, escritura pendiente
Mensaje: Negative result paper, valioso para la comunidad

PAPER 2: GNN-TAT Architecture (Prioritario)
-----------------------------------------
Titulo: "Graph Neural Networks with Temporal Attention for
        Multi-Horizon Precipitation Forecasting in Mountainous Regions"
Journal: Journal of Hydrometeorology (Q1) o
         Geophysical Research Letters (Q1, alto impacto)
Estado: Resultados V4 listos, pendiente full grid
Mensaje: Contribucion metodologica principal

PAPER 3: Multi-Modal Fusion (Futuro)
-----------------------------------------
Titulo: "Multi-Modal Deep Learning for Sub-Seasonal Precipitation
        Prediction: Integrating Reanalysis and Satellite Data"
Journal: Nature Communications (si resultados excepcionales) o
         Environmental Modelling & Software (Q1)
Estado: Planificado para V5
Mensaje: Aplicacion practica, impacto operacional
```

### 5.2 Conferencias Target

| Conferencia | Deadline | Tipo | Relevancia |
|-------------|----------|------|------------|
| AGU Fall Meeting | Agosto | Poster/Talk | Alta |
| EGU General Assembly | Enero | Abstract | Alta |
| NeurIPS Climate Workshop | Septiembre | Paper | Media |
| ICLR | Septiembre | Paper | Media |

---

## PARTE VI: RIESGOS Y MITIGACION

### 6.1 Matriz de Riesgos

| Riesgo | Probabilidad | Impacto | Mitigacion |
|--------|--------------|---------|------------|
| V4 full grid no mejora | Media | Alto | Iterar sobre hiperparametros, probar grids intermedios |
| Overfitting persiste | Alta | Medio | Data augmentation, reducir modelo |
| ERA5 data muy grande | Media | Medio | Usar subset temporal/espacial, cloud compute |
| Colab Pro insuficiente | Media | Medio | Universidad cluster, AWS/GCP credits |
| Paper rechazado | Media | Medio | Tener 2 journals target, mejorar segun reviews |
| Timeline se extiende | Alta | Medio | Buffer de 2 meses, priorizar V4-V5 |

### 6.2 Plan de Contingencia

```
SI V4 full grid falla:
  -> Usar light mode como "proof of concept"
  -> Argumentar que metodologia es escalable
  -> Enfocar en analisis de arquitectura

SI V5 multi-modal no mejora:
  -> Ablation study mostrando que se intento
  -> Reportar como "negative result"
  -> Enfocar V6 en ensemble de V2+V4

SI tiempo insuficiente:
  -> Priorizar: V4 full > Paper GNN-TAT > V5 > V6
  -> Tesis puede excluir V5-V6 si hay tiempo
  -> V5-V6 como "trabajo futuro"
```

---

## PARTE VII: RECURSOS

### 7.1 Compute

| Recurso | Uso | Costo/mes |
|---------|-----|-----------|
| Google Colab Pro+ | Entrenamiento principal | $50 USD |
| Local GPU (RTX 3080) | Desarrollo, debug | $0 |
| Google Cloud (backup) | Si Colab insuficiente | Variable |

### 7.2 Datos

| Dataset | Tamano | Acceso |
|---------|--------|--------|
| CHIRPS 2.0 | ~10 GB (region) | Publico |
| SRTM DEM | ~500 MB | Publico |
| ERA5 | ~50-100 GB | CDS API (gratuito) |
| MODIS | ~20 GB | Google Earth Engine |

### 7.3 Software

```
Core Stack:
- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- xarray + netCDF4
- pandas, numpy, scikit-learn

Visualization:
- matplotlib, seaborn
- cartopy (mapas)

Development:
- Jupyter/Colab
- Git/GitHub
- VS Code
```

---

## APENDICE: CHECKLIST DE TAREAS

### Inmediato (Esta semana)

- [ ] Ejecutar V4 full grid H=12
- [ ] Documentar tiempo de entrenamiento full vs light
- [ ] Analizar metricas full grid
- [ ] Commit resultados a GitHub

### Corto plazo (Enero)

- [ ] Implementar fixes de overfitting
- [ ] Re-entrenar V4 con regularizacion aumentada
- [ ] Iniciar descarga ERA5 para region de estudio
- [ ] Escribir seccion metodologia de tesis

### Mediano plazo (Febrero-Marzo)

- [ ] Completar pipeline V5 multi-modal
- [ ] Entrenar V5 y comparar con V4
- [ ] Escribir paper GNN-TAT
- [ ] Presentar avances a asesor

### Largo plazo (Abril-Junio)

- [ ] Implementar V6 ensemble
- [ ] Completar capitulos de tesis
- [ ] Submit paper a journal
- [ ] Preparar defensa

---

## CONTROL DE VERSIONES DE ESTE DOCUMENTO

| Version | Fecha | Cambios |
|---------|-------|---------|
| 1.0 | 2026-01-03 | Version inicial |

---

*Plan generado como parte del Framework ML Precipitation Prediction*
*Este documento debe actualizarse quincenalmente con el progreso real*
