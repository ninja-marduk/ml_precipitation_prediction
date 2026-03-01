# V2 Enhanced Models — H=6 (modo light 5x5)

Análisis a partir de `models/output/V2_Enhanced_Models` para el horizonte H=6 entrenado en modo light (rejilla 5x5 con 25 píxeles). Los artefactos generados se guardan en este directorio para facilitar su uso en el paper.

## Setup observado en el notebook base
- Notebook: `models/base_models_Conv_STHyMOUNTAIN_V2.ipynb`.
- Configuración: `input_window=60`, `horizon=6`, `epochs=150`, `batch_size=2`, `learning_rate=1e-3`, `patience=50`, `light_mode=True`, `light_grid_size=5`.
- Subconjunto espacial usado: lat `[28:33]`, lon `[30:35]` (25 píxeles).
- Dataset cargado con 518 timesteps y 5x5 píxeles; variables incluyeron `total_precipitation`, extremos diarios y `month_sin` (entre otras).
- Tests rápidos: losses y capas custom pasaron; ConvGRU2D no disponible (modelos ConvGRU omitidos).
- Experimentos ejecutados: BASIC, KCE y PAFC (10 modelos por experimento).

## Archivos generados aquí
- `metrics_summary.csv`: resumen por experimento/modelo (media y desviación de RMSE, MAE, R², sesgos).
- `best_by_rmse.csv`, `best_by_r2.csv`: mejores modelos por experimento.
- `training_best_epochs.csv`: época con mínimo `val_loss` por modelo/experimento.
- Figuras: `rmse_by_model_experiment.png`, `r2_by_model_experiment.png`, `bias_by_model_experiment.png`, `best_val_loss_matrix.png`.

## Hallazgos principales (H=6, light 5x5)
- **Rendimiento global**: BASIC mantiene menor error medio (RMSE≈96) y R² medio positivo (0.45). KCE se degrada (RMSE medio≈165, R² medio≈-2.07) por modelos con gran sesgo negativo. PAFC queda intermedio (RMSE medio≈116, R² medio≈0.15).
- **Mejores modelos por experimento** (ver `best_by_rmse.csv`):
  - BASIC: `ConvLSTM_Bidirectional` — RMSE=81.9, MAE=58.9, R²=0.61, sesgo=-24.8 mm.
  - KCE: `ConvLSTM_Residual` — RMSE=92.6, MAE=72.9, R²=0.50, sesgo=+4.2 mm.
  - PAFC: `ConvLSTM_EfficientBidir` — RMSE=92.1, MAE=72.7, R²=0.51, sesgo=+2.4 mm.
- **Sesgos**:
  - Sesgo medio por experimento: BASIC -17.6 mm, KCE -44.8 mm, PAFC -15.4 mm (predicciones tienden a subestimar, especialmente en KCE).
  - Mayor subestimación: `ConvLSTM_Enhanced` en KCE (sesgo -333 mm; `bias_by_model_experiment.png`).
  - Solo algunos modelos KCE muestran ligera sobreestimación (sesgo +3 a +8 mm).
- **Convergencia** (ver `training_best_epochs.csv` y `best_val_loss_matrix.png`):
  - Los mejores `val_loss` aparecen muy temprano (épocas 0–2) para los top models por experimento, sugiriendo que el aprendizaje se estabiliza rápido con 25 píxeles.
  - Distribución de mínimos `val_loss`: media BASIC 0.61, KCE 0.86, PAFC 0.89; varianza mayor en PAFC.

## Ideas para el paper
- Destacar que el modo light (5x5) permite entrenamiento rápido sin sacrificar el desempeño del mejor modelo BASIC (R²>0.6).
- Contrastar BASIC vs KCE/PAFC: KCE introduce sesgo negativo notable y R² negativos en varios modelos; PAFC mejora respecto a KCE pero aún por debajo de BASIC.
- Resaltar que los modelos bidireccionales y residuales lideran por experimento; los modelos con atención meteorológica muestran sobreajuste (val_loss más altos y sesgos).
- Incluir las figuras generadas (RMSE, R², sesgo) para mostrar la variabilidad entre modelos, y la matriz de `val_loss` para evidenciar la rapidez de convergencia.

## Rutas clave de datos crudos
- Métricas espaciales consolidadas: `models/output/V2_Enhanced_Models/metrics_spatial_v2_refactored_h6.csv`.
- Logs de entrenamiento y curvas: `models/output/V2_Enhanced_Models/h6/<EXPERIMENTO>/training_metrics/`.
