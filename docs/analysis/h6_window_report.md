# Reporte de resultados - Ventana H=6

## Contexto de entrenamiento
- Notebook base: `models/base_models_Conv_STHyMOUNTAIN_V2.ipynb` (celdas 4-13) con `input_window=60`, `batch_size=2` (pero el `train_model` entrena con `batch_size//2 = 1`), `epochs=150`, `patience=50`, `learning_rate=1e-3` y splits cronologicos 80/20 tras construir ventanas deslizantes de 60+H pasos.
- Dataset (`data/output/complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc`) contiene 518 meses y grilla 6165 con 15 variables; las caracteristicas por experimento son:
  - **BASIC**: 12 features (temporalidad + estadisticos diarios + terreno).
  - **KCE**: BASIC + one-hot de clases de elevacion.
  - **PAFC**: KCE + `total_precipitation_lag{1,2,12}`.
- Los artefactos de entrenamiento para H=6 viven en `models/output/V2_Enhanced_Models/h6/{BASIC,KCE,PAFC}/training_metrics`. El archivo `models/output/V2_Enhanced_Models/metrics_spatial_v2_refactored_h6.csv` consolida las metricas espaciales.

## Resumen cuantitativo de `metrics_spatial_v2_refactored_h6.csv`
- **Mejores combinaciones por horizonte** (RMSE en mm/mes):

| Experiment   | Model                     |   H |    RMSE |    MAE |    R2 |   Mean_True_mm |   Mean_Pred_mm |
|:-------------|:--------------------------|----:|--------:|-------:|------:|---------------:|---------------:|
| BASIC        | ConvLSTM                  |   1 |  64.60  | 48.78  |  0.61 |        201.83  |        220.78  |
| BASIC        | ConvRNN_Enhanced          |   2 |  48.39  | 36.12  |  0.75 |        247.05  |        238.10  |
| BASIC        | ConvLSTM_Residual         |   3 |  57.51  | 46.97  |  0.38 |        243.39  |        224.16  |
| BASIC        | ConvLSTM                  |   4 |  46.92  | 39.14  |  0.73 |        117.04  |        144.28  |
| BASIC        | ConvLSTM_EfficientBidir   |   5 |  32.92  | 25.43  |  0.38 |         76.59  |         64.97  |
| BASIC        | ConvRNN_Enhanced          |   6 |  77.03  | 50.16  |  0.55 |        125.40  |        114.90  |
| KCE          | ConvLSTM                  |   1 |  75.02  | 64.32  |  0.47 |        201.83  |        234.19  |
| KCE          | ConvLSTM_Bidirectional    |   2 |  58.51  | 46.01  |  0.63 |        247.05  |        232.97  |
| KCE          | ConvLSTM                  |   3 |  53.46  | 43.06  |  0.47 |        243.39  |        229.33  |
| KCE          | ConvLSTM_Residual         |   4 |  66.46  | 56.24  |  0.46 |        117.04  |        149.13  |
| KCE          | ConvLSTM_Residual         |   5 |  60.63  | 54.26  | -1.10 |         76.59  |        129.19  |
| KCE          | ConvLSTM_Residual         |   6 | 105.50  | 76.74  |  0.15 |        125.40  |        106.99  |
| PAFC         | ConvLSTM_Bidirectional    |   1 |  82.55  | 73.82  |  0.36 |        201.83  |        223.05  |
| PAFC         | ConvLSTM_Bidirectional    |   2 |  52.25  | 40.10  |  0.71 |        247.05  |        247.64  |
| PAFC         | ConvLSTM_Residual         |   3 |  53.15  | 41.14  |  0.47 |        243.39  |        229.47  |
| PAFC         | ConvLSTM_Residual         |   4 |  71.61  | 60.99  |  0.37 |        117.04  |        162.47  |
| PAFC         | ConvLSTM_Bidirectional    |   5 |  75.47  | 65.32  | -2.25 |         76.59  |        136.13  |
| PAFC         | ConvLSTM_Residual         |   6 | 102.20  | 84.19  |  0.20 |        125.40  |        162.31  |

- **Tendencias por experimento**:
  - *BASIC*: mejores RMSE y R2 en todos los horizontes. R2 positivo hasta H=6 (0.55) pero se vuelve negativo para varios modelos a H=5 cuando se consideran todas las variantes; vies sistematico en media (p. ej. +23 % en H=4 y -15 % en H=5 para los ganadores).
  - *KCE*: la inclusion de one-hot de elevacion empeora la generalizacion (val_loss minimos 0.67 frente a 0.31-0.36 en BASIC). Sesgos muy altos (+69 % en H=5) y R2 negativo para H=5 aun en el mejor modelo; `Transformer_Baseline` explota con RMSE >34 000 y `TotalPrecipitation_Pred`  -1.2108 mm, evidencia de inversion de escala fallida.
  - *PAFC*: los lags de precipitacion estabilizan H=2-3 (R2=0.71 y 0.47), pero los horizontes largos introducen sobreestimaciones (+78 % en H=5) y R2 negativos.
- **Distribucion global**:
  - Media de RMSE por experimento/horizonte (ver calculo en `docs/analysis/h6_window_report.md` fuente `metrics_spatial_v2_refactored_h6.csv`): BASIC 64 mm para H5 y 107 mm en H=6; KCE y PAFC superan 70-120 mm incluso en horizontes cortos.
  - El entrenamiento `ConvLSTM` basico logra mejores val_loss tempranas (epoca 6; `BASIC/.../ConvLSTM_training_log_h6.csv`), mientras que modelos con atencion o transformadores no convergen antes de la paciencia configurada.

## Observaciones sobre los artefactos de entrenamiento (`models/output/V2_Enhanced_Models/h6/**`)
- Las curvas de aprendizaje muestran sobreajuste temprano: en BASIC, `val_loss` toca 0.33 en la epoca 6 (`ConvLSTM_training_log_h6.csv`) pero el entrenamiento continua hasta la 56 con `val_loss` >0.41.
- KCE y PAFC nunca bajan de `val_loss0.64-0.69` incluso con los modelos mas simples, lo que sugiere que las nuevas features introducen ruido o desbalance; ademas, los historicos registran mejores epocas en 4 iteraciones pero el scheduler sigue reduciendo LR hasta 1e-6 sin mejora sustancial (`ConvLSTM_Attention_training_log_h6.csv`, etc.).
- Falta el `ConvLSTM_Enhanced_training_log_h6.csv` para KCE aunque existen `*_history.json`; esto impide reproducir la convergencia y apunta a fallos en el callback de logging.
- Los hiperparametros son identicos entre experimentos (`*_hyperparameters.json`), por lo que la degradacion proviene de los datos y no de configuraciones distintas.

## Hallazgos clave en `models/base_models_Conv_STHyMOUNTAIN_V2.ipynb`
1. **Loop de horizontes mal indentado** (`cell 13`): el bloque `for exp_name in CONFIG['feature_sets']` esta fuera del `for horizon ...`, por lo que solo se entrenan/guardan resultados para el ultimo horizonte del listado. Los artefactos H=6 existen porque se reconfiguro temporalmente `CONFIG['enabled_horizons']=[6]`, pero el codigo actual no puede iterar multiples horizontes.
2. **Evaluacion parcial**: la funcion de inferencia usa `y_hat_sc = batch_predict(model, X_va[-1:], ...)`; las metricas en `metrics_spatial_v2_refactored_h6.csv` provienen solo del ultimo bloque de validacion, no del conjunto completo. El sesgo observado podria deberse a que el sample final no es representativo.
3. **Division temporal con fuga**: `windowed_arrays` genera ventanas solapadas y el split 80/20 se aplica *despues* de construir las secuencias. Ventanas de entrenamiento cerca del corte incluyen targets dentro del segmento de validacion (ej.: inicio 296  usa pasos 296-365). Esto explica parte del optimismo en R2 para H=1-2 y la pobre extrapolacion en horizontes altos.
4. **Batch efectivo = 1**: `train_model` usa `batch_size=config['batch_size']//2`, que para la configuracion por defecto da 1. La varianza del gradiente es alta y ralentiza el GPU; ademas, el valor en `*_hyperparameters.json` no coincide con la realidad.
5. **Seleccion de loss inconsistente**: para KCE/PAFC se construyen `horizon_weights` constantes, pero enseguida se reasignan dos veces con `compute_horizon_weights`, y `consistency_weight` se fija siempre en 0.1 independientemente del experimento. Revisar la intencion original (probablemente ponderar horizontes segun importancia y aplicar regularizacion temporal diferenciada).
6. **Transformador sin inversion correcta**: los valores masivos negativos en `Transformer_Baseline` (ver CSV) indican que el `scaler.inverse_transform` fallo o se aplico en el orden equivocado; revisar `batch_predict` y `scaler` compartidos.
7. **Preprocesamiento redundante**: `preprocess_data` se ejecuta para cada experimento y horizonte sin caching; ademas, los parametros `lat`, `lon` nunca se usan. Considerar construir una cache `data_splits[(exp, horizon)]` y, si procede, recortar espacialmente por lat/lon para bajar la memoria.
8. **Registro de metricas incompleto**: algunos experimentos carecen de CSVs (p.ej. `KCE/ConvLSTM_Enhanced`), y los callbacks guardan `.h5` legacy, lo que dispara warnings recurrentes.

## Recomendaciones
1. **Corregir el bucle de horizontes y evaluar multiples muestras** para que `results` cubra todos los H y todas las ventanas de validacion; aprovechar `model.predict(X_va, batch_size=...)` y acumular metricas espaciales promedio y por percentil.
2. **Rehacer el split temporal evitando fugas**: dividir antes de construir ventanas (p. ej. usar el indice de tiempo para cortar en 80 %) o usar `TimeSeriesSplit`/`GroupKFold` sobre el indice de inicio de ventana.
3. **Rebalancear las features adicionales**: los resultados muestran que KCE/PAFC introducen sesgos positivos fuertes. Normalizar cada bloque categorico/lag por separado, anadir dropout espacial o seleccionar automaticamente las features mas informativas (SHAP/permutation) antes de concatenar.
4. **Revisar `Transformer_Baseline`**: confirmar que el escalador usado para invertir predicciones coincide con el del target; en KCE los valores negativos gigantes sugieren overflow o dtype incorrecto.
5. **Usar `max(1, config['batch_size']//2)` o, mejor, exponer un `effective_batch_size` con gradient accumulation** para estabilizar el training sin forzar lotes de tamano 1.
6. **Loggear y versionar correctamente**: garantizar que todos los `training_log_h6.csv` se escriban y que los JSON incluyan `best_epoch`, `best_val_loss` y `train/val_gap` para automatizar la comparacion entre features.
7. **Validar metricas espaciales agregadas**: anadir scripts que comparen `TotalPrecipitation` vs `Pred` y alerten cuando el sesgo supere 10 %; eso hubiese resaltado las salidas negativas del transformador y los sobreajustes de KCE/PAFC.

## Proximos pasos sugeridos
1. Repetir los experimentos BASIC/KCE/PAFC tras corregir el loop de horizontes y evaluar todo el set de validacion para confirmar si el sesgo se mantiene.
2. Introducir regularizacion especifica para los nuevos features (p. ej. `LayerNormalization` + `Dropout` tras concatenar las one-hot de elevacion y los lags).
3. Ajustar la funcion de perdida compuesta para que refleje las prioridades reales por horizonte (p. ej. enfocarse en H3 con mayores pesos) y medir el impacto sobre RMSE/R2.

