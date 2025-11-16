# Reporte de resultados – Ventana H=12 (versión actual)

## Contexto y diferencias contra la corrida previa
- **Configuración declarada** (`models/base_models_Conv_STHyMOUNTAIN_V2.ipynb:5`): `input_window=60`, `batch_size=2`, `epochs=150`, `train_val_split=0.8`, `effective_batch_size=None`, `prediction_batch_size=1`, `enabled_horizons=[12]`. Sin embargo, la celda impresa en el notebook muestra `Prediction horizon: 12 months` aunque el campo `CONFIG['horizon']` en el código está seteado en `6`; esto confirma que las salidas almacenadas pertenecen a una ejecución anterior y ya no reflejan el código fuente actual.
- **Preprocesamiento actual** (`cell 9`): divide la serie temporal antes de generar ventanas mediante `compute_split_indices`, aplica escalado independiente a los bloques continuos y de lags, y deja las one-hot acotadas. Esta lógica nunca se ejecutó durante la corrida cuyos logs están en la notebook (de lo contrario, el número de ventanas reportado no sería 357/90 y las predicciones no tendrían tamaño 1×12).
- **Ejecución registrada** (`cell 13 outputs`): todos los mensajes siguen indicando “BASIC processed: 357 train, 90 validation samples” y la inferencia se hace sobre `X_va[-1:]` (única ventana). Con el código actual se esperarían 343 ventanas de entrenamiento y 33 de validación sin fuga temporal. Conclusión: las métricas y modelos del directorio `models/output/V2_Enhanced_Models/h12` pertenecen a la versión previa del pipeline y deberían regenerarse tras re-ejecutar el notebook actualizado.

## Análisis de métricas oficiales (`models/output/V2_Enhanced_Models/metrics_spatial_v2_refactored_h12.csv`)
- **Cobertura**: 360 filas (3 experimentos × 10 modelos × 12 horizontes). El archivo no incluye la columna `mean_bias_pct` ni las alertas de sesgo que el nuevo código imprimiría, evidencia adicional de que las salidas no fueron regeneradas.
- **Mejores combinaciones por horizonte (RMSE en mm/mes)**:

| Experiment | Model                    | H | RMSE | R²       | Bias (%) |
|------------|-------------------------|---|------|----------|----------|
| BASIC      | ConvLSTM                | 1 | 24.8 | -0.72    | +19.7    |
| BASIC      | ConvRNN_Enhanced        | 2 | 17.0 | 0.82     | −3.6     |
| BASIC      | ConvLSTM_Bidirectional  | 3 | 20.8 | 0.90     | −5.0     |
| BASIC      | Transformer_Baseline    | 4 | 87.4 | −1.27    | −34.8    |
| BASIC      | ConvLSTM_Enhanced       | 5 | 19.1 | 0.60     | +14.3    |
| ...        | ...                     | … | ...  | ...      | ...      |
| KCE        | ConvRNN_Enhanced        | 1 | 33.1 | −2.08    | +9.3     |
| KCE        | ConvLSTM_EfficientBidir | 2 | 19.3 | 0.76     | +4.6     |
| ...        | ...                     | … | ...  | ...      | ...      |
| PAFC       | ConvRNN                 | 1 | 26.3 | −0.94    | +10.7    |
| PAFC       | ConvLSTM                | 2 | 25.8 | 0.57     | +3.8     |
| ...        | ...                     | … | ...  | ...      | ...      |

  - **Sesgos importantes**: incluso los mejores modelos para varios horizontes muestran desbalances >10 % (BASIC H1/5, KCE H5–7, PAFC H1/7/10/12). Esto valida la necesidad del chequeo de sesgo que añade la versión actual del notebook.
  - **R² negativos severos**: Transformer_Baseline y algunas variantes KCE/PAFC entregan R²≤−1 y hasta −7 (PAFC H11), lo que evidencia inestabilidad numérica.
- **Totales de precipitación**: tres combinaciones (dos en PAFC y una en KCE) producen `TotalPrecipitation_Pred < 0`, es decir, precipitación acumulada negativa a escala mensual-regional, algo físicamente imposible y señal de errores de escala (`models/output/V2_Enhanced_Models/metrics_spatial_v2_refactored_h12.csv` filas 236, 350, 356).
- **Media vs. predicción**: en horizontes largos (H≥9) los modelos BASIC mantienen RMSE ≤ 40 mm, pero los sesgos superan ±20 % cuando interviene el Transformer. Los experimentos KCE y PAFC degradan notablemente las métricas respecto a BASIC pese a añadir más features, replicando la conclusión del reporte H=6.

## Observaciones sobre los artefactos en `models/output/V2_Enhanced_Models/h12`
- Los `*_training_log_h12.csv` muestran que `val_loss` alcanza su mínimo antes de la época 10–20 y aun así el entrenamiento continúa hasta la paciencia completa (100 + épocas). Aunque la nueva versión guarda `best_epoch` y `train_val_gap` en los JSON, esos campos no existen en los archivos actuales (p.ej. `h12/BASIC/training_metrics/ConvLSTM_history.json`), indicando que se generaron con la lógica antigua.
- Existen subcarpetas `*_logs` pero no hay rastro de los avisos de sesgo ni del control de escala añadido en el código nuevo. Algunos modelos deberían haber sido descartados por la razón `scale_ratio > 50` o por sesgo >10 %; al no estar vigente la verificación, los CSV contienen columnas con valores físicamente incoherentes.
- No hay registro alguno para `models/output/V2_Enhanced_Models/h12/KCE/ConvLSTM_Enhanced_training_log_h12.csv` en `git`, pero en el filesystem está presente. Conviene asegurarse de que estos logs se versionen sólo si son necesarios.

## Incongruencias detectadas
1. **Notebook vs. salidas**: el código actual espera múltiples ventanas de validación y calcula métricas sobre todas ellas, pero las salidas guardadas corresponden a una ejecución que evaluó una sola ventana (`batch_predict` sólo imprimió 1/1). Se requiere re-ejecutar tras los cambios para validar que la nueva lógica funciona.
2. **Configuración divergente**: `CONFIG['horizon']` quedó en 6 mientras que `enabled_horizons=[12]`. El valor impreso en la celda y los resultados exportados usan 12 meses; cualquier lector que abra la notebook sin ejecutar podría asumir erróneamente que la corrida actual fue a 6 meses.
3. **Features extendidos siguen degradando**: los experimentos KCE/PAFC muestran RMSE >50 mm y R² negativos para múltiples horizontes; al no existir un benefit claro, se recomienda revisar la normalización por bloques (ahora incluida) y repetir la corrida para confirmar si persisten los sesgos.
4. **Transformer_Baseline**: en KCE/PAFC produce totales negativos y R² extremos, pero el script no lo detectó. Tras re-ejecutar con la nueva guardia de escala, se espera que dichos modelos sean descartados o al menos reportados.

## Recomendaciones inmediatas
1. **Re-ejecutar el notebook actualizado** para regenerar los artefactos de `h12` (y, si aplica, `h6`). Esto asegurará que los splits sin fuga, el escalado por bloques y los chequeos de sesgo/escala entren en vigor.
2. **Validar los tamaños de ventana producidos por `preprocess_data`** (imprimir `X_tr.shape[0]` y `X_va.shape[0]` tras la nueva división). Si el conteo no coincide con lo esperado (343/33 para H=12), ajustar `compute_split_indices`.
3. **Revisar los modelos con sesgo sistemático**: usar los nuevos logs de `mean_bias_pct` para descartar automáticamente horizontes con |bias|>10 % o aplicar regularización adicional en los experimentos KCE/PAFC antes de considerarlos para despliegue.
