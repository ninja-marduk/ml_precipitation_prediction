# V2 Enhanced Models - H=12 (light 5x5)

Analisis de `models/output/V2_Enhanced_Models` para el horizonte H=12 entrenado en modo light (rejilla 5x5, 25 pixeles). Incluye resumen del notebook base, estadistica de desempeno y comparativa contra H=6.

## Contexto del notebook base
- Notebook: `models/base_models_Conv_STHyMOUNTAIN_V2.ipynb`.
- Configuracion: `input_window=60`, `horizon=12`, `epochs=150`, `batch_size=2`, `learning_rate=1e-3`, `patience=50`, `light_mode=True`, `light_grid_size=5`.
- Subconjunto espacial: lat `[28:33]`, lon `[30:35]` (25 pixeles), 518 timesteps, 12 features de entrada (precipitacion total, max/min diarios, std diaria, month_sin, etc.).
- Tests rapidos: losses y capas custom OK; ConvGRU2D ausente (modelos ConvGRU omitidos). Prediccion por lotes y losses unitarios marcaron SUCCESS.
- Experimentos ejecutados: BASIC, KCE y PAFC (10 arquitecturas por experimento).

## Archivos generados en `docs/models/V2_Enhanced_Models/h12`
- `metrics_summary.csv`: medias y desvios de RMSE/MAE/R2/sesgo por experimento-modelo (promediado sobre todos los horizontes 1-12).
- `best_by_rmse.csv`, `best_by_r2.csv`: mejor fila por experimento segun RMSE minimo o R2 maximo.
- `training_best_epochs.csv`: epoca con menor `val_loss` por experimento-modelo.
- Figuras: `rmse_by_model_experiment.png`, `r2_by_model_experiment.png`, `bias_by_model_experiment.png`, `best_val_loss_matrix.png`.
- Comparativa h6 vs h12: `comparison_h6_vs_h12.csv`, `rmse_delta_h12_vs_h6.png`, `r2_delta_h12_vs_h6.png`.
- Visuales ya existentes en `models/output/V2_Enhanced_Models/comparisons/` utiles para el paper: `metrics_evolution_by_horizon_h12.png`, `normalized_metrics_comparison_h12.png`, `metrics_summary_table.png`.

## Hallazgos clave H=12 (light 5x5)
- Rendimiento medio por experimento (RMSE_mean / R2_mean / sesgo medio mm):
  - BASIC: 63.4 / 0.56 / -5.9
  - KCE: 96.5 / -0.36 / -32.1
  - PAFC: 115.2 / -1.47 / -53.8
- Mejores modelos por RMSE (ver `best_by_rmse.csv`):
  - BASIC: `ConvRNN` en H=1 (RMSE=51.63, MAE=33.28, R2=0.71, sesgo=-12.0 mm).
  - KCE: `ConvLSTM_MeteoAttention` en H=9 (RMSE=58.87, MAE=42.46, R2=0.62, sesgo=-2.0 mm).
  - PAFC: `ConvLSTM_Residual` en H=3 (RMSE=59.17, MAE=42.09, R2=0.63, sesgo=-4.0 mm).
- Sesgos: BASIC mantiene bias cercano a cero; KCE y PAFC subestiman precipitacion con bias medios -32 mm y -54 mm. Modelos con atencion meteorologica en KCE reducen el sesgo.
- Convergencia (ver `training_best_epochs.csv` y `best_val_loss_matrix.png`):
  - Mejor `val_loss` promedio: BASIC 0.69, KCE 1.04, PAFC 1.08.
  - Minimos por experimento: ConvRNN BASIC (epoch 74, val_loss 0.59), ConvLSTM_MeteoAttention KCE (epoch 9, val_loss 0.94), ConvRNN PAFC (epoch 10, val_loss 0.96).

## Comparativa H=6 vs H=12 (mean sobre horizontes)
- Deltas calculados como h12 - h6 (`comparison_h6_vs_h12.csv`, figuras `rmse_delta_h12_vs_h6.png`, `r2_delta_h12_vs_h6.png`):
  - RMSE medio mejora (deltas negativas) en BASIC -32.7, KCE -75.2, PAFC -10.3.
  - R2 medio sube en BASIC (+0.10) y KCE (+1.86, aun queda negativo), pero cae en PAFC (-1.43).
  - Mayor salto positivo en R2: `ConvLSTM_Enhanced` KCE (+15.52, -371.76 RMSE) indicando fuerte correccion respecto a H=6 aunque con escala de error distinta.
  - Modelo con delta mas controlada y consistente: `Transformer_Baseline` BASIC (delta_RMSE -56.25, delta_R2 +0.39) y `ConvRNN` PAFC (delta_RMSE -54.85, delta_R2 +0.36).
- Implicacion: para modo light 5x5, extender el horizonte a 12 meses no degrada el error en BASIC y KCE (incluso mejora), mientras PAFC pierde poder explicativo (R2 negativo) y requiere ajuste o regularizacion adicional.

## Ideas para el paper
- Resaltar que el modo light (5x5) permite entrenar 10 arquitecturas x 3 experimentos con curvas de aprendizaje estables y `val_loss` bajos; la mejor configuracion BASIC logra R2>0.7 en H=12.
- En KCE, los modelos con atencion meteorologica son los unicos que mantienen R2 positivo y sesgo cercano a cero; las variantes residuales/bidireccionales siguen con sesgo negativo.
- PAFC muestra caida de R2 al pasar a H=12; se sugiere investigar balance de clases o normalizacion regional para ese subconjunto.
- Usar las figuras de comparacion (`rmse_by_model_experiment.png`, `r2_by_model_experiment.png`, `rmse_delta_h12_vs_h6.png`) para ilustrar efectos del horizonte y justificar la eleccion de modelos para despliegue.
