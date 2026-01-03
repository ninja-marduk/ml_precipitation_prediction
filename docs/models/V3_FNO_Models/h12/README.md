# Reporte H=12 (V3 FNO Models)

Fuente: `models/output/V3_FNO_Models/metrics_spatial_v2_refactored_h12.csv` (corrida full en Colab). Modelos: FNO_ConvLSTM_Hybrid y FNO_Pure; experimentos BASIC, KCE, PAFC.

## Resumen
- Promedios (todas las combinaciones): BASIC RMSE 102.55 / MAE 77.74 / R^2 0.39; KCE 123.75 / 95.10 / 0.12; PAFC 127.16 / 98.47 / 0.05.
- Mejores por horizonte (RMSE mínimo): BASIC usa FNO_ConvLSTM_Hybrid en casi todos los horizontes (R^2 hasta 0.63 en H1, 0.58 en H12). KCE/PAFC mantienen R^2 ≤ 0.44 y RMSE > 96 mm.
- Sesgo: subestimación marcada (Mean_Pred_mm < Mean_True_mm), especialmente en KCE/PAFC.
- Rango R^2: −1.84 a 0.63; rango RMSE: 78.8 a 223.4 mm.

## Hallazgos clave
- Frente a V2, FNO no mejora ningún horizonte en RMSE; los mejores RMSE por horizonte son siempre superiores a los de ConvLSTM/ConvRNN de V2.
- La degradación media vs V2 es de +4–8 mm RMSE y −0.05 a −0.10 R^2 según el experimento.
- Los sesgos negativos se agravan en KCE/PAFC (Mean_Pred_mm muy por debajo de Mean_True_mm).

## Artefactos y rutas
- Métricas: `models/output/V3_FNO_Models/metrics_spatial_v2_refactored_h12.csv`
- Mapas/GIF: `models/output/V3_FNO_Models/map_exports/H{H}/{exp}/{model}/` (misma plantilla que V2).
- Logs: `models/output/V3_FNO_Models/h12/{EXP}/training_metrics/FNO_*_training_log_h12.csv`

## Nota
- Autoentrenamiento y exportes de mapas usan la celda MAP EXPORTS unificada (PNG 300 dpi, GIF streaming, reuse de NPY/JSON).
