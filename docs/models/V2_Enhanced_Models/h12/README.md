# Reporte actualizado H=12 (V2 Enhanced Models)

Fuente: `models/output/V2_Enhanced_Models/metrics_spatial_v2_refactored_h12.csv` (corrida full en Colab).

## Resumen
- Experimentos: BASIC, KCE, PAFC. Modelos: variantes ConvLSTM/ConvRNN.
- Promedios (todas las combinaciones): BASIC RMSE 98.17 / MAE 73.92 / R^2 0.44; KCE 120.01 / 93.85 / 0.13; PAFC 119.43 / 92.79 / 0.15.
- Mejores por horizonte (RMSE mínimo): dominan variantes ConvLSTM (Bidirectional, EfficientBidir, Residual). No se necesitan Transformers.
- Sesgo: subestimación general (Mean_Pred_mm < Mean_True_mm); BASIC presenta menor sesgo que KCE/PAFC.
- Rango R^2: −1.34 a 0.65; los peores casos son horizontes altos e inestables.

## Hallazgos clave
- BASIC logra los mejores R^2 (≈0.65 en H1, ≈0.63 en H12) con RMSE 76–84 mm en horizontes cortos y 83–94 mm en largos.
- KCE y PAFC empeoran RMSE y R^2 frente a BASIC pese a más features; el sesgo negativo aumenta.
- Los valores de totales y medias están físicamente consistentes; no se detectan totales negativos en esta corrida full.

## Artefactos y rutas
- Métricas: `models/output/V2_Enhanced_Models/metrics_spatial_v2_refactored_h12.csv`
- Mapas/GIF (formato alineado con V3): `models/output/V2_Enhanced_Models/map_exports/H{H}/{exp}/{model}/`
- Logs: `models/output/V2_Enhanced_Models/h12/{EXP}/training_metrics/*_training_log_h12.csv`

## Uso
- La celda MAP EXPORTS (notebook V2) está sincronizada con V3: PNG a 300 dpi, GIF streaming (`loop=0`), reutiliza NPY/JSON si existen y permite ciclar todas las ventanas o elegir `MAP_SAMPLE_INDEX`.
