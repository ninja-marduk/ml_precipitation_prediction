# Benchmark figures

Multi-architecture comparison figures for the eight model families.

| Script | Output(s) | Input data |
|--------|-----------|------------|
| `suite.py` | `models/output/final_figures/*.png` (horizon_degradation, feature_heatmap, parameter_efficiency, model_ranking, training_dynamics, cross_architecture, summary) | `models/output/V*/metrics_spatial_*.csv` |
| `reviewer_annotations.py` | Annotated benchmark figures for journal reviewer responses (DEM elevation map, spatial R², scatter, elevation strata, time series) | `models/output/V*/`, `data/output/complete_dataset_*.nc` |
| `radar_chart.py` | `.docs/papers/5/figures/radar_chart.png` (+ delivery copy) | Headline metrics dict (in-script) |
