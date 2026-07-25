# Analysis figures

Data, feature and study-area analysis figures.

| Script | Output(s) | Input data |
|--------|-----------|------------|
| `bimodal_climograph.py` | `.docs/conferences/EGU26/poster/figures/bimodal_cycle.png` | `data/output/complete_dataset_*.nc` |
| `k_selection_sweep.py` | `.docs/thesis/figures/k_selection_sweep.png` + `models/scripts/output/k_selection_sweep_metrics.csv` | `data/output/complete_dataset_*.nc` |
| `kce_cluster_analysis.py` | `.docs/thesis/figures/kce_k3_clusters_map.png` + `.docs/thesis/figures/monthly_cluster_profiles.png` | `data/output/complete_dataset_*.nc`, `data/input/shapes/MGN_Departamento.shp` |
| `spatial_r2.py` | `.docs/papers/5/figures/spatial_r2_map_3panel.png` | `models/output/V2_*`, `V4_*`, `V10_*` map exports |
| `study_area.py` | `.docs/papers/5/figures/boyaca.png` (study area DEM map) | `notebooks/data/output/complete_dataset_*.nc`, `data/input/MGN_Departamento.shp` |
