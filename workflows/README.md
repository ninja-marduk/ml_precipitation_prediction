# Workflows - End-to-End Pipeline

Reproducible pipeline for the ML Precipitation Prediction project, covering data preparation through model evaluation.

## Quick Start

```bash
# Full pipeline (stages 1-9)
python workflows/run_pipeline.py

# Only post-training stages (V10 fusion + benchmarks + figures)
python workflows/run_pipeline.py --from 7

# Paper 5: Use intra-cell DEM models instead of Paper 4 defaults
python workflows/run_pipeline.py --from 7 --intracell-dem

# Paper 5 with a specific feature bundle
python workflows/run_pipeline.py --from 7 --intracell-dem --bundle BASIC_PCA6

# Specific stages
python workflows/run_pipeline.py --stages 8 9

# Dry run (preview without executing)
python workflows/run_pipeline.py --dry-run

# Skip GPU-dependent stages
python workflows/run_pipeline.py --skip-gpu
```

---

## Paper 4 vs Paper 5: `--intracell-dem`

The pipeline supports two modes via the `--intracell-dem` flag:

| Mode | Command | Prediction Paths | Description |
|------|---------|-----------------|-------------|
| **Paper 4** (default) | `--from 7` | `V2_Enhanced_Models/`, `V4_GNN_TAT_Models/` | Original BASIC/KCE/PAFC features |
| **Paper 5** | `--from 7 --intracell-dem` | `V2_Enhanced_Models_intracell_dem/`, `V4_GNN_TAT_Models_intracell_dem/` | Intra-cell DEM features |

When `--intracell-dem` is active, stages 7-9 automatically use the Paper 5 prediction directories:

- **Stage 7 (V10 Fusion):** Reads from `_intracell_dem/` predictions, saves to `V10_Late_Fusion_intracell_dem/`
- **Stage 8 (Benchmarks):** Computes ACC, FSS, elevation metrics from `_intracell_dem/` predictions. Results saved to `scripts/benchmark/output/intracell_dem/{bundle}/`
- **Stage 9 (Figures):** Generates tables and figures from the intracell_dem benchmark results

**Available bundles** (use `--bundle`):

| Bundle | Features | Count |
|--------|----------|:-----:|
| `BASIC_D10` (default) | BASIC + 10 elevation deciles | 22 |
| `BASIC_PCA6` | BASIC + 6 PCA components | 18 |
| `BASIC_D10_STATS` | BASIC + deciles + statistics | 27 |

---

## Pipeline Stages

| Stage | Script | Description | GPU |
|:-----:|--------|-------------|:---:|
| 1 | `01_download_chirps.py` | Download and crop CHIRPS daily precipitation | No |
| 2 | `02_aggregate_monthly.py` | Aggregate daily to monthly totals + climatology | No |
| 3 | `03_merge_dem.py` | Merge DEM elevation data with CHIRPS grid | No |
| 4 | `04_feature_engineering.py` | Elevation/precipitation clustering + features | No |
| 5 | `05_train_v2_convlstm.py` | Train V2 ConvLSTM (TensorFlow) | **Yes** |
| 6 | `06_train_v4_gnn_tat.py` | Train V4 GNN-TAT (PyTorch + PyG) | **Yes** |
| 7 | `07_late_fusion_v10.py` | V10 Ridge regression fusion (V2 + V4) | No |
| 8 | `08_benchmark_metrics.py` | ACC, FSS, elevation-stratified, Friedman | No |
| 9 | `09_generate_figures.py` | Publication-quality figures + LaTeX tables | No |

### Stage dependencies

```
1 (CHIRPS) --> 2 (Monthly) --> 3 (DEM) --> 4 (Features)
                                              |
                               +--------------+--------------+
                               |                             |
                          5 (V2 ConvLSTM)              6 (V4 GNN-TAT)
                               |                             |
                               +-------------+---------------+
                                             |
                                        7 (V10 Fusion)
                                             |
                                        8 (Benchmarks)
                                             |
                                        9 (Figures)
```

---

## Reproducibility Guide

This project spans two papers. Each has its own training configuration and output directories.

### Paper 4 (Accepted - MDPI Hydrology)

**Title:** A Data-Driven Deep Learning Framework for Monthly Precipitation Prediction in Complex Mountainous Terrain: Systematic Evaluation of Hybrid Architectures

**Feature sets:** BASIC (12), KCE (15), PAFC (18)

**Original notebooks (preserved for reproducibility):**
- `models/base_models_conv_sthymountain_v2.ipynb` - V2 ConvLSTM
- `models/base_models_gnn_tat_v4.ipynb` - V4 GNN-TAT
- `models/base_models_late_fusion_v10.ipynb` - V10 Late Fusion

**Output directories:**
```
models/output/
  V2_Enhanced_Models/         # Paper 4 V2 predictions
  V4_GNN_TAT_Models/          # Paper 4 V4 predictions
  V10_Late_Fusion/            # Paper 4 V10 fusion (R2=0.668)
```

**To reproduce Paper 4 results (no GPU needed for stages 7-9):**
```bash
python workflows/run_pipeline.py --from 7
```

### Paper 5 (In Preparation)

**Title:** Systematic Evaluation of Hybrid Architectures, Ensemble Strategies, and Emerging Paradigms

**New feature sets:** BASIC_D10 (22), BASIC_PCA6 (18), BASIC_D10_STATS (27)

**Technique:** Intra-cell DEM heterogeneity. Each CHIRPS cell (~5.5 km) contains ~3,477 DEM pixels at 90m resolution. Features capture sub-grid topographic heterogeneity through elevation deciles and PCA.

**Colab notebooks (GPU training):**
- `models/intracell_dem/train_v2_convlstm_intracell_dem.ipynb` - V2 with new features
- `models/intracell_dem/train_v4_gnn_tat_intracell_dem.ipynb` - V4 with new features
- `models/intracell_dem/evaluate_v10_fusion_intracell_dem.ipynb` - V10 fusion + evaluation

**Output directories:**
```
models/output/
  V2_Enhanced_Models_intracell_dem/   # Paper 5 V2 predictions
  V4_GNN_TAT_Models_intracell_dem/   # Paper 5 V4 predictions
  V10_Late_Fusion_intracell_dem/     # Paper 5 V10 fusion
```

**To reproduce Paper 5 feature engineering (no GPU):**
```bash
# Step 1: Compute intra-cell DEM features from 90m GeoTIFF
python preprocessing/dem_intra_cell_features.py \
    --dem data/input/dem/dem_boyaca_90m.tif \
    --integrate --figures

# Step 2: Run DS1 vs DS2 bidirectional analysis
python preprocessing/ds1_ds2_analysis.py
```

**To reproduce Paper 5 training (GPU required):**
1. Upload the extended dataset to Google Drive
2. Open Colab notebooks from `models/intracell_dem/`
3. Run in order: V2 -> V4 -> V10 evaluation

**To reproduce Paper 5 post-training evaluation (no GPU):**
```bash
# After downloading Colab predictions to models/output/*_intracell_dem/
python workflows/run_pipeline.py --from 7 --intracell-dem --bundle BASIC_D10
```

---

## Feature Engineering (Paper 5)

### Intra-Cell DEM Features

Each CHIRPS cell at 0.05 deg resolution covers ~5.5 km and contains ~3,477 DEM pixels at 90m. Instead of using a single nearest-neighbor elevation value (Paper 4), Paper 5 computes the full intra-cell elevation distribution.

**Script:** `preprocessing/dem_intra_cell_features.py`

**Posibilidad 2 - Elevation Deciles (15 features):**
- 10 percentiles: p10, p20, ..., p100
- 5 statistics: mean, std, skewness, kurtosis, range

**Posibilidad 3 - PCA (6 features):**
- 6 principal components from sorted DEM pixel vectors
- PC1 (99.3%) = mean elevation (redundant with existing feature)
- PC2 (0.67%) = intra-cell heterogeneity (r=-0.794 with std)
- PC3 (0.05%) = distribution asymmetry (r=0.559 with skewness)

**Feature Bundles:**

| Bundle | Features | Count | Description |
|--------|----------|:-----:|-------------|
| BASIC | Paper 4 original | 12 | Temporal + precip stats + single elevation |
| BASIC_D10 | BASIC + deciles | 22 | + 10 elevation percentiles |
| BASIC_PCA6 | BASIC + PCA | 18 | + 6 PCA components |
| BASIC_D10_STATS | BASIC + deciles + stats | 27 | + 10 percentiles + 5 statistics |

### DS1 vs DS2 Analysis

Bidirectional analysis of the precipitation-elevation relationship.

**Script:** `preprocessing/ds1_ds2_analysis.py`

| Direction | Finding |
|-----------|---------|
| DS1 (CHIRPS -> DEM) | Pearson r = -0.700 (strong negative correlation) |
| DS2 (DEM -> CHIRPS) | Linear R2 = 0.490 (elevation explains 49% spatial variance) |

---

## Configuration

All pipeline parameters are centralized in `workflows/config.yaml`:
- Geographic region (Boyaca lat/lon bounds, grid dimensions)
- Data paths (CHIRPS, DEM, processed datasets)
- Model hyperparameters (V2, V4, V10)
- Feature engineering settings (clusters, intra-cell DEM)
- Benchmark settings (ACC, FSS thresholds/neighborhoods)

Override per environment:
```bash
python workflows/run_pipeline.py --config workflows/config_colab.yaml
```

---

## Directory Structure

```
workflows/
├── README.md                  # This file
├── config.yaml                # Central configuration
├── run_pipeline.py            # Orchestrator (stages 1-9)
├── run_pipeline.sh            # Shell wrapper (conda activation)
├── 01_download_chirps.py      # Stage 1: CHIRPS download
├── 02_aggregate_monthly.py    # Stage 2: Monthly aggregation
├── 03_merge_dem.py            # Stage 3: DEM merge
├── 04_feature_engineering.py  # Stage 4: Clustering
├── 05_train_v2_convlstm.py   # Stage 5: V2 training wrapper
├── 06_train_v4_gnn_tat.py    # Stage 6: V4 training wrapper
├── 07_late_fusion_v10.py      # Stage 7: V10 Ridge fusion
├── 08_benchmark_metrics.py    # Stage 8: All benchmarks
└── 09_generate_figures.py     # Stage 9: Figures + tables

models/intracell_dem/          # Colab notebooks (GPU training, Paper 5)
├── train_v2_convlstm_intracell_dem.ipynb
├── train_v4_gnn_tat_intracell_dem.ipynb
└── evaluate_v10_fusion_intracell_dem.ipynb
```

---

## Requirements

Core dependencies (see `requirements.txt`):
- Python >= 3.10
- xarray, netCDF4, h5netcdf
- numpy, scipy, scikit-learn
- matplotlib

GPU training (Colab):
- TensorFlow >= 2.12 (V2 ConvLSTM)
- PyTorch >= 2.0 + PyTorch Geometric (V4 GNN-TAT)

DEM feature engineering:
- rasterio (for GeoTIFF loading)
