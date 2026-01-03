# Technical Specifications - Data-Driven Framework for Precipitation Prediction
## ML Precipitation Prediction - Technical Specification Document

**Version:** 2.0
**Date:** January 2026
**Context:** Doctoral Thesis - Monthly Precipitation Prediction in Mountainous Areas

---

## 1. VISION AND OBJECTIVES

### 1.1 Purpose

This framework implements a **Data-Driven (DD)** system for monthly precipitation prediction in the Colombian Andes mountainous regions, using advanced spatiotemporal Deep Learning techniques within a **Specification-Driven Development (SDD)** methodology.

### 1.2 Research Hypotheses

The framework must validate these hypotheses from the doctoral thesis proposal:

| ID | Hypothesis | Validation Metric | Current Status |
|----|------------|-------------------|----------------|
| H1 | Hybrid GNN-Temporal models outperform ConvLSTM in spatial prediction | R² > 0.60, RMSE < 70mm | **VALIDATED (V4)** |
| H2 | Topographic features improve prediction accuracy | Compare BASIC vs PAFC | **VALIDATED** |
| H3 | Non-Euclidean spatial relations capture orographic dynamics better | GNN vs CNN metrics | **IN VALIDATION** |
| H4 | Multi-scale temporal attention improves long horizons (H6-H12) | R² degradation < 20% | **PARTIALLY VALIDATED** |

### 1.3 Problem Domain

```
GEOGRAPHIC CONTEXT:
+------------------------------------------+
|  STUDY AREA: Colombian Andes (Boyaca)    |
|  - Altitude: 500m - 4000m                |
|  - Variability: High (orography)         |
|  - Regime: Bimodal (2 wet seasons)       |
|  - Resolution: 0.05 deg (~5km)           |
+------------------------------------------+

SPECIFIC CHALLENGES:
1. Pronounced altitudinal gradients
2. Complex orographic effects
3. Interannual variability (ENSO)
4. Sparse in-situ data
5. Non-linear precipitation-elevation patterns
```

---

## 2. DEVELOPMENT FRAMEWORKS

### 2.1 SDD (Specification-Driven Development)

The SDD methodology ensures specifications are defined BEFORE implementation:

```
┌─────────────┐
│   DEFINE    │ ← spec.md: standards, requirements
└─────┬───────┘
      ↓
┌─────────────┐
│   DESIGN    │ ← plan.md: implementation approach
└─────┬───────┘
      ↓
┌─────────────┐
│  DEVELOP    │ ← notebooks: implementation
└─────┬───────┘
      ↓
┌─────────────┐
│  DOCUMENT   │ ← thesis.tex: methodology
└─────┬───────┘
      ↓
┌─────────────┐
│  DELIVER    │ ← paper.tex: results
└─────┬───────┘
      ↓
┌─────────────┐
│  ITERATE    │ → back to DEFINE
└─────────────┘
```

**SDD Rules:**
1. No implementation without specification
2. Changes require spec.md update first
3. All notebooks follow spec.md structure
4. Documentation reflects implementation

### 2.2 DD (Data-Driven) Framework

The DD methodology validates hypotheses through empirical evidence:

```
┌─────────────┐
│ HYPOTHESIS  │ ← Define H1-H4 from thesis proposal
└─────┬───────┘
      ↓
┌─────────────┐
│ EXPERIMENT  │ ← Design V1-V6 controlled experiments
└─────┬───────┘
      ↓
┌─────────────┐
│  MEASURE    │ ← Collect RMSE, MAE, R², Bias
└─────┬───────┘
      ↓
┌─────────────┐
│  ANALYZE    │ ← Statistical tests (Friedman, Nemenyi)
└─────┬───────┘
      ↓
┌─────────────┐
│  CONCLUDE   │ ← Accept/reject hypotheses
└─────┬───────┘
      ↓
┌─────────────┐
│  DOCUMENT   │ ← thesis.tex + paper.tex
└─────────────┘
```

**DD Rules:**
1. Empirical evidence over theoretical assumptions
2. Standardized metrics across all experiments
3. Statistical significance required for claims
4. Reproducible experiments with version control

---

## 3. FRAMEWORK ARCHITECTURE

### 3.1 Directory Structure

```
ml_precipitation_prediction/
│
├── CLAUDE.md                      # Project rules (THIS FILE)
│
├── data/                          # Input data
│   ├── raw/                       # Original data (NetCDF, GeoTIFF)
│   ├── processed/                 # Preprocessed data
│   └── dataset_monthly_*.nc       # Consolidated datasets
│
├── models/                        # Model implementations
│   ├── spec.md                    # Technical specifications
│   ├── plan.md                    # Development roadmap
│   ├── base_models_*.ipynb        # Main notebooks (V1-V6)
│   ├── custom_layers/             # Custom layers
│   └── output/                    # Training outputs
│       ├── V2_Enhanced_Models/    # V2 results
│       ├── V4_GNN_TAT_Models/     # V4 results (current)
│       └── ...
│
├── notebooks/                     # Exploratory analysis
├── scripts/                       # Automation scripts
│   └── benchmark/                 # Comparative analysis
│
├── docs/                          # Documentation
│   ├── framework/                 # Technical specs
│   ├── models/                    # Model documentation
│   ├── tesis/                     # Thesis documents
│   │   └── thesis.tex             # Doctoral thesis
│   └── papers/                    # Papers
│       └── 4/paper.tex            # Comparative paper
│
├── preprocessing/                 # Data preprocessing
└── utils/                         # Common utilities
```

### 3.2 Model Versioning

| Version | Name | Base Architecture | Status |
|---------|------|-------------------|--------|
| V1 | Baseline | ConvLSTM, ConvGRU, ConvRNN | Complete |
| V2 | Enhanced | V1 + Attention + Bidirectional + Residual | Complete |
| V3 | FNO | Fourier Neural Operators | Complete (underperformed) |
| **V4** | **GNN-TAT** | **Graph Neural Networks + Temporal Attention** | **Current** |
| V5 | Multi-Modal | V4 + ERA5 + Satellite | Planned |
| V6 | Ensemble | Best of V2-V5 + Meta-learning | Planned |

---

## 4. NOTEBOOK STANDARDS

### 4.1 Mandatory Cell Structure

Every model notebook must follow this structure:

```
CELL 1: HEADER & METADATA
--------------------------
- Notebook title
- Version and date
- Author
- Experiment description

CELL 2: ENVIRONMENT SETUP
--------------------------
- Colab/Local detection
- Dependency installation
- Path configuration
- Version matching (netCDF4, h5py, PyTorch)

CELL 3-4: IMPORTS
--------------------------
- Standard libraries (numpy, pandas, xarray)
- Deep Learning (torch, torch_geometric)
- Visualization (matplotlib, seaborn)

CELL 5: CONFIGURATION (CONFIG dict)
--------------------------
CONFIG = {
    'input_window': 60,      # Input months
    'horizon': 12,           # Prediction months
    'epochs': 150,
    'batch_size': 4,
    'learning_rate': 0.001,
    'patience': 50,
    'train_val_split': 0.8,
    'light_mode': False,     # True for quick tests
    'enabled_horizons': [1, 3, 6, 12],
    'gnn_config': {...},     # Model-specific config
}

CELL 6-8: DATA LOADING & VALIDATION
--------------------------
- NetCDF loading
- Feature validation
- Dimension verification

CELL 9-11: PREPROCESSING
--------------------------
- Graph construction (for GNN)
- Feature engineering
- Temporal windowing
- Train/Val split

CELL 12-15: MODEL DEFINITION
--------------------------
- Custom layer classes
- Model architecture
- Forward pass

CELL 16-18: TRAINING LOOP
--------------------------
- Training function
- Early stopping
- Checkpointing

CELL 19: MAIN EXPERIMENT LOOP
--------------------------
- Iterate over feature sets (BASIC, KCE, PAFC)
- Iterate over horizons
- Save metrics

CELL 20: RESULTS AGGREGATION
--------------------------
- Consolidate metrics
- Save final CSV
- Memory cleanup
```

### 4.2 File Naming Conventions

```
Notebooks:
  base_models_{architecture}_{version}.ipynb
  Example: base_models_gnn_tat_v4.ipynb

Saved Models:
  {model_name}_best_h{horizon}.pt
  Example: gnn_tat_gat_best_h12.pt

Metrics:
  metrics_spatial_{version}_{model}_h{horizon}.csv
  Example: metrics_spatial_v4_gnn_tat_h12.csv

Training Logs:
  {model_name}_training_log_h{horizon}.csv
  {model_name}_history.json
```

---

## 5. OUTPUT STANDARDS

### 5.1 Output Directory Structure

```
models/output/{VERSION}_{MODEL}/
│
├── experiment_state_{version}.json    # Complete experiment state
│
├── h{HORIZON}/                        # Per horizon
│   └── {EXPERIMENT}/                  # BASIC, KCE, PAFC
│       └── training_metrics/
│           ├── {model}_best_h{H}.pt        # Best model
│           ├── {model}_history.json        # Training history
│           └── {model}_training_log_h{H}.csv
│
├── metrics_spatial_{version}_h{H}.csv  # Consolidated metrics
└── graph_visualization.png             # Graph visualization
```

### 5.2 Spatial Metrics CSV Format

```csv
TotalHorizon,Experiment,Model,H,RMSE,MAE,R^2,Mean_True_mm,Mean_Pred_mm,TotalPrecipitation,TotalPrecipitation_Pred,mean_bias_mm,mean_bias_pct
12,BASIC,GNN_TAT_GAT,1,55.28,38.41,0.6642,154.84,150.90,127743.43,124495.39,-3.94,-2.54
...
```

**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| TotalHorizon | int | Total experiment horizon |
| Experiment | str | Feature set (BASIC/KCE/PAFC) |
| Model | str | Model name |
| H | int | Specific horizon (1-12) |
| RMSE | float | Root Mean Squared Error (mm) |
| MAE | float | Mean Absolute Error (mm) |
| R^2 | float | Coefficient of determination |
| Mean_True_mm | float | Average actual precipitation |
| Mean_Pred_mm | float | Average predicted precipitation |
| mean_bias_mm | float | Mean bias (mm) |
| mean_bias_pct | float | Percentage bias |

### 5.3 History JSON Format

```json
{
  "model_name": "GNN_TAT_GAT",
  "experiment": "BASIC",
  "horizon": 12,
  "best_epoch": 23,
  "best_val_loss": 0.6059,
  "final_train_loss": 0.0862,
  "final_val_loss": 0.7986,
  "total_epochs": 73,
  "parameters": 97932
}
```

---

## 6. FEATURE SETS

### 6.1 Feature Set Definitions

| Set | Features | Purpose |
|-----|----------|---------|
| **BASIC** | Temporal + Precipitation + Base topography | Minimal baseline |
| **KCE** | BASIC + K-means elevation clusters | Orographic regimes |
| **PAFC** | KCE + Temporal lags | Autocorrelation |

### 6.2 Detailed Features

```python
FEATURE_SETS = {
    'BASIC': [
        # Temporal
        'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        # Precipitation
        'max_daily_precipitation', 'min_daily_precipitation',
        'daily_precipitation_std',
        # Topography
        'elevation', 'slope', 'aspect'
    ],

    'KCE': [
        *BASIC,
        # Elevation clusters (one-hot)
        'elev_high', 'elev_med', 'elev_low'
    ],

    'PAFC': [
        *KCE,
        # Temporal lags
        'total_precipitation_lag1',   # Previous month
        'total_precipitation_lag2',   # 2 months before
        'total_precipitation_lag12'   # Same month last year
    ]
}
```

---

## 7. MODEL ARCHITECTURE (V4 GNN-TAT)

### 7.1 Architecture Diagram

```
INPUT: (batch, seq_len, n_nodes, n_features)
        │
        ▼
┌──────────────────┐
│ SPATIAL ENCODER  │  GCN/GAT/SAGE layers
│ (GNN per timestep)│  - Message passing
└──────────────────┘  - Edge weights: distance + elevation + correlation
        │
        ▼
┌──────────────────┐
│ TEMPORAL ENCODER │  Multi-head Self-Attention
│ (Attention)      │  - 4 heads
└──────────────────┘  - Residual connection + LayerNorm
        │
        ▼
┌──────────────────┐
│ SEQUENCE MODEL   │  Bidirectional LSTM
│ (LSTM)           │  - 2 layers
└──────────────────┘  - Dropout 0.1
        │
        ▼
┌──────────────────┐
│ OUTPUT PROJECTION│  Linear layers
└──────────────────┘  - Per-node, per-horizon output
        │
        ▼
OUTPUT: (batch, n_nodes, horizon)
```

### 7.2 Hyperparameter Configuration

```python
GNN_CONFIG = {
    # Dimensions
    'hidden_dim': 64,
    'temporal_hidden_dim': 64,
    'lstm_hidden_dim': 64,

    # Layers
    'num_gnn_layers': 2,
    'num_temporal_heads': 4,
    'num_lstm_layers': 2,

    # Regularization
    'dropout': 0.1,
    'temporal_dropout': 0.1,

    # Graph construction
    'edge_threshold': 0.3,
    'max_neighbors': 8,
    'use_distance_edges': True,
    'use_elevation_edges': True,
    'use_correlation_edges': True,

    # Similarity weights
    'distance_scale_km': 10.0,
    'elevation_scale': 0.2,
    'elevation_weight': 0.3,
    'correlation_weight': 0.5,
    'min_edge_weight': 0.01
}
```

### 7.3 GNN Variants

| Variant | Description | Pros | Cons |
|---------|-------------|------|------|
| **GCN** | Graph Convolutional Network | Stable, uses edge_weight | Limited receptive field |
| **GAT** | Graph Attention Network | Adaptive attention | No native edge_weight |
| **SAGE** | GraphSAGE | Efficient, sampling | No native edge_weight |

---

## 8. TRAINING PIPELINE

### 8.1 Training Flow

```
1. PREPARATION
   ├── Load NetCDF data
   ├── Validate available features
   ├── Build spatial graph
   ├── Create temporal windows (input_window -> horizon)
   └── Train/Val split (80/20 temporal)

2. EXPERIMENT LOOP
   FOR each feature_set in [BASIC, KCE, PAFC]:
       FOR each model_type in [GAT, SAGE, GCN]:
           FOR each horizon in enabled_horizons:
               ├── Instantiate model
               ├── Train with early stopping
               ├── Save best checkpoint
               ├── Evaluate on validation
               └── Save metrics

3. EVALUATION
   ├── Calculate per-cell spatial metrics
   ├── Aggregate to consolidated CSV
   ├── Generate visualizations
   └── Save experiment_state.json
```

### 8.2 Evaluation Metrics

```python
def evaluate_model(y_true, y_pred):
    """
    Standard framework metrics.

    Args:
        y_true: (N, horizon, lat, lon) - Actual values
        y_pred: (N, horizon, lat, lon) - Predictions

    Returns:
        dict with per-horizon metrics
    """
    metrics = {}
    for h in range(horizon):
        metrics[f'H{h+1}'] = {
            'RMSE': sqrt(mean((y_true[:, h] - y_pred[:, h])**2)),
            'MAE': mean(abs(y_true[:, h] - y_pred[:, h])),
            'R2': 1 - SS_res / SS_tot,
            'Bias': mean(y_pred[:, h] - y_true[:, h]),
            'Bias_pct': 100 * Bias / mean(y_true[:, h])
        }
    return metrics
```

### 8.3 Early Stopping Configuration

```python
EARLY_STOPPING_CONFIG = {
    'monitor': 'val_loss',
    'patience': 50,
    'min_delta': 1e-4,
    'mode': 'min',
    'restore_best_weights': True
}

LEARNING_RATE_SCHEDULER = {
    'type': 'ReduceLROnPlateau',
    'factor': 0.5,
    'patience': 20,
    'min_lr': 1e-6
}
```

---

## 9. DOCUMENTATION SYNCHRONIZATION

### 9.1 thesis.tex Mapping

| Notebook Section | Thesis Chapter |
|------------------|----------------|
| Data loading | Ch. 2: Data Acquisition |
| Preprocessing | Ch. 2: Preprocessing Pipeline |
| Feature engineering | Ch. 3: Feature Engineering |
| Model definition | Ch. 4/5: Model Architecture |
| Training | Ch. 4/5: Training Protocol |
| Evaluation | Ch. 4/5: Evaluation |
| Results | Ch. 4/5: Results |

### 9.2 paper.tex Mapping

| Model Version | Paper Section |
|---------------|---------------|
| V2 Enhanced | Results: Baselines |
| V3 FNO | Results: Physics-Informed |
| V4 GNN-TAT | Results: Hybrid Models |
| Statistical Tests | Discussion |

---

## 10. SUCCESS CRITERIA

### 10.1 Acceptance Criteria

| Metric | Baseline (V2) | Target (V4+) | Excellent |
|--------|---------------|--------------|-----------|
| R² (H1-H6) | 0.44 | > 0.60 | > 0.70 |
| R² (H7-H12) | 0.30 | > 0.50 | > 0.60 |
| RMSE (mm) | 98.17 | < 70 | < 55 |
| MAE (mm) | 44.19 | < 50 | < 40 |
| Parameters | 2M+ | < 150K | < 100K |
| Train ratio (val/train loss) | - | < 10x | < 5x |

### 10.2 Current V4 GNN-TAT Results

```
BEST CONFIGURATION: GNN_TAT_SAGE + KCE @ H=3
- R²: 0.707 (vs baseline 0.437) -> +62% improvement
- RMSE: 52.45mm (vs baseline 98.17mm) -> -47% improvement
- Parameters: ~98K (vs 2M+) -> -95% reduction

RANKING BY MODEL (all horizons average):
1. GNN_TAT_GAT + PAFC:  R²=0.628, RMSE=58.4mm
2. GNN_TAT_GCN + PAFC:  R²=0.625, RMSE=58.7mm
3. GNN_TAT_SAGE + KCE:  R²=0.618, RMSE=59.1mm
```

---

## 11. BEST PRACTICES

### 11.1 Reproducibility

```python
# Always set seeds
import torch
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
```

### 11.2 Memory Management

```python
# Clean memory between experiments
import gc

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Call after each model
cleanup()
```

### 11.3 Logging and Checkpointing

```python
# Save checkpoint only if improved
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': CONFIG
    }, checkpoint_path)

# Load with weights_only=False (PyTorch 2.6+)
checkpoint = torch.load(path, weights_only=False)
```

---

## 12. GLOSSARY

| Term | Definition |
|------|------------|
| **GNN** | Graph Neural Network - Network operating on graphs |
| **TAT** | Temporal Attention - Temporal attention mechanism |
| **ConvLSTM** | Convolutional LSTM - LSTM with spatial convolutions |
| **GAT** | Graph Attention Network - GNN with attention |
| **SAGE** | GraphSAGE - GNN with neighbor sampling |
| **GCN** | Graph Convolutional Network - Basic GNN |
| **H** | Prediction horizon (months ahead) |
| **BASIC** | Minimal feature set |
| **KCE** | K-means Cluster Elevation features |
| **PAFC** | Precipitation Auto-correlation Features Complete |
| **R²** | Coefficient of determination |
| **RMSE** | Root Mean Square Error |
| **MAE** | Mean Absolute Error |
| **Orographic** | Related to elevation effects on precipitation |

---

## 13. DOCUMENTATION FORMATTING STANDARDS

### 13.1 Official Thesis Title

**Extended Title (Aligned with Doctoral Proposal):**
> "Computational Model for Spatiotemporal Prediction of Monthly Precipitation in Mountainous Areas: A Hybrid Deep Learning Approach Using Graph Neural Networks with Temporal Attention"

### 13.2 Doctoral Thesis Chapter Structure

Following ML/Hydrology doctoral thesis standards:

| Chapter | Title | Pages | Key Content |
|---------|-------|-------|-------------|
| 1 | Introduction | 20 | Problem Statement, Hypotheses H1-H4, Objectives |
| 2 | Literature Review | 40 | State of the Art, Deep Learning, GNNs, Research Gaps |
| 3 | Theoretical Framework | 30 | Graph Theory, Attention Mechanisms, Statistics |
| 4 | Materials and Methods | 40 | Study Area, Data Sources, Preprocessing, Models |
| 5 | Results | 50 | V1-V4 Results, Statistical Tests, Validation |
| 6 | Discussion | 30 | Interpretation, SOTA Comparison, Limitations |
| 7 | Conclusions | 15 | Contributions, Future Work |
| - | References | - | 100-150 Q1 entries |
| - | Appendices | - | Code, Tables, Figures |

### 13.3 LaTeX Figure Width Standards

**CRITICAL:** All figures must stay within page margins:

```latex
\newcommand{\fullwidth}{0.95\textwidth}     % Single full-width figure
\newcommand{\halfwidth}{0.47\textwidth}     % Two figures side-by-side
\newcommand{\thirdwidth}{0.31\textwidth}    % Three figures in row
\newcommand{\quarterwidth}{0.23\textwidth}  % Four figures in row
```

### 13.4 Light Mode Notation

When presenting results from reduced grid experiments (5×5 grid subset), include this disclaimer:

```latex
\textbf{Important Note:} Results were obtained using \textit{light mode}
(5×5 grid subset) for rapid prototyping. Full-grid validation (61×65)
is pending and will be reported in Section~\ref{sec:full-grid-results}.
```

### 13.5 Bibliography Requirements

- **Location:** `docs/tesis/references.bib`
- **Minimum Entries:** 100+ Q1 references
- **Categories:**
  - Graph Neural Networks (25 refs)
  - Precipitation/Weather ML (25 refs)
  - Climate/Hydrology Data (15 refs)
  - Deep Learning Fundamentals (20 refs)
  - Statistical Methods (10 refs)
  - Spatiotemporal Modeling (15 refs)

---

## 14. FILE SCOPE RULES

### 14.1 Project Scope Definition

This project focuses EXCLUSIVELY on the doctoral thesis: **"Computational Model for Spatiotemporal Prediction of Monthly Precipitation in Mountainous Areas"**

All files must directly support:
1. Model training and evaluation (V1-V6)
2. Data preprocessing pipeline (CHIRPS, SRTM, ERA5)
3. Thesis and paper documentation
4. Statistical analysis and benchmarking
5. Feature engineering (BASIC, KCE, PAFC)

### 14.2 Directory Scope Classification

| Directory | Scope | Purpose |
|-----------|-------|---------|
| `models/*.ipynb` | **CORE** | Model implementations |
| `models/output/` | **CORE** | Training outputs |
| `data/` | **CORE** | ETL pipeline |
| `docs/tesis/` | **CORE** | Doctoral thesis |
| `docs/papers/4/` | **CORE** | Comparative paper |
| `scripts/benchmark/` | **CORE** | Results analysis |
| `notebooks/` | **CORE** | EDA and preprocessing |
| Other `docs/`, `utils/` | **SUPPORTING** | Pipeline utilities |
| `.venv/`, `.git/` | **SYSTEM** | Auto-generated |

### 14.3 OUT-OF-SCOPE Criteria

**DO NOT CREATE files that:**
1. Are backups or copies (`*_backup.py`, `*_old.ipynb`)
2. Serve non-thesis AI tasks
3. Are temporary exploration unrelated to precipitation prediction
4. Duplicate existing functionality
5. Are in languages other than English

### 14.4 Before Creating New Files

1. Verify the file serves thesis objectives (H1-H4)
2. Check if existing file can be modified instead
3. Follow naming conventions (Section 6)
4. Place in appropriate directory (Section 14.2)
5. Document purpose if new pattern

---

## 15. REFERENCES

### 15.1 Fundamental Papers

1. Kipf & Welling (2017) - "Semi-Supervised Classification with Graph Convolutional Networks"
2. Velickovic et al. (2018) - "Graph Attention Networks"
3. Hamilton et al. (2017) - "Inductive Representation Learning on Large Graphs" (GraphSAGE)
4. Shi et al. (2015) - "Convolutional LSTM Network" (ConvLSTM)
5. Vaswani et al. (2017) - "Attention Is All You Need"

### 15.2 Datasets

- **CHIRPS 2.0**: Climate Hazards Group InfraRed Precipitation with Station data
- **SRTM DEM**: Shuttle Radar Topography Mission Digital Elevation Model
- **ERA5**: ECMWF Reanalysis v5 (planned for V5)

---

*Document generated as part of the ML Precipitation Prediction Framework*
*Last updated: January 2026*
