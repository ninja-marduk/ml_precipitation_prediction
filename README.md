# ML Precipitation Prediction

## Computational Model for Spatiotemporal Prediction of Monthly Precipitation in Mountainous Areas

**A Hybrid Deep Learning Approach Using Graph Neural Networks with Temporal Attention**

---

## Project Overview

This repository contains the implementation of a doctoral thesis project developing hybrid deep learning models for monthly precipitation prediction in the mountainous terrain of Boyaca, Colombia. The research follows a Data-Driven (DD) scientific methodology with rigorous statistical validation.

### Research Publications

This work is organized into three complementary studies, each addressing a different aspect of the research:

| Study | Focus | Journal | Status |
|-------|-------|---------|--------|
| **Systematic Review** (Paper 1) | Survey of hybrid DL models for precipitation prediction | Hydrology Research (IWA) | Under R3 review |
| **Hybrid Architecture Benchmark** (Paper 4) | ConvLSTM vs FNO vs GNN-TAT comparison | Hydrology (MDPI) | Accepted |
| **Sub-grid Feature Engineering** (Paper 5) | Intra-cell DEM topographic heterogeneity | Hydrology (MDPI) | In preparation |

Throughout this documentation, papers are referenced by their descriptive name or by number (e.g., "Paper 4") following this convention.

### Model Performance Summary (H=12, Full Grid Results)

| Metric | V2 ConvLSTM | V4 GNN-TAT | V5 Stacking | V9 BiMamba | **V10 Late Fusion** |
|--------|-------------|------------|-------------|------------|---------------------|
| **R²** | 0.628 | 0.597 | 0.212 | 0.200 | **0.668** ✅ |
| **RMSE (mm)** | 81.05 | 84.40 | 117.93 | 111.18 | **76.67** ✅ |
| **MAE (mm)** | 58.91 | 59.74 | 92.41 | 87.33 | **56.12** ✅ |
| **Bias (mm)** | -10.50 | -28.79 | -- | 8.23 | **-0.002** |
| **Status** | Baseline | Efficient | ❌ Failed | ❌ Failed | **Best** ✅ |

**V10 Late Fusion Success:** V10 combines V2 and V4 predictions through Ridge regression, achieving +6.2% improvement over the best baseline. Learned weights: w_V2=0.446, w_V4=0.710, bias=-5.53mm. Both models contribute complementary information.

**Key Insight:** Late fusion (V10) succeeds where early fusion (V5) failed. When combining heterogeneous architectures, fuse predictions rather than intermediate features.

**Recommendation for Thesis:** Use **V10 Late Fusion Ridge Ensemble** as final best model (R²=0.668).

### Value Proposition

GNN-TAT achieves **comparable predictive performance** to ConvLSTM baselines while offering:
1. **95% parameter reduction** (98K vs 500K-2.1M parameters)
2. **Interpretable spatial relationships** through explicit graph structure
3. **Significantly lower mean RMSE** across all configurations (p=0.015)

### Key Research Findings

**1. Hybridization Rescue Effect (H5 - Validated):**
Pure spectral methods (FNO) fail for precipitation (R²=0.206), but hybridization rescues performance through component integration:
- Pure FNO: R²=0.206
- FNO-ConvLSTM Hybrid: R²=0.582
- **Improvement: 182%**

**2. When Stacking Fails (H6 - Rejected):**
Complex fusion architectures don't guarantee better results. V5 Stacking attempted to combine ConvLSTM and GNN-TAT but performed catastrophically worse:
- V2 ConvLSTM (individual): R²=0.628, RMSE=81mm
- V4 GNN-TAT (individual): R²=0.516, RMSE=92mm
- **V5 Stacking (ensemble): R²=0.212, RMSE=118mm** ❌

**Root Cause:** GridGraphFusion mixed branch features BEFORE predictions, destroying branch identity and preventing effective meta-learning.

**Lesson:** Simpler models (V2 ConvLSTM) often outperform sophisticated ensembles when the fusion mechanism isn't well-designed. Fusion timing and architecture matter more than complexity.

**3. When Ensemble Stratification Works vs Fails (V6 - Complete):**
V6 Multi-Dimensional Ensemble Matrix tested 8 ensemble strategies across 4 stratification dimensions to rescue the ensemble approach. Results (on validation set):

**Stratification Dimensions Tested:**
- **Elevation:** High (>3000m), Medium (2000-3000m), Low (<2000m)
- **Precipitation Magnitude:** Light, Moderate, Heavy
- **Season:** DJF (Winter), MAM (Spring), JJA (Summer), SON (Autumn)
- **Forecast Horizon:** Short (H1-4), Medium (H5-8), Long (H9-12)

**Results Across ALL Dimensions:**
| Dimension | V2 R² Range | V4 R² Range | Winner |
|-----------|-------------|-------------|--------|
| Elevation (3 zones) | 0.136-0.240 | 0.568-0.610 | V4 all zones |
| Season (4 seasons) | -3.67-0.250 | 0.264-0.611 | V4 all seasons |
| Horizon (3 groups) | 0.103-0.218 | 0.583-0.608 | V4 all groups |

**Ensemble Strategies Performance:**
- Simple Average (50/50): R²=0.478 (-20% vs V4) ❌
- All Stratified Ensembles: R²=0.597 (equals V4, no improvement)

**Critical Finding:** Ensemble stratification CANNOT improve performance when one model dominates universally. V4 outperforms V2 across ALL tested dimensions, so optimal weights are 100% V4, 0% V2.

**Theoretical Lesson:** Successful ensembles require **complementary strengths** - different models excelling in different conditions. V2 vs V4 lack complementarity, making ensemble futile.

---

## Research Hypotheses

| ID | Hypothesis | Status | Evidence |
|----|------------|--------|----------|
| H1 | Hybrid GNN-Temporal models achieve comparable or better accuracy than ConvLSTM | **PARTIALLY VALIDATED** | V4 R²=0.516 vs V2 R²=0.628; GNN captures spatial structure but ConvLSTM superior overall |
| H2 | Topographic features improve prediction accuracy | **VALIDATED** | KCE features improve V4 GNN performance (p<0.05) |
| H3 | Non-Euclidean spatial relations capture orographic dynamics | **VALIDATED** | 3,965 nodes, 500,000 edges successfully trained in V4 |
| H4 | Multi-scale temporal attention improves long horizons | **VALIDATED** | R² degradation 9.6% (H1→H12), below 20% threshold |
| H5 | Hybridization rescues architectural limitations | **VALIDATED** | Pure FNO R²=0.206 → Hybrid R²=0.582 (182% improvement) |
| H6 | Stacking improves upon best individual models | **❌ REJECTED** | V5 Stacking R²=0.212 vs V2 R²=0.628 (66% worse); GridGraphFusion destroyed information |
| H7 | Ensemble stratification can leverage complementary strengths | **❌ REJECTED** | V6 tested 8 strategies × 4 dimensions; V4 dominates all strata, no complementarity exists |

---

## Model Versions

| Version | Architecture | Purpose | Status | Best R² | Recommendation |
|---------|--------------|---------|--------|---------|----------------|
| V1 | ConvLSTM, ConvGRU, ConvRNN | Baselines | Complete | 0.642 | Superseded by V2 |
| V2 | Enhanced + Attention + Bidirectional | Improved baselines | Complete | 0.628 | Baseline |
| V3 | Fourier Neural Operators (FNO) | Physics-informed | Complete | 0.312 | ❌ Underperformed |
| V4 | GNN-TAT | Hybrid spatial-temporal | Complete | 0.628 | Efficient (95% less params) |
| V5 | GNN-ConvLSTM Stacking | Early fusion ensemble | Complete | 0.212 | ❌ Failed - early fusion |
| V6 | Stratified Ensemble Matrix | Ensemble strategies | Complete | 0.597 | No improvement |
| V7 | AMES (Multi-Expert System) | MoE + Physics-guided | In Development | TBD | Research |
| V8 | GNN-Mamba | State Space Models | In Development | TBD | Research |
| V9 | GNN-BiMamba | Bidirectional Mamba | Complete | 0.200 | ❌ Failed - SSM |
| **V10** | **Late Fusion Ridge Ensemble** | **Prediction-level fusion** | **Complete** | **0.668** | **✅ BEST MODEL** |

**Note on V6:** V6 tested 8 ensemble strategies across 4 stratification dimensions (elevation, magnitude, season, horizon). *Results on validation set show V4 (R²=0.597) superior to V2 (R²=0.175). All ensemble strategies either equal or worsen V4's performance because V4 dominates across all tested dimensions. **Finding:** Ensemble stratification cannot improve when one model is universally superior. See [V6 Multi-Dimensional Ensemble README](docs/models/V6_Multi_Dimensional_Ensemble/README.md) for complete analysis.

### V9-V10: Completed Advanced Architectures

**V9: GNN-BiMamba (FAILED)**
- Bidirectional Mamba State Space Models for temporal processing
- **Result:** R²=0.200, RMSE=111.18mm
- **Status:** Failed - SSM paradigm does not transfer to precipitation prediction
- Flat R² profile across horizons indicates degenerate solution
- Mamba's selective state space mechanism unsuited for chaotic, threshold-driven precipitation dynamics

**V10: Late Fusion Ridge Ensemble (SUCCESS - BEST MODEL)**
- Combines V2 ConvLSTM and V4 GNN-TAT predictions through Ridge regression
- **Result:** R²=0.668, RMSE=76.67mm, MAE=56.12mm
- **Improvement:** +6.2% over best baseline (V2)
- Learned weights: w_V2=0.446, w_V4=0.710, bias=-5.53mm
- Near-zero bias (-0.002mm) demonstrates effective bias correction
- **Key finding:** Late fusion preserves component strengths that early fusion (V5) destroyed

### V7-V8: In Development

**V7: AMES (Adaptive Multi-Expert Ensemble System)**
- Mixture of Experts with physics-guided routing
- Expert specialization by elevation zones
- Target: Must exceed V10's R²=0.668 to justify complexity

**V8: GNN-Mamba**
- Unidirectional Mamba State Space Models
- Based on V9 failure, expectations are tempered
- Resources may be better allocated to extending V10 framework

---

## Project Structure

```
ml_precipitation_prediction/
├── data/                         # Input data (CHIRPS, SRTM DEM)
├── models/
│   ├── base_models_*.ipynb       # Model notebooks (V1-V10)
│   └── output/                   # Training outputs
├── notebooks/                    # Exploratory analysis
├── preprocessing/                # Feature engineering scripts
│   ├── dem_intra_cell_features.py  # Intra-cell DEM feature extraction
│   └── ds1_ds2_analysis.py         # Bidirectional analysis
├── scripts/                      # Benchmark and evaluation
│   └── benchmark/                # ACC, FSS, elevation metrics
├── workflows/                    # End-to-end pipeline (see below)
│   ├── 01-09_*.py                # Pipeline stages
│   ├── colab/                    # Colab notebooks (GPU training)
│   └── config.yaml               # Central configuration
└── utils/                        # Utility functions
```

### Reproducible Pipeline

The [`workflows/`](workflows/) directory contains an end-to-end pipeline (9 stages) for reproducing all results. See the **[Workflows README](workflows/README.md)** for full documentation.

```bash
# Reproduce hybrid architecture benchmark results (stages 7-9, no GPU)
python workflows/run_pipeline.py --from 7

# Intra-cell DEM feature engineering (no GPU)
python preprocessing/dem_intra_cell_features.py --dem data/input/dem/dem_boyaca_90m.tif --integrate --figures

# DEM-enhanced model training: see models/intracell_dem/ notebooks
```

| Study | Notebooks | Feature Sets | Output Directory |
|-------|-----------|--------------|------------------|
| **Hybrid Architecture Benchmark** | `models/base_models_*_v2/v4/v10.ipynb` | BASIC, KCE, PAFC | `models/output/V2_Enhanced_Models/` |
| **Sub-grid Feature Engineering** | `models/intracell_dem/*_intracell_dem.ipynb` | BASIC_D10, BASIC_PCA6 | `models/output/V2_Enhanced_Models_intracell_dem/` |

---

## Data Sources

- **CHIRPS 2.0**: Climate Hazards InfraRed Precipitation with Stations (0.05° resolution)
- **SRTM DEM**: Shuttle Radar Topography Mission elevation data (90m)
- **ERA5**: ECMWF Reanalysis v5 (planned for V5)

### Study Area
- **Region**: Boyaca, Colombian Andes
- **Grid**: 61 x 65 cells (0.05° resolution)
- **Temporal**: 518 monthly steps
- **Horizons**: H = 1, 3, 6, 12 months

---

## Feature Sets

### Baseline Feature Sets (Hybrid Architecture Benchmark)

| Set | Features | Description |
|-----|----------|-------------|
| BASIC | 12 | Temporal encodings + precipitation stats + base topography |
| KCE | 15 | BASIC + K-means elevation clusters |
| PAFC | 18 | KCE + precipitation autocorrelation lags (t-1, t-2, t-12) |

### Intra-Cell DEM Feature Sets (Sub-grid Feature Engineering)

| Set | Features | Description |
|-----|----------|-------------|
| BASIC_D10 | 22 | BASIC + 10 elevation deciles (p10-p100) per CHIRPS cell |
| BASIC_PCA6 | 18 | BASIC + 6 PCA components of intra-cell DEM pixel distributions |
| BASIC_D10_STATS | 27 | BASIC_D10 + mean, std, skewness, kurtosis, range |

Each CHIRPS cell (~5.5 km) contains ~3,477 DEM pixels at 90m. Deciles capture intra-cell topographic heterogeneity that a single elevation value misses.

---

## Installation

### Dataset Download

Download the Boyacá precipitation dataset:
- **Google Drive**: [Boyacá Dataset (CHIRPS + SRTM)](https://drive.google.com/file/d/13INBvB654a3iDhQFLWC1WiQo3vZ9l5HN/view?usp=drive_link)
- **Extract to**: `data/` directory in the repository

### Local Environment

```bash
# Clone repository
git clone https://github.com/ninja-marduk/ml_precipitation_prediction.git
cd ml_precipitation_prediction

# Create environment
conda create -n precipitation python=3.10
conda activate precipitation

# Install dependencies
pip install -r requirements.txt

# For V4 GNN-TAT (PyTorch Geometric)
pip install torch-geometric
```

---

## Usage

### Execution Modes

#### Light Mode (CPU/Small GPU)
- **Environment**: Google Colab Free Tier (CPU with High RAM)
- **RAM**: 12+ GB
- **Dataset**: 5×5 grid subset for rapid prototyping
- **Use Case**: Testing, development, debugging
- **Notebooks**: All V1-V5 notebooks support light mode

#### Full Mode (Production Training)
- **Environment**: Google Colab Pro/Pro+ with GPU
- **GPU Required**: A100 (40GB) or H100 (80GB) recommended
- **RAM**: 40+ GB
- **Dataset**: Full 61×65 grid (3,965 nodes, 500,000 edges for GNN)
- **Use Case**: Final model training, benchmark experiments
- **Training Time**: 2-8 hours depending on model complexity

### Running Notebooks

#### Local Execution
1. **V4 GNN-TAT** (recommended):
   ```bash
   jupyter notebook models/base_models_gnn_tat_v4.ipynb
   ```

2. **V5 GNN-ConvLSTM Stacking** (latest):
   ```bash
   jupyter notebook models/base_models_gnn_convlstm_stacking_v5.ipynb
   ```

#### Google Colab Execution
1. Open notebook from GitHub in Colab
2. **For Light Mode**:
   - Runtime → Change runtime type → CPU
   - Edit → Notebook settings → Hardware accelerator: None
   - Runtime settings → High RAM
3. **For Full Mode**:
   - Runtime → Change runtime type → GPU
   - Select: A100 GPU or H100 GPU (Colab Pro+)
   - Runtime settings → High RAM

All notebooks include automatic GPU detection and PyTorch Geometric installation.

---

## Evaluation Metrics

- **RMSE**: Root Mean Square Error (mm)
- **MAE**: Mean Absolute Error (mm)
- **R²**: Coefficient of Determination
- **Bias**: Mean prediction bias (mm, %)
- **ACC**: Anomaly Correlation Coefficient
- **FSS**: Fractions Skill Score at 1/5/10mm thresholds
- **Statistical Tests**: Friedman + Nemenyi post-hoc


---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:
- **Share**: Copy and redistribute the material in any medium or format
- **Adapt**: Remix, transform, and build upon the material for any purpose, including commercial use

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made

See [LICENSE](LICENSE) file for complete terms.

---

## Citation

If you use this code, dataset, or methodology in your research, please cite:

### Software Citation (BibTeX)

```bibtex
@software{Perez2026MLPrecipitation,
  author       = {Perez Reyes, Manuel Ricardo},
  title        = {{ML Precipitation Prediction: Hybrid Deep Learning
                   for Spatiotemporal Forecasting in Mountainous Areas}},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/ninja-marduk/ml_precipitation_prediction},
  note         = {Doctoral Thesis Project - UPTC}
}
```

### Dataset Citation (BibTeX)

```bibtex
@dataset{Perez2026BoyacaDataset,
  author       = {Perez Reyes, Manuel Ricardo},
  title        = {{Boyacá Precipitation Dataset (CHIRPS + SRTM)}},
  year         = {2026},
  publisher    = {Google Drive},
  url          = {https://drive.google.com/file/d/13INBvB654a3iDhQFLWC1WiQo3vZ9l5HN/view},
  note         = {61×65 grid, 518 monthly steps, Colombian Andes}
}
```

### Systematic Review — Paper 1 (BibTeX)

```bibtex
@article{PerezReyes2025Review,
  author       = {P\'erez Reyes, Manuel Ricardo and Su\'arez Bar\'on, Marco Javier
                  and Garc\'ia Cabrejo, \'Oscar Javier},
  title        = {{Hybrid Deep Learning Models for Monthly Precipitation
                   Prediction: A Systematic Review}},
  journal      = {Hydrology Research},
  year         = {2025},
  note         = {Under review (R3 submitted 2026-02-11)}
}
```

### Hybrid Architecture Benchmark — Paper 4 (BibTeX)

```bibtex
@article{PerezReyes2026Hybrid,
  author       = {P\'erez Reyes, Manuel Ricardo and Su\'arez Bar\'on, Marco Javier
                  and Garc\'ia Cabrejo, \'Oscar Javier},
  title        = {{A Data-Driven Deep Learning Framework for Monthly Precipitation
                   Prediction in Complex Mountainous Terrain: Systematic Evaluation
                   of Hybrid Architectures}},
  journal      = {Hydrology},
  year         = {2026},
  note         = {Accepted at MDPI Hydrology (hydrology-4131162)}
}
```

### Sub-grid Feature Engineering — Paper 5 (BibTeX)

```bibtex
@article{PerezReyes2026Ensemble,
  author       = {P\'erez Reyes, Manuel Ricardo and Su\'arez Bar\'on, Marco Javier
                  and Garc\'ia Cabrejo, \'Oscar Javier},
  title        = {{A Data-Driven Deep Learning Framework for Monthly Precipitation
                   Prediction in Complex Mountainous Terrain: Systematic Evaluation
                   of Hybrid Architectures, Ensemble Strategies, and Emerging Paradigms}},
  journal      = {Hydrology},
  year         = {2026},
  note         = {Ready for submission to MDPI Hydrology}
}
```

### Doctoral Thesis Citation (BibTeX)

```bibtex
@phdthesis{PerezThesis2026,
  author       = {Perez Reyes, Manuel Ricardo},
  title        = {{Computational Model for Spatiotemporal Prediction of
                   Monthly Precipitation in Mountainous Areas: A Hybrid
                   Deep Learning Approach Using Graph Neural Networks
                   with Temporal Attention}},
  school       = {Pedagogical and Technological University of Colombia (UPTC)},
  year         = {2026},
  note         = {Doctoral Program in Engineering}
}
```

### Academic Use

For academic publications using this work:
1. Cite the software repository (required)
2. Cite the relevant paper(s) for the methodology you use:
   - **Systematic review of hybrid models** → cite the Systematic Review (Paper 1)
   - **ConvLSTM, GNN-TAT, FNO architectures** → cite the Hybrid Architecture Benchmark (Paper 4)
   - **Ensemble strategies, late fusion, SSM results** → cite the Sub-grid Feature Engineering study (Paper 5)
3. Cite the dataset if you use Boyaca data (required)
4. Cite the doctoral thesis once published (recommended)

### Commercial Use

Commercial use is permitted under CC BY 4.0 with proper attribution:
1. Include citation in product documentation
2. Acknowledge original author and institution
3. Provide link to this repository
4. For collaboration inquiries, contact author directly

---

## Funding Acknowledgment

This research is supported by:
- **Institution**: Pedagogical and Technological University of Colombia (UPTC)
- **Program**: Doctoral Program in Engineering

---

## Contact

**Author**: Manuel Ricardo Perez Reyes
**ORCID**: [0009-0003-2963-1631](https://orcid.org/0009-0003-2963-1631)
**Email**: manuelricardo.perez@uptc.edu.co
**Institution**: Pedagogical and Technological University of Colombia (UPTC)
**Program**: Doctoral Program in Engineering

**For Collaboration Inquiries**:
- Research collaboration: manuelricardo.perez@uptc.edu.co
- Technical questions: GitHub Issues
- Citations and academic use: Contact via ORCID profile

---

*Last Updated: March 2026*
*Project Status: V10 Late Fusion (R2=0.668) is the best model. Systematic Review under R3 review (IWA), Hybrid Architecture Benchmark accepted (MDPI Hydrology), Sub-grid Feature Engineering study in preparation.*
