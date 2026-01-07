# ML Precipitation Prediction

## Computational Model for Spatiotemporal Prediction of Monthly Precipitation in Mountainous Areas

**A Hybrid Deep Learning Approach Using Graph Neural Networks with Temporal Attention**

---

## Project Overview

This repository contains the implementation of a doctoral thesis project developing hybrid deep learning models for monthly precipitation prediction in the mountainous terrain of Boyaca, Colombia. The research follows a Data-Driven (DD) scientific methodology with rigorous statistical validation.

### Key Achievement: GNN-TAT (V4) - Full Grid Results

| Metric | V2 ConvLSTM | V4 GNN-TAT | Comparison |
|--------|-------------|------------|------------|
| R² | 0.642 | **0.628** | Comparable (-2.2%) |
| RMSE | 77.55mm | 82.29mm | Comparable |
| Parameters | 500K-2.1M | **98K** | **95% fewer** |
| Mean RMSE | 112.02mm | **92.12mm** | **17.8% lower*** |

*Statistical significance: Mann-Whitney U=57.00, p=0.015, Cohen's d=1.03 (large effect)

### Value Proposition

GNN-TAT achieves **comparable predictive performance** to ConvLSTM baselines while offering:
1. **95% parameter reduction** (98K vs 500K-2.1M parameters)
2. **Interpretable spatial relationships** through explicit graph structure
3. **Significantly lower mean RMSE** across all configurations (p=0.015)

---

## Research Hypotheses

| ID | Hypothesis | Status | Evidence |
|----|------------|--------|----------|
| H1 | Hybrid GNN-Temporal models achieve comparable or better accuracy than ConvLSTM | **PARTIALLY VALIDATED** | R²=0.628 vs 0.642; Mean RMSE significantly lower (p=0.015) |
| H2 | Topographic features improve prediction accuracy | **VALIDATED** | KCE and PAFC significantly improve GNN performance (p<0.05) |
| H3 | Non-Euclidean spatial relations capture orographic dynamics | **VALIDATED** | 3,965 nodes, 500,000 edges successfully trained |
| H4 | Multi-scale temporal attention improves long horizons | **VALIDATED** | R² degradation 9.6% (H1→H12), below 20% threshold |

---

## Model Versions

| Version | Architecture | Purpose | Status | Best R² |
|---------|--------------|---------|--------|---------|
| V1 | ConvLSTM, ConvGRU, ConvRNN | Baselines | Complete | 0.642 |
| V2 | Enhanced + Attention + Bidirectional | Improved baselines | Complete | 0.653 |
| V3 | Fourier Neural Operators (FNO) | Physics-informed | Complete | 0.312 (underperformed) |
| **V4** | **GNN-TAT** | **Hybrid spatial-temporal** | **Complete** | **0.628** |
| V5 | Multi-Modal | ERA5 + Satellite integration | Planned | TBD |
| V6 | Ensemble | Meta-learning | Planned | TBD |

---

## Project Structure

```
ml_precipitation_prediction/
├── CLAUDE.md                     # Project rules and standards
├── data/                         # Input data (CHIRPS, SRTM)
├── docs/
│   ├── tesis/                    # Doctoral thesis (thesis.tex)
│   │   └── references.bib        # 110+ Q1 references
│   └── papers/4/                 # Comparative paper (paper.tex)
├── models/
│   ├── spec.md                   # Technical specifications
│   ├── plan.md                   # Development roadmap
│   ├── base_models_*.ipynb       # Model notebooks (V1-V4)
│   └── output/                   # Training outputs
├── notebooks/                    # Exploratory analysis
└── scripts/                      # Automation scripts
```

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

| Set | Features | Description |
|-----|----------|-------------|
| BASIC | 12 | Temporal encodings + precipitation stats + base topography |
| KCE | 15 | BASIC + K-means elevation clusters |
| PAFC | 18 | KCE + precipitation autocorrelation lags (t-1, t-2, t-12) |

---

## Installation

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

### Running Notebooks

1. **V4 GNN-TAT** (recommended):
   ```bash
   jupyter notebook models/base_models_GNN_TAT_V4.ipynb
   ```

2. **V2 Enhanced Models**:
   ```bash
   jupyter notebook models/base_models_Enhanced_V2.ipynb
   ```

### Google Colab

V4 notebook is optimized for Colab with automatic GPU detection and PyTorch Geometric installation.

---

## Evaluation Metrics

- **RMSE**: Root Mean Square Error (mm)
- **MAE**: Mean Absolute Error (mm)
- **R²**: Coefficient of Determination
- **Bias**: Mean prediction bias (mm, %)
- **Statistical Tests**: Friedman + Nemenyi post-hoc

---

## Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Thesis | `docs/tesis/thesis.tex` | Complete methodology |
| Paper | `docs/papers/4/paper.tex` | Comparative results |
| Specifications | `models/spec.md` | Technical standards |
| Plan | `models/plan.md` | Development roadmap |
| Rules | `CLAUDE.md` | Project governance |

---

## Citation

```bibtex
@phdthesis{perez2026precipitation,
  title={Computational Model for Spatiotemporal Prediction of Monthly
         Precipitation in Mountainous Areas: A Hybrid Deep Learning
         Approach Using Graph Neural Networks with Temporal Attention},
  author={P{\'e}rez Reyes, Manuel Ricardo},
  year={2026},
  school={Pedagogical and Technological University of Colombia (UPTC)}
}
```

---

## Key References

1. Kipf & Welling (2017) - Graph Convolutional Networks
2. Velickovic et al. (2018) - Graph Attention Networks
3. Shi et al. (2015) - ConvLSTM for Precipitation Nowcasting
4. Vaswani et al. (2017) - Attention Is All You Need
5. Funk et al. (2015) - CHIRPS Precipitation Dataset

Full bibliography: `docs/tesis/references.bib` (110+ Q1 references)

---

## License

MIT License - See LICENSE file for details.

---

## Contact

**Author**: Manuel Ricardo Perez Reyes
**Institution**: Pedagogical and Technological University of Colombia (UPTC)
**Program**: Doctoral Program in Engineering

---

*Last Updated: January 2026*
*Project Status: V4 GNN-TAT complete (full-grid 61x65), thesis and paper ready for submission*
