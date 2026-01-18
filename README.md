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

### Hybridization Rescue Effect (H5)

A key finding: **Pure spectral methods (FNO) fail for precipitation** (R²=0.206), but **hybridization rescues performance** through component integration:

- Pure FNO: R²=0.206
- FNO-ConvLSTM Hybrid: R²=0.582
- **Improvement: 182%**

This validates the component-combination approach used across all V2-V4 architectures.

---

## Research Hypotheses

| ID | Hypothesis | Status | Evidence |
|----|------------|--------|----------|
| H1 | Hybrid GNN-Temporal models achieve comparable or better accuracy than ConvLSTM | **PARTIALLY VALIDATED** | R²=0.628 vs 0.642; Mean RMSE significantly lower (p=0.015) |
| H2 | Topographic features improve prediction accuracy | **VALIDATED** | KCE and PAFC significantly improve GNN performance (p<0.05) |
| H3 | Non-Euclidean spatial relations capture orographic dynamics | **VALIDATED** | 3,965 nodes, 500,000 edges successfully trained |
| H4 | Multi-scale temporal attention improves long horizons | **VALIDATED** | R² degradation 9.6% (H1→H12), below 20% threshold |
| H5 | Hybridization rescues architectural limitations | **VALIDATED** | Pure FNO R²=0.206 → Hybrid R²=0.582 (182% improvement) |

---

## Model Versions

| Version | Architecture | Purpose | Status | Best R² |
|---------|--------------|---------|--------|---------|
| V1 | ConvLSTM, ConvGRU, ConvRNN | Baselines | Complete | 0.642 |
| V2 | Enhanced + Attention + Bidirectional | Improved baselines | Complete | 0.653 |
| V3 | Fourier Neural Operators (FNO) | Physics-informed | Complete | 0.312 (underperformed) |
| **V4** | **GNN-TAT** | **Hybrid spatial-temporal** | **Complete** | **0.628** |
| V5 | **GNN-ConvLSTM Stacking** | Dual-branch ensemble | In Progress | TBD |
| V6 | Ensemble | Meta-learning | Planned | TBD |

---

## Project Structure

```
ml_precipitation_prediction/
├── data/                         # Input data (CHIRPS, SRTM)
├── models/
│   ├── base_models_*.ipynb       # Model notebooks (V1-V4)
│   └── output/                   # Training outputs
├── notebooks/                    # Exploratory analysis
├── scripts/                      # Automation scripts
└── utils/                        # Utility functions
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
2. Cite the dataset if you use Boyacá data (required)
3. Cite the doctoral thesis once published (recommended)
4. Cite related publications with DOIs when available (check repository for updates)

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
**ORCID**: [Add your ORCID ID]
**Email**: manuelricardo.perez@uptc.edu.co
**Institution**: Pedagogical and Technological University of Colombia (UPTC)
**Program**: Doctoral Program in Engineering

**For Collaboration Inquiries**:
- Research collaboration: manuelricardo.perez@uptc.edu.co
- Technical questions: GitHub Issues
- Citations and academic use: Contact via ORCID profile

---

*Last Updated: January 18, 2026*
*Project Status: V4 complete (R²=0.628, 61×65 full-grid), V5 stacking implementation in progress*
