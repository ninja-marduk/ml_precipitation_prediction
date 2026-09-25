# ML Precipitation Prediction

## Computational Model for the Spatiotemporal Prediction of Monthly Precipitation in Mountainous Areas Using Machine Learning Techniques

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21576207.svg)](https://doi.org/10.5281/zenodo.21576207)

Code, analysis scripts and outputs of a doctoral thesis (Doctoral Program in Engineering, Pedagogical and Technological University of Colombia, UPTC) on monthly precipitation prediction over the mountainous terrain of Boyaca, Colombia. The repository also holds the reference implementation of **AnchorGate v1.0**, an anchored, seed-resolved evaluation protocol for data-driven environmental prediction.

---

## Publications

| Article | Focus | Journal | Status |
|---------|-------|---------|--------|
| **Article 1** | Systematic review of hybrid deep learning models for monthly precipitation prediction | Hydrology Research (Elsevier) | Published 2026 ([doi:10.1016/j.hydrch.2026.100008](https://doi.org/10.1016/j.hydrch.2026.100008)) |
| **Article 2** | Architectural benchmark (ConvLSTM family, FNO-ConvLSTM, GNN-TAT) | Hydrology (MDPI), 13(3), 98 | Published 2026 ([doi:10.3390/hydrology13030098](https://doi.org/10.3390/hydrology13030098)) |
| **Article 3** | Evaluation protocol (AnchorGate v1.0) and re-analysis of the benchmark and ensembles | Geoscientific Model Development (Copernicus) | Submitted 8 September 2026, under review (egusphere-2026-5272) |

---

## Main Findings

The central finding concerns evaluation rather than architecture. Consecutive forecast windows share eleven of their twelve target months, so shuffled fold schemes let evaluation months leak into what a model is fitted on. The thesis builds a ladder of fold schemes that removes this overlap step by step, ending in a contiguous holdout purged by an eleven-window embargo.

- **No architecture or ensemble beats a per-cell monthly climatology** once target-month overlap is removed. The climatology has no fitted parameters and reaches R² = 0.730 over the validation windows (0.732 averaged per window) and R² = 0.763 on the purged holdout block. It has the lowest RMSE in 33 of 33 validation windows.
- **On the purged holdout** (three seeds): best base learner R² = 0.639 ± 0.060; Late Fusion (Ridge over Enhanced ConvLSTM and GNN-TAT) R² = 0.598 ± 0.022. Late Fusion's paired advantage over its own best base learner is -0.041 ± 0.041, so the fusion gain does not survive purging.
- **Main hypothesis not supported.** The approved main hypothesis, that hybrid graph-temporal models improve on established baselines, is not supported once evaluation leakage is removed.
- **Seed variance matters.** Seed choice alone moves R² by a median of 0.045 (up to 0.114), more than most architecture differences. After Holm correction, none of the five pairwise architecture comparisons is significant.
- **Released-fold figures (leaky, for reference only).** On the released, shuffled folds (about 100 % target overlap) Late Fusion reaches pooled R² = 0.672 for seed 42 and 0.655 as the mean over three seeds. Under blocked folds (99.5 % overlap) it reaches 0.640 ± 0.006. These numbers are not evidence of skill over climatology.

### Architecture notes

| Model | Result | Note |
|-------|--------|------|
| Enhanced ConvLSTM (Bidirectional) | Peak R² = 0.653 | Single run, pre-correction pipeline |
| GNN-TAT (GAT encoder) | Peak R² = 0.628 at H=5 (best horizon of best of 3 seeds); seed-resolved mean 0.446-0.510 over H=1-12 | 98K parameters, 34 % fewer than the best ConvLSTM (148K). Its lower mean RMSE (p = 0.015 uncorrected) is not significant after Holm correction |
| FNO / FNO-ConvLSTM | R² = 0.206 / 0.582 at the 12-month lead | The ConvLSTM decoder recovers most of the loss; spectral truncation smooths sharp orographic gradients |
| Stacking (early fusion) | R² = 0.212 | Collapses towards each cell's mean |
| Stratified ensemble | R² = 0.597 | No gain over the dominant base model (pre-correction run) |
| GNN-BiMamba | R² = 0.18 (validation windows) | Collapses towards each cell's mean |

Sub-hypotheses: feature hybridization (KCE, PAFC over BASIC) is not supported; advanced ConvLSTM variants are partially supported (small, non-significant gains); physics-data hybrids (pure FNO) are rejected for precipitation.

---

## Data

- **CHIRPS 2.0** monthly precipitation (0.05°) and **SRTM** elevation (90 m).
- **Period:** 518 months, January 1982 to February 2025.
- **Domain:** a 61 x 65 rectangle (3,965 cells at 0.05°, 4.375-7.375 N, 74.925-71.725 W) that encloses Boyaca; 757 cells lie inside the department. All metrics are computed over the full rectangle.
- **Lead times:** 1 to 12 months.
- **Graph (GNN-TAT):** 3,965 nodes with a 500,000-edge budget.

### Feature sets

| Set | Features | Description |
|-----|----------|-------------|
| BASIC | 12 | Temporal encodings + precipitation statistics + base topography |
| KCE | 15 | BASIC + K-means elevation clusters (k = 3) |
| PAFC | 18 | KCE + precipitation lags (t-1, t-2, t-12) |

### Sub-cell DEM features (negative result)

Three intra-cell DEM bundles (BASIC_D10: elevation deciles, 22 features; BASIC_PCA6: 18; BASIC_D10_STATS: 27) were tested. Every bundle degrades every model relative to BASIC (mean per-cell R² in the Low elevation zone, below 1,500 m, pooled over the twelve leads; single runs on the pre-correction pipeline):

| Feature bundle | Enhanced ConvLSTM | GNN-TAT | Late Fusion |
|----------------|-------------------|---------|-------------|
| BASIC (baseline) | 0.511 | 0.473 | 0.549 |
| BASIC_D10 | 0.321 (-37.2 %) | 0.400 (-15.4 %) | 0.498 (-9.2 %) |
| BASIC_PCA6 | 0.278 (-45.5 %) | 0.287 (-39.2 %) | 0.449 (-18.1 %) |
| BASIC_D10_STATS | 0.180 (-64.7 %) | 0.293 (-38.1 %) | 0.357 (-35.0 %) |

These are single-run figures from before the graph-construction correction; the direction of the result is reliable, the exact percentages are not at the precision of the seed-resolved numbers above.

---

## Repository Structure

```
ml_precipitation_prediction/
├── data/                  # Input data and processed NetCDF
├── models/
│   ├── base_models_*.ipynb    # Model notebooks
│   ├── intracell_dem/         # Sub-cell DEM experiments
│   ├── scripts/analysis/      # Evaluation protocol and re-analysis scripts
│   └── output/                # Training outputs and prediction arrays
├── preprocessing/         # Feature engineering
├── scripts/benchmark/     # Benchmark metrics and figures
├── workflows/             # End-to-end pipeline (stages 1-9)
└── utils/
```

| Model | Notebook | Output directory |
|-------|----------|------------------|
| Enhanced ConvLSTM | `models/base_models_conv_sthymountain_v2.ipynb` | `models/output/V2_Enhanced_Models/` |
| FNO / FNO-ConvLSTM | `models/base_models_conv_sthymountain_v3_fno.ipynb` | |
| GNN-TAT | `models/base_models_gnn_tat_v4.ipynb` | `models/output/V4_GNN_TAT_Models/` |
| Stacking | `models/base_models_gnn_convlstm_stacking_v5.ipynb` | |
| GNN-BiMamba | `models/base_models_gnn_bimamba_v9.ipynb` | |
| Late Fusion | `models/base_models_late_fusion_v10.ipynb` | `models/output/V10_Late_Fusion/` |

---

## Installation and Usage

```bash
git clone https://github.com/ninja-marduk/ml_precipitation_prediction.git
cd ml_precipitation_prediction

conda create -n precipitation python=3.12
conda activate precipitation
pip install -r requirements.txt   # requirements-lock.txt pins every dependency
```

Reference environment: Python 3.12, PyTorch 2.6.0 (CUDA 12.4), PyTorch Geometric 2.7.0. Training was run on an NVIDIA A100 GPU; every analysis script runs on CPU.

```bash
# Full pipeline (stages 1-9)
python workflows/run_pipeline.py

# Post-training stages only (fusion, benchmarks, figures; no GPU)
python workflows/run_pipeline.py --from 7
```

See the [Workflows README](workflows/README.md) for all options.

### Dataset

The analysis-ready dataset (CHIRPS + SRTM with the BASIC, KCE and PAFC feature bundles) is published on Kaggle: [CHIRPS-DEM Boyaca monthly precipitation (ML-ready)](https://www.kaggle.com/datasets/engricardoperez/chirps-dem-boyaca-monthly-precipitation-ml). Extract it to `data/`.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff).

### Software

```bibtex
@software{PerezReyes2026AnchorGate,
  author    = {P\'erez Reyes, Manuel Ricardo and Su\'arez Bar\'on, Marco Javier
               and Garc\'ia Cabrejo, \'Oscar Javier and Castillo-Reyes, Octavio},
  title     = {{AnchorGate v1.0: an anchored, seed-resolved evaluation protocol
                for data-driven environmental prediction}},
  year      = {2026},
  version   = {v1.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22216993},
  url       = {https://doi.org/10.5281/zenodo.22216993},
  note      = {Concept DOI (all versions): 10.5281/zenodo.21576207}
}
```

### Dataset

```bibtex
@dataset{PerezReyes2026BoyacaDataset,
  author    = {P\'erez Reyes, Manuel Ricardo},
  title     = {{CHIRPS-DEM Boyac\'a Monthly Precipitation (ML-ready)}},
  year      = {2026},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/engricardoperez/chirps-dem-boyaca-monthly-precipitation-ml},
  note      = {61x65 grid, 518 monthly steps, January 1982 to February 2025}
}
```

### Articles

```bibtex
@article{PerezReyes2026Review,
  author    = {P\'erez Reyes, Manuel Ricardo and Su\'arez Bar\'on, Marco Javier
               and Garc\'ia Cabrejo, \'Oscar Javier},
  title     = {{Hybrid Deep Learning Models for Monthly Precipitation
                Prediction: A Systematic Review}},
  journal   = {Hydrology Research},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.hydrch.2026.100008}
}

@article{PerezReyes2026Hybrid,
  author    = {P\'erez Reyes, Manuel Ricardo and Su\'arez Bar\'on, Marco Javier
               and Garc\'ia Cabrejo, \'Oscar Javier},
  title     = {{A Data-Driven Deep Learning Framework for Monthly Precipitation
                Prediction in Complex Mountainous Terrain: Systematic Evaluation
                of Hybrid Architectures}},
  journal   = {Hydrology},
  volume    = {13},
  number    = {3},
  pages     = {98},
  year      = {2026},
  publisher = {MDPI},
  doi       = {10.3390/hydrology13030098}
}

@article{PerezReyes2026Fusion,
  author    = {P\'erez Reyes, Manuel Ricardo and Su\'arez Bar\'on, Marco Javier
               and Garc\'ia Cabrejo, \'Oscar Javier and Castillo-Reyes, Octavio},
  title     = {{Spatiotemporal Prediction of Monthly Precipitation in Mountainous
                Terrain: A Benchmark of Hybrid Deep Learning Architectures}},
  journal   = {Geoscientific Model Development},
  year      = {2026},
  note      = {Submitted 8 September 2026, under review; manuscript egusphere-2026-5272}
}
```

### Doctoral thesis

```bibtex
@phdthesis{PerezThesis2026,
  author    = {P\'erez Reyes, Manuel Ricardo},
  title     = {{Computational Model for the Spatiotemporal Prediction of Monthly
                Precipitation in Mountainous Areas Using Machine Learning Techniques}},
  school    = {Pedagogical and Technological University of Colombia (UPTC)},
  year      = {2026},
  note      = {Doctoral Program in Engineering}
}
```

---

## Funding

Development partially supported by Becas de Excelencia Doctoral del Bicentenario (MinCiencias Colombia, BPIN 2021000100031, Plan Bienal FCTeI, Sistema General de Regalias), Universidad Pedagogica y Tecnologica de Colombia, grant 2021-SGR-00478 (Generalitat de Catalunya, AGAUR), and in-kind compute on the Barcelona Supercomputing Center MareNostrum 5 supercomputer.

---

## Contact

**Author:** Manuel Ricardo Perez Reyes
**ORCID:** [0009-0003-2963-1631](https://orcid.org/0009-0003-2963-1631)
**Email:** manuelricardo.perez@uptc.edu.co
**Institution:** Pedagogical and Technological University of Colombia (UPTC), Doctoral Program in Engineering

Technical questions: GitHub Issues.

---

*Last updated: 2026-09-25*
