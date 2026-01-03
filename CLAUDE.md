# ML Precipitation Prediction - Project Rules

## Project Context

**Title:** Computational model for the spatiotemporal prediction of monthly precipitation in mountainous areas using machine learning techniques

**Institution:** Pedagogical and Technological University of Colombia (UPTC)

**Domain:** Doctoral Thesis in Engineering - Hybrid Deep Learning for Hydrology

---

## 1. LANGUAGE STANDARD

- **ALL content MUST be in English**: code, comments, documentation, commits, filenames
- Exception: Spanish summaries in thesis only if required by university regulations
- Variable names, function names, and class names must be in English
- Git commit messages must be in English

---

## 2. FILE NAMING CONVENTIONS

### 2.1 General Rules
- Use **snake_case** exclusively: `file_name.extension`
- NO spaces, NO camelCase, NO special characters
- Use lowercase only (except for README.md, CLAUDE.md)

### 2.2 Specific Patterns

| Type | Pattern | Example |
|------|---------|---------|
| Notebooks | `base_models_{architecture}_{version}.ipynb` | `base_models_gnn_tat_v4.ipynb` |
| Metrics CSV | `metrics_spatial_{version}_{model}_h{horizon}.csv` | `metrics_spatial_v4_gnn_tat_h12.csv` |
| Training logs | `{model}_training_log_h{horizon}.csv` | `gnn_tat_gat_training_log_h12.csv` |
| History JSON | `{model}_history.json` | `gnn_tat_gat_history.json` |
| Checkpoints | `{model}_best_h{horizon}.pt` | `gnn_tat_gat_best_h12.pt` |
| Scripts | `{action}_{target}.py` | `generate_latex_tables.py` |
| Figures | `{type}_{metric}_{context}.png` | `heatmap_rmse_v4_comparison.png` |

---

## 3. DOCUMENTATION SYNCHRONIZATION

### 3.1 thesis.tex (Doctoral Thesis)
- **Location:** `docs/tesis/thesis.tex`
- **Purpose:** Complete methodology documentation
- **Content:** ALL notebook processes, EDA, preprocessing, feature engineering, model architectures
- **Update trigger:** Any methodology change in notebooks

### 3.2 paper.tex (Comparative Paper)
- **Location:** `docs/papers/4/paper.tex`
- **Purpose:** Model comparison (V2 vs V3 vs V4, baselines vs hybrids)
- **Content:** Benchmark results, statistical tests, conclusions
- **Update trigger:** New model results or comparative analysis

### 3.3 spec.md (Technical Specifications)
- **Location:** `models/spec.md`
- **Purpose:** Technical standards and framework definition
- **Content:** Notebook structure, output formats, evaluation metrics
- **Update trigger:** New standards or architectural changes

### 3.4 plan.md (Development Roadmap)
- **Location:** `models/plan.md`
- **Purpose:** Project timeline and milestone tracking
- **Content:** Progress status, hypothesis validation, deliverables
- **Update trigger:** Milestone completion or plan changes

### 3.5 Synchronization Rules

When modifying `models/*.ipynb`:
1. Update `thesis.tex` with methodology changes
2. Update `paper.tex` with benchmark results
3. Update `spec.md` with new standards
4. Update `plan.md` with progress

---

## 4. RESEARCH HYPOTHESES

### H1: Hybrid GNN-Temporal > ConvLSTM
- **Claim:** Hybrid GNN-Temporal models outperform ConvLSTM in spatial precipitation prediction
- **Validation Metric:** R² > 0.60, RMSE < 70mm
- **Current Status:** **VALIDATED** (V4: R²=0.707, RMSE=52.45mm)

### H2: Topographic Features Improve Prediction
- **Claim:** Incorporating topographic features (elevation, slope, aspect) improves prediction accuracy
- **Validation Method:** Compare BASIC vs KCE vs PAFC feature sets
- **Current Status:** **VALIDATED** (PAFC consistently best)

### H3: Non-Euclidean Spatial Relations
- **Claim:** Graph-based (non-Euclidean) spatial relations capture orographic dynamics better than CNNs
- **Validation Method:** GNN vs CNN architecture comparison
- **Current Status:** **IN VALIDATION**

### H4: Multi-Scale Temporal Attention
- **Claim:** Multi-scale temporal attention mechanisms improve long-horizon forecasting
- **Validation Metric:** R² degradation < 20% from H1 to H12
- **Current Status:** **PARTIALLY VALIDATED**

---

## 5. DEVELOPMENT FRAMEWORKS

### 5.1 SDD (Specification-Driven Development)

```
DEFINE    → spec.md: standards, requirements, constraints
    ↓
DESIGN    → plan.md: implementation approach, timeline
    ↓
DEVELOP   → notebooks: model implementation
    ↓
DOCUMENT  → thesis.tex: methodology documentation
    ↓
DELIVER   → paper.tex: results publication
    ↓
ITERATE   → back to DEFINE based on findings
```

**Key Principle:** Specifications must be defined BEFORE implementation begins.

### 5.2 DD (Data-Driven) Framework

```
HYPOTHESIS  → Define testable research questions (H1-H4)
    ↓
EXPERIMENT  → Design controlled experiments (V1-V6)
    ↓
MEASURE     → Collect standardized metrics (RMSE, MAE, R², Bias)
    ↓
ANALYZE     → Statistical validation (Friedman, Nemenyi tests)
    ↓
CONCLUDE    → Accept/reject hypotheses with evidence
    ↓
DOCUMENT    → Update thesis.tex and paper.tex
```

**Key Principle:** Empirical evidence drives all conclusions.

---

## 6. MODEL VERSIONING

| Version | Name | Architecture | Purpose | Status |
|---------|------|--------------|---------|--------|
| V1 | Baseline | ConvLSTM, ConvGRU, ConvRNN | Initial baselines | Complete |
| V2 | Enhanced | V1 + Attention + Bidirectional + Residual | Improved baselines | Complete |
| V3 | FNO | Fourier Neural Operators | Physics-informed | Complete (underperformed) |
| **V4** | **GNN-TAT** | **Graph Neural Networks + Temporal Attention** | **Hybrid spatial-temporal** | **In Progress** |
| V5 | Multi-Modal | V4 + ERA5 + Satellite data | Multi-source fusion | Planned |
| V6 | Ensemble | Best of V2-V5 + Meta-learning | Ensemble optimization | Planned |

---

## 7. OUTPUT STANDARDS

### 7.1 Metrics CSV Format
Required columns:
- `TotalHorizon`, `Experiment`, `Model`, `H`
- `RMSE`, `MAE`, `R2`
- `Mean_True_mm`, `Mean_Pred_mm`
- `mean_bias_mm`, `mean_bias_pct`

### 7.2 Training History JSON
Required fields:
- `model_name`, `experiment`, `horizon`
- `best_epoch`, `total_epochs`
- `best_val_loss`, `final_train_loss`, `final_val_loss`
- `parameters`

### 7.3 Directory Structure
```
models/output/{VERSION}_{MODEL}/
├── experiment_state_{version}.json
├── metrics_spatial_{version}_h{H}.csv
├── h{H}/
│   └── {EXPERIMENT}/
│       └── training_metrics/
│           ├── {model}_best_h{H}.pt
│           ├── {model}_history.json
│           └── {model}_training_log_h{H}.csv
```

### 7.4 Figure Standards
- Resolution: **700 DPI minimum** for publications
- Format: PNG for raster, PDF for vector
- Naming: `{type}_{metric}_{context}.png`

---

## 8. FEATURE SETS

| Set | Features | Purpose |
|-----|----------|---------|
| **BASIC** | Temporal encodings + Precipitation stats + Base topography | Minimal baseline |
| **KCE** | BASIC + K-means elevation clusters (one-hot) | Orographic regimes |
| **PAFC** | KCE + Precipitation lags (t-1, t-2, t-12) | Temporal autocorrelation |

---

## 9. EVALUATION METRICS

### Primary Metrics
- **RMSE** (Root Mean Square Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Robust to outliers
- **R²** (Coefficient of Determination): Explained variance

### Secondary Metrics
- **Bias (mm)**: Systematic over/under-estimation
- **Bias (%)**: Relative bias

### Statistical Tests
- **Friedman Test**: Non-parametric comparison across models
- **Nemenyi Post-hoc**: Pairwise significance at α=0.05

---

## 10. SUCCESS CRITERIA

| Metric | Baseline (V2) | Target | Excellent |
|--------|---------------|--------|-----------|
| R² (H1-H6) | 0.44 | > 0.60 | > 0.70 |
| R² (H7-H12) | 0.30 | > 0.50 | > 0.60 |
| RMSE (mm) | 98.17 | < 70 | < 55 |
| MAE (mm) | 44.19 | < 50 | < 40 |
| Parameters | 2M+ | < 150K | < 100K |

---

## 11. DOCUMENTATION FORMATTING STANDARDS

### 11.1 LaTeX Figure Width Standards

**CRITICAL:** All figures must stay within page margins. Use these predefined commands:

```latex
% Safe width specifications for A4 paper with 25mm margins
\newcommand{\fullwidth}{0.95\textwidth}     % Single full-width figure
\newcommand{\halfwidth}{0.47\textwidth}     % Two figures side-by-side
\newcommand{\thirdwidth}{0.31\textwidth}    % Three figures in row
\newcommand{\quarterwidth}{0.23\textwidth}  % Four figures in row
```

**Figure Template:**
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\fullwidth]{filename.png}
    \caption{Descriptive caption with units and data source.}
    \label{fig:meaningful-label}
\end{figure}
```

### 11.2 Light Mode Notation

When presenting results from reduced grid experiments (e.g., 5×5 grid subset), include this disclaimer:

```latex
\textbf{Important Note:} Results presented in this section were obtained using
\textit{light mode} (5×5 grid subset) for rapid prototyping. Full-grid validation
(61×65) is pending and will be reported in Section~\ref{sec:full-grid-results}.
```

### 11.3 Official Thesis Title

**Extended Title (Aligned with Doctoral Proposal):**
> "Computational Model for Spatiotemporal Prediction of Monthly Precipitation in Mountainous Areas: A Hybrid Deep Learning Approach Using Graph Neural Networks with Temporal Attention"

**Original Proposal Title:**
> "Computational model for the spatiotemporal prediction of monthly precipitation in mountainous areas using machine learning techniques"

### 11.4 Citation Standards

- Bibliography must contain 100+ Q1 references
- Use `\citep{}` for parenthetical citations, `\citet{}` for textual
- Every major claim must have a supporting citation
- Reference categories: GNN, Precipitation ML, Hydrology Data, DL Fundamentals, Statistics

### 11.5 Doctoral Thesis Structure (ML/Hydrology Standard)

```
Chapter 1: Introduction (20 pages)
  - Problem Statement, Hypotheses, Objectives, Contributions

Chapter 2: Literature Review (40 pages)
  - State of the Art, Research Gaps

Chapter 3: Theoretical Framework (30 pages)
  - Graph Theory, Attention Mechanisms, Statistical Framework

Chapter 4: Materials and Methods (40 pages)
  - Study Area, Data Sources, Preprocessing, Model Architectures

Chapter 5: Results (50 pages)
  - V1-V4 Results, Statistical Tests, Hypothesis Validation

Chapter 6: Discussion (30 pages)
  - Interpretation, Comparison with SOTA, Limitations

Chapter 7: Conclusions (15 pages)
  - Contributions, Future Work

References: 100-150 entries
Appendices: Code, Tables, Figures
```

---

## 12. FILE SCOPE RULES

### 12.1 Project Scope Definition

This project focuses EXCLUSIVELY on the doctoral thesis: **"Computational Model for Spatiotemporal Prediction of Monthly Precipitation in Mountainous Areas"**

**IN-SCOPE files must directly support:**
1. Model training and evaluation (V1-V6)
2. Data preprocessing pipeline (CHIRPS, SRTM, ERA5)
3. Thesis and paper documentation
4. Statistical analysis and benchmarking
5. Feature engineering (BASIC, KCE, PAFC)

### 12.2 Directory Scope Classification

| Directory | Scope | Purpose |
|-----------|-------|---------|
| `models/*.ipynb` | **CORE** | Model implementations (V1-V6) |
| `models/output/` | **CORE** | Training outputs and metrics |
| `data/` | **CORE** | ETL pipeline and datasets |
| `docs/tesis/` | **CORE** | Doctoral thesis |
| `docs/papers/4/` | **CORE** | Comparative paper |
| `scripts/benchmark/` | **CORE** | Results analysis |
| `notebooks/` | **CORE** | EDA and preprocessing |
| `utils/`, `preprocessing/`, `process/` | **SUPPORTING** | Pipeline utilities |
| `docs/models/` | **SUPPORTING** | Model comparison outputs |
| `docs/framework/` | **SUPPORTING** | Strategy documentation |
| `docs/architecture/` | **SUPPORTING** | PlantUML diagrams |
| `figures/`, `images/` | **SUPPORTING** | Visualizations |
| `docs/adrs/`, `docs/deas/`, `docs/rfcs/` | **TEMPLATES** | Empty template directories |
| `.venv/`, `.git/`, `.pytest_cache/` | **SYSTEM** | Auto-generated |

### 12.3 OUT-OF-SCOPE Criteria

**DO NOT CREATE files that:**
1. Are backups or copies (e.g., `*_backup.py`, `*_old.ipynb`)
2. Serve non-thesis AI tasks (e.g., AI assistant experiments)
3. Are temporary exploration unrelated to precipitation prediction
4. Duplicate existing functionality without clear purpose
5. Are in languages other than English (except legacy Spanish files)

### 12.4 Legacy Spanish Files (To Be Translated)

The following files require English translation:
- `notebooks/analisis_bimodal_boyaca.ipynb` → `bimodal_analysis_boyaca.ipynb`
- `notebooks/analisis_correlacion.ipynb` → `correlation_analysis.ipynb`
- `notebooks/conv_lstm_boyaca_comparado.ipynb` → `conv_lstm_boyaca_comparison.ipynb`
- `docs/architecture/README_diagramas.md` → `README_diagrams.md`
- Spanish `.puml` files in `docs/architecture/`

### 12.5 CamelCase Files (To Be Renamed)

The following violate snake_case convention:
- `base_models_Conv_STHyMOUNTAIN*.ipynb` → `base_models_conv_sthymountain*.ipynb`
- `base_models_ST-HybridWaveStack.ipynb` → `base_models_st_hybrid_wave_stack.ipynb`
- `hybrid_models_ElevClusConvPrecipMetaNet.ipynb` → `hybrid_models_elev_clus_conv_precip_meta_net.ipynb`
- `hybrid_models_TopoRain_NET*.ipynb` → `hybrid_models_topo_rain_net*.ipynb`
- Output directories: `Advanced_Spatial/`, `Spatial_CONVRNN/`, `TS_CNN_ConvLSTM/`

### 12.6 Orphan Files (Review Required)

Files at root level that may need relocation or deletion:
- `cluster_one_hot_encoding.png` → Move to `figures/` or `docs/`
- `nul` → Delete (accidental Windows artifact)
- `output/` at root → Consider merging with `models/output/`

### 12.7 Before Creating New Files

1. Verify the file serves thesis objectives (H1-H4)
2. Check if existing file can be modified instead
3. Follow naming conventions (Section 2)
4. Place in appropriate directory (Section 12.2)
5. Document purpose in spec.md if new pattern

---

## 13. GIT CONVENTIONS

### Branch Naming
- `main`: Production-ready code
- `feature/{feature-name}`: New features
- `fix/{bug-description}`: Bug fixes
- `experiment/{experiment-name}`: Experimental branches

### Commit Messages
Format: `{type}: {description}`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Tests
- `data`: Data changes

Example: `feat: add GNN-TAT V4 memory optimization`

---

## 15. REFERENCES

### Bibliography Location
- **File:** `docs/tesis/references.bib`
- **Total Entries:** 110+ Q1 references

### Reference Categories

| Category | Count | Key Authors |
|----------|-------|-------------|
| Graph Neural Networks | 25 | Kipf, Velickovic, Hamilton, Wu, Scarselli |
| Precipitation/Weather ML | 25 | Shi, Ravuri, Reichstein, Lam, Bi |
| Climate/Hydrology Data | 15 | Funk (CHIRPS), Hersbach (ERA5), Farr (SRTM) |
| Deep Learning Fundamentals | 20 | Vaswani, Hochreiter, He, Goodfellow |
| Statistical Methods | 10 | Friedman, Nemenyi, Nash, Gupta |
| Spatiotemporal Modeling | 15 | Wang, Lin, Gao, Tan |

### Datasets
- **CHIRPS 2.0**: Precipitation (0.05° resolution, 518 monthly steps)
- **SRTM DEM**: Elevation data (90m resolution)
- **ERA5**: Atmospheric reanalysis (planned for V5)

---

*Document Version: 2.0*
*Last Updated: January 2026*
*Project: ML Precipitation Prediction - Doctoral Thesis*
*Phase 2: Documentation & Formatting Standards Applied*
