# PhD Dissertation Specification

## Computational Model for the Spatiotemporal Prediction of Monthly Precipitation in Mountainous Areas Using Machine Learning Techniques

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Document ID** | SPEC-PHD-PRECIP-2025 |
| **Version** | 2.0.0 |
| **Status** | Active |
| **Author** | Manuel Ricardo Pérez Reyes |
| **Advisor** | PhD. Marco Javier Suárez Barón |
| **Co-Advisor** | PhD. Oscar Javier García Cabrejo |
| **Institution** | Universidad Pedagógica y Tecnológica de Colombia (UPTC) |
| **Faculty** | Facultad de Ingeniería |
| **Program** | Doctorado en Ingeniería |
| **Research Group** | GALASH |
| **Research Lines** | Artificial Intelligence (AI), Computational Hydrology |
| **Last Updated** | January 2026 |

---

## 1. Executive Summary

This doctoral thesis addresses the critical challenge of predicting monthly precipitation in mountainous regions using machine learning techniques. The research focuses on the Department of Boyacá, Colombia, characterized by altitudes ranging from 145 to 5,490 meters above sea level. Climate change and phenomena such as El Niño and La Niña have disrupted traditional precipitation patterns, making accurate prediction essential for water resource management, agricultural planning, and disaster preparedness.

The research develops and validates hybrid deep learning models that combine Graph Neural Networks with Temporal Attention mechanisms (GNN-TAT) to capture both spatial topographic relationships and temporal dependencies in precipitation patterns.

---

## 2. Research Problem

### 2.1 Problem Statement

Precipitation prediction is inherently complex due to three key factors:
1. **Non-linearity**: Precipitation patterns are influenced by complex interactions among multiple climatic variables
2. **Non-stationarity**: Statistical properties of precipitation time series change over time due to climate change
3. **Stochastic nature**: High randomness and uncertainty in long-term predictions

In Boyacá, Colombia:
- 25% of municipalities faced drought emergencies (IDEAM, 2014)
- Limited spatial uniformity of ground-based sensor stations
- Temporal gaps in historical records
- Complex orography affecting precipitation patterns

### 2.2 Research Questions

**Primary Question:**
> How can machine learning models, combined with time series analysis and data preprocessing methods, significantly improve the accuracy of monthly precipitation predictions in mountainous areas?

**Secondary Questions:**
1. What hybrid architectures combining spatial and temporal modeling yield optimal performance?
2. How does feature engineering (topographic, climatic indices) impact prediction accuracy?
3. What is the degradation rate of prediction accuracy across extended horizons (H=1 to H=12)?
4. How do Graph Neural Networks compare to traditional ConvLSTM architectures?

---

## 3. Research Hypothesis and Objectives

### 3.1 Main Hypothesis

> "Applying machine learning models, combined with time series analysis and data preprocessing methods, will significantly improve the accuracy of monthly precipitation predictions in mountainous areas."

### 3.2 Derived Hypotheses

| ID | Hypothesis | Validation Criteria | Status |
|----|-----------|---------------------|--------|
| **H1** | Hybrid GNN-Temporal models achieve comparable or better accuracy than ConvLSTM | R² ≥ 0.60, statistical significance (p < 0.05) | PARTIALLY VALIDATED |
| **H2** | Topographic features (KCE, PAFC) improve prediction accuracy | Compare BASIC vs KCE vs PAFC feature sets | VALIDATED |
| **H3** | Non-Euclidean spatial relations capture orographic dynamics | GNN edge weights correlate with physical processes | VALIDATED |
| **H4** | Multi-scale temporal attention improves long horizons | R² degradation < 20% from H=1 to H=12 | VALIDATED (9.6%) |

### 3.3 Objectives

#### Overall Objective

To design and implement a computational model for predicting monthly precipitation in mountainous areas, specifically the Department of Boyacá, Colombia, using machine learning techniques.

#### Specific Objectives

| ID | Objective | Deliverable | Success Metric |
|----|-----------|-------------|----------------|
| **SO1** | Develop an end-to-end data-driven pipeline integrating CHIRPS 2.0 data and DEM for Boyacá | Working pipeline + documentation | Complete preprocessing chain |
| **SO2** | Benchmark ConvRNN/ConvLSTM architectures against hybrid GNN-TAT models | Comparison study + paper | Statistical significance |
| **SO3** | Quantify the value of feature engineering (BASIC, KCE, PAFC) relative to model capacity | Feature analysis report | Ablation study results |
| **SO4** | Validate results with statistical significance tests | Statistical report | p < 0.05 on key comparisons |

---

## 4. Thesis Structure (IMRD Extended)

### Chapter Organization

```
FRONT MATTER
├── Title Page
├── Dedication
├── Acknowledgments
├── Abstract (English and Spanish)
├── Table of Contents
├── List of Figures
├── List of Tables
├── List of Abbreviations

CHAPTER 1: INTRODUCTION (20-25 pages)
├── 1.1 Research Context and Motivation
├── 1.2 Problem Formulation
├── 1.3 Research Questions
├── 1.4 Main Hypothesis and Derived Hypotheses (H1-H4)
├── 1.5 Objectives (Overall + Specific)
├── 1.6 Scope and Limitations
├── 1.7 Contributions
└── 1.8 Thesis Structure Overview

CHAPTER 2: THEORETICAL FRAMEWORK AND STATE OF THE ART (40-50 pages)
├── 2.1 Precipitation Physics in Mountainous Regions
│   ├── 2.1.1 Orographic Effects
│   ├── 2.1.2 ENSO Influence on Colombian Precipitation
│   └── 2.1.3 Non-stationarity and Climate Change
├── 2.2 Traditional Precipitation Prediction Methods
│   ├── 2.2.1 Statistical Methods (ARIMA, SARIMA)
│   └── 2.2.2 Numerical Weather Prediction (NWP)
├── 2.3 Machine Learning for Precipitation Prediction
│   ├── 2.3.1 Shallow Learning (SVM, RF, XGBoost)
│   ├── 2.3.2 Deep Learning Fundamentals
│   ├── 2.3.3 Convolutional Neural Networks (CNN)
│   ├── 2.3.4 Recurrent Neural Networks (LSTM, GRU)
│   ├── 2.3.5 ConvLSTM and Spatiotemporal Models
│   ├── 2.3.6 Attention Mechanisms and Transformers
│   └── 2.3.7 Graph Neural Networks (GCN, GAT, SAGE)
├── 2.4 Hybrid Predictive Models
│   ├── 2.4.1 Data Preprocessing-based Hybrids (CEEMD, Wavelet)
│   ├── 2.4.2 Parameter Optimization-based Hybrids (PSO, GA)
│   ├── 2.4.3 Component Combination-based Hybrids
│   └── 2.4.4 Postprocessing-based Hybrids
├── 2.5 Systematic Literature Review
│   ├── 2.5.1 PRISMA Methodology
│   ├── 2.5.2 Quantitative Synthesis (85 studies, 2020-2025)
│   └── 2.5.3 Research Gaps Identified
└── 2.6 Colombian Context and Related Work

CHAPTER 3: MATERIALS AND METHODS (40-50 pages)
├── 3.1 Study Area: Boyacá, Colombia
│   ├── 3.1.1 Geographic Description
│   ├── 3.1.2 Climate Characteristics
│   └── 3.1.3 Grid Configuration (61×65 = 3,965 nodes)
├── 3.2 Data Sources
│   ├── 3.2.1 CHIRPS 2.0 Precipitation Data
│   ├── 3.2.2 SRTM Digital Elevation Model
│   └── 3.2.3 Data Quality and Limitations
├── 3.3 Data Preprocessing Pipeline
│   ├── 3.3.1 Quality Control
│   ├── 3.3.2 Feature Engineering
│   │   ├── BASIC: Raw precipitation + temporal features
│   │   ├── KCE: K-means Cluster Elevation features
│   │   └── PAFC: Precipitation Autocorrelation Features
│   ├── 3.3.3 Windowing Strategy (W=12)
│   └── 3.3.4 Normalization Methods
├── 3.4 Model Architectures
│   ├── 3.4.1 V1: Baseline Models (RNN, GRU, LSTM)
│   ├── 3.4.2 V2: Enhanced ConvLSTM with Attention
│   ├── 3.4.3 V3: Fourier Neural Operators (FNO)
│   └── 3.4.4 V4: GNN-TAT (Main Contribution)
│       ├── Graph Construction from DEM
│       ├── GNN Variants (GCN, GAT, SAGE)
│       └── Temporal Attention Mechanism
├── 3.5 Training Protocol
│   ├── 3.5.1 Train/Val/Test Split (70/15/15)
│   ├── 3.5.2 Optimizer Configuration (Adam)
│   ├── 3.5.3 Early Stopping Strategy
│   └── 3.5.4 Reproducibility Details (Seeds, Versions)
├── 3.6 Evaluation Framework
│   ├── 3.6.1 Metrics (RMSE, MAE, R², Bias)
│   ├── 3.6.2 Horizon Analysis (H=1,3,6,12)
│   └── 3.6.3 Statistical Tests (Mann-Whitney U, Cohen's d)
└── 3.7 Computational Environment
    └── 3.7.1 Google Colab A100 GPU

CHAPTER 4: RESULTS (50-60 pages)
├── 4.1 V1-V2 Baseline Results
│   ├── 4.1.1 ConvLSTM Performance
│   ├── 4.1.2 Enhanced ConvLSTM with Attention
│   └── 4.1.3 Best Baseline: R²=0.653 (ConvLSTM_Bidir)
├── 4.2 V3 FNO Results
│   ├── 4.2.1 Architecture Experiments
│   └── 4.2.2 Negative Result: R²=0.312 (Documented)
├── 4.3 V4 GNN-TAT Results (Main Results)
│   ├── 4.3.1 Full-Grid Results (3,965 nodes)
│   ├── 4.3.2 Best Configuration: GAT + BASIC, H=5 (R²=0.628)
│   ├── 4.3.3 Per-Experiment Analysis (BASIC, KCE, PAFC)
│   ├── 4.3.4 Per-Horizon Degradation (9.6% at H=12)
│   └── 4.3.5 Spatial Prediction Maps
├── 4.4 Cross-Version Comparison
│   ├── 4.4.1 V2 vs V4 Statistical Tests
│   ├── 4.4.2 Mean RMSE: GNN-TAT 92.12mm vs ConvLSTM 112.02mm
│   └── 4.4.3 Mann-Whitney U=57.00, p=0.015
├── 4.5 Ablation Studies
│   ├── 4.5.1 GNN Type Impact (GCN vs GAT vs SAGE)
│   ├── 4.5.2 Feature Set Impact (BASIC vs KCE vs PAFC)
│   └── 4.5.3 Parameter Efficiency (98K vs 2.1M parameters)
└── 4.6 Visualization and Interpretability
    ├── 4.6.1 Graph Edge Weight Analysis
    └── 4.6.2 Attention Mechanism Visualization

CHAPTER 5: DISCUSSION (25-30 pages)
├── 5.1 Interpretation of Results
│   ├── 5.1.1 Why ConvLSTM Achieves Higher Peak R²
│   ├── 5.1.2 GNN-TAT Parameter Efficiency Advantage
│   └── 5.1.3 Interpretability Value
├── 5.2 Hypothesis Validation Summary
│   ├── 5.2.1 H1: Partially Validated (Comparable, not superior)
│   ├── 5.2.2 H2: Validated (Feature engineering helps)
│   ├── 5.2.3 H3: Validated (Graph structure captures spatial)
│   └── 5.2.4 H4: Validated (9.6% degradation < 20%)
├── 5.3 Comparison with State of the Art
│   ├── 5.3.1 vs Hybrid Models in Literature
│   └── 5.3.2 vs Regional Studies
├── 5.4 Limitations and Threats to Validity
│   ├── 5.4.1 Data Limitations (CHIRPS resolution)
│   ├── 5.4.2 Model Limitations (Single-run results)
│   └── 5.4.3 Geographic Generalization
├── 5.5 Practical Implications
│   ├── 5.5.1 Water Resource Management
│   └── 5.5.2 Agricultural Planning
└── 5.6 Future Work
    ├── 5.6.1 V5: ERA5 Multi-Modal Integration
    ├── 5.6.2 V6: Ensemble Meta-Learning
    └── 5.6.3 Operational Deployment

CHAPTER 6: CONCLUSIONS (15-20 pages)
├── 6.1 Summary of Contributions
├── 6.2 Fulfillment of Specific Objectives
│   ├── SO1: Data pipeline - COMPLETED
│   ├── SO2: Architecture benchmark - COMPLETED
│   ├── SO3: Feature engineering quantification - COMPLETED
│   └── SO4: Statistical validation - COMPLETED
├── 6.3 Answers to Research Questions
├── 6.4 Theoretical and Practical Contributions
├── 6.5 Recommendations
└── 6.6 Final Remarks

BACK MATTER
├── References (100-150 entries, Q1/Q2 journals)
├── Appendix A: Complete Results Tables
├── Appendix B: Hyperparameter Configurations
├── Appendix C: Statistical Test Details
├── Appendix D: Code Repository Documentation
└── Appendix E: Glossary of Terms
```

---

## 5. Reproducibility Standards

### 5.1 Random Seeds

```python
# Required seed documentation
RANDOM_SEEDS = {
    "pytorch": 42,
    "numpy": 42,
    "python": 42,
    "cuda": 42
}

# Implementation
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

### 5.2 Hyperparameter Documentation

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-3 | Constant (no scheduling) |
| Optimizer | Adam | β₁=0.9, β₂=0.999, ε=10⁻⁸ |
| Batch Size | 2 | Limited by GPU memory |
| Epochs | 150 | With early stopping |
| Patience | 50 | On validation MAE |
| Window Size | 12 | 12 months lookback |
| Horizons | 1, 3, 6, 12 | Months ahead |

### 5.3 Data Split

| Split | Period | Percentage |
|-------|--------|------------|
| Training | 1981-2010 | 70% |
| Validation | 2011-2017 | 15% |
| Testing | 2018-2024 | 15% |

**Note:** No shuffling applied to preserve temporal order.

---

## 6. Data Analysis and Feature Engineering Details

### 6.1 Dataset Overview

#### CHIRPS-2.0 Precipitation Data
| Attribute | Value |
|-----------|-------|
| Source | Climate Hazards Group InfraRed Precipitation with Station data |
| Version | 2.0 |
| Temporal Span | January 1981 - June 2024 (530 months) |
| Spatial Resolution | 0.05° (~5.5 km) |
| Grid Dimensions | 61 latitudes × 65 longitudes |
| Total Grid Cells | 3,965 unique coordinates |
| Geographic Extent | 74.93°W to 71.73°W, 4.375°N to 7.375°N |
| East-West Distance | ~354 km |
| North-South Distance | ~333.58 km |

#### SRTM Digital Elevation Model
| Attribute | Value |
|-----------|-------|
| Source | Shuttle Radar Topography Mission |
| Resolution | 30m (resampled to CHIRPS grid) |
| Elevation Range | 58 - 4,728 m |
| Mean Elevation | 1,286 m (SD: 1,139 m) |

### 6.2 Precipitation Statistics

#### Monthly Averages
| Month | Range (mm) | Characteristics |
|-------|------------|-----------------|
| Jan-Mar | 90-185 | Dry season transition |
| Apr-May | 270+ | First rainy season peak |
| Jun-Aug | 100-200 | Mid-year moderate |
| Sep-Nov | 250+ | Second rainy season peak |
| Dec | ~150 | Transition period |

#### Global Statistics
| Statistic | Value |
|-----------|-------|
| Mean Monthly Precipitation | 185.88 mm |
| Standard Deviation | 111.51 mm |
| Maximum Daily | 116.11 mm |
| Minimum Daily | 0.13 mm |
| Maximum Monthly Total | 1,265.90 mm |

### 6.3 K-Means Clustering Analysis

#### Elevation Clustering Configuration
| Parameter | Value |
|-----------|-------|
| Algorithm | K-Means |
| Number of Clusters | 3 (Low, Medium, High) |
| Evaluation Method | Elbow method + Silhouette score |
| K Range Tested | 2-10 |
| Random State | 42 |
| n_init | auto |

#### Elevation Categories
| Category | Min (m) | Max (m) | Mean (m) | SD (m) | Label |
|----------|---------|---------|----------|--------|-------|
| Level 1 (Low) | 58 | 1,129 | 400 | 289 | elev_low |
| Level 2 (Mid) | 1,120 | 2,293 | 1,692 | 343 | elev_med |
| Level 3 (High) | 2,274 | 4,728 | 3,459 | 625 | elev_high |

#### Cross-Cluster Statistical Analysis
| Test | Value | Interpretation |
|------|-------|----------------|
| Chi-square statistic | 1,816.278 | |
| Degrees of freedom | 6 | |
| P-value | < 0.0001 | Highly significant |
| Conclusion | Strong association between elevation and precipitation clusters |

#### Contingency Table (Precipitation × Elevation)
| Elevation | Precip Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|-----------|-----------------|-----------|-----------|-----------|
| Low (0) | 646 (31.45%) | 924 (44.99%) | 171 | 313 |
| Mid (1) | 106 | 33 | 852 (85.63%) | 4 |
| High (2) | 223 | 268 | 369 (40.28%) | 56 |

### 6.4 Temporal Analysis (PACF and Seasonality)

#### PACF Analysis Results
| Lag | Significance | Implication |
|-----|--------------|-------------|
| Lag 1 | Significant | Month-to-month persistence |
| Lag 2 | Significant | Two-month memory |
| Lag 12 | Significant | Annual cycle |

#### Seasonal Decomposition
| Component | Method | Period |
|-----------|--------|--------|
| Trend | Additive | - |
| Seasonal | Additive | 12 months |
| Residual | Additive | - |

#### Monthly Correlation (Precipitation vs. Elevation)
- Range: -0.15 to 0.45
- Highest positive correlation: Dry season months
- Pattern: Variable seasonal response by elevation level

### 6.5 Feature Engineering Pipeline

#### Feature Set Definitions

**BASIC (12 features):**
| Feature | Type | Description |
|---------|------|-------------|
| year | Temporal | Calendar year |
| month | Temporal | Month (1-12) |
| month_sin | Temporal | sin(2π × month/12) |
| month_cos | Temporal | cos(2π × month/12) |
| doy_sin | Temporal | sin(2π × day_of_year/365) |
| doy_cos | Temporal | cos(2π × day_of_year/365) |
| max_daily_precipitation | Precipitation | Maximum daily value in month |
| min_daily_precipitation | Precipitation | Minimum daily value in month |
| daily_precipitation_std | Precipitation | Daily standard deviation |
| elevation | Topographic | SRTM elevation (m) |
| slope | Topographic | Terrain slope (degrees) |
| aspect | Topographic | Terrain aspect (degrees) |

**KCE (15 features) = BASIC + Elevation Clusters:**
| Feature | Type | Description |
|---------|------|-------------|
| elev_high | Cluster | High elevation indicator (one-hot) |
| elev_med | Cluster | Medium elevation indicator (one-hot) |
| elev_low | Cluster | Low elevation indicator (one-hot) |

**PAFC (18 features) = KCE + Autocorrelation Lags:**
| Feature | Type | Description |
|---------|------|-------------|
| lag1 | Autocorrelation | Precipitation at t-1 month |
| lag2 | Autocorrelation | Precipitation at t-2 months |
| lag12 | Autocorrelation | Precipitation at t-12 months (annual) |

### 6.6 Preprocessing Techniques

#### EMD/CEEMDAN Decomposition
| Parameter | Value |
|-----------|-------|
| Method | Complete Ensemble EMD with Adaptive Noise |
| Processing Chunks | 4 latitude bands |
| Processing Time per Chunk | ~55-58 minutes |
| Max IMFs | 8 modes per signal |

#### Wavelet Analysis
| Parameter | Value |
|-----------|-------|
| Wavelet Type | Daubechies 4 (db4) |
| Decomposition Level | 4 |
| Energy Classification | Alta/Media/Baja based on energy distribution |

#### Frequency Classification Criteria
| Class | Criterion |
|-------|-----------|
| Alta (High) | L1 energy dominant |
| Media (Medium) | L2 energy > L3 and Approx |
| Baja (Low) | Approximate energy dominant |

### 6.7 Data Windowing Configuration

| Parameter | Value |
|-----------|-------|
| Input Window (T_in) | 60 months |
| Prediction Horizons (H) | 1, 3, 6, 12 months |
| Stride | 1 month |
| Input Shape | (60, 61, 65, F) where F = features |
| Output Shape | (H, 61, 65, 1) |
| Train Windows (H=12) | 343 |
| Validation Windows (H=12) | 33 |

### 6.8 Normalization

| Method | Application |
|--------|-------------|
| StandardScaler | Continuous features |
| Fit Data | 80% training set |
| Transform | Train, validation, test |
| Binary Features | Cluster masks (no scaling) |
| Float Precision | float32 |

---

### 5.4 Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.10.12 |
| PyTorch | 2.1.0 |
| PyTorch Geometric | 2.4.0 |
| TensorFlow | 2.15.0 |
| NumPy | 1.24.3 |
| Pandas | 2.0.3 |

---

## 6. Statistical Rigor Requirements

### 6.1 Required Statistical Tests

| Test | Purpose | When to Use |
|------|---------|-------------|
| Mann-Whitney U | Compare GNN-TAT vs ConvLSTM | Main comparison |
| Wilcoxon signed-rank | Paired model comparison | Within-architecture |
| Cohen's d | Effect size | Report with p-values |
| Bootstrap CI | Confidence intervals | Per-metric uncertainty |

### 6.2 Reporting Standards

- Report mean ± standard deviation for all metrics
- Include 95% confidence intervals where applicable
- Report effect sizes alongside p-values
- Use Bonferroni correction for multiple comparisons
- Minimum 3 runs per configuration (ideally 30+)

### 6.3 Current Statistical Results

```
Mann-Whitney U Test (GNN-TAT vs ConvLSTM):
- U statistic: 57.00
- p-value: 0.015 (significant at α=0.05)
- Effect size (Cohen's d): 1.03 (large)

Mean RMSE Comparison:
- GNN-TAT: 92.12 mm (SD=6.48)
- ConvLSTM: 112.02 mm (SD=27.16)
- Reduction: 17.8%
```

---

## 7. Model Versioning and Architecture Details

### 7.1 Version Summary

| Version | Architecture | Purpose | Status | Best R² | Parameters |
|---------|--------------|---------|--------|---------|------------|
| V1 | ConvLSTM/GRU/RNN | Baseline | Complete | 0.642 | 16K-79K |
| V2 | Enhanced + Attention | Improved baseline | Complete | 0.653 | 79K-41.8M |
| V3 | FNO | Physics-informed | Complete | 0.312 | 4.6K-106K |
| V4 | GNN-TAT | Hybrid spatial-temporal | **Current** | 0.628 | 97K-106K |
| V5 | Multi-Modal | ERA5 + Satellite | Planned | TBD | TBD |
| V6 | Ensemble | Meta-learning | Planned | TBD | TBD |

### 7.2 V1 Baseline Models (ConvLSTM/ConvGRU/ConvRNN)

#### Configuration
| Parameter | Value |
|-----------|-------|
| Input Window | 60 months |
| Prediction Horizon | 3 months |
| Spatial Grid | 61 × 65 (3,965 cells) |
| Batch Size | 8 |
| Epochs | 150 |
| Early Stopping Patience | 80 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| Loss Function | MSE |

#### Model Architectures

**ConvLSTM (78,732 parameters):**
```
Input: (batch, 60, 61, 65, n_feats)
→ ConvLSTM2D: 32 filters, (3×3) kernel, padding='same', return_sequences=True
→ ConvLSTM2D: 16 filters, (3×3) kernel, padding='same', return_sequences=False
→ Conv2D: 3 filters (1×1), linear activation
→ Output: (batch, 3, 61, 65, 1)
```

**ConvGRU (Custom Implementation):**
- Update gate (z): sigmoid activation
- Reset gate (r): sigmoid activation
- Candidate hidden state (h): tanh activation
- Kernel initialization: Glorot uniform (input), Orthogonal (recurrent)

**ConvRNN:**
- TimeDistributed(Conv2D): 32 filters → 16 filters
- SimpleRNN: 128 units, tanh activation
- Dense: horizon × 61 × 65

### 7.3 V2 Enhanced Models with Attention

#### Configuration
| Parameter | Value |
|-----------|-------|
| Input Window | 60 months |
| Prediction Horizon | 3, 6, 12 months |
| Batch Size | 4 (reduced for memory) |
| Epochs | 120 |
| Early Stopping Patience | 100 |
| Learning Rate | 1e-3 |
| L2 Weight Decay | 1e-5 |
| Dropout | 0.2 |

#### Model Variants

| Model | Parameters | Key Features |
|-------|------------|--------------|
| ConvLSTM_Attention | 315,941 | CBAM attention, channel attention |
| ConvLSTM_Enhanced | 78,732 | Residual connections, normalization |
| ConvLSTM_Residual | Variable | Skip connections between layers |
| ConvLSTM_Bidirectional | Variable | Forward + backward temporal encoding |
| ConvGRU_Residual | 240,005 | GRU with residual blocks |
| Transformer_Baseline | ~41.8M | Multi-head self-attention, positional encoding |

### 7.4 V3 Fourier Neural Operator (FNO) Models

#### Configuration
| Parameter | Value |
|-----------|-------|
| Input Window | 60 months |
| Prediction Horizon | 12 months |
| Batch Size | 2 (memory efficiency) |
| Epochs | 80 |
| Early Stopping Patience | 30 |
| Training Batch Size | 1 (gradient accumulation) |

#### Model Architectures

**FNO_Pure (4,612 parameters):**
- Fourier space: FFT → Spectral Conv → IFFT
- Global receptive field
- Extremely lightweight

**FNO_ConvLSTM_Hybrid (106,292 parameters):**
- FNO backbone for spatial modeling
- ConvLSTM for temporal dynamics
- Dense decoder for output projection

### 7.5 V4 GNN-TAT Architecture (Main Contribution)

#### Global Configuration
| Parameter | Value |
|-----------|-------|
| Input Window | 60 months |
| Prediction Horizon | 12 months |
| Grid Dimensions | 61 × 65 |
| Total Nodes | 3,965 |
| Total Edges | ~500,000 |
| Batch Size | 2 |
| Epochs | 150 |
| Early Stopping Patience | 50 |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-5 |

#### Graph Construction Parameters
| Edge Type | Scale | Weight |
|-----------|-------|--------|
| Distance-based | 10.0 km | 0.3 |
| Elevation-based | 0.2 | 0.3 |
| Correlation-based | threshold=0.3 | 0.5 |
| Max neighbors per node | 8 | - |
| Min edge weight | 0.01 | - |

#### GNN Variants

| Model | Parameters (BASIC) | Best Val Loss | Best Epoch |
|-------|-------------------|---------------|------------|
| GNN_TAT_GAT | 97,932 | 0.4553 | 3 |
| GNN_TAT_GCN | 97,676 | 0.4627 (PAFC) | 3 |
| GNN_TAT_SAGE | 105,868 | 0.5603 (PAFC) | 4 |

#### Temporal Attention Transformer
| Component | Value |
|-----------|-------|
| Hidden Dimension | 64 |
| GNN Layers | 2 |
| Attention Heads | 4 |
| Temporal Attention Heads | 4 |
| LSTM Layers | 2 |
| Dropout | 0.1 |

---

## 8. Evaluation Metrics

### 8.1 Primary Metrics

| Metric | Formula | Optimal |
|--------|---------|---------|
| R² | 1 - SS_res/SS_tot | 1.0 |
| RMSE | sqrt(mean((y-ŷ)²)) | 0 |
| MAE | mean(\|y-ŷ\|) | 0 |
| Bias | mean(ŷ-y) | 0 |

### 8.2 Horizon-Specific Evaluation

Evaluate all metrics at horizons: H=1, H=3, H=6, H=12 months.

### 8.3 Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|----------|
| R² | ≥ 0.60 | ✅ 0.628 |
| Horizon degradation | < 20% | ✅ 9.6% |
| Statistical significance | p < 0.05 | ✅ p=0.015 |
| Parameter efficiency | 10x reduction | ✅ 95% fewer |

---

## 9. Publication Plan

### 9.1 Required Publications (UPTC Doctorate)

| # | Type | Target Venue | Status | Title |
|---|------|--------------|--------|-------|
| 1 | Review | Q1 Journal | Under Review | "Spatiotemporal Prediction of Monthly Precipitation: A Systematic Review of Hybrid Models" |
| 2 | Full Paper | MDPI Hydrology (Q2) | Ready | "Hybrid Data-Driven Framework for Multi-Month Precipitation Forecasting" |
| 3 | Full Paper | Q1 Journal | Planned | "GNN-TAT: Graph Neural Networks with Temporal Attention for Precipitation Prediction" |

### 9.2 Target Venues

**Tier 1 (Q1):**
- Journal of Hydrology (IF: 6.4)
- Water Resources Research (IF: 5.4)
- Nature Machine Intelligence (IF: 23.8)

**Tier 2 (Q2):**
- MDPI Hydrology
- Atmospheric Research
- Environmental Modelling & Software

**Conferences:**
- NeurIPS Climate Change AI Workshop
- AGU Fall Meeting
- AMS Annual Meeting

---

## 10. INTERNAL TECHNICAL PRACTICES (NOT FOR THESIS)

This section documents internal development practices that should NOT appear in the thesis document. They are for project management only.

### 10.1 SDD Framework (Internal Use Only)

**IMPORTANT:** The Specification-Driven Development (SDD) framework is a personal methodology for project organization. It MUST NOT be included in the thesis document.

```
SDD Workflow (Internal):
1. DEFINE → spec.md
2. DESIGN → plan.md
3. DEVELOP → notebooks/code
4. DOCUMENT → thesis.tex, paper.tex
5. DELIVER → publications
6. ITERATE → update specs
```

**Visibility:** Project management files only (CLAUDE.md, spec.md, plan.md)

### 10.2 Figure DPI Requirements (Internal Standard)

**IMPORTANT:** The "700 DPI" requirement is an internal quality standard. Do NOT mention DPI values in the thesis.

| Context | DPI | Notes |
|---------|-----|-------|
| Internal standard | 700 | Quality assurance target |
| Journal minimum | 300 | MDPI/Q1 journal requirement |
| Thesis display | N/A | Do NOT mention in thesis text |

**Correct thesis language:** "High-resolution figures suitable for publication"
**Incorrect thesis language:** "Figures at 700 DPI for Q1/Q2 journals"

### 10.3 Data-Driven (DD) Framework (PUBLIC - Include in Thesis)

The DD framework IS appropriate for the thesis as it represents the scientific validation methodology:

```
DD Methodology (Include in thesis):
1. HYPOTHESIS: Define testable research questions
2. EXPERIMENT: Design controlled experiments
3. MEASURE: Collect standardized metrics
4. ANALYZE: Apply statistical validation
5. CONCLUDE: Accept/reject hypotheses
6. DOCUMENT: Record findings
```

**Thesis Guidance:** When describing methodology, use "Data-Driven methodology" with proper academic context and references to support the approach.

---

## 11. Writing Style Guidelines

### 11.1 Language and Voice

| Aspect | Guideline |
|--------|-----------|
| Language | English (academic) |
| Voice | First-person preferred ("I developed...") per APA 7th edition |
| Tense | Past tense for completed work, present for general truths |
| Formality | Avoid contractions, colloquialisms |

### 11.2 Quantification

- Use precise, quantifiable descriptions
- "R² improved by 15%" NOT "significantly improved"
- Include units for all measurements
- Report decimal places consistently (3 for R², 2 for RMSE)

### 11.3 Hedging Language

Use appropriate hedging:
- "suggests" instead of "proves"
- "indicates" instead of "shows"
- "may" instead of "will"

### 11.4 Content Depth Requirements

**CRITICAL:** Every section must have sufficient depth and context. Avoid:
- Short phrases without explanation
- Statements without supporting references
- Technical terms without definitions
- Claims without evidence

**Minimum requirements per section:**
- Introduction of the topic with context
- Supporting references (minimum 2-3 per major claim)
- Connection to previous/next sections
- Clear conclusions or takeaways

**Example - WRONG (too brief):**
```
We use CHIRPS-2.0, a precipitation product at 0.05° resolution.
```

**Example - CORRECT (adequate depth):**
```
The Climate Hazards Group InfraRed Precipitation with Station data
(CHIRPS) version 2.0 was selected as the primary precipitation source
\citep{funk2015chirps}. CHIRPS integrates satellite imagery with
in-situ station data to produce gridded rainfall estimates at 0.05°
(~5km) spatial resolution, spanning 1981 to near-present. This
dataset has been extensively validated in tropical regions,
demonstrating strong agreement with ground observations in
Colombia's Andean region \citep{sun2018review}. The choice of
CHIRPS over alternatives such as GPM-IMERG or PERSIANN was based
on its longer temporal record (enabling robust model training)
and demonstrated performance in mountainous terrain.
```

### 11.5 Reference Requirements

Every major section must include appropriate citations:

| Section Type | Minimum References |
|--------------|-------------------|
| Data source description | 2-3 (original paper + validations) |
| Method description | 3-5 (foundational + recent applications) |
| Claim/assertion | 1-2 (supporting evidence) |
| Comparison with literature | 3-5 (comparable studies) |

**Key data sources requiring citations:**
- CHIRPS 2.0: \citep{funk2015chirps}
- SRTM DEM: \citep{farr2007shuttle}
- ERA5: \citep{hersbach2020era5}

---

## 12. Graphics and Visualization Standards

### 12.1 Figure Quality Standards

| Requirement | Specification |
|-------------|---------------|
| Resolution | Minimum 300 DPI (internal target: 700 DPI) |
| Format | PNG for raster, PDF for vector |
| Font size | Minimum 8pt for axis labels |
| Color scheme | Colorblind-friendly palettes |

### 12.2 Figure Margin Compliance (CRITICAL)

**MANDATORY:** All figures MUST fit within page margins. Never use widths exceeding these limits:

```latex
% SAFE WIDTH SPECIFICATIONS - NEVER EXCEED
\newcommand{\fullwidth}{0.95\textwidth}     % Single full-width
\newcommand{\halfwidth}{0.47\textwidth}     % Two side-by-side
\newcommand{\thirdwidth}{0.31\textwidth}    % Three in row
\newcommand{\quarterwidth}{0.23\textwidth}  % Four in row
```

**Common violations to avoid:**
- TikZ diagrams with unconstrained node positioning
- Subfigures without proper width constraints
- Tables exceeding text width
- Multi-panel figures with inadequate spacing

### 12.3 Diagram Type Selection

Choose the appropriate visualization for each content type:

| Content Type | Recommended Diagram | Tool |
|--------------|---------------------|------|
| **Methodology pipeline** | Flowchart (horizontal) | TikZ, Mermaid |
| **Architecture** | Block diagram with layers | TikZ |
| **Data flow** | Sequence diagram | Mermaid |
| **Comparisons** | Bar charts, heatmaps | Matplotlib, Seaborn |
| **Temporal analysis** | Line plots with confidence | Matplotlib |
| **Statistical tests** | Critical Difference plots | Python (autorank) |
| **Geographic data** | Maps with boundaries | Cartopy, QGIS |

### 12.4 Figure Frequency Guidelines

**Avoid diagram overuse.** Guidelines:

| Chapter Type | Max Figures | Notes |
|--------------|-------------|-------|
| Introduction | 1-2 | Study area map only |
| Literature Review | 0-2 | Only if essential taxonomy |
| Methodology | 3-5 | Pipeline, architecture |
| Results | 8-12 | Key comparisons |
| Discussion | 1-2 | Summary visualizations |

### 12.5 Quality Checklist (Pre-submission)

Before including any figure, verify:
- [ ] Text is readable (no overlapping labels)
- [ ] Fits within page margins
- [ ] Legend is complete and positioned correctly
- [ ] Axes have labels with units
- [ ] Colors are distinguishable
- [ ] Resolution is adequate for print

---

## 13. Schedule and Budget (Thesis Appendices)

The thesis MUST include Schedule and Budget sections as appendices, aligned with the doctoral proposal.

### 13.1 Schedule Requirements

Include a Gantt chart or timeline showing:
- Research phases (V1→V2→V3→V4)
- Publication milestones
- Defense preparation timeline

### 13.2 Budget Summary (Based on Proposal)

**Total Budget:** $66,600,000 COP

| Category | UPTC | Own Resources | Total |
|----------|------|---------------|-------|
| Professional Services | $19,200,000 | - | $19,200,000 |
| Equipment Purchase | - | $6,000,000 | $6,000,000 |
| Software | - | $400,000 | $400,000 |
| Publications | - | $15,000,000 | $15,000,000 |
| Travel & Per Diem | $8,000,000 | $12,000,000 | $20,000,000 |
| Academic Events | $6,000,000 | - | $6,000,000 |

### 13.3 Computational Resources (Actual Expenses)

Detailed breakdown of cloud computing resources used during the research:

| Resource | Period | Monthly Cost | Total Cost |
|----------|--------|--------------|------------|
| Google Colab Pro+ (Account 1) | 24 months | $50,000 COP | $1,200,000 COP |
| Google Colab Pro+ (Account 2) | 3 months | $50,000 COP | $150,000 COP |
| **Colab Pro+ Subtotal** | | | **$1,350,000 COP** |
| Cloud storage (Google Drive) | 24 months | $15,000 COP | $360,000 COP |
| GitHub Pro | 24 months | $12,500 COP | $300,000 COP |
| **Total Computational** | | | **$2,010,000 COP** |

**Note:** Two parallel Colab Pro+ accounts were required during intensive V4 GNN-TAT training (3 months) to enable concurrent experiments with different GNN variants (GAT, GCN, SAGE) and feature sets (BASIC, KCE, PAFC).

### 13.4 GPU Hardware Specifications (Google Colab Pro+)

| Specification | Value |
|---------------|-------|
| GPU Model | NVIDIA A100 (40GB HBM2) |
| CUDA Cores | 6,912 |
| Tensor Cores | 432 (3rd generation) |
| Memory Bandwidth | 1,555 GB/s |
| FP32 Performance | 19.5 TFLOPS |
| TF32 Tensor Core | 156 TFLOPS |
| Typical Session Duration | 8-12 hours |

### 13.5 Additional Costs

| Item | Estimated Cost | Status |
|------|----------------|--------|
| Grammarly Premium | $300,000/year | Optional |
| APC Article 1 (Review) | $5,000,000 | Budgeted |
| APC Article 2 (MDPI) | $10,000,000 | Budgeted |
| APC Article 3 (Q1) | $15,000,000 | NOT budgeted |
| Conference registration | $6,000,000 | Budgeted |
| Additional conference | $4,000,000 | NOT budgeted |

**Note:** Review actual expenses against proposal and update budget section in thesis accordingly.

---

## 14. File Organization

```
docs/thesis/
├── spec.md                 # This specification
├── plan.md                 # Execution plan
├── thesis.tex              # Main LaTeX document
├── references.bib          # Bibliography (100+ entries)
├── figures/                # High-resolution figures (≥300 DPI)
├── tables/                 # Generated tables
└── appendices/             # Supplementary material

models/
├── spec.md                 # Model specifications
├── plan.md                 # Model development plan
└── output/
    ├── V1_Baseline_Models/
    ├── V2_Enhanced_Models/
    ├── V3_FNO_Models/
    └── V4_GNN_TAT_Models/

scripts/benchmark/
├── statistical_significance_tests.py
├── generate_latex_tables.py
└── generate_csv_tables.py
```

---

## 15. Acceptance Criteria

### 15.1 UPTC Requirements

- [ ] Pass candidacy examination
- [ ] Research group endorsement (GALASH)
- [ ] Director acceptance letter
- [ ] Minimum 1 Q1/Q2 publication as first author
- [ ] International presentation (conference)
- [ ] Second language proficiency (English)
- [ ] External internship completion

### 15.2 Technical Success Criteria

- [x] R² ≥ 0.60 for best configuration
- [x] Statistical significance (p < 0.05) in model comparison
- [x] Complete reproducibility documentation
- [x] 4+ model architectures benchmarked
- [x] 120+ references in bibliography (aligned with doctoral proposal: 123 refs)
- [ ] Thesis document ≥ 150 pages

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-01 | Manuel Pérez | Initial specification |
| 2.0.0 | 2026-01 | Manuel Pérez | Aligned with proposal, added V4 results, updated hypotheses |

---

*This specification serves as the authoritative source for the PhD dissertation. All development, writing, and research activities should align with these requirements.*

**Document Owner:** Manuel Ricardo Pérez Reyes
**Approved By:** PhD. Marco Javier Suárez Barón (Pending)
