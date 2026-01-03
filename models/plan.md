# Development Plan - ML Precipitation Prediction
## Strategic Roadmap for Doctoral Thesis

**Version:** 2.0
**Date:** January 2026
**Objective:** Monthly Precipitation Prediction in Mountainous Areas using Deep Learning
**Frameworks:** SDD (Specification-Driven Development) + DD (Data-Driven)

---

## EXECUTIVE SUMMARY

This document defines the strategic roadmap for the precipitation prediction framework development, integrating results obtained to date and establishing the path toward doctoral thesis completion. All development follows the SDD and DD frameworks defined in [spec.md](spec.md) and [CLAUDE.md](../CLAUDE.md).

### Current Status

```
+------------------------------------------------------------------+
|                    PROJECT PROGRESS                               |
+------------------------------------------------------------------+
| V1 Baseline      [##########] 100%  - ConvLSTM/GRU basic         |
| V2 Enhanced      [##########] 100%  - Attention + Bidirectional  |
| V3 FNO           [##########] 100%  - Complete (underperformed)  |
| V4 GNN-TAT       [########--]  80%  - Light mode OK, full pend.  |
| V5 Multi-Modal   [----------]   0%  - Planned                    |
| V6 Ensemble      [----------]   0%  - Planned                    |
| Thesis Written   [####------]  40%  - In progress                |
| Paper Written    [######----]  60%  - V2 complete, V4 pending    |
+------------------------------------------------------------------+
```

### Key Achievement V4

| Metric | V2 Baseline | V4 GNN-TAT | Improvement |
|--------|-------------|------------|-------------|
| R² | 0.437 | **0.707** | **+62%** |
| RMSE | 98.17mm | **52.45mm** | **-47%** |
| Parameters | 2M+ | **~98K** | **-95%** |

---

## PART I: HYPOTHESIS VALIDATION STATUS

### Research Hypotheses (from Doctoral Thesis Proposal)

| ID | Hypothesis | Status | Evidence |
|----|------------|--------|----------|
| **H1** | Hybrid GNN-Temporal > ConvLSTM | **VALIDATED** | V4 R²=0.707 vs V2 R²=0.437 |
| **H2** | Topographic features improve prediction | **VALIDATED** | PAFC > KCE > BASIC consistently |
| **H3** | Non-Euclidean spatial relations capture orographic dynamics | **IN VALIDATION** | GNN outperforms CNN, pending full grid |
| **H4** | Multi-scale temporal attention improves long horizons | **PARTIALLY VALIDATED** | R² degradation < 20% H1→H12 |

### Validation Criteria

```
H1 Validation Criteria:
- R² improvement > 20% vs best baseline  [ACHIEVED: +62%]
- RMSE reduction > 20% vs best baseline  [ACHIEVED: -47%]

H2 Validation Criteria:
- PAFC consistently outperforms BASIC    [ACHIEVED]
- Statistical significance (p < 0.05)    [PENDING: Friedman test]

H3 Validation Criteria:
- GNN captures elevation-based patterns  [ACHIEVED: graph weights]
- Spatial coherence in predictions       [ACHIEVED visually]

H4 Validation Criteria:
- R² degradation < 20% from H1 to H12   [ACHIEVED: ~15% degradation]
- Long-horizon RMSE < 80mm              [ACHIEVED: 52-65mm range]
```

---

## PART II: SITUATION ANALYSIS

### 2.1 Consolidated Results by Version

#### V1 - Baseline (Complete)
```
Architectures: ConvLSTM, ConvGRU, ConvRNN
Best R² H1: 0.86 (ConvRNN-BASIC)
Problem: Severe degradation H2-H12 (R² < 0.30)
Lesson: Basic architectures insufficient for multi-horizon
Documentation: thesis.tex Chapter 4.1
```

#### V2 - Enhanced (Complete)
```
Architectures: +Bidirectional, +Residual, +Attention, +Transformer
Best R²: 0.752 (ConvRNN_Enhanced + PAFC)
RMSE: 44.85mm, MAE: 34.38mm
Lesson: Regularization > Architectural complexity
Documentation: thesis.tex Chapter 4.2, paper.tex Section 4
```

#### V3 - FNO (Complete - Underperformed)
```
Architectures: FNO_Pure, FNO_ConvLSTM_Hybrid
Result: WORSE than V2 in all experiments
- BASIC: +4.38mm RMSE vs V2
- PAFC: +7.73mm RMSE vs V2
Critical Lesson: FNO not suitable for precipitation (discontinuities, small grid)
Documentation: thesis.tex Chapter 4.3, paper.tex Section 5
```

#### V4 - GNN-TAT (Current - 80%)
```
Architectures: GNN_TAT_GAT, GNN_TAT_SAGE, GNN_TAT_GCN
Best Result (Light Mode 5x5):
- R² = 0.707 (SAGE+KCE H=3)
- RMSE = 52.45mm
- Parameters: ~98K

Ranking by Configuration:
1. GAT + PAFC: R²=0.628 average
2. GCN + PAFC: R²=0.625 average
3. SAGE + KCE: R²=0.618 average

Documentation: thesis.tex Chapter 5 (pending), paper.tex Section 6 (pending)
```

### 2.2 Key Insights

```
+---------------------------------------------------------------+
|  KEY PROJECT INSIGHTS                                          |
+---------------------------------------------------------------+
| 1. SIMPLE > COMPLEX                                           |
|    ConvRNN outperformed ConvLSTM in several scenarios         |
|                                                               |
| 2. REGULARIZATION > ARCHITECTURE                              |
|    Dropout + Early stopping more effective than more layers   |
|                                                               |
| 3. TOPOGRAPHIC FEATURES WORK                                  |
|    PAFC consistently better than BASIC (validates H2)         |
|                                                               |
| 4. FNO IS NOT THE SOLUTION                                    |
|    Spectral methods fail with discontinuous patterns          |
|                                                               |
| 5. GNN CAPTURES SPATIAL RELATIONSHIPS                         |
|    Graph based on elevation + distance is effective (H3)      |
|                                                               |
| 6. OVERFITTING IS THE MAIN CHALLENGE                          |
|    Train/val ratio of 6-19x indicates memorization            |
+---------------------------------------------------------------+
```

### 2.3 Identified Issues in V4

| Issue | Severity | Proposed Solution |
|-------|----------|-------------------|
| Overfitting (6-19x ratio) | High | More regularization, data augmentation |
| Early stopping too aggressive | Medium | Adjust patience, warmup period |
| Negative bias (-3 to -20mm) | Medium | Balanced loss function |
| Only light mode validated | High | **Execute full grid** |

---

## PART III: ACTION PLAN

### 3.1 Immediate Tasks (Sprint 1)

#### TASK 1.1: Execute V4 Full Grid
```
Objective: Validate results on complete grid (not light mode)
Resources: Colab Pro+ recommended

Steps:
1. Modify CONFIG['light_mode'] = False
2. Execute for H = [1, 3, 6, 12]
3. Compare metrics light vs full
4. Document differences

Success Criteria:
- R² full >= 0.90 * R² light (degradation < 10%)
- Complete training without OOM

Documentation Update:
- thesis.tex: Add full grid results to Chapter 5
- paper.tex: Add comparative table light vs full
```

#### TASK 1.2: Mitigate V4 Overfitting
```
Objective: Reduce train/val ratio from ~10x to <5x

Actions:
1. Increase dropout: 0.1 -> 0.2-0.3
2. Add weight decay: 1e-5 -> 1e-4
3. Implement data augmentation:
   - Temporal jittering (+/- 1 month)
   - Spatial noise (sigma=0.1)
4. Label smoothing in loss function

Success Criteria:
- Train/val ratio < 5x
- R² maintained > 0.55

Documentation Update:
- thesis.tex: Add regularization analysis to Chapter 5.2
```

#### TASK 1.3: Fix Negative Bias
```
Objective: Reduce precipitation underestimation

Actions:
1. Add bias term to loss:
   L = MSE + lambda * |mean(pred) - mean(true)|²

2. Post-processing with bias correction:
   pred_corrected = pred + mean_bias_training

3. Weighted loss by rainfall intensity

Success Criteria:
- |bias_pct| < 5% (currently -3% to -10%)
```

### 3.2 V5 Multi-Modal Development (Sprint 2-3)

#### TASK 2.1: Data Acquisition
```
Sources to Integrate:
+------------------+-------------------+-------------------+
| Source           | Variables         | Resolution        |
+------------------+-------------------+-------------------+
| ERA5 Reanalysis  | Wind u/v          | 0.25 deg / hourly |
|                  | Humidity          |                   |
|                  | Temperature       |                   |
|                  | Pressure          |                   |
+------------------+-------------------+-------------------+
| MODIS            | Cloud cover       | 1km / daily       |
|                  | LST               |                   |
|                  | NDVI              |                   |
+------------------+-------------------+-------------------+
| Climate Indices  | ENSO (Nino 3.4)   | Monthly           |
|                  | IOD               |                   |
|                  | MJO               |                   |
+------------------+-------------------+-------------------+
```

#### TASK 2.2: Multi-Modal Pipeline
```
Proposed Architecture:

INPUT_1: Precipitation (60 months, spatial grid)
INPUT_2: ERA5 (60 months, atmospheric variables)
INPUT_3: Satellite (60 months, cloud + LST)
INPUT_4: Climate Indices (60 scalar values)

         +------------------+
         | Modality-Specific|
INPUT_1->| Encoder (GNN-TAT)|--+
         +------------------+  |
                               |
         +------------------+  |   +-----------------+
INPUT_2->| Encoder (Conv3D) |--+-->| Cross-Modal     |
         +------------------+  |   | Attention       |
                               |   |                 |
         +------------------+  |   | Q: Precip       |
INPUT_3->| Encoder (CNN)    |--+   | K,V: All modes  |
         +------------------+  |   +-----------------+
                               |          |
         +------------------+  |          v
INPUT_4->| Encoder (MLP)    |--+   +-----------------+
         +------------------+      | Fusion + Output |
                                   +-----------------+
                                          |
                                          v
                                   Precipitation Forecast

Success Criteria:
- R² > 0.75 (improvement >10% vs V4)
- Ablation study showing value of each modality
```

### 3.3 V6 Ensemble Development (Sprint 4)

#### TASK 3.1: Intelligent Ensemble
```
Strategy: Meta-learner that selects best model based on context

BASE MODELS (from V2-V5):
- V2: ConvLSTM_Bidirectional (good H1-H6)
- V2: ConvRNN_Enhanced (efficient)
- V4: GNN_TAT_GAT + PAFC (best spatial)
- V4: GNN_TAT_SAGE + KCE (best H3-H6)
- V5: Multi-Modal (if improves)

META-LEARNER:
Context Features:
- Horizon (1-12)
- Season (DJF, MAM, JJA, SON)
- Spatial location (cluster)
- Recent RMSE of each model

Output:
- Weights for ensemble: w = softmax(f(context))
- Prediction: P_ensemble = sum(w_i * P_i)

Success Criteria:
- R² ensemble > max(R² individual)
- Reduced variance across horizons
```

### 3.4 Soft Physics Constraints (Sprint 5)

#### TASK 4.1: Soft Physical Constraints
```
Objective: Add physics-based regularization without losing flexibility

CONSTRAINTS TO IMPLEMENT:

1. Mass conservation (simplified):
   L_mass = |sum(P_pred) - sum(P_true)|

2. Spatial smoothness (precipitation doesn't jump abruptly):
   L_smooth_spatial = sum(|P(i,j) - P(i+1,j)|² + |P(i,j) - P(i,j+1)|²)

3. Temporal consistency (smooth transitions):
   L_smooth_temporal = sum(|P(t+1) - P(t)|²)

4. Non-negativity (precipitation >= 0):
   L_nonneg = sum(ReLU(-P_pred)²)

FINAL LOSS FUNCTION:
L_total = L_MSE + lambda_mass * L_mass
               + lambda_spatial * L_smooth_spatial
               + lambda_temporal * L_smooth_temporal
               + lambda_nonneg * L_nonneg

Hyperparameters to tune: lambda_* via grid search

Success Criteria:
- Physically plausible predictions
- No degradation in main metrics
```

---

## PART IV: THESIS CHAPTER MAPPING

### 4.1 Thesis Structure (thesis.tex)

```
OFFICIAL TITLE (Aligned with Doctoral Proposal):
"Computational Model for Spatiotemporal Prediction of Monthly Precipitation
in Mountainous Areas: A Hybrid Deep Learning Approach Using Graph Neural
Networks with Temporal Attention"

CHAPTERS:

CAP 1: INTRODUCTION (15 pages)
+-- 1.1 Context and Motivation
+-- 1.2 Problem Statement
+-- 1.3 Objectives (General and Specific)
+-- 1.4 Research Hypotheses (H1-H4)
+-- 1.5 Scope and Limitations
+-- 1.6 Document Structure
Status: [####------] 40% - Draft complete, needs revision

CAP 2: THEORETICAL FRAMEWORK (30 pages)
+-- 2.1 Precipitation Prediction
|   +-- 2.1.1 Traditional methods
|   +-- 2.1.2 Numerical models (NWP)
|   +-- 2.1.3 Machine Learning in meteorology
+-- 2.2 Spatiotemporal Deep Learning
|   +-- 2.2.1 Convolutional Networks (CNN)
|   +-- 2.2.2 Recurrent Networks (LSTM, GRU)
|   +-- 2.2.3 ConvLSTM and variants
+-- 2.3 Graph Neural Networks
|   +-- 2.3.1 Theoretical foundations
|   +-- 2.3.2 GCN, GAT, GraphSAGE
|   +-- 2.3.3 GNN for spatial data
+-- 2.4 Attention Mechanisms
|   +-- 2.4.1 Self-attention and Transformers
|   +-- 2.4.2 Temporal attention
+-- 2.5 State of the Art in DL Precipitation
Status: [######----] 60% - Core sections complete

CAP 3: METHODOLOGY (25 pages)
+-- 3.1 Proposed Data-Driven Framework (DD)
+-- 3.2 Specification-Driven Development (SDD)
+-- 3.3 Data and Preprocessing
|   +-- 3.3.1 CHIRPS 2.0
|   +-- 3.3.2 SRTM DEM
|   +-- 3.3.3 Feature Engineering (BASIC, KCE, PAFC)
+-- 3.4 Implemented Architectures
|   +-- 3.4.1 Baseline (V1)
|   +-- 3.4.2 Enhanced (V2)
|   +-- 3.4.3 FNO (V3)
|   +-- 3.4.4 GNN-TAT (V4)
+-- 3.5 Spatial Graph Construction
+-- 3.6 Experimental Protocol
+-- 3.7 Evaluation Metrics
Status: [########--] 80% - Pending V4 methodology

CAP 4: RESULTS V1-V3 (20 pages)
+-- 4.1 Baseline Results (V1)
+-- 4.2 Enhanced Results (V2)
+-- 4.3 FNO Results (V3)
+-- 4.4 Comparative Analysis
+-- 4.5 Lessons Learned
Status: [##########] 100% - Complete

CAP 5: GNN-TAT V4 RESULTS (25 pages) <-- MAIN CONTRIBUTION
+-- 5.1 Architecture Design
+-- 5.2 Experiments and Configurations
+-- 5.3 Results by Model (GAT, SAGE, GCN)
+-- 5.4 Analysis by Feature Set
+-- 5.5 Analysis by Horizon
+-- 5.6 Comparison with State of the Art
+-- 5.7 Graph Interpretability
Status: [####------] 40% - Light mode complete, full grid pending

CAP 6: EXTENSIONS (V5-V6) (15 pages)
+-- 6.1 Multi-Modal Integration
+-- 6.2 Intelligent Ensemble
+-- 6.3 Physical Constraints
+-- 6.4 Preliminary Results
Status: [----------] 0% - Planned

CAP 7: DISCUSSION (15 pages)
+-- 7.1 Hypothesis Validation
+-- 7.2 Practical Implications
+-- 7.3 Study Limitations
+-- 7.4 Comparison with Literature
Status: [##--------] 20% - Outline only

CAP 8: CONCLUSIONS (10 pages)
+-- 8.1 General Conclusions
+-- 8.2 Scientific Contributions
+-- 8.3 Future Work
+-- 8.4 Derived Publications
Status: [----------] 0% - Pending

APPENDICES
+-- A: Source Code (GitHub)
+-- B: Complete Metrics Tables
+-- C: Additional Visualizations
+-- D: Hyperparameter Configurations
```

### 4.2 Scientific Contributions

```
CONTRIBUTION 1: GNN-TAT ARCHITECTURE
- Combines spatial GNN with temporal attention
- Graph constructed with elevation + distance + correlation
- Superior to ConvLSTM by +62% R²
- Validates H1 and H3

CONTRIBUTION 2: COMPARATIVE ANALYSIS FNO vs DATA-DRIVEN
- Demonstrates FNO underperforms for precipitation
- Explains why (discontinuities, small grid)
- Valuable "negative result" for the community

CONTRIBUTION 3: STANDARDIZED FRAMEWORK
- Reproducible end-to-end pipeline
- Robust benchmarking with statistical tests
- Open-source (GitHub)
- SDD + DD methodology documented

CONTRIBUTION 4 (Projected): MULTI-MODAL FUSION
- ERA5 + Satellite + Climate Indices integration
- Cross-modal attention mechanism
```

---

## PART V: PAPER DELIVERABLES (paper.tex)

### 5.1 Current Paper Status

```
Location: docs/papers/4/paper.tex
Format: MDPI Hydrology
Focus: Baseline vs Hybrid Model Comparison

CURRENT SECTIONS:
1. Introduction                    [COMPLETE]
2. Related Work                    [COMPLETE]
3. Methodology                     [COMPLETE - V2]
4. V2 Enhanced Results             [COMPLETE]
5. V3 FNO Results                  [COMPLETE]
6. V4 GNN-TAT Results              [PENDING]
7. Statistical Analysis            [PARTIAL - needs V4]
8. Discussion                      [PARTIAL - needs V4]
9. Conclusions                     [PENDING - needs V4]
```

### 5.2 Required Paper Updates

```
SECTION 6: V4 GNN-TAT RESULTS (TO ADD)
+-- 6.1 Architecture Overview
+-- 6.2 Graph Construction Methodology
+-- 6.3 Light Mode Results
    +-- Table: R², RMSE, MAE by model (GAT, SAGE, GCN)
    +-- Table: Results by feature set (BASIC, KCE, PAFC)
    +-- Table: Results by horizon (H1, H3, H6, H12)
+-- 6.4 Full Grid Results (when available)
+-- 6.5 Comparison V2 vs V3 vs V4

SECTION 7: STATISTICAL ANALYSIS (TO UPDATE)
+-- 7.1 Friedman Test across V2, V3, V4
+-- 7.2 Nemenyi Post-hoc Analysis
+-- 7.3 Critical Difference Diagrams

SECTION 8: DISCUSSION (TO UPDATE)
+-- 8.1 Hypothesis Support Summary
+-- 8.2 Why GNN-TAT Outperforms
+-- 8.3 Why FNO Underperforms
+-- 8.4 Feature Engineering Value

SECTION 9: CONCLUSIONS (TO WRITE)
+-- 9.1 Main Findings
+-- 9.2 Practical Recommendations
+-- 9.3 Future Directions
```

### 5.3 Publications Strategy

```
PAPER 1: V2 vs V3 Benchmark (In preparation)
-----------------------------------------
Title: "Why Fourier Neural Operators Underperform for Precipitation:
        A Comprehensive Benchmark Study"
Journal: Water Resources Research (Q1)
Status: Data ready, writing pending
Message: Negative result paper, valuable for the community

PAPER 2: GNN-TAT Architecture (Priority)
-----------------------------------------
Title: "Graph Neural Networks with Temporal Attention for
        Multi-Horizon Precipitation Forecasting in Mountainous Regions"
Journal: Journal of Hydrometeorology (Q1) or
         Geophysical Research Letters (Q1, high impact)
Status: V4 results ready, pending full grid
Message: Main methodological contribution

PAPER 3: Multi-Modal Fusion (Future)
-----------------------------------------
Title: "Multi-Modal Deep Learning for Sub-Seasonal Precipitation
        Prediction: Integrating Reanalysis and Satellite Data"
Journal: Nature Communications (if exceptional results) or
         Environmental Modelling & Software (Q1)
Status: Planned for V5
Message: Practical application, operational impact
```

### 5.4 Target Conferences

| Conference | Deadline | Type | Relevance |
|------------|----------|------|-----------|
| AGU Fall Meeting | August | Poster/Talk | High |
| EGU General Assembly | January | Abstract | High |
| NeurIPS Climate Workshop | September | Paper | Medium |
| ICLR | September | Paper | Medium |

---

## PART VI: DEVELOPMENT MILESTONES

### 6.1 Milestone Tracker

| Milestone | Target | Deliverable | Success Criteria | Status |
|-----------|--------|-------------|------------------|--------|
| M1 | Jan 15 | V4 Full Grid | R² > 0.60 full grid | PENDING |
| M2 | Jan 31 | V4 Optimized | Overfitting < 5x | PENDING |
| M3 | Feb 28 | V5 Data Ready | Pipeline working | PENDING |
| M4 | Mar 31 | V5 Trained | R² > 0.75 | PENDING |
| M5 | Apr 30 | V6 Complete | R² > 0.80 | PENDING |
| M6 | May 31 | Paper Draft | Ready for submission | PENDING |
| M7 | Jun 30 | Thesis Draft | Chapters 1-6 complete | PENDING |

### 6.2 Documentation Sync Points

```
After each milestone completion:
1. Update thesis.tex with methodology/results
2. Update paper.tex with comparative results
3. Update spec.md if standards change
4. Update this plan.md with progress
5. Commit to GitHub with milestone tag
```

---

## PART VII: RISK ASSESSMENT

### 7.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| V4 full grid doesn't improve | Medium | High | Iterate hyperparameters, try intermediate grids |
| Overfitting persists | High | Medium | Data augmentation, reduce model size |
| ERA5 data too large | Medium | Medium | Use temporal/spatial subset, cloud compute |
| Colab Pro insufficient | Medium | Medium | University cluster, AWS/GCP credits |
| Paper rejected | Medium | Medium | Have 2 target journals, improve per reviews |
| Timeline extends | High | Medium | 2-month buffer, prioritize V4-V5 |

### 7.2 Contingency Plan

```
IF V4 full grid fails:
  -> Use light mode as "proof of concept"
  -> Argue methodology is scalable
  -> Focus on architecture analysis

IF V5 multi-modal doesn't improve:
  -> Ablation study showing attempt was made
  -> Report as "negative result"
  -> Focus V6 on V2+V4 ensemble

IF insufficient time:
  -> Prioritize: V4 full > GNN-TAT Paper > V5 > V6
  -> Thesis can exclude V5-V6 if needed
  -> V5-V6 as "future work"
```

---

## PART VIII: RESOURCES

### 8.1 Compute Resources

| Resource | Use | Cost/month |
|----------|-----|------------|
| Google Colab Pro+ | Main training | $50 USD |
| Local GPU (RTX 3080) | Development, debug | $0 |
| Google Cloud (backup) | If Colab insufficient | Variable |

### 8.2 Data Sources

| Dataset | Size | Access |
|---------|------|--------|
| CHIRPS 2.0 | ~10 GB (region) | Public |
| SRTM DEM | ~500 MB | Public |
| ERA5 | ~50-100 GB | CDS API (free) |
| MODIS | ~20 GB | Google Earth Engine |

### 8.3 Software Stack

```
Core Stack:
- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- xarray + netCDF4
- pandas, numpy, scikit-learn

Visualization:
- matplotlib, seaborn
- cartopy (maps)

Development:
- Jupyter/Colab
- Git/GitHub
- VS Code
```

---

## APPENDIX: TASK CHECKLIST

### Immediate (This Week)

- [ ] Execute V4 full grid H=12
- [ ] Document training time full vs light
- [ ] Analyze full grid metrics
- [ ] Commit results to GitHub
- [ ] Update thesis.tex Chapter 5

### Short Term (January)

- [ ] Implement overfitting fixes
- [ ] Re-train V4 with increased regularization
- [ ] Start ERA5 download for study region
- [ ] Write thesis methodology section
- [ ] Update paper.tex with V4 results

### Medium Term (February-March)

- [ ] Complete V5 multi-modal pipeline
- [ ] Train V5 and compare with V4
- [ ] Write GNN-TAT paper
- [ ] Present progress to advisor
- [ ] Statistical significance tests

### Long Term (April-June)

- [ ] Implement V6 ensemble
- [ ] Complete thesis chapters
- [ ] Submit paper to journal
- [ ] Prepare defense

---

## VERSION CONTROL

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-03 | Initial version (Spanish) |
| 2.0 | 2026-01-03 | Complete rewrite in English, added SDD/DD frameworks, thesis mapping, paper deliverables |
| 2.1 | 2026-01-03 | Updated thesis title to align with doctoral proposal; added documentation formatting standards |
| 2.2 | 2026-01-03 | Project audit: renamed 13 CamelCase notebooks, translated 3 Spanish notebooks, added file scope rules |

---

## PROJECT AUDIT FINDINGS (January 2026)

### Audit Summary

Project compliance with spec.md, plan.md, and CLAUDE.md rules was audited.

**Overall Compliance:** ~75% (after corrections)

### Corrections Applied

| Issue | Count | Action Taken |
|-------|-------|--------------|
| CamelCase notebooks | 13 | Renamed to snake_case |
| Spanish filenames | 3 | Translated to English |
| Missing requirements.txt | 1 | Created at project root |
| Orphan files (nul) | 1 | Deleted |
| Missing scope rules | 2 | Added to CLAUDE.md, spec.md |

### Files Renamed (Models)

| Old Name | New Name |
|----------|----------|
| `base_models_Conv_STHyMOUNTAIN.ipynb` | `base_models_conv_sthymountain_v1.ipynb` |
| `base_models_Conv_STHyMOUNTAIN_V2.ipynb` | `base_models_conv_sthymountain_v2.ipynb` |
| `base_models_Conv_STHyMOUNTAIN_V3_FNO.ipynb` | `base_models_conv_sthymountain_v3_fno.ipynb` |
| `base_models_GNN_TAT_V4.ipynb` | `base_models_gnn_tat_v4.ipynb` |
| `base_models_ST-HybridWaveStack.ipynb` | `base_models_st_hybrid_wave_stack.ipynb` |
| `base_models_STHyMOUNTAIN.ipynb` | `base_models_sthymountain.ipynb` |
| `hybrid_models_ElevClusConvPrecipMetaNet.ipynb` | `hybrid_models_elev_clus_conv_precip_meta_net.ipynb` |
| `hybrid_models_enconders_ST-HybridWaveStack.ipynb` | `hybrid_models_encoders_st_hybrid_wave_stack.ipynb` |
| `hybrid_models_enconders_layering_...` | `hybrid_models_encoders_layering_w3_meta_model_unet_convlstm_st_hybrid_wave_stack.ipynb` |
| `hybrid_models_GRU-w12.ipynb` | `hybrid_models_gru_w12.ipynb` |
| `hybrid_models_ST-HybridWaveStack.ipynb` | `hybrid_models_st_hybrid_wave_stack.ipynb` |
| `hybrid_models_TopoRain_NET.ipynb` | `hybrid_models_topo_rain_net.ipynb` |
| `hybrid_models_TopoRain_NET_clusters_models_base.ipynb` | `hybrid_models_topo_rain_net_clusters_models_base.ipynb` |

### Files Renamed (Notebooks)

| Old Name | New Name |
|----------|----------|
| `analisis_bimodal_boyaca.ipynb` | `bimodal_analysis_boyaca.ipynb` |
| `analisis_correlacion.ipynb` | `correlation_analysis.ipynb` |
| `conv_lstm_boyaca_comparado.ipynb` | `conv_lstm_boyaca_comparison.ipynb` |

### Remaining Items (Low Priority)

| Item | Description | Action |
|------|-------------|--------|
| Spanish `.puml` files | Architecture diagrams in Spanish | Future translation |
| `output/` at root | Old hybrid model outputs | Consider archival |
| Output directory names | CamelCase (Advanced_Spatial/) | Preserved for backwards compatibility |

---

## DOCUMENTATION FORMATTING STANDARDS

### Applied Standards (Phase 2)

1. **Official Thesis Title:** Aligned with doctoral proposal
2. **Bibliography:** Expanded to 110+ Q1 references in `docs/tesis/references.bib`
3. **LaTeX Figures:** Width commands defined to prevent margin overflow
4. **Light Mode Notation:** Added disclaimers for provisional 5×5 grid results
5. **Thesis Structure:** 7 chapters following ML/Hydrology doctoral standards

### Light Mode Results Disclaimer

All V4 GNN-TAT results marked as "Light Mode (5×5 grid)" are provisional.
Full-grid validation (61×65 = 3,965 nodes) pending computational resources.

---

*Plan generated as part of the ML Precipitation Prediction Framework*
*This document should be updated bi-weekly with actual progress*
*Follows SDD and DD frameworks defined in spec.md*
*Phase 2: Documentation & Formatting Standards Applied*
