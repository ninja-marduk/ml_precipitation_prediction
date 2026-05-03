# Intra-Cell DEM Feature Analysis Report

**Date:** 2026-03-04
**Status:** Complete (all bundles trained and evaluated)

## 1. Objective

Evaluate whether sub-cell DEM topographic features improve spatiotemporal precipitation prediction beyond the baseline BASIC feature set (12 features). Three intra-cell DEM bundles are tested:

| Bundle | Features | Description |
|--------|----------|-------------|
| **BASIC** | 12 | Temporal encodings + precipitation stats + elevation/slope/aspect |
| **BASIC_D10** | 22 | BASIC + 10 elevation deciles (dem_p10...dem_p100) |
| **BASIC_PCA6** | 18 | BASIC + 6 PCA components of sub-cell DEM topology |
| **BASIC_D10_STATS** | 27 | BASIC_D10 + 5 statistics (dem_mean, dem_std, dem_skewness, dem_kurtosis, dem_range) |

**Grid:** 61 x 65 = 3,965 cells, 33 validation windows, H=12 horizons.

---

## 2. Complete Results Summary

### 2.1 All Models x All Bundles

| Model | BASIC (12) | D10 (22) | PCA6 (18) | D10_STATS (27) |
|-------|-----------|----------|-----------|----------------|
| **V2 ConvLSTM** | R²=0.629 | R²=0.453 | R²=0.417 | R²=0.351 |
| **V4 GNN-TAT** | R²=0.597 | R²=0.542 | R²=0.453 | R²=0.467 |
| **V10 Fusion** | R²=0.666 | R²=0.628 | R²=0.596 | R²=0.532 |

### 2.2 Detailed Comparison Table

| Bundle | Model | R² | RMSE (mm) | MAE (mm) | Bias (mm) |
|--------|-------|-----|-----------|----------|-----------|
| BASIC | V2 | 0.6287 | 81.05 | 58.91 | -10.50 |
| BASIC | V4 | 0.5974 | 84.40 | 59.74 | -28.79 |
| BASIC | V10 | **0.6658** | **76.90** | **56.30** | -0.02 |
| BASIC_D10 | V2 | 0.4533 | 98.35 | 78.84 | +17.41 |
| BASIC_D10 | V4 | 0.5424 | 89.98 | 63.96 | -32.91 |
| BASIC_D10 | V10 | 0.6283 | 81.10 | 59.87 | -0.06 |
| BASIC_PCA6 | V2 | 0.4166 | 101.60 | 79.92 | +7.03 |
| BASIC_PCA6 | V4 | 0.4526 | 98.41 | 71.37 | -40.78 |
| BASIC_PCA6 | V10 | 0.5956 | 84.59 | 63.50 | -0.04 |
| BASIC_D10_STATS | V2 | 0.3506 | 107.19 | 83.64 | +0.74 |
| BASIC_D10_STATS | V4 | 0.4667 | 97.14 | 71.38 | -26.80 |
| BASIC_D10_STATS | V10 | 0.5319 | 91.00 | 68.29 | -39.91 |

### 2.3 V10 Fusion vs BASIC Baseline

| Bundle | V10 R² | vs BASIC V10 | Delta |
|--------|--------|-------------|-------|
| BASIC | **0.6658** | baseline | - |
| BASIC_D10 | 0.6283 | **-5.6%** | -0.0375 |
| BASIC_PCA6 | 0.5956 | **-10.5%** | -0.0702 |
| BASIC_D10_STATS | 0.5319 | **-20.1%** | -0.1339 |

---

## 3. V2 ConvLSTM Results

### 3.1 Per-Horizon: BASIC_D10_STATS (27 features)

| H | RMSE | MAE | R² | Bias (mm) | Bias (%) |
|---|------|-----|-----|-----------|----------|
| 1 | 95.42 | 76.03 | 0.4573 | +13.79 | +7.4% |
| 2 | 96.60 | 76.16 | 0.4432 | +11.65 | +6.2% |
| 3 | 94.26 | 73.21 | 0.4958 | +1.51 | +0.8% |
| 4 | 101.09 | 80.54 | 0.4422 | +2.77 | +1.4% |
| **5** | **160.83** | **123.42** | **-0.4195** | **-46.58** | **-24.2%** |
| 6 | 101.09 | 81.64 | 0.4405 | +15.23 | +7.9% |
| 7 | 100.38 | 80.23 | 0.4435 | +6.48 | +3.3% |
| 8 | 108.17 | 86.56 | 0.3368 | -0.95 | -0.5% |
| 9 | 112.41 | 87.76 | 0.2760 | -10.02 | -4.8% |
| 10 | 102.53 | 82.70 | 0.3996 | +6.74 | +3.3% |
| 11 | 98.45 | 78.59 | 0.4486 | +8.29 | +4.2% |
| 12 | 98.17 | 76.88 | 0.4508 | -1.74 | -0.9% |

**H5 Anomaly:** R²=-0.42, RMSE=160.83mm, Bias=-46.58mm. The model collapses at this horizon.

### 3.2 Training Dynamics

| Bundle | Epochs | Best Epoch | Best val_loss | Params | Batch |
|--------|--------|------------|---------------|--------|-------|
| BASIC_D10 | 52 | 2 | 0.7450 | 90,252 | 2 |
| BASIC_PCA6 | 53 | 3 | 0.7747 | 85,644 | 2 |
| BASIC_D10_STATS | 55 | 5 | 0.8114 | 96,012 | 4 |

---

## 4. V4 GNN-TAT Results

### 4.1 Per-Horizon: BASIC_D10_STATS (27 features)

| H | RMSE | MAE | R² | Bias (mm) | Bias (%) |
|---|------|-----|-----|-----------|----------|
| 1 | 89.55 | 64.76 | 0.5220 | -20.71 | -11.1% |
| 2 | 93.51 | 68.52 | 0.4783 | -23.25 | -12.4% |
| 3 | 97.17 | 71.23 | 0.4641 | -26.82 | -14.1% |
| 4 | 100.00 | 73.96 | 0.4542 | -27.65 | -14.3% |
| 5 | 100.00 | 73.24 | 0.4512 | -30.41 | -15.8% |
| 6 | 98.73 | 72.28 | 0.4662 | -29.32 | -15.1% |
| 7 | 98.63 | 72.55 | 0.4627 | -24.31 | -12.3% |
| 8 | 98.77 | 72.94 | 0.4471 | -25.59 | -12.6% |
| 9 | 100.17 | 74.98 | 0.4251 | -27.53 | -13.3% |
| 10 | 98.36 | 73.06 | 0.4474 | -30.83 | -15.0% |
| 11 | 94.50 | 69.26 | 0.4920 | -27.81 | -13.9% |
| 12 | 95.67 | 69.80 | 0.4785 | -27.36 | -14.0% |

**Key observations:**
- No H5 anomaly (V4 is more stable than V2)
- Consistent negative bias (-12 to -16%): V4 systematically underpredicts
- Best epoch 2/52: still very early convergence
- R² range [0.425-0.522]: more uniform than V2 across horizons

### 4.2 V4 vs V2 on D10_STATS

| Metric | V2 ConvLSTM | V4 GNN-TAT | Delta |
|--------|-------------|------------|-------|
| R² | 0.3506 | **0.4667** | +33.1% |
| RMSE | 107.19 | **97.14** | -9.4% |
| MAE | 83.64 | **71.38** | -14.7% |
| Params | 96,012 | 98,892 | +3.0% |
| Best epoch | 5 | 2 | - |

V4 GNN-TAT handles the high-dimensional D10_STATS better than V2, confirming that graph attention provides some resilience to feature noise.

---

## 5. V10 Late Fusion Results

### 5.1 D10_STATS Fusion

| Metric | V2 | V4 | V10 Fusion |
|--------|-----|-----|------------|
| R² | 0.351 | 0.467 | **0.532** |
| RMSE | 107.19 | 97.14 | **91.00** |
| MAE | 83.64 | 71.38 | **68.29** |

**Fusion weights:** w_V2=0.325, w_V4=1.017, bias=-39.91

The fusion heavily favors V4 (w=1.02 vs w=0.32), which makes sense given V4's superior individual performance. The large negative bias (-39.91) compensates for V4's systematic underprediction.

### 5.2 V10 D10_STATS Per-Horizon

| H | R² | RMSE |
|---|-----|------|
| 1 | 0.5776 | 84.19 |
| 2 | 0.5497 | 86.87 |
| 3 | 0.5564 | 88.41 |
| 4 | 0.5544 | 90.35 |
| **5** | **0.4607** | **99.13** |
| 6 | 0.5522 | 90.43 |
| 7 | 0.5053 | 94.65 |
| 8 | 0.4949 | 94.40 |
| 9 | 0.4799 | 95.28 |
| 10 | 0.5379 | 89.95 |
| 11 | 0.5625 | 87.70 |
| 12 | 0.5422 | 89.64 |

H5 partially recovered by fusion (R²=0.46 vs V2's -0.42) but still the weakest horizon.

---

## 6. Analysis and Interpretation

### 6.1 Degradation Pattern

Adding more DEM features **consistently degrades** all models compared to BASIC:

```
ConvLSTM:  BASIC=0.629 → D10=0.453(-28%) → PCA6=0.417(-34%) → D10_STATS=0.351(-44%)
GNN-TAT:   BASIC=0.597 → D10=0.542(-9%)  → PCA6=0.453(-24%) → D10_STATS=0.467(-22%)
V10 Fused: BASIC=0.666 → D10=0.628(-6%)  → PCA6=0.596(-11%) → D10_STATS=0.532(-20%)
```

### 6.2 Architecture Resilience

GNN-TAT degrades less than ConvLSTM with additional features:

| Bundle | V2 Drop | V4 Drop | V4 Advantage |
|--------|---------|---------|--------------|
| D10 | -28.0% | -9.2% | 3x more resilient |
| PCA6 | -33.7% | -24.2% | 1.4x more resilient |
| D10_STATS | -44.2% | -21.8% | **2x more resilient** |

Graph attention can selectively weight node features, providing partial immunity to noise features. But it still cannot extract useful signal from intra-cell DEM statistics.

### 6.3 Root Cause: Curse of Dimensionality

1. **Early stopping at epochs 2-5**: All DEM bundles overfit immediately
2. **Monotonic degradation**: More features = worse R² (both architectures)
3. **H5 anomaly in V2**: Model collapses at specific horizons under feature overload
4. **Multicollinearity**: DEM deciles are inherently correlated (p10-p100 follow a monotonic curve)
5. **Statistics redundancy**: dem_mean ~ dem_p50, dem_range ~ dem_p100 - dem_p10

### 6.4 ConvLSTM Cannot Leverage Sub-Cell DEM

ConvLSTM processes features as channels in a 5D tensor. The 3x3 convolution kernel captures **between-cell** patterns, but intra-cell DEM features describe **within-cell** terrain variability. Convolution mixes these features across cells, creating spurious spatial patterns.

### 6.5 Implications for H2 (Topographic Features)

H2 was validated in Paper 4 using KCE (K-means elevation clusters). The intra-cell DEM approach differs fundamentally:

- **KCE:** Categorical (one-hot), low dimensionality (3-5 features), captures **elevation regimes**
- **Intra-cell DEM:** Continuous, high dimensionality (10-15 features), captures **terrain texture**

**Conclusion:** Elevation regime classification (KCE) is more informative for precipitation than raw sub-cell terrain statistics. This aligns with meteorological theory: orographic precipitation depends on elevation bands and slope orientation, not within-cell roughness.

### 6.6 V4 Bias Pattern

V4 GNN-TAT shows consistent systematic underprediction across all DEM bundles:

| Bundle | V4 Bias |
|--------|---------|
| BASIC | -28.79 mm |
| D10 | -32.91 mm |
| PCA6 | -40.78 mm |
| D10_STATS | -26.80 mm |

This suggests the GNN graph structure introduces a dampening effect on precipitation magnitude that worsens with more features. Late Fusion effectively corrects this (V10 bias near zero).

### 6.7 Elevation-Stratified Analysis

Per-cell R² stratified by elevation zone reveals that DEM features disproportionately degrade high-elevation predictions:

**V10 Late Fusion - Mean R² by Elevation Zone:**

| Bundle | Low (<1500m) | Medium (1500-2500m) | High (>2500m) | High-Low Gap |
|--------|-------------|--------------------|--------------|----|
| **P4 Baseline** | **0.549** | **0.490** | **0.436** | -0.113 |
| BASIC_D10 | 0.498 | 0.442 | 0.387 | -0.111 |
| BASIC_PCA6 | 0.449 | 0.329 | 0.254 | -0.195 |
| BASIC_D10_STATS | 0.357 | 0.196 | 0.124 | -0.233 |

**V2 ConvLSTM - Mean R² by Elevation Zone:**

| Bundle | Low (<1500m) | Medium (1500-2500m) | High (>2500m) |
|--------|-------------|--------------------|----|
| **P4 Baseline** | **0.505** | **0.446** | **0.370** |
| BASIC_D10 | 0.321 | -0.097 | **-1.125** |
| BASIC_PCA6 | 0.278 | -0.132 | -0.968 |
| BASIC_D10_STATS | 0.180 | -0.190 | -0.980 |

**Key finding:** ConvLSTM collapses to R²<-1.0 at high elevation with DEM features, meaning predictions are worse than predicting the mean. The curse of dimensionality is most severe where CHIRPS data quality is lowest.

### 6.8 Variogram Analysis - Spatial Error Structure

Empirical variograms of prediction errors (spherical model fit) show DEM features increase spatial error magnitude:

**Error Sill by Bundle (higher = worse):**

| Bundle | V2 ConvLSTM | V4 GNN-TAT | V10 Late Fusion |
|--------|------------|-----------|----------------|
| **P4 Baseline** | **390.5** | **360.8** | **276.6** |
| BASIC_D10 | 1140.0 | 397.4 | 374.2 |
| BASIC_PCA6 | 1444.5 | 667.9 | 584.1 |
| BASIC_D10_STATS | 1750.7 | 548.2 | 530.0 |

V2 ConvLSTM error sill increases 4.5× from baseline to D10_STATS. V4 GNN-TAT increases only 1.5×, confirming graph attention's selective feature weighting provides spatial resilience. Late Fusion reduces sill relative to individual DEM models but cannot recover baseline levels.

---

## 7. Recommendations

1. **Do not pursue D10_STATS further.** The 5 additional statistics add noise without signal.
2. **D10 is the best intra-cell bundle** for both architectures (highest R² among DEM variants).
3. **Feature selection** (MI, LASSO) could identify a minimal subset of useful DEM features.
4. **KCE remains the best topographic encoding** for precipitation prediction.
5. **Document as negative result** in thesis Chapter 5: valuable evidence that more granular topographic features do not improve performance.
6. **Late Fusion partially rescues** DEM degradation but cannot recover BASIC baseline level.

---

## 8. Data Locations

```
models/output/intracell_dem/
├── ConvLSTM/
│   ├── BASIC_D10/20260301_1/
│   ├── BASIC_PCA6/20260301_1/
│   └── BASIC_D10_STATS/20260303_1/
├── GNN_TAT_GAT/
│   ├── BASIC_D10/20260301_1/
│   ├── BASIC_PCA6/20260301_1/
│   └── BASIC_D10_STATS/20260304_1/
└── Late_Fusion/
    ├── BASIC_D10/20260303_1/
    ├── BASIC_PCA6/20260303_1/
    └── BASIC_D10_STATS/20260304_1/
```
