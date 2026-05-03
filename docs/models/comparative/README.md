# V2 vs V3 Comparative Benchmark Analysis

This directory contains a comprehensive benchmark analysis comparing V2 Enhanced Models (ConvLSTM-based) against V3 FNO Models (Fourier Neural Operators) for precipitation prediction.

## Overview

**Purpose:** Publication-quality comparative analysis for hybrid models research paper

**Scope:**
- Model Families: V2 Enhanced (10 architectures) vs V3 FNO (2 architectures)
- Prediction Horizon: H=12 months
- Feature Sets: BASIC, KCE, PAFC
- Metrics: RMSE, MAE, R², Bias
- Statistical Tests: Paired t-test, Wilcoxon, Cohen's d

## Directory Structure

```
comparative/
├── README.md                          # This file
├── KEY_FINDINGS.md                    # Main results summary (START HERE)
│
├── data/                              # Intermediate data files
│   ├── unified_metrics_h12.csv        # Combined V2+V3 metrics
│   ├── per_horizon_comparison_h12.csv # Per-horizon V2 vs V3
│   ├── aggregate_statistics_h12.csv   # Mean/std by experiment
│   ├── delta_metrics_h12.csv          # Performance deltas
│   ├── convergence_summary.csv        # Training dynamics
│   ├── epochs_to_convergence.csv      # Training efficiency
│   ├── statistical_tests_summary.csv  # Hypothesis test results
│   ├── v2_notebook_outputs.csv        # Extracted V2 notebook metrics
│   ├── v3_notebook_outputs.csv        # Extracted V3 notebook metrics
│   └── extraction_log.txt             # Extraction summary
│
└── tables/                            # Publication-ready tables
    ├── significance_matrix.tex        # LaTeX significance table
    ├── table1_overall_summary.tex     # LaTeX overall performance
    ├── table2_best_per_horizon.tex    # LaTeX per-horizon comparison
    ├── table3_training_efficiency.tex # LaTeX training metrics
    ├── summary_statistics.csv         # Comprehensive stats
    ├── best_models_ranking.csv        # Model rankings
    ├── horizon_degradation_rates.csv  # Degradation analysis
    └── feature_set_impact.csv         # Feature engineering impact
```

## Quick Start

### View Results

**Start with:** [KEY_FINDINGS.md](KEY_FINDINGS.md) - Comprehensive analysis answering 6 research questions

**Key Tables:**
- [summary_statistics.csv](tables/summary_statistics.csv) - Overall performance by experiment
- [statistical_tests_summary.csv](data/statistical_tests_summary.csv) - Significance tests
- [significance_matrix.tex](tables/significance_matrix.tex) - LaTeX table for paper

### Reproduce Analysis

```bash
# Navigate to project root
cd d:/github.com/ninja-marduk/ml_precipitation_prediction

# Run complete analysis
python scripts/benchmark/run_full_benchmark_analysis.py --core-only

# Skip notebook extraction (use existing data)
python scripts/benchmark/run_full_benchmark_analysis.py --skip-extraction --core-only
```

**Execution Time:** ~15 minutes first run, ~30 seconds subsequent runs

## Key Results Summary

### Main Finding

**V2 Enhanced ConvLSTM models significantly outperform V3 FNO models** (BASIC experiment: p < 0.001, Cohen's d = -1.97)

### Performance Comparison (BASIC H=12)

| Metric | V2 Enhanced | V3 FNO | Delta | p-value |
|--------|-------------|--------|-------|---------|
| **RMSE** | 98.17 mm | 102.55 mm | +4.38 mm | **< 0.001*** |
| **MAE** | 73.92 mm | 77.74 mm | +3.82 mm | **< 0.001*** |
| **R²** | 0.437 | 0.385 | -0.052 | **< 0.001*** |

### Feature Set Impact

**BASIC features provide best performance** - engineered features (KCE, PAFC) degrade performance ~22% for both families

| Feature Set | V2 RMSE | V3 RMSE | Performance |
|-------------|---------|---------|-------------|
| **BASIC** | 98.17 mm | 102.55 mm | **Best** |
| KCE | 120.01 mm | 123.75 mm | -22% worse |
| PAFC | 119.43 mm | 127.16 mm | -22% worse |

### Recommendations

**For Production:** Use **V2 ConvLSTM_Bidirectional with BASIC features**
- RMSE: ~80-85 mm at H=12
- R²: ~0.60-0.64
- Statistically validated (p < 0.001)

## Data Files Description

### Raw Metrics (`data/`)

**unified_metrics_h12.csv**
- Combined V2 and V3 metrics
- Columns: TotalHorizon, Experiment, Model, H, RMSE, MAE, R², Mean_True_mm, Mean_Pred_mm, Version
- 1000+ rows (all models × experiments × horizons)

**per_horizon_comparison_h12.csv**
- Direct V2 vs V3 comparison per horizon
- Columns: Experiment, H, RMSE_V2, RMSE_V3, delta_RMSE, percent_change_RMSE, etc.
- 36 rows (3 experiments × 12 horizons)

**aggregate_statistics_h12.csv**
- Mean/std/min/max by experiment and model family
- Columns: Experiment, Model_Family, RMSE_mean, RMSE_std, MAE_mean, etc.
- 6 rows (3 experiments × 2 families)

**delta_metrics_h12.csv**
- Overall performance deltas
- Columns: Experiment, V2_RMSE_mean, V3_RMSE_mean, delta_RMSE_mean, etc.
- 3 rows (BASIC, KCE, PAFC)

**statistical_tests_summary.csv**
- Hypothesis test results
- Columns: experiment, metric, t_statistic, p_value, cohens_d, significant_alpha_0.05, etc.
- 9 rows (3 experiments × 3 metrics)

**convergence_summary.csv**
- Training dynamics aggregated
- Columns: version, experiment, epochs_to_best_mean, best_val_loss_mean, training_stability_mean
- 6 rows (3 experiments × 2 versions)

### Tables (`tables/`)

**LaTeX Tables** (.tex files):
- Ready for `\input{}` in LaTeX documents
- Uses booktabs package for professional formatting
- Includes table captions and labels

**CSV Tables** (.csv files):
- Intermediate analysis results
- Ready for Excel, Python, R analysis
- Self-documenting column names

## Statistical Methods

### Tests Performed

1. **Paired t-test:** Compare V2 vs V3 on same horizons (parametric)
2. **Wilcoxon Signed-Rank:** Non-parametric alternative
3. **Cohen's d:** Effect size calculation
4. **95% Confidence Intervals:** Uncertainty quantification

### Significance Levels

- *** p < 0.001 (highly significant)
- ** p < 0.01 (very significant)
- * p < 0.05 (significant)
- ns = not significant (p ≥ 0.05)

### Effect Size Interpretation (Cohen's d)

- |d| < 0.2: Negligible
- 0.2 ≤ |d| < 0.5: Small
- 0.5 ≤ |d| < 0.8: Medium
- |d| ≥ 0.8: Large

## Scripts & Reproducibility

### Analysis Pipeline

All scripts located in `scripts/benchmark/`:

1. **01_extract_notebook_outputs.py** - Parse Jupyter notebooks
2. **02_consolidate_metrics.py** - Merge and calculate deltas
3. **03_training_dynamics_analysis.py** - Training efficiency
4. **04_statistical_tests.py** - Hypothesis testing
5. **11_generate_latex_tables.py** - LaTeX table generation
6. **12_generate_csv_tables.py** - CSV table generation
7. **run_full_benchmark_analysis.py** - Master script

### Requirements

```python
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
nbformat>=5.7.0
```

### Installation

```bash
pip install pandas numpy scipy nbformat
```

## Input Data Sources

### V2 Enhanced Models
- **Metrics:** `models/output/V2_Enhanced_Models/metrics_spatial_v2_refactored_h12.csv`
- **Training Logs:** `models/output/V2_Enhanced_Models/h12/{BASIC,KCE,PAFC}/training_metrics/*_training_log_h12.csv`
- **Notebook:** `models/base_models_Conv_STHyMOUNTAIN_V2.ipynb`

### V3 FNO Models
- **Metrics:** `models/output/V3_FNO_Models/metrics_spatial_v2_refactored_h12.csv`
- **Training Logs:** `models/output/V3_FNO_Models/h12/{BASIC,KCE,PAFC}/training_metrics/*_training_log_h12.csv`
- **Notebook:** `models/base_models_Conv_STHyMOUNTAIN_V3_FNO.ipynb`

## Citation

If using this benchmark analysis in publications, please cite:

```bibtex
@article{precipitation_hybrid_models_2025,
  title={Comparative Analysis of Hybrid Deep Learning Models for Multi-Month Precipitation Prediction},
  author={[Authors]},
  journal={[Journal Name]},
  year={2025},
  note={V2 vs V3 Benchmark Analysis}
}
```

## Changelog

**Version 1.0 (2025-12-31)**
- Initial comprehensive benchmark analysis
- Statistical significance testing implemented
- LaTeX and CSV tables generated
- KEY_FINDINGS document created

## Contact & Support

**Project:** ML Precipitation Prediction
**Analysis Date:** December 31, 2025

For questions or issues:
1. Check [KEY_FINDINGS.md](KEY_FINDINGS.md) for detailed results
2. Review statistical tests in [data/statistical_tests_summary.csv](data/statistical_tests_summary.csv)
3. Consult main project documentation

---

**Last Updated:** December 31, 2025
**Document Version:** 1.0
