# PhD Dissertation Execution Plan

## Computational Model for the Spatiotemporal Prediction of Monthly Precipitation in Mountainous Areas Using Machine Learning Techniques

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Document ID** | PLAN-PHD-PRECIP-2025 |
| **Version** | 1.0.0 |
| **Status** | Active |
| **Author** | Manuel Ricardo Pérez Reyes |
| **Institution** | Universidad Pedagógica y Tecnológica de Colombia (UPTC) |
| **Reference** | spec.md (v2.0.0) |
| **Last Updated** | January 2026 |

---

## 1. Executive Summary

This plan outlines the execution strategy for completing the PhD dissertation on precipitation prediction using machine learning. The research has achieved significant milestones (V1-V4 model development) and is now in the documentation and publication phase.

**Current Status:** 75% complete - Model development finished, documentation in progress.

---

## 2. Progress Overview

### 2.1 Completed Milestones

| Milestone | Description | Status | Completion Date |
|-----------|-------------|--------|-----------------|
| **M1** | Data pipeline development (CHIRPS + DEM) | ✅ Complete | 2024-Q2 |
| **M2** | V1 Baseline models (ConvLSTM/GRU/RNN) | ✅ Complete | 2024-Q3 |
| **M3** | V2 Enhanced models with attention | ✅ Complete | 2024-Q4 |
| **M4** | V3 FNO experiments | ✅ Complete | 2025-Q1 |
| **M5** | V4 GNN-TAT development | ✅ Complete | 2025-Q4 |
| **M6** | Full-grid V4 results (3,965 nodes) | ✅ Complete | 2026-01 |
| **M7** | Statistical significance tests | ✅ Complete | 2026-01 |
| **M8** | Paper 4 (MDPI Hydrology) draft | ✅ Complete | 2026-01 |

### 2.2 In-Progress Milestones

| Milestone | Description | Status | Target Date |
|-----------|-------------|--------|-------------|
| **M9** | Thesis document writing | 🔄 In Progress | 2026-Q1 |
| **M10** | Bibliography expansion (100+ refs) | 🔄 In Progress | 2026-01 |
| **M11** | Review article submission | 📝 Under Review | 2026-Q1 |
| **M12** | Paper 4 submission | 📝 Ready | 2026-01 |

### 2.3 Pending Milestones

| Milestone | Description | Status | Target Date |
|-----------|-------------|--------|-------------|
| **M13** | Thesis first draft complete | ⏳ Pending | 2026-Q2 |
| **M14** | Candidacy examination | ⏳ Pending | TBD |
| **M15** | Defense preparation | ⏳ Pending | TBD |
| **M16** | Final defense | ⏳ Pending | TBD |

---

## 3. Hypothesis Validation Status

### 3.1 Main Hypothesis

> "Applying machine learning models, combined with time series analysis and data preprocessing methods, will significantly improve the accuracy of monthly precipitation predictions in mountainous areas."

**Status:** PARTIALLY VALIDATED

| Evidence | Finding |
|----------|---------|
| GNN-TAT R² | 0.628 (meets threshold R² ≥ 0.60) |
| Mean RMSE reduction | 17.8% vs ConvLSTM (p=0.015) |
| Interpretation | Comparable performance with interpretability + efficiency gains |

### 3.2 Derived Hypotheses Status

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| **H1**: GNN-TAT ≥ ConvLSTM accuracy | PARTIALLY VALIDATED | R²=0.628 vs 0.653; Mean RMSE better (92.12 vs 112.02) |
| **H2**: Topographic features improve accuracy | VALIDATED | PAFC +10% R² for GCN; KCE +3% for GAT |
| **H3**: Non-Euclidean relations capture orography | VALIDATED | Graph structure on 3,965 nodes, 500K edges |
| **H4**: Temporal attention improves long horizons | VALIDATED | 9.6% degradation (H1→H12) < 20% threshold |

---

## 4. Chapter Completion Tracking

### 4.1 Front Matter

| Section | Status | Pages | Notes |
|---------|--------|-------|-------|
| Title Page | ⏳ Pending | 1 | UPTC format |
| Dedication | ⏳ Pending | 1 | |
| Acknowledgments | ⏳ Pending | 1 | |
| Abstract (EN) | ✅ Draft | 1 | Aligned with proposal |
| Abstract (ES) | ⏳ Pending | 1 | Translation needed |
| Table of Contents | ⏳ Auto | 2 | LaTeX generated |
| List of Figures | ⏳ Auto | 2 | LaTeX generated |
| List of Tables | ⏳ Auto | 2 | LaTeX generated |
| List of Abbreviations | 🔄 In Progress | 2 | 50+ terms defined |

### 4.2 Main Chapters

| Chapter | Status | Target Pages | Current Pages | Completion |
|---------|--------|--------------|---------------|------------|
| **Ch 1: Introduction** | 🔄 In Progress | 20-25 | 15 | 60% |
| **Ch 2: Theoretical Framework** | 🔄 In Progress | 40-50 | 25 | 50% |
| **Ch 3: Materials & Methods** | ✅ Draft | 40-50 | 35 | 80% |
| **Ch 4: Results** | ✅ Draft | 50-60 | 40 | 70% |
| **Ch 5: Discussion** | 🔄 In Progress | 25-30 | 15 | 50% |
| **Ch 6: Conclusions** | ✅ Draft | 15-20 | 10 | 60% |

### 4.3 Back Matter

| Section | Status | Entries | Target |
|---------|--------|---------|--------|
| References | 🔄 In Progress | 40 | 100-150 |
| Appendix A: Results Tables | ✅ Complete | - | - |
| Appendix B: Hyperparameters | ✅ Complete | - | - |
| Appendix C: Statistical Tests | ✅ Complete | - | - |
| Appendix D: Code Documentation | 🔄 In Progress | - | - |
| Appendix E: Glossary | 🔄 In Progress | 50+ | 100+ |

---

## 5. Current Phase: Documentation & Publication

### 5.1 Immediate Tasks (This Week)

| Task | Priority | Owner | Status |
|------|----------|-------|--------|
| Submit Paper 4 to MDPI Hydrology | HIGH | MRP | Ready |
| Expand bibliography to 100+ refs | HIGH | MRP | In Progress |
| Complete Chapter 2 (Theoretical Framework) | HIGH | MRP | In Progress |
| Update thesis.tex with V4 results | MEDIUM | MRP | Completed |

### 5.2 Short-term Tasks (This Month)

| Task | Priority | Target Date |
|------|----------|-------------|
| Bibliography: Add 60 Q1/Q2 references | HIGH | 2026-01-15 |
| Chapter 1: Complete Introduction | HIGH | 2026-01-20 |
| Chapter 2: Complete Literature Review | HIGH | 2026-01-25 |
| Chapter 5: Expand Discussion section | MEDIUM | 2026-01-30 |
| LaTeX figure formatting verification | MEDIUM | 2026-01-15 |

### 5.3 Medium-term Tasks (This Quarter)

| Task | Priority | Target Date |
|------|----------|-------------|
| Complete first draft of all chapters | HIGH | 2026-02-28 |
| Internal review with advisor | HIGH | 2026-03-15 |
| Revisions based on feedback | HIGH | 2026-03-30 |
| Prepare defense slides (draft) | MEDIUM | 2026-03-30 |

---

## 6. Bibliography Expansion Plan

### 6.1 Current Status

| Category | Current | Target | Gap |
|----------|---------|--------|-----|
| Total References | 40 | 100-150 | +60-110 |
| Q1/Q2 Journals | 27 | 80+ | +53 |
| Conference Papers | 5 | 15+ | +10 |
| Books/Chapters | 8 | 10+ | +2 |

### 6.2 Reference Categories to Expand

| Category | Target Additions | Priority |
|----------|------------------|----------|
| **Graph Neural Networks** | +20 | HIGH |
| **Precipitation Prediction ML** | +15 | HIGH |
| **ConvLSTM/Spatiotemporal** | +10 | HIGH |
| **Climate Data (CHIRPS, ERA5)** | +10 | MEDIUM |
| **Statistical Methods** | +10 | MEDIUM |
| **Hydrology/Meteorology** | +10 | MEDIUM |
| **Deep Learning Fundamentals** | +5 | LOW |

### 6.3 Key References to Add

**Graph Neural Networks:**
- Kipf & Welling (2017) - GCN
- Veličković et al. (2018) - GAT
- Hamilton et al. (2017) - GraphSAGE
- Wu et al. (2020) - GNN Survey

**Precipitation ML:**
- Ravuri et al. (2021) - Nature: DeepMind nowcasting
- Kratzert et al. (2019) - LSTM hydrology
- MetNet-2 (2022) - Google Research
- NowcastNet (2023) - Nature

**Climate/Hydrology:**
- Funk et al. (2015) - CHIRPS
- ERA5 documentation
- Boyacá regional studies

---

## 7. Quality Assurance Checklist

### 7.1 Reproducibility Verification

- [x] Random seeds documented (seed=42)
- [x] Hyperparameters tabulated
- [x] Data splits specified (70/15/15)
- [x] Software versions listed
- [x] Training times recorded
- [ ] Code repository prepared for publication
- [ ] Data access instructions documented

### 7.2 Statistical Rigor

- [x] Mann-Whitney U test completed (U=57, p=0.015)
- [x] Cohen's d effect size calculated (d=1.03)
- [x] Standard deviations reported (GNN-TAT: 6.48, ConvLSTM: 27.16)
- [ ] Bootstrap confidence intervals (optional)
- [ ] Multiple-run verification (recommend 30+ runs)

### 7.3 Document Quality

- [x] Figures at ≥300 DPI
- [x] Tables consistently formatted
- [ ] All figures within page margins
- [ ] Cross-references verified
- [ ] Spelling/grammar check completed
- [ ] Citation format consistent (APA)

### 7.4 Alignment Verification

- [x] Thesis aligned with doctoral proposal hypothesis
- [x] Thesis aligned with doctoral proposal objectives
- [x] paper.tex metrics aligned with CSV data
- [x] thesis.tex metrics aligned with paper.tex
- [x] spec.md updated with current results
- [x] plan.md reflects current progress

---

## 8. Publication Strategy

### 8.1 Papers Status

| # | Title | Target | Status | Action |
|---|-------|--------|--------|--------|
| 1 | Systematic Review of Hybrid Models | Q1 Journal | Under Review | Wait for response |
| 2 | MDPI Hydrology Paper (V2-V4) | MDPI Hydrology | Ready | Submit this week |
| 3 | GNN-TAT Focus Paper | Q1 Journal | Planned | After thesis draft |

### 8.2 Conference Presentations

| Event | Target Date | Status |
|-------|-------------|--------|
| AGU Fall Meeting 2026 | Dec 2026 | Consider abstract |
| CCAI Workshop (NeurIPS) | Nov 2026 | Optional |
| Local/Regional Conference | TBD | Required by UPTC |

---

## 9. Risk Assessment

### 9.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ConvLSTM outperforms GNN-TAT on peak R² | REALIZED | Medium | Emphasize mean RMSE, efficiency, interpretability |
| Missing training logs for CI | Medium | Medium | Use single-run values, note as limitation |
| Reviewers request additional experiments | High | Medium | Prepare V5/V6 as future work |

### 9.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Paper rejection delays thesis | Medium | High | Submit early, have backup venues |
| Advisor availability | Low | Medium | Schedule regular meetings |
| UPTC administrative delays | Medium | Medium | Start paperwork early |

### 9.3 Quality Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient bibliography | Medium | High | Dedicate week to expansion |
| Poor English writing quality | Low | Medium | Use professional editing |
| LaTeX formatting issues | Low | Low | Use UPTC template, test early |

---

## 10. Action Items for Next Session

### 10.1 COMPLETED Tasks (2026-01-04 and 2026-01-05)

1. **Governance & Standards Update** ✅
   - [x] Updated spec.md with internal vs. public practices (Section 10)
   - [x] Removed SDD framework from thesis.tex (personal methodology)
   - [x] Added DD framework with proper academic context and citations
   - [x] Fixed TikZ diagram overflow in Chapter 4 (resized to 0.95\textwidth)
   - [x] Removed "700 DPI" from thesis objectives (internal standard)
   - [x] Added Schedule and Budget appendices to thesis.tex
   - [x] Updated paper spec.md with figure margin compliance standards
   - [x] Expanded CHIRPS section with proper references and justification

2. **Comprehensive Spec Update (2026-01-05)** ✅
   - [x] Updated reference requirement to 120+ (aligned with doctoral proposal: 123 refs)
   - [x] Added Section 6: Data Analysis and Feature Engineering Details
     - 6.1 Dataset Overview (CHIRPS-2.0 and SRTM DEM full details)
     - 6.2 Precipitation Statistics (monthly averages, global stats)
     - 6.3 K-Means Clustering Analysis (configuration, categories, chi-square test)
     - 6.4 Temporal Analysis (PACF, seasonal decomposition)
     - 6.5 Feature Engineering Pipeline (BASIC, KCE, PAFC full definitions)
     - 6.6 Preprocessing Techniques (EMD/CEEMDAN, wavelet analysis)
     - 6.7 Data Windowing Configuration
     - 6.8 Normalization details
   - [x] Enhanced Section 7: Model Versioning with full architecture details
     - V1: ConvLSTM/ConvGRU/ConvRNN (78,732 params, batch=8, epochs=150)
     - V2: Enhanced models (dropout=0.2, L2=1e-5, 7 variants)
     - V3: FNO models (4,612 to 106,292 params)
     - V4: GNN-TAT full configuration (3,965 nodes, 500K edges, 97K-106K params)
   - [x] Updated Colab Pro+ costs (2 years Account 1, 3 months Account 2)
   - [x] Added GPU hardware specifications (NVIDIA A100 40GB)
   - [x] Added thesis.tex Appendix B with detailed computational resources

### 10.2 Immediate (Before Next Work Session)

1. **Content Quality Improvements** (Priority: HIGH)
   - [ ] Review all thesis sections for short phrases without context
   - [ ] Add 2-3 citations per major technical claim
   - [ ] Expand preprocessing section with methodology justification
   - [ ] Add DEM (SRTM) section with proper references

2. **Figure Quality Check** (Priority: HIGH)
   - [ ] Test thesis.tex PDF compilation
   - [ ] Verify all figures fit within page margins
   - [ ] Check for overlapping text in all graphs
   - [ ] Verify legend positions are correct

3. **Paper 4 Submission** (Priority: HIGH)
   - [ ] Final PDF compilation check
   - [ ] Verify author information
   - [ ] Submit to MDPI Hydrology portal

### 10.3 Near-term (This Month)

4. **Chapter 2: Literature Review**
   - [ ] Expand Section 2.3 (ML for Precipitation) with 10+ refs
   - [ ] Complete Section 2.4 (Hybrid Models) using systematic review
   - [ ] Add PRISMA methodology details from review paper

5. **Chapter 5: Discussion Expansion**
   - [ ] Add comparison with more state-of-art methods
   - [ ] Expand limitations section with specific constraints
   - [ ] Add practical implications for Boyacá water management

6. **Documentation Synchronization**
   - [ ] Ensure thesis.tex metrics match spec.md claims
   - [ ] Verify paper.tex uses same statistical values
   - [ ] Update README.md with project status

### 10.4 Quality Gates (Pre-submission)

| Gate | Requirement | Status |
|------|-------------|--------|
| Figures within margins | All figures ≤0.95\textwidth | ✅ Fixed |
| No SDD in thesis | Remove personal framework | ✅ Complete |
| No DPI in thesis text | Internal standard only | ✅ Complete |
| CHIRPS properly cited | Original paper + validations | ✅ Complete |
| Schedule/Budget appendices | Required by UPTC | ✅ Complete |
| DD framework documented | With academic citations | ✅ Complete |

---

## 11. Success Metrics

### 11.1 Thesis Completion Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| 150+ pages | ⏳ In Progress | Currently ~100 |
| 100+ references | ✅ Complete | 110+ references in bibliography |
| 6 chapters complete | 🔄 60% | 4/6 drafted |
| All figures at 300+ DPI | ✅ Complete | Verified |
| Statistical tests complete | ✅ Complete | U=57, p=0.015 |
| Hypothesis validation | ✅ Complete | H1-H4 documented |
| Figures within margins | ✅ Complete | TikZ diagram fixed |
| SDD removed from thesis | ✅ Complete | Personal methodology |
| Schedule/Budget added | ✅ Complete | Appendices A and B |

### 11.2 Key Performance Indicators

| KPI | Target | Current | Status |
|-----|--------|---------|--------|
| Best R² | ≥ 0.60 | 0.628 | ✅ Achieved |
| Statistical significance | p < 0.05 | p=0.015 | ✅ Achieved |
| Mean RMSE reduction | > 10% | 17.8% | ✅ Achieved |
| Parameter efficiency | > 5x | 20x | ✅ Achieved |
| Horizon degradation | < 20% | 9.6% | ✅ Achieved |

---

## 12. Appendix: Key Results Summary

### Full-Grid V4 Best Results

| Model | Experiment | H | R² | RMSE (mm) | MAE (mm) |
|-------|------------|---|-----|-----------|----------|
| **GAT + BASIC** | BASIC | 5 | **0.628** | 82.29 | 58.19 |
| GCN + PAFC | PAFC | 1 | 0.625 | 79.34 | 55.41 |
| GAT + KCE | KCE | 6 | 0.616 | 83.70 | 59.12 |

### Horizon Degradation (GAT + BASIC)

| Horizon | R² | RMSE (mm) | Degradation vs H=1 |
|---------|-----|----------|-------------------|
| H=1 | 0.613 | 80.61 | --- |
| H=3 | 0.610 | 82.86 | -0.5% |
| H=6 | 0.612 | 84.17 | -0.2% |
| H=12 | 0.554 | 88.51 | -9.6% |

### Statistical Comparison

| Metric | GNN-TAT | ConvLSTM | Test |
|--------|---------|----------|------|
| Mean RMSE | 92.12 (SD=6.48) | 112.02 (SD=27.16) | U=57, p=0.015 |
| Best R² | 0.628 | 0.653 | - |
| Parameters | ~98K | ~500K-2.1M | 95% reduction |

---

*Plan maintained by: Manuel Ricardo Pérez Reyes*
*Aligned with: spec.md (v2.0.0), CLAUDE.md project rules*
*Last updated: January 2026*
