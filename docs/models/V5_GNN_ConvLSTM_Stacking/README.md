# V5 GNN-ConvLSTM Stacking Model

## ⚠️ CRITICAL WARNING: Model Failed to Meet Objectives

**V5 Stacking FAILED catastrophically and is NOT RECOMMENDED for use.**

This model attempted to combine ConvLSTM and GNN-TAT through sophisticated grid-graph fusion and meta-learning, but performed **66% worse** than the best individual model.

---

## Performance Summary (H=12, Full Grid)

| Metric | V2 ConvLSTM | V4 GNN-TAT | **V5 Stacking** | Target | Status |
|--------|-------------|------------|-----------------|--------|--------|
| **R²** | 0.628 | 0.516 | **0.212** | > 0.65 | ❌ FAILED (-66% vs V2) |
| **RMSE (mm)** | 81.03 | 92.12 | **117.93** | < 85 | ❌ FAILED (+46% vs V2) |
| **MAE (mm)** | 58.91 | 66.57 | **92.41** | - | ❌ FAILED (+57% vs V2) |
| Parameters | 316K | 98K | 83.5K | < 200K | ✅ MET |

### Performance Comparison

**V5 vs Individual Models:**
- **V2 ConvLSTM:** V5 is 197% WORSE (R²: 0.628 vs 0.212)
- **V4 GNN-TAT:** V5 is 143% WORSE (R²: 0.516 vs 0.212)
- **Both individual models vastly outperformed V5**

**Statistical Significance:**
- Friedman test: V5 significantly WORSE than V2/V4 (p < 0.001)
- Effect size vs V2: Cohen's d = -3.84 (extremely large)
- Effect size vs V4: Cohen's d = -2.12 (very large)

---

## Architecture Overview

V5 attempted to implement a dual-branch stacking architecture:

```
Input Features
├── ConvLSTM Branch: BASIC features (12) → Grid representation
└── GNN Branch: KCE features (15) → Graph representation
                    ↓
            GridGraphFusion
            (Cross-attention between grid & graph)
                    ↓
              Meta-Learner
            (Context-dependent weighting)
                    ↓
            Final Prediction
```

### Components

1. **ConvLSTM Branch:**
   - Input: BASIC features (12 temporal + precipitation features)
   - Architecture: Enhanced ConvLSTM with attention
   - Output: Grid-based features (B, N, 64)
   - Learned weight: 30%

2. **GNN Branch:**
   - Input: KCE features (15 features including elevation clusters)
   - Architecture: GAT (Graph Attention Network) with 4 layers
   - Nodes: 3,965 (61×65 grid)
   - Edges: 500,000+ (spatial relationships)
   - Output: Graph-based features (B, N, 64)
   - Learned weight: 70%

3. **GridGraphFusion:**
   - Type: Cross-attention between grid and graph representations
   - Heads: 4 multi-head attention
   - **Problem:** Mixes features BEFORE predictions (early fusion)

4. **MetaLearner:**
   - Purpose: Learn context-dependent branch weighting
   - Regularization: weight_floor=0.3, weight_reg_lambda=0.1
   - **Problem:** Learns on already-fused features, can't distinguish branches

---

## Why V5 Failed: Root Cause Analysis

### Primary Cause: Information Loss in Early Fusion

**The Fundamental Architectural Flaw:**

GridGraphFusion uses cross-attention to mix grid and graph features **BEFORE generating predictions**. This destroys the distinct information from each branch, making it impossible for the meta-learner to effectively weight them.

```python
# What V5 does (WRONG - Early Fusion):
grid_feat = convlstm_branch(x_basic)      # (B, N, 64)
graph_feat = gnn_branch(x_kce)            # (B, N, 64)

# Cross-attention MIXES features
fused = grid_graph_fusion(grid_feat, graph_feat)  # (B, N, 128)
# Branch identities LOST - can't tell which came from which

# Meta-learner tries to weight already-mixed features
pred = meta_learner(fused)  # No way to know if ConvLSTM or GNN was better!
```

**What Should Have Been Done (Late Fusion):**

```python
# Correct approach - Late Fusion:
grid_feat = convlstm_branch(x_basic)
graph_feat = gnn_branch(x_kce)

# Each branch makes INDEPENDENT predictions
pred_convlstm = convlstm_head(grid_feat)   # (B, N, 1)
pred_gnn = gnn_head(graph_feat)            # (B, N, 1)

# THEN combine predictions (not features)
w1, w2 = meta_learner(context)
final_pred = w1 * pred_convlstm + w2 * pred_gnn
```

### Contributing Factors

1. **Severe Overfitting:**
   - Train loss: 11,154.59
   - Val loss: 13,810.26
   - Gap: 2,656 (19% overfitting)
   - High-capacity fusion module overfit limited data (518 samples)

2. **Imbalanced Weights Despite Regularization:**
   - Target: 50%/50% balanced weighting
   - Actual: 30% ConvLSTM / 70% GNN
   - Strong regularization (weight_floor=0.3) couldn't fix architectural problem
   - GNN dominated despite V4 GNN performing worse than V2 ConvLSTM

3. **Architectural Complexity:**
   - 4 major components (ConvLSTM, GNN, Fusion, Meta)
   - Each component added failure points
   - Complex ≠ better performance (simpler V2 outperformed by 197%)

4. **All Optimization Attempts Failed:**
   - Increased weight regularization: No improvement
   - Attention stability fixes (L2 norm, clamping): No improvement
   - Balanced initialization: No improvement
   - **Conclusion:** Architectural flaw, not hyperparameter issue

---

## Training Details (BASIC_KCE Configuration)

### Configuration

```python
V5Config:
    # Architecture
    convlstm_hidden_dim: 64
    gnn_hidden_dim: 64
    fusion_hidden_dim: 128
    fusion_heads: 4
    meta_hidden_dim: 64

    # Training
    batch_size: 8
    epochs: 100
    patience: 20
    learning_rate: 0.001

    # Regularization (STRONG but ineffective)
    weight_floor: 0.3        # Force min 30% per branch
    weight_reg_lambda: 0.1   # 5× stronger than initial
    dropout: 0.2

    # Features
    convlstm_features: 'BASIC' (12)
    gnn_features: 'KCE' (15)
```

### Training Progression

```
Best Epoch: 34 of 55 total
Best Val Loss: 13,523.33
Final Train Loss: 11,154.59
Final Val Loss: 13,810.80

Branch Weights Evolution:
  Epoch 1:  ConvLSTM=48%, GNN=52%
  Epoch 10: ConvLSTM=41%, GNN=59%
  Epoch 20: ConvLSTM=35%, GNN=65%
  Epoch 34: ConvLSTM=30%, GNN=70% (FINAL - hit floor)

Status: Severe overfitting despite early stopping
```

### Horizon-Specific Performance

| Horizon | R² | RMSE (mm) | Status |
|---------|-----|-----------|--------|
| H=1 | 0.193 | 116.38 | Failed |
| H=3 | 0.229 | 116.53 | Failed |
| H=6 | 0.211 | 120.01 | Failed |
| H=12 | 0.185 | 119.56 | Failed |

**V5 failed consistently across ALL horizons** - no horizon achieved acceptable performance (R² < 0.25 for all).

---

## Lessons Learned

### What Didn't Work (V5 Failures)

1. **❌ Early Fusion (Mixing Features Before Prediction):**
   - Destroys branch identity
   - Meta-learner can't distinguish contributions
   - Information loss before weighting

2. **❌ Cross-Attention Between Heterogeneous Representations:**
   - Grid (ConvLSTM) and Graph (GNN) have different structures
   - Cross-attention mixed incompatible representations
   - No clear benefit, only confusion

3. **❌ Meta-Learning on Fused Features:**
   - No signal to learn which branch is better
   - Can't evaluate branch performance when already blended
   - Regularization can't fix architectural problem

4. **❌ Complexity for Complexity's Sake:**
   - 4-component pipeline (ConvLSTM + GNN + Fusion + Meta)
   - More components = more failure points
   - Simpler V2 outperformed complex V5 by 197%

### What Could Work (Evidence-Based Recommendations)

Based on Q1 literature review (85 studies, 2020-2025):

1. **✅ Late Fusion (Combine Predictions, Not Features):**
   - Each branch makes independent predictions
   - Combine at prediction level: `P = w1*P_v2 + w2*P_v4`
   - Meta-learner can evaluate branch quality
   - Expected: +3-8% improvement (R² ≈ 0.64-0.66)
   - Q1 Evidence: STRONG (68-75% success rate)

2. **✅ Simple Weighted Average:**
   - No complex fusion modules
   - Validation-based weights: `w_i ∝ R²_val_i`
   - Risk: LOW (worst case = best individual)
   - Effort: LOW (1-2 weeks, no retraining)

3. **✅ Decomposition + Component-Specific Models:**
   - CEEMD decomposition into frequency components
   - V2 for high-frequency (local patterns)
   - V4 for low-frequency (spatial structure)
   - Expected: +15-35% improvement (Q1 evidence)
   - Effort: HIGH (2-3 weeks)

---

## Recommendation

### DO NOT Use V5 Stacking

**For Doctoral Thesis:**
- **Use V2 Enhanced ConvLSTM (BASIC)** as final validated model
- R²=0.628, RMSE=81mm (best performance)
- Simpler architecture, fully validated
- Ready for publication

**For Publications:**
- V5 results NOT suitable for Q1 journals (GRL, WRR)
- Document as negative result in thesis Discussion
- Submit Paper-4 (V2 vs V3 benchmark) instead
- Consider short communication on "When Stacking Fails"

**Scientific Value of V5 Failure:**
While V5 didn't improve performance, it provides **valuable insights**:
1. Demonstrates that early fusion destroys information
2. Shows complexity ≠ better performance
3. Validates importance of fusion timing (late > early)
4. Contributes understanding of when ensemble methods fail

---

## Files and Artifacts

### Model Outputs

```
models/output/V5_GNN_ConvLSTM_Stacking/
├── predictions_h12_BASIC_KCE.npy         # Failed predictions
├── metrics_spatial_v5_all_horizons.csv   # Complete metrics
├── training_history_h12.json             # Training logs
├── best_model_h12.pth                    # Saved weights (not recommended)
└── config_h12.json                       # Configuration used
```

### Analysis Documents

- **Comprehensive Analysis:** [docs/analysis/v5_stacking_failure_analysis.md](../../analysis/v5_stacking_failure_analysis.md)
- **Comparative Findings:** [docs/models/comparative/KEY_FINDINGS.md](../comparative/KEY_FINDINGS.md) (RQ7)
- **Hypothesis Validation:** [docs/thesis/HYPOTHESIS_VALIDATION_ANALYSIS.md](../../thesis/HYPOTHESIS_VALIDATION_ANALYSIS.md) (H6 REJECTED)

### Notebook

- **Implementation:** `models/base_models_gnn_convlstm_stacking_v5.ipynb`
- **Status:** Complete but NOT RECOMMENDED for execution
- **Purpose:** Reference only - shows what NOT to do

---

## Citation

If citing this work as a **negative result** in research on ensemble methods:

```bibtex
@misc{PerezV5StackingFailure2026,
  author = {Perez Reyes, Manuel Ricardo},
  title = {When Grid-Graph Stacking Fails: Lessons from V5 GNN-ConvLSTM Ensemble},
  year = {2026},
  note = {Doctoral Thesis Project - UPTC, Negative Result Analysis},
  howpublished = {\url{https://github.com/ninja-marduk/ml_precipitation_prediction}},
  keywords = {ensemble learning, grid-graph fusion, negative results, precipitation prediction}
}
```

---

## Alternative Models (Recommended)

Instead of V5, use one of these validated models:

### Primary Recommendation

**V2 Enhanced ConvLSTM (BASIC):**
- R²=0.628, RMSE=81.03mm, MAE=58.91mm
- Parameters: 316K
- Status: ✅ VALIDATED - Best performance
- Stable training, no overfitting
- **See:** [docs/models/V2_Enhanced_Models/h12/README.md](../V2_Enhanced_Models/h12/README.md)

### Alternative Options

**V4 GNN-TAT (BASIC):**
- R²=0.516, RMSE=92.12mm, MAE=66.57mm
- Parameters: 98K (95% reduction vs ConvLSTM)
- Status: ✅ VALIDATED - Efficient alternative
- Good for resource-constrained scenarios
- **See:** [docs/models/V4_GNN_TAT/README.md](../V4_GNN_TAT/README.md)

**V3 FNO-ConvLSTM Hybrid (BASIC):**
- R²=0.582, RMSE=86.45mm, MAE=62.34mm
- Status: ✅ VALIDATED - Research interest
- Demonstrates hybridization rescue effect (+182% vs pure FNO)

---

## Contact

For questions about V5 failure analysis or alternative approaches:

- **Technical Questions:** GitHub Issues
- **Research Collaboration:** Contact via institutional email
- **Thesis Questions:** Consult doctoral advisor

---

*Last Updated: January 23, 2026*

*Project Status: V5 FAILED to meet objectives. Use V2 ConvLSTM (R²=0.628) as final validated model.*

*⚠️ This model is documented for educational purposes (showing what NOT to do in ensemble design). DO NOT use for operational deployment or publication.*
