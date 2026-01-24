# V5 Robust Improvement Plan

## Goals
- Reproducible V5 pipeline with leakage-safe splits, consistent artifacts, and auditable metrics.
- Avoid V2-V4 regressions: split leakage, stale outputs, scale inversion errors, and missing logs.

## Phase 0: Data readiness (blockers)
- Standardize dataset path (local and Colab) and fail fast if missing.
- Validate required variables exist before training (time, lat, lon, total_precipitation, elevation, slope, aspect, lags, one-hot).
- Record dataset hash and dimensions in the experiment state.

## Phase 1: Leakage-free splits
- Split by time index BEFORE windowing.
- Exclude any window that crosses the train/val boundary (add an explicit gap if needed).
- Compute feature normalization stats on train only.
- Build correlation-based graph edges using train period only (no target leakage).

## Phase 2: Training and evaluation reliability
- Restore best model before evaluation and log best_epoch and train_val_gap.
- Evaluate on the full validation set (all windows, all horizons).
- Enforce quality gates:
  - mean_bias_pct <= 10
  - scale_ratio <= 50
  - no negative total precipitation
- Save metrics per horizon and branch weights per horizon.

## Phase 3: V5 model improvements
- Keep branch inputs explicit: BASIC -> ConvLSTM, KCE -> GNN.
- Add branch-weight regularization (entropy or min/max clamp) to avoid collapse.
- Optional bias calibration per horizon after evaluation (log before/after).

## Phase 4: Artifact management
- Output root: `models/output/V5_GNN_ConvLSTM_Stacking`.
- Save:
  - `metrics_spatial_v5_all_horizons.csv`
  - `experiment_state_v5.json` (config + dataset hash + split indices)
  - `v5_stacking_training_log_h{h}.csv`
  - `v5_results_summary.png`, `v5_training_curves.png`
- Clean partial runs to avoid mixing artifacts.

## Phase 5: Tests and checks
- Unit tests for:
  - window creation counts
  - split leakage (no overlap)
  - metric calculations
  - graph construction determinism
- Smoke test on light_mode (5x5 grid) to validate pipeline end-to-end.
