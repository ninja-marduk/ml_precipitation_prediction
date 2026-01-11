# AGENTS.md

This repo contains multiple notebook-driven pipelines (V1-V5). Use the guardrails below to avoid regressions seen in V2-V4 and keep V5 reproducible.

## Non-regression guardrails
- Split by time BEFORE windowing to avoid overlap leakage across train/val.
- Normalize features with training-only stats; persist stats and log them.
- If using correlation edges, compute them on the training period only (no target leakage).
- Evaluate on ALL validation windows, never a single sample.
- Reject runs with invalid outputs (negative precipitation, scale_ratio > 50, bias_pct > 10).

## Artifact discipline
- Always write outputs under `models/output/<version>/...` and log:
  - config, dataset path, dataset hash, split indices
  - best_epoch, best_val_loss, train_val_gap
- Do not keep stale notebook outputs; re-run after code changes or clear outputs.
- Avoid hard-coded Colab paths in outputs; use BASE_PATH and local-friendly paths.

## V5-specific checks
- Dataset expected locally:
  `data/output/complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc`
- Output root:
  `models/output/V5_GNN_ConvLSTM_Stacking`
- Fail fast if dataset missing or if output dir is empty after a run.

## Review checklist
- Window count matches `T - input_window - horizon + 1`.
- No overlap leakage across train/val (gap or split before windowing).
- Metrics computed per horizon on full validation set.
- Bias and scale checks logged and enforced.
- Branch weights reported per horizon; no collapse to a single branch without explanation.
