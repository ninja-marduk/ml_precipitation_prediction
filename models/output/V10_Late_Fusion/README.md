# V10 Late Fusion — released metrics

Canonical artefacts for the Late Fusion (Ridge) ensemble reported in the manuscript.

## Which run is canonical

Use the seed-scoped directories. The manuscript's Equation for the fusion weights
and all reported Late Fusion figures come from `SEED42/`:

| File | Base learners | w_ConvLSTM | w_GNN | bias (mm) | pooled R2 |
|---|---|---|---|---|---|
| `SEED42/v10_summary.json` | ConvLSTM-Bidirectional (148K) + GNN-TAT-GAT | 0.5092 | 0.6524 | -6.371 | 0.6716 |
| `SEED123/v10_summary.json` | same | 0.601 | 0.747 | -7.69 | 0.6564 |
| `SEED456/v10_summary.json` | same | 0.598 | 0.829 | -26.67 | 0.6364 |

Three-seed mean: R2 = 0.655 +/- 0.018.

## Superseded runs (not released)

An earlier pipeline state fused the tuned 79K ConvLSTM (`v2_convlstm`) instead of the
Bidirectional variant, yielding different weights (0.446, 0.710). Those `v10_summary.legacy.json`
files and the repository-root `v10_metrics.csv` are superseded and are deliberately not part of
this release, to avoid ambiguity about which model the manuscript reports.

## Files

- `metrics_multiseed_consolidated.csv` — per-horizon mean/sd over seeds {42, 123, 456}
- `SEED*/v10_metrics.csv` — per-horizon R2, RMSE, MAE, Bias for that seed
- `SEED*/v10_summary.json` — baselines, fusion variants, learned Ridge weights
