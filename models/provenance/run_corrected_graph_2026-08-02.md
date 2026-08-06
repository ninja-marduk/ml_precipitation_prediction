# V4 GNN-TAT, corrected graph, three seeds (2026-08-02)

Retrain of GNN_TAT_GAT on BASIC features, H=12, on the corrected graph. This run
supersedes every V4 GAT/BASIC number in the manuscript. The metrics it replaces
are preserved under `archive_pre_correction.md`.

## What changed relative to the archived run

1. **Correlation edges are estimated on the training period only** (notebook
   v4.1). Previously they used the full record, so the fixed spatial prior
   encoded the held-out months.
2. **The edge budget is now explicit** (v4.7). It was previously a constant
   behind a 1,000,000-edge trigger.

**Graph size is not a confounder.** `models/scripts/analysis/graph_edge_budget_audit.py`
rebuilds the adjacency on CPU and shows the thresholded graph holds 15,673,588
edges, 99.7% of the complete graph, because the elevation similarity term is
dense. The 500,000-edge budget therefore binds under both the leaked and the
corrected correlation. Both runs trained on exactly 500,000 edges and only
10,666 of them (2.13%) differ.

What is still confounded is the leakage fix and the v4.6 training-loop rewrite,
which changed batch ordering, loss accumulation and TF32. The model early-stops
at epoch 3 to 5, where the trajectory is still sensitive to batch order, so this
is not negligible. `ABLATION_LEAKED_GRAPH` (v4.9) isolates it by running the
current code with the full-record correlation.

## Results

Per-horizon R2 averaged over H=1..12, and the best single horizon:

| Seed | mean R2 | peak R2 | H at peak | R2 at H=12 | mean RMSE (mm) | epochs | best epoch |
|------|---------|---------|-----------|------------|----------------|--------|------------|
| 42   | 0.3882  | 0.4398  | 1         | 0.3560     | 103.90         | 18     | 3          |
| 123  | 0.4956  | 0.5342  | 6         | 0.4874     |  94.30         | 19     | 4          |
| 456  | 0.4549  | 0.5570  | 1         | 0.4951     |  98.00         | 20     | 5          |

Across seeds: mean R2 **0.4462 +/- 0.0542**, peak R2 **0.5103 +/- 0.0621** (sd, n=3).

Like for like against the archived GAT/BASIC run (`metrics_spatial_v4_gnn_tat_h12.csv`,
single seed, leaked graph): mean 0.5963 and peak 0.6284 at H=5. The correction
costs 0.150 of mean R2 and 0.118 of peak R2.

The per-cell climatology on the same target scores 0.739. The corrected model
therefore sits **0.23 below the zero-cost baseline at its best horizon**, where
the archived number sat 0.11 below it.

Systematic dry bias throughout: -17% to -27% depending on horizon.

The inter-seed spread (sd 0.054 on mean R2) is more than three times the
combination term the fusion decomposition attributes to ensembling (+0.0161),
which is consistent with that analysis rather than in tension with it.

## Provenance

Every number above is reproducible from the released arrays:
`SEED*/map_exports/H12/BASIC/GNN_TAT_GAT/{predictions,targets}.npy`. Recomputing
R2 from them returns the CSV values to four decimals for all three seeds.

Compute, from `SEED*/h12/BASIC/training_metrics/GNN_TAT_GAT_history.json`:

- NVIDIA A100-SXM4-80GB, fp32 with TF32 matmuls, no autocast
- 97,932 parameters; 500,000 edges over 3,965 nodes (126 per node)
- batch size 4, GNN chunk 30, gradient checkpointing on
- peak GPU memory **27.05 GB**; **322 s/epoch**; 1h37m, 1h42m and 1h47m per seed
- stopped by early stopping at epochs 18, 19 and 20 (patience 15)

Note for the compute table: the manuscript reports 2.1 GB and 28 minutes. The
measured cost of this configuration is 27.05 GB and about 100 minutes per seed.

## Coverage

This run covers GAT + BASIC only (`enabled_variants: ['GAT']`,
`enabled_features: ['BASIC']`). GCN, SAGE, KCE and PAFC still hold archived,
leaked-graph numbers and are not yet corrected.
