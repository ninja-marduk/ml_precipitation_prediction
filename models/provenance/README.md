# Provenance of the reported results

These documents record what changed in the V4 GNN-TAT results between January and
August 2026, and why. They are kept under version control, unlike the metric files
and checkpoints themselves, which `.gitignore` excludes for size.

| file | what it records |
|------|-----------------|
| `investigation_headline_r2_2026-08-03.md` | Why the previously reported R2 = 0.628 could not be reproduced, what was ruled out, and what the number should be. Read this first. |
| `run_corrected_graph_2026-08-02.md` | The three-seed retrain on the corrected graph: results, measured compute, and what it supersedes. |
| `ablation_leaked_graph.md` | Paired three-seed ablation measuring the target leakage at -0.002 in R2. |
| `ablation_batch_size_2.md` | Batch size tested and ruled out; the decomposition of the gap into calibration and correlation. |
| `archive_pre_correction.md` | What the superseded metric files are and why they are not citable as published results. |
| `benchmark_multiseed.csv` | Every configuration with a mean, a spread and the inflation from quoting its best seed. Produced by `models/scripts/analysis/multiseed_benchmark.py`. |

The scripts that produce and verify these findings are in
`models/scripts/analysis/`: `graph_edge_budget_audit.py`,
`graph_structure_diagnostic.py`, `multiseed_benchmark.py`, `multiregion_ceiling.py`
and `graph_leakage_audit.py`. Each runs on CPU from the released metric files and
prints its own conclusions.

The per-seed metric files, checkpoints and prediction arrays these documents refer
to are not in the repository. The curated metric files needed to recompute every
reported number are included in the Zenodo archive.
