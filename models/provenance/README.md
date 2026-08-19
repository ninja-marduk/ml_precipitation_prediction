# Provenance of the reported results

These documents record what changed in the V4 GNN-TAT results between January and
August 2026, and why. They are kept under version control, unlike the metric files
and checkpoints themselves, which `.gitignore` excludes for size.

## The check that enforces this

`models/scripts/analysis/manuscript_numbers_audit.py` binds every load-bearing
number in the manuscript to the file that produces it, and fails if the two
disagree. Run it before any submission:

```
python models/scripts/analysis/manuscript_numbers_audit.py
```

It runs four kinds of check. **Anchors** match a regular expression over the LaTeX
sources and compare each match against a named quantity in a store built from the
records below; a pattern that matches nothing fails too, because the sentence it
guarded has moved. **Uniqueness** fails when one anchor finds two different values,
which is how a quantity comes to be stated two ways. **Prohibitions** fail when a
claim withdrawn in one section is still standing in another, which no equality check
finds. **Structure** recomputes every summary row of every table from the rows above
it, weighted where a count column exists, and requires a caption to say so when a
summary is deliberately not the aggregate of the rows shown.

Coverage is the registry, not the manuscript: a number no anchor names is not
checked. Add an anchor when you add a claim.

The stdout of each analysis script is captured here as `*.txt` and parsed by the
audit, so regenerating a record and re-running the audit is the loop.

## Two arrays for one model

Three directories hold more than one prediction array for what the manuscript calls
one model, written at different times:

| model | superseded | canonical |
|-------|-----------|-----------|
| GNN-TAT-GAT, seed 42 | `V4_GNN_TAT_Models/map_exports/...` (leaked graph, pooled 0.597) | `V4_GNN_TAT_Models/SEED42/map_exports/...` (corrected, 0.389) |
| Late Fusion, seed 42 | `V10_Late_Fusion/` (pooled 0.668) | `V10_Late_Fusion/SEED42/` (0.672) |

Every analysis script now names the seed in its path rather than falling back from a
parent directory, because the fallback silently chose the older array for one model
while other scripts read the newer one for another. That is what made the stratified
tables, the variogram table and the fusion table describe different model sets.

| file | what it records |
|------|-----------------|
| `naive_baselines.txt` | The three zero-cost references on the 33 released forecast windows, and every model scored against them under one pooled definition. |
| `anchoring_case_study.txt` | The same anchoring engine over all 92 admissible origins of the regional grid, for comparison with the 33-window figures. |
| `anchoring_eight_regimes.csv` | The anchoring step over eight precipitation regimes, global CHIRPS. Every row is engine output; the audit fails if one is not. |
| `fusion_decomposition_multiseed.txt` | The calibration-versus-combination decomposition, the per-seed blocked refit, and the split of the published-minus-refit difference into a fold-scheme term and an unattributed remainder. |
| `fusion_purged_holdout.txt` | The one out-of-sample estimate this design allows, on a contiguous holdout with an embargo. |
| `window_overlap_audit.txt` | Why blocking folds by window is not out of sample, and the two arrays for one model. |
| `stratified_comparison.txt` | Per-cell scores by elevation band, season and horizon group, plus the per-cell complementarity count. |
| `statistical_tests_audit.txt` | Every reported statistic against the admissible range of its own test, and Holm over the family that survives. |
| `factorial_blocked_analysis.txt` | The feature-by-architecture factorial re-analysed as a randomised complete block with seed as the block. |
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
