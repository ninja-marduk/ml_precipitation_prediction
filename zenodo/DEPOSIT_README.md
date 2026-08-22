# AnchorBench v1.0 archive

This record holds the code, the inputs and the intermediate results needed to
recompute every number in the accompanying manuscript. Nothing is held back and
nothing is offered on request.

## What is here

| file | what it holds |
|------|---------------|
| `anchorbench-v1.3.0-code.zip` | the repository as tracked by git: the model notebooks, the protocol implementation, every analysis script, and the provenance record |
| `anchorbench-v1.3.0-predictions-and-metrics.zip` | per-seed prediction arrays and targets for the three architectures at seeds 42, 123 and 456; the per-seed and per-horizon metric files behind the multi-seed factorial; `models/provenance/` |
| `complete_dataset_...clean.nc` | the engineered feature set the CPU analyses consume (504 MB), CHIRPS precipitation and SRTM-derived terrain on the 61x65 Boyaca grid, 518 monthly steps from January 1982 to February 2025 |
| `anchorbench-v1.3.0-checkpoints.zip` | trained weights. These are not needed to check any reported number; they are what a reader needs to run the models rather than to verify the results |
| `MANIFEST.sha256` | SHA-256 of each file above |

## Reconstructing the tree

```
unzip anchorbench-v1.3.0-code.zip                    -d anchorbench
unzip anchorbench-v1.3.0-predictions-and-metrics.zip -d anchorbench
unzip anchorbench-v1.3.0-checkpoints.zip             -d anchorbench     # optional
mkdir -p anchorbench/notebooks/data/output
mv complete_dataset_*.nc anchorbench/notebooks/data/output/
cd anchorbench
python -m pip install -r requirements-lock.txt
```

The zips share one directory layout, so they unpack over each other into a single
tree that matches the paths the scripts open.

## Verifying the reported numbers

Every analysis runs on CPU from the files above and prints its own conclusions. No
retraining is required, and none of these needs a GPU.

```
python models/scripts/figures/benchmark/naive_baselines.py        # zero-cost references
python models/scripts/analysis/anchoring_protocol.py --case-study # the same, all origins
python models/scripts/figures/benchmark/fusion_decomposition.py   # calibration vs combination
python models/scripts/analysis/fusion_purged_holdout.py           # the out-of-sample estimate
python models/scripts/analysis/window_overlap_audit.py            # why the blocked folds are not
python models/scripts/figures/analysis/stratified_comparison.py   # per-cell scores by stratum
python models/scripts/analysis/statistical_tests_audit.py         # tests against their own definitions
python models/scripts/analysis/factorial_blocked_analysis.py      # the factorial as a blocked design
python models/scripts/analysis/graph_structure_diagnostic.py      # what the graph connects
python models/scripts/analysis/graph_leakage_audit.py             # the corrected correlation edges
```

Their stdout is committed under `models/provenance/`, so a run that differs from the
record is a discrepancy worth reporting to us.

The eight-regime anchoring step reads global CHIRPS remotely and is the one analysis
that needs a network connection:

```
python models/scripts/analysis/anchoring_protocol.py --regions --csv out.csv
```

## The check that ties the manuscript to these files

```
python models/scripts/analysis/manuscript_numbers_audit.py
```

This binds each reported quantity to the record that produces it and exits non-zero
if they disagree. It also fails when a quantity is stated two ways, when a claim
withdrawn in one section is left standing in another, and when a table's summary row
is not the aggregate of the rows above it. It needs the manuscript sources, which are
not part of this record; it is included because it documents which numbers were
checked and against what.

## Two arrays for one model

Some directories hold more than one prediction array for what the manuscript calls
one model, written at different times. The manuscript reads the later one in every
case, and both are shipped so the difference can be inspected rather than taken on
trust:

| model, seed 42 | superseded | used in the manuscript |
|---|---|---|
| GNN-TAT-GAT | `V4_GNN_TAT_Models/map_exports/...`, pre-correction graph, pooled R2 0.597 | `V4_GNN_TAT_Models/SEED42/map_exports/...`, pooled R2 0.389 |
| Late Fusion | `V10_Late_Fusion/`, pooled R2 0.668 | `V10_Late_Fusion/SEED42/`, pooled R2 0.672 |

`models/provenance/investigation_headline_r2_2026-08-03.md` records why the graph
results changed and what was ruled out on the way. Note that the protocol is called
CeilBench in the older provenance documents; it was renamed to AnchorBench when the
manuscript stopped claiming a bound on predictability, and the documents are left as
written because they are dated records.

## Licence

MIT for the code, CC-BY-4.0 for the derived data.

The earlier wording restricted the derived feature set to "reproduction of the
reported analyses", which is a use restriction and does not conform to the Open
Source Definition that GMD requires of a deposit its papers depend on. It also
attached to the one file every CPU analysis reads, so the reproducibility claim
rested on a file nobody was licensed to reuse. CHIRPS v2.0 and SRTM both permit
redistribution of derived products, so the restriction bought nothing and is
removed.
