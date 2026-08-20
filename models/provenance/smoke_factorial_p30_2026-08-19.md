# Rehearsal of the p30 factorial, 2026-08-19

`smoke_p30_42_basic`, the light twin of the first chunk: seed 42, BASIC, the three
graph operators, on the 5x5 centre subset for two epochs. It exists to find a broken
path before 59 hours of GPU time are committed, not to produce a result.

## What it confirms

The configuration reaches the training loop as the plan declares it. From
`experiment_state_v4.json` in the smoke tree:

| setting | recorded |
|---|---|
| light_mode / grid | True, 5x5, 25 nodes, 600 edges |
| patience | 30 |
| lr_patience | 7 |
| epochs | 2 |
| batch_size | 4 |
| gnn_chunk_size | 30 |
| out_root | `.../V4_GNN_TAT_Models_smoke` |

All three operators trained, exported and wrote metrics:

| variant | epochs | parameters | s/epoch | peak GPU |
|---------|--------|-----------|---------|----------|
| GAT | 2 | 97,932 | 10.61 | 0.15 GB |
| GCN | 2 | 97,676 | 5.78 | 0.15 GB |
| SAGE | 2 | 105,868 | 4.75 | 0.15 GB |

Parameter counts match the full-grid runs exactly, which is the expected result: the
model's weights do not depend on the number of nodes. Prediction and target arrays
come out at (33, 12, 5, 5, 1), so the 33 validation windows survive: light mode
subsets space and leaves the time axis alone.

Two earlier fixes are confirmed in the record rather than in the source. The chunk
size is 30, not the 60 the old grid-size heuristic would have chosen for a 5x5 grid,
so the probe model now reads CONFIG. And `lr_patience` stayed at 7 while `patience`
moved to 30, so raising the stopping rule did not drag the learning-rate schedule
with it.

## What it does not tell us

The scores. Two epochs on 25 nodes give R2 between 0.21 and 0.36 across horizons for
GAT, and those numbers mean nothing: they are a rehearsal of the plumbing.

The cost, and the memory. Peak GPU is 0.15 GB against the 27.05 GB of a full run,
and the graph holds 600 edges, so the 500,000-edge budget never binds. Nothing here
would have caught an out-of-memory failure at full scale.

## What it suggests about the cost, weakly

GCN and SAGE run at 0.54 and 0.45 of GAT's rate here. If those ratios held at full
scale the eighteen runs would come to about 40 A100-hours rather than the 59 the
GAT-rate bound gives. They probably do not hold: at 25 nodes the epoch is dominated
by fixed overhead, while at 3,965 nodes and 500,000 edges it is dominated by message
passing, where the three operators are closer together. The bound stays 59 hours and
the true figure is somewhere below it. The first full chunk will settle it, and its
history files carry `sec_per_epoch_mean` for each operator.
