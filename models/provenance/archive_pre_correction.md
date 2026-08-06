# V4 GNN-TAT metrics predating the graph correction

Snapshot taken 2026-08-02, before the corrected retrain was copied in. Every
`.csv` and `.json` under `models/output/V4_GNN_TAT_Models/` at that moment, with
the original directory layout preserved.

These are the numbers the manuscript was written against, and they are **not
usable as published results**. Two defects apply to them:

1. **Target leakage in the graph.** Correlation edges were estimated over the
   full record, so the fixed spatial prior encoded the held-out months. Fixed in
   notebook v4.1.
2. **A graph twice the size, by accident.** The edge budget was applied only
   above a 1,000,000-edge trigger. The leaked graph had 998,715 edges and slipped
   under it; the corrected graph crosses it and was clamped to 500,000. The
   budget is now an explicit hyperparameter applied unconditionally (v4.7), so
   both effects are separable in principle but are confounded in these files.

Kept because the manuscript's current tables were computed from them and the
change has to be traceable, not because they should be cited.

Superseded by the run of 2026-08-02 (`SEED{42,123,456}/`, GAT + BASIC, H=12,
500,000 edges), which is recorded in `run_corrected_graph_2026-08-02.md`.
