# Ablation: leaked graph (2026-08-02)

Diagnostic run. **These numbers must never be reported as model performance.**

Identical to `../V4_GNN_TAT_Models/` in every respect except one: correlation
edges are estimated over the full record rather than the training period, which
restores the target leakage that notebook v4.1 removed. Same seeds, same budget,
same chunk, same loop, same 500,000 edges (15,678,252 before the budget, against
15,673,588 for the corrected graph).

## Result: the leakage had no measurable effect

| Seed | corrected mean / peak | leaked mean / peak | delta mean | delta peak |
|------|-----------------------|--------------------|------------|------------|
| 42   | 0.3882 / 0.4398 | 0.3874 / 0.4393 | -0.0008 | -0.0005 |
| 123  | 0.4956 / 0.5342 | 0.4942 / 0.5329 | -0.0013 | -0.0013 |
| 456  | 0.4549 / 0.5570 | 0.4509 / 0.5542 | -0.0040 | -0.0028 |
| **mean** | **0.4462 / 0.5103** | **0.4442 / 0.5088** | **-0.0021** | **-0.0015** |

The effect is negative, an order of magnitude smaller than the seed-to-seed
standard deviation of 0.054, and consistent in sign across all three seeds only
in the sense that it is indistinguishable from zero. The leaked graph is if
anything marginally worse.

This is not an argument for keeping the leakage. It is a defect in the
methodology and stays fixed. What it establishes is narrower and more useful:
**the drop from the archived R2 = 0.628 to the corrected 0.510 is not caused by
removing the leakage.**

Consistent with the CPU audit
(`models/scripts/analysis/graph_structure_diagnostic.py`), which found that
correlation dominates the selection of 81.7% of retained edges but that the two
correlation estimates disagree on only 10,666 of 500,000 edges, 2.13%.

## What is still open

The archived GAT/BASIC run reached `best_val_loss = 0.4553` at epoch 3 over 53
epochs. The three current seeds reach 0.6879, 0.5599 and 0.6293. The gap is in
the loss itself, not only in the derived R2, so it is a training-trajectory
difference rather than an evaluation artefact.

The archived run used `batch_size = 2`; the current runs use 4. Because the model
peaks at epoch 3 to 5, the number of optimizer steps taken before that peak is
decisive, and batch_size 2 takes 172 steps per epoch against 86, which is exactly
double by epoch 3. That is the next thing to test (`ABLATION = 'batch2'`).
