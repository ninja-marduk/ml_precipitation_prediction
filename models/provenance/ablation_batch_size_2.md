# Ablation: batch_size 2 (2026-08-03)

Diagnostic run. Corrected graph, `batch_size = 2` instead of 4, which is what the
archived run used. Everything else identical.

## Result: reproduces the archived training, not the archived skill

The training curve matches the archived run almost exactly:

| epoch | archived train | batch2 seed 42 train |
|-------|----------------|----------------------|
| 1 | 0.5904 | 0.5886 |
| 2 | 0.3687 | 0.3647 |
| 3 | 0.3079 | 0.3095 |
| 4 | 0.2764 | 0.2764 |
| 5 | 0.2657 | 0.2643 |
| 6 | 0.2581 | 0.2579 |

The validation does not:

| variant | R2 mean | R2 peak |
|---------|---------|---------|
| corrected, batch 4 | 0.4462 +/- 0.054 | 0.5103 +/- 0.062 |
| leaked graph, batch 4 | 0.4442 +/- 0.054 | 0.5088 +/- 0.061 |
| **corrected, batch 2** | **0.3476 +/- 0.024** | **0.4288 +/- 0.030** |
| archived, single run | 0.5963 | 0.6284 |

batch_size 2 is worse, not better. Ruled out.

## What has been ruled out so far

| candidate | test | effect |
|-----------|------|--------|
| graph target leakage | paired 3-seed ablation | -0.002 R2 |
| graph size | CPU audit of the budget | none, 500,000 edges in both |
| validation data | byte comparison of `targets.npy` | identical arrays |
| validation batch averaging | recomputed from saved windows | -0.9% |
| batch_size | this run | -0.099 R2, wrong direction |

## What the gap actually is

Against the same targets, the archived predictions have standard deviation 93.7
against the observed 133.0; the corrected ones have 67.2. The two prediction sets
correlate at 0.927 with each other, so they agree on the pattern and disagree on
the amplitude.

Decomposing per horizon, with an in-sample affine recalibration applied equally
to all variants:

| variant | correlation | sd / sd_obs | R2 raw | R2 recalibrated |
|---------|-------------|-------------|--------|-----------------|
| archived | 0.809 | 0.704 | 0.5963 | 0.6553 |
| corrected batch 4 | 0.746 | 0.506 | 0.3882 | 0.5567 |
| corrected batch 2 | 0.724 | 0.520 | 0.3418 | 0.5238 |

Roughly half the gap is under-dispersion, which recalibration recovers. The other
half is a genuine correlation difference: the archived model has more skill, not
just better scaling.

## Remaining candidates

1. **TF32**, enabled in v4.6, which truncates the matmul mantissa from 24 bits to
   10. The last systematic difference between the pipelines. Testable with
   `ABLATION = 'no_tf32'`.
2. **Run-to-run variance plus selection.** The archived run was unseeded and is a
   single draw. The model early-stops at epoch 3 to 6 against a 33-window
   validation set whose loss swings between 0.45 and 0.81 from epoch to epoch. If
   an unknown number of development runs were made and the best kept, 0.628 is
   the maximum of that set rather than an expectation. This cannot be falsified
   without the development history, but more seeds would show whether 0.628 lies
   inside the distribution.

Note that the conclusion of the manuscript does not depend on which is right: the
per-cell climatology scores 0.739, above the archived 0.628 as well as the
corrected 0.510.
