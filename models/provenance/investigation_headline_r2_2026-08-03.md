# Why R2 = 0.628 could not be reproduced, and what the number should be

Closed 2026-08-03 after four GPU experiments and three CPU audits. No further
compute is required: the question is answered from data already in hand.

## The answer

**0.628 is the best horizon of the best seed of the archived run.** It is a
maximum over 36 values (3 seeds x 12 horizons), reported as though it were the
result. The archived data always contained three seeds; only one of them was ever
quoted.

Archived GAT + BASIC, from `_archive_2026-04_leaked_graph/metrics_factorial_consolidated.csv`:

| seed | mean R2 | peak R2 |
|------|---------|---------|
| 42 | 0.5963 | **0.6284** |
| 123 | 0.4624 | 0.5343 |
| 456 | 0.3852 | 0.4949 |
| **across seeds** | **0.4813 +/- 0.107** | **0.5525 +/- 0.069** |

Corrected pipeline, same three seeds: **0.4462 +/- 0.054** mean, **0.5103 +/- 0.062** peak.

The difference between the archived and corrected pipelines is **-0.035 mean and
-0.042 peak**, smaller than the seed standard deviation of either. Paired by seed
it is not even consistent in sign:

| seed | peak archived | peak corrected | delta |
|------|---------------|----------------|-------|
| 42 | 0.6284 | 0.4398 | -0.189 |
| 123 | 0.5343 | 0.5342 | -0.000 |
| 456 | 0.4949 | 0.5570 | **+0.062** |

Seed 42 was a fortunate draw in the archived run and an unfortunate one in the
corrected run. That is the entire effect.

## What was ruled out on the way

| candidate | how it was tested | effect |
|-----------|-------------------|--------|
| graph target leakage | paired 3-seed ablation, full-record correlation | **-0.002** |
| graph size | CPU rebuild of the adjacency, budget analysis | none; 500,000 edges in both |
| validation data | byte comparison of `targets.npy` | identical arrays |
| validation batch averaging | recomputed from per-window losses | -0.9% |
| batch_size (2 vs 4) | 3-seed ablation | -0.099, wrong direction |
| TF32 matmuls | 3-seed ablation | 0.000 |
| run-to-run determinism | batch2 repeated unchanged | reproduces to 4 decimals |

Twelve runs across four configurations, peak R2 between 0.401 and 0.557. None
reaches 0.628, which sits 2.3 standard deviations above their mean. Given a seed
the pipeline is exactly reproducible, so the spread is model variance, not noise
in the harness.

## Two consequences that change the science, not just the number

**1. The best configuration is not the one reported.** Ranking the archived
factorial by multi-seed peak R2 rather than by best single seed:

| variant | features | mean R2 | peak R2 | best single seed |
|---------|----------|---------|---------|------------------|
| GAT | PAFC | 0.5440 +/- 0.019 | **0.6052 +/- 0.009** | 0.6150 |
| SAGE | PAFC | 0.5340 +/- 0.031 | 0.5864 +/- 0.021 | 0.6078 |
| GCN | PAFC | 0.5377 +/- 0.062 | 0.5803 +/- 0.060 | 0.6248 |
| GCN | BASIC | 0.5020 +/- 0.028 | 0.5545 +/- 0.021 | 0.5746 |
| GAT | BASIC | 0.4813 +/- 0.107 | 0.5525 +/- 0.069 | **0.6284** |
| SAGE | BASIC | 0.4045 +/- 0.027 | 0.4607 +/- 0.028 | 0.4864 |

GAT + BASIC, the configuration the manuscript reports as best, is **fifth of six**
on the multi-seed peak. It won the single-seed comparison because it has by far
the widest seed spread (0.107 against 0.019 to 0.062 for the others), so its
maximum is the highest even though its expectation is not.

The configuration that actually wins is **GAT + PAFC**, and it wins with the
tightest spread in the table. PAFC beats BASIC for all three graph variants, which
is both a cleaner result and an interpretable one: the precipitation lags carry
information the topographic features do not.

**2. This is the manuscript's own thesis, demonstrated on the manuscript's own
data.** CeilBench argues that model-versus-model tables built from single runs
cannot separate architecture from initialisation. Here a single seed produced a
number 2.3 standard deviations above the mean, it entered a published abstract as
evidence that one architecture matches another, and it inverted the ranking of the
factorial. The paper does not need a hypothetical example of the failure it
describes.

## What should be reported

- **GAT + BASIC, corrected graph, three seeds: R2 = 0.4462 +/- 0.0542 mean,
  0.5103 +/- 0.0621 peak.** Never a bare 0.628.
- The per-cell climatology scores **0.739**. It was above the archived number and
  is further above the corrected one, so the predictability-ceiling conclusion is
  unchanged and strengthened.
- Compute, measured: A100-SXM4-80GB, **27.05 GB peak, 322 s/epoch, 97 to 107 min
  per seed**, against the 2.1 GB and 28 min currently in the table.

## What this means for the published comparative paper

The 0.628 in `10.3390/hydrology13030098` is not fabricated and is not the product
of the leakage, which costs 0.002. It is the best seed's best horizon, selected
and reported as the result. That is a reporting practice, not a data error, and
it is the same practice the present manuscript sets out to correct.

Whether that warrants any communication with the journal is the authors' call.
The defensible minimum is that the new manuscript reports multi-seed values
throughout and does not repeat the single-seed figure.

## Coverage still missing

The archived factorial covers GAT, GCN and SAGE across BASIC and PAFC, with three
seeds each: 216 rows, all usable, since the leakage effect is measured at -0.002.
**KCE is absent from it** and exists only as single-run root-level numbers. The
corrected retrain covers GAT + BASIC only.

With no GPU budget remaining, the honest presentation is the archived three-seed
factorial for the architecture comparison, the corrected three-seed run for
GAT + BASIC, the measured leakage effect to justify using the former, and KCE
either dropped or marked explicitly as single-run.
