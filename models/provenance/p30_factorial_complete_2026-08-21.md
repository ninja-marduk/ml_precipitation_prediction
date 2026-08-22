# The corrected factorial, complete: 18 cells, one protocol

Six configurations by three seeds, all on the corrected graph at patience 30,
in `V4_GNN_TAT_Models_p30`. About 29 A100-hours across six sessions. This replaces
the archived factorial, whose cells were produced by a pipeline the retrain shows
cannot be compared with this one.

## The cells

| cell | seed 42 | seed 123 | seed 456 | mean | s.d. | inflation |
|------|--------:|---------:|---------:|-----:|-----:|----------:|
| GAT/PAFC | 0.5374 | 0.4846 | 0.5083 | **0.5101** | 0.0264 | 0.0375 |
| SAGE/PAFC | 0.4715 | 0.4366 | 0.6134 | **0.5072** | 0.0936 | 0.1140 |
| GCN/PAFC | 0.4869 | 0.4251 | 0.5479 | 0.4866 | 0.0614 | 0.0436 |
| GAT/BASIC | 0.3882 | 0.4952 | 0.4552 | 0.4462 | 0.0541 | 0.0470 |
| GCN/BASIC | 0.4387 | 0.3652 | 0.4440 | 0.4160 | 0.0441 | 0.0422 |
| SAGE/BASIC | 0.4480 | 0.4483 | 0.3083 | 0.4015 | 0.0807 | 0.0507 |

All three PAFC cells sit above all three BASIC cells on the expectation.

## What resolves

Randomised complete block ANOVA, seed as block, n=18:

| source | SS | df | F | p |
|--------|---:|---:|--:|--:|
| block (seed) | 0.0041 | 2 | 0.46 | 0.646 |
| feature bundle | 0.0289 | 1 | **6.40** | **0.030** |
| variant | 0.0026 | 2 | 0.29 | 0.756 |
| bundle x variant | 0.0015 | 2 | 0.17 | 0.849 |
| residual | 0.0451 | 10 | | |

A permutation test shuffling only within seed agrees: bundle p=0.008, variant
p=0.835, over 20,000 permutations.

The archived factorial gave the same qualitative answer, bundle p=0.015 and variant
p=0.271, so this conclusion is one of the few that survives the pipeline change
intact. It is now established on the pipeline the paper actually released.

Of the fifteen pairwise comparisons, **one** clears |t| > 4.303: GAT/PAFC over
GCN/BASIC, +0.0941 with a paired spread of 0.0279, t=5.85. The next closest is
GCN/BASIC against GCN/PAFC at t=-4.17, just under the bar.

## What does not resolve, and this changes a claim in the manuscript

**GAT with PAFC is not the strongest configuration.** It leads SAGE with PAFC by
0.0029, with a paired spread of 0.0940 across the seeds: t=0.05. The two are
indistinguishable, and the manuscript's practical recommendation that GAT with the
precipitation-lag bundle is strongest, and its supporting parenthetical about being
the most stable, cannot be carried over. GAT/PAFC is in fact the tightest cell in
the design at +/-0.0264, so the stability half of the claim survives; the ranking
half does not.

The variant effect is absent by every test available: the ANOVA at p=0.756, the
permutation at p=0.835, and no pairwise comparison among operators within either
bundle. Which message-passing operator is used cannot be shown to matter here.

## Single-run inflation is worse than the archived factorial said

The median inflation from quoting a configuration's best seed instead of its
expectation is 0.045 on the corrected factorial, against 0.026 on the archived one,
and the worst cell is SAGE with PAFC at 0.114 rather than GAT with BASIC at 0.076.
The manuscript's argument about single-run reporting is therefore understated by
its own current numbers.

## The instability is not where two seeds said it was

SAGE with BASIC scored 0.4480 and 0.4483 at the first two seeds and 0.3083 at the
third: the tightest cell in the design on two seeds becomes one of the widest on
three. SAGE with PAFC does the same in the other direction, 0.4715 and 0.4366
before 0.6134. Any statement about which configuration is stable rests on n=3 and
should be read with the same caution the paper asks of everyone else.

## Reproducibility

GAT with BASIC over three seeds gives 0.4462 +/- 0.0541 here, against
0.4462 +/- 0.0542 for the earlier corrected run at patience 15, on a different Colab
account and across a change of stopping rule. Every cell peaked by epoch 6; patience
30 was never binding, and the archived cell that peaked at epoch 25 peaks at epoch 2
on this pipeline.
