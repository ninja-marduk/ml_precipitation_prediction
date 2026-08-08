"""Re-analyse the feature-by-architecture factorial as a blocked design.

Two problems with the analysis in the manuscript, both raised in review.

First, it contradicts the paper's own stated method. Section 3.8 adopts
non-parametric tests throughout on the grounds that the sample size "precludes
reliable normality assumptions", and the factorial is then analysed by two-way
ANOVA with Tukey HSD on eighteen cells, justified by a Shapiro-Wilk test on
eighteen residuals, where that test has almost no power to reject.

Second, and more consequential, the eighteen cells are not independent. The design
is fully crossed: the same three seeds run every variant and every feature bundle.
Seed is a blocking factor, and ignoring it inflates the error term, which makes the
feature main effect look weaker or stronger than it is depending on how the seeds
happen to fall. The same mistake appears wherever the manuscript compares a
difference against a marginal standard deviation: for a crossed design the
relevant dispersion is that of the per-seed difference.

This script analyses the factorial as a randomised complete block design with seed
as the block, checks the result against a permutation test that permutes only
within blocks, and recomputes every configuration comparison as a paired
difference.

Usage: python models/scripts/analysis/factorial_blocked_analysis.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FACTORIAL = ROOT / "models" / "output" / "V4_GNN_TAT_Models" / \
    "_archive_2026-04_leaked_graph" / "metrics_factorial_consolidated.csv"
N_PERM = 20000
RNG = np.random.default_rng(42)


def cell_means(df):
    """One response per (seed, variant, feature): the mean R2 over horizons."""
    g = df.groupby(["seed", "variant", "feat"])["R^2"].mean().reset_index()
    return g.rename(columns={"R^2": "y"})


def rcbd(g):
    """Randomised complete block ANOVA: block = seed, factors = feature, variant."""
    y = g["y"].values
    n = len(y)
    gm = y.mean()

    def ss(labels):
        s = 0.0
        for lv in np.unique(labels):
            m = labels == lv
            s += m.sum() * (y[m].mean() - gm) ** 2
        return s

    seeds = g["seed"].values
    feats = g["feat"].values
    vars_ = g["variant"].values

    ss_tot = ((y - gm) ** 2).sum()
    ss_blk = ss(seeds)
    ss_f = ss(feats)
    ss_v = ss(vars_)

    # interaction: cell means of feature x variant, pooled over blocks
    ss_fv = 0.0
    for f in np.unique(feats):
        for v in np.unique(vars_):
            m = (feats == f) & (vars_ == v)
            if m.any():
                ss_fv += m.sum() * (y[m].mean() - gm) ** 2
    ss_int = ss_fv - ss_f - ss_v
    ss_err = ss_tot - ss_blk - ss_f - ss_v - ss_int

    df_blk = len(np.unique(seeds)) - 1
    df_f = len(np.unique(feats)) - 1
    df_v = len(np.unique(vars_)) - 1
    df_int = df_f * df_v
    df_err = n - 1 - df_blk - df_f - df_v - df_int
    ms_err = ss_err / df_err if df_err > 0 else np.nan

    from scipy import stats
    out = {}
    for name, s, d in (("block (seed)", ss_blk, df_blk),
                       ("feature bundle", ss_f, df_f),
                       ("variant", ss_v, df_v),
                       ("feature x variant", ss_int, df_int)):
        F = (s / d) / ms_err if (d > 0 and ms_err > 0) else np.nan
        p = 1 - stats.f.cdf(F, d, df_err) if np.isfinite(F) else np.nan
        out[name] = (s, d, F, p)
    out["_err"] = (ss_err, df_err, np.nan, np.nan)
    out["_tot"] = (ss_tot, n - 1, np.nan, np.nan)
    return out


def perm_within_block(g, factor, n_perm=N_PERM):
    """Permutation test for a factor, shuffling labels only inside each seed."""
    y = g["y"].values
    seeds = g["seed"].values
    lab = g[factor].values

    def stat(labels):
        gm = y.mean()
        s = 0.0
        for lv in np.unique(labels):
            m = labels == lv
            s += m.sum() * (y[m].mean() - gm) ** 2
        return s

    obs = stat(lab)
    hits = 0
    perm = lab.copy()
    for _ in range(n_perm):
        for sd in np.unique(seeds):
            m = seeds == sd
            perm[m] = RNG.permutation(lab[m])
        if stat(perm) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    df = pd.read_csv(FACTORIAL)
    g = cell_means(df)
    print(f"{len(g)} cells: {g['seed'].nunique()} seeds x "
          f"{g['variant'].nunique()} variants x {g['feat'].nunique()} feature bundles\n")

    print("=" * 74)
    print("RANDOMISED COMPLETE BLOCK ANOVA  (block = seed)")
    print("=" * 74)
    res = rcbd(g)
    print(f"{'source':<22}{'SS':>10}{'df':>5}{'F':>9}{'p':>9}")
    print("-" * 74)
    for k in ("block (seed)", "feature bundle", "variant", "feature x variant"):
        s, d, F, p = res[k]
        print(f"{k:<22}{s:>10.4f}{d:>5}{F:>9.2f}{p:>9.4f}")
    for k, lab in (("_err", "residual"), ("_tot", "total")):
        s, d, _, _ = res[k]
        print(f"{lab:<22}{s:>10.4f}{d:>5}")

    blk_share = res["block (seed)"][0] / res["_tot"][0]
    print(f"\nthe seed block absorbs {100*blk_share:.1f}% of the total sum of squares.")
    print("Treating the eighteen cells as independent puts that variance into the")
    print("error term, which is what the unblocked analysis in the manuscript does.")

    print()
    print("=" * 74)
    print("PERMUTATION TEST, shuffling only within seed")
    print("=" * 74)
    for f in ("feat", "variant"):
        p = perm_within_block(g, f)
        print(f"  {f:<10} p = {p:.4f}   ({N_PERM:,} permutations)")
    print("This makes no distributional assumption, which is what Section 3.8 said")
    print("the sample size required.")

    print()
    print("=" * 74)
    print("PAIRED COMPARISONS BETWEEN CONFIGURATIONS")
    print("=" * 74)
    from scipy import stats
    key = g.assign(cfg=g["variant"] + "/" + g["feat"])
    wide = key.pivot(index="seed", columns="cfg", values="y")
    cfgs = list(wide.columns)
    print(f"{'comparison':<30}{'paired mean':>13}{'paired sd':>11}"
          f"{'marginal sd':>13}{'t':>7}{'p':>8}")
    print("-" * 84)
    rows = []
    for a, b in combinations(cfgs, 2):
        d = (wide[a] - wide[b]).values
        m, sd = d.mean(), d.std(ddof=1)
        marg = np.sqrt(wide[a].std(ddof=1) ** 2 + wide[b].std(ddof=1) ** 2)
        t = m / (sd / np.sqrt(len(d))) if sd > 0 else np.nan
        p = 2 * (1 - stats.t.cdf(abs(t), len(d) - 1)) if np.isfinite(t) else np.nan
        rows.append((abs(t), a, b, m, sd, marg, t, p))
    for _, a, b, m, sd, marg, t, p in sorted(rows, reverse=True)[:8]:
        print(f"{a+' vs '+b:<30}{m:>+13.4f}{sd:>11.4f}{marg:>13.4f}{t:>7.2f}{p:>8.4f}")

    n_tighter = sum(1 for _, _, _, _, sd, marg, _, _ in rows if sd < marg)
    print(f"\nin {n_tighter} of {len(rows)} comparisons the paired dispersion is smaller")
    print("than the marginal one. Using the marginal spread as the evidential")
    print("threshold, as the manuscript does, is therefore conservative in most")
    print("cases and arbitrary in all of them: the paired quantity is the one the")
    print("crossed design supplies.")


if __name__ == "__main__":
    main()
