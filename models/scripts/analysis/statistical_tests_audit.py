"""Audit the reported significance tests against their own definitions.

Three defects were raised in review and all three reproduce from the table alone,
without needing the underlying data:

  1. Three rows labelled "Wilcoxon" report statistics of 2.10, 0.85 and 1.23. The
     Wilcoxon signed-rank statistic W is a non-negative integer bounded above by
     n(n+1)/2, which is 78 for n=12. Those three values are not W. The other two
     Wilcoxon rows report 0.00 and 21.00, which are admissible, so one table
     mixes two different statistics under one label.
  2. The Nemenyi critical difference is reported as 2.83. CD = q_alpha *
     sqrt(k(k+1)/(6N)) evaluates to 1.83 for the k=8, N=33 of the Friedman row
     above it, and to 0.58 for the k=3 the caption implies. Neither is 2.83.
  3. The manuscript declares nine pairwise tests; the table contains eight. The
     Holm accounting downstream inherits the error.

This script recomputes what can be recomputed from the reported values and states
what cannot. Rows whose statistic is inadmissible cannot be repaired here: they
have to be recomputed from the paired metric files or removed.

Usage: python models/scripts/analysis/statistical_tests_audit.py
"""
from __future__ import annotations

import numpy as np

# Studentised range statistic divided by sqrt(2), alpha = 0.05, as used for Nemenyi
Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
              7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

# (comparison, test, n, reported statistic, reported p, reported verdict)
ROWS = [
    ("GNN-TAT vs ConvLSTM (RMSE)",  "Mann-Whitney", 12, 57.00, 0.015, "GNN-TAT better"),
    ("GNN-TAT vs ConvLSTM (R2)",    "Mann-Whitney", 12, 72.50, 0.089, "Not significant"),
    ("FNO vs FNO-ConvLSTM",         "Wilcoxon",     12,  0.00, 0.0009, "Hybrid significant"),
    ("KCE vs BASIC (GNN)",          "Wilcoxon",     12,  2.10, 0.036, "KCE better"),
    ("KCE vs BASIC (ConvLSTM)",     "Wilcoxon",     12,  0.85, 0.395, "Not significant"),
    ("PAFC vs BASIC (all)",         "Wilcoxon",     36,  1.23, 0.218, "Not significant"),
    ("Stacking vs Best Individual", "Mann-Whitney", 12,  0.00, 0.0009, "Failure confirmed"),
    ("Stratified vs GNN-TAT",       "Wilcoxon",     12, 21.00, 0.907, "No improvement"),
]


def admissible(test, n, stat):
    """Range check against the statistic's own definition."""
    if test == "Wilcoxon":
        hi = n * (n + 1) / 2
        ok = float(stat).is_integer() and 0 <= stat <= hi
        return ok, f"W integer in [0, {hi:.0f}]"
    if test == "Mann-Whitney":
        hi = n * n
        ok = 0 <= stat <= hi
        return ok, f"U in [0, {hi:.0f}]"
    return True, ""


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def nemenyi_cd(k, n, alpha_q=None):
    q = alpha_q if alpha_q is not None else Q_ALPHA_05.get(k)
    if q is None:
        return None
    return q * np.sqrt(k * (k + 1) / (6 * n))


def main():
    print("=" * 78)
    print("1. ARE THE REPORTED STATISTICS ADMISSIBLE?")
    print("=" * 78)
    print(f"{'comparison':<30}{'test':<14}{'stat':>8}   verdict")
    print("-" * 78)
    bad = []
    for name, test, n, stat, p, res in ROWS:
        ok, rule = admissible(test, n, stat)
        if not ok:
            bad.append(name)
        print(f"{name:<30}{test:<14}{stat:>8.2f}   "
              f"{'ok' if ok else 'INADMISSIBLE (' + rule + ')'}")
    print(f"\n{len(bad)} of {len(ROWS)} rows report a value the named test cannot produce:")
    for b in bad:
        print(f"  - {b}")
    print("These are z or t statistics reported under a W label, or a different test.")
    print("They must be recomputed from the paired metric files, or the rows removed.")

    print()
    print("=" * 78)
    print("2. HOLM-BONFERRONI OVER THE PAIRWISE FAMILY")
    print("=" * 78)
    names = [r[0] for r in ROWS]
    praw = [r[4] for r in ROWS]
    padj = holm(praw)
    print(f"family size: {len(ROWS)} pairwise tests "
          f"(the manuscript declares 9; the table contains {len(ROWS)})\n")
    print(f"{'comparison':<30}{'raw p':>9}{'Holm p':>10}   {'survives':<10}reported verdict")
    print("-" * 88)
    flipped = []
    for (name, _, _, _, p, res), a in zip(ROWS, padj):
        surv = a < 0.05
        claims = res not in ("Not significant", "No improvement")
        if claims and not surv:
            flipped.append((name, res))
        print(f"{name:<30}{p:>9.4f}{a:>10.4f}   {'yes' if surv else 'no':<10}{res}")
    print(f"\n{sum(padj < 0.05)} of {len(ROWS)} survive Holm.")
    if flipped:
        print("\nRows whose Result column asserts an effect that does not survive:")
        for n, r in flipped:
            print(f"  - {n:<30} says \"{r}\"")
        print("The Result column must state the post-correction verdict.")

    print()
    print("=" * 78)
    print("3. NEMENYI CRITICAL DIFFERENCE")
    print("=" * 78)
    print(f"reported: CD = 2.83, attached to p = 0.052\n")
    print(f"{'k':>4}{'N':>6}{'q_0.05':>9}{'CD':>9}   reading")
    print("-" * 62)
    for k, n, why in ((8, 33, "the k=8 Friedman row above it"),
                      (3, 33, "the 'top 3' the caption implies"),
                      (8, 12, "if N were the 12 configurations"),
                      (3, 12, "k=3 over 12 configurations")):
        cd = nemenyi_cd(k, n)
        print(f"{k:>4}{n:>6}{Q_ALPHA_05[k]:>9.3f}{cd:>9.3f}   {why}")
    print("\nNone reaches 2.83. State k and N explicitly and report the CD they give.")
    print("A critical-difference procedure also yields a decision, not a single p:")
    print("the p = 0.052 attached to it does not follow from it and should be dropped")
    print("or replaced by per-comparison adjusted p-values.")

    print()
    print("=" * 78)
    print("4. WHAT TO WRITE")
    print("=" * 78)
    print("- recompute the three inadmissible statistics from the paired metric files")
    print("  and report W itself, or drop those rows;")
    print(f"- correct the declared family size from 9 to {len(ROWS)} in Section 3.8 and")
    print("  in the Conclusions;")
    print("- add a Holm-adjusted column and set Result from it, which flips")
    print(f"  {len(flipped)} row(s);")
    print("- give k and N for the Nemenyi CD and quote the value they produce.")


if __name__ == "__main__":
    main()
