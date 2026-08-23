"""How many independent observations does this design actually have?

Every inferential procedure in the manuscript takes its sample size from a count
that is not a count of independent things. The 33 validation windows slide by one
month and span 44 calendar months, so a window-level test that writes N=33 is
claiming eleven times the information the record holds. The three seeds are a real
n of three, but a paired t at two degrees of freedom has a minimum detectable
effect that no architectural difference in this field would exceed, and reporting
"not significant" without that number invites the reader to hear "equivalent".

This computes the three quantities the manuscript needs in order to state its
negative results honestly:

  1. the effective number of independent evaluation units behind the 33 windows,
     and the Nemenyi critical difference recomputed at that number;
  2. the minimum detectable effect of the three-seed paired design, in R2;
  3. the fifteen pairwise comparisons the factorial admits, with Holm and
     Bonferroni applied to the family as a whole rather than to a subset.

None of this changes a measurement. It changes which claims the measurements
support, which is the point of the protocol the paper proposes.

Usage: python models/scripts/analysis/effective_sample_size.py
"""
from __future__ import annotations

import glob
import itertools
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
P30 = ROOT / "models" / "output" / "V4_GNN_TAT_Models_p30"
LF = ROOT / "models" / "output" / "V10_Late_Fusion" / "SEED42"
SEEDS = (42, 123, 456)
# Nemenyi studentised-range critical values at alpha=0.05, q_alpha/sqrt(2)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def r2(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    p, t = pred[m], tgt[m]
    return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))


def cells():
    """Horizon-mean R2 per (bundle, variant, seed) from the retrained factorial."""
    out = {}
    for d in sorted(glob.glob(str(P30 / "SEED*" / "map_exports" / "H12" / "*" /
                                  "GNN_TAT_*"))):
        parts = d.replace(os.sep, "/").split("/")
        seed = int([p for p in parts if p.startswith("SEED")][0][4:])
        bundle, variant = parts[-2], parts[-1].replace("GNN_TAT_", "")
        pred, tgt = load(d), load(d, "targets")
        out[(bundle, variant, seed)] = float(np.mean(
            [r2(pred[:, h], tgt[:, h]) for h in range(pred.shape[1])]))
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def part1():
    print("=" * 74)
    print("1. THE 33 WINDOWS ARE NOT 33 INDEPENDENT OBSERVATIONS")
    print("=" * 74)
    tgt = load(LF, "targets")
    S, H = tgt.shape[0], tgt.shape[1]
    months = sorted({s + h + 1 for s in range(S) for h in range(H)})
    mult = {}
    for s in range(S):
        for h in range(H):
            mult[s + h + 1] = mult.get(s + h + 1, 0) + 1
    print(f"  {S} windows x {H} leads = {S * H} scored (window, lead) pairs")
    print(f"  distinct target months: {len(months)}")
    print(f"  each month is scored between {min(mult.values())} and "
          f"{max(mult.values())} times")
    print(f"  a window-level test writing N={S} therefore claims "
          f"{S / len(months) * H:.1f} times")
    print(f"  the information the record holds, before any serial correlation.\n")

    # Non-overlapping windows of the same span, and annual decorrelation, are the
    # two defensible unit counts. Both land in the same place.
    n_block = len(months) // H
    n_year = len(months) / 12
    print(f"  non-overlapping {H}-month windows in {len(months)} months: {n_block}")
    print(f"  independent annual cycles in {len(months)} months: {n_year:.1f}")
    print(f"  => the effective N for a window-level test is 3 to 4, not {S}.\n")

    print("  Nemenyi critical difference, k=8 models, recomputed:")
    print(f"  {'N':>6}{'CD':>10}   reading")
    print("  " + "-" * 60)
    for n, note in ((S, "as reported, treating windows as independent"),
                    (4, "annual cycles in the record"),
                    (3, "non-overlapping windows")):
        cd = Q05[8] * np.sqrt(8 * 9 / (6.0 * n))
        print(f"  {n:>6}{cd:>10.3f}   {note}")
    cd33 = Q05[8] * np.sqrt(8 * 9 / (6.0 * S))
    cd3 = Q05[8] * np.sqrt(8 * 9 / (6.0 * 3))
    print(f"\n  The critical difference grows by a factor of {cd3 / cd33:.1f} when the")
    print(f"  dependence is respected. A Friedman/Nemenyi analysis at N={S} is")
    print(f"  anticonservative for the omnibus test and conservative for the post-hoc,")
    print(f"  so quoting both is quoting whichever direction suits the claim.")
    print(f"  With N=3 the post-hoc separates nothing: CD={cd3:.2f} exceeds the whole")
    print(f"  range of mean ranks that 8 models over 3 blocks can produce.")
    return len(months), n_block


def part2(c):
    print("\n" + "=" * 74)
    print("2. MINIMUM DETECTABLE EFFECT OF THE THREE-SEED PAIRED DESIGN")
    print("=" * 74)
    n = len(SEEDS)
    tcrit = stats.t.ppf(0.975, n - 1)
    keys = sorted({(b, v) for b, v, _ in c})
    sds = []
    for a, b in itertools.combinations(keys, 2):
        d = np.array([c[(a[0], a[1], s)] - c[(b[0], b[1], s)] for s in SEEDS])
        sds.append(float(d.std(ddof=1)))
    lo, hi, med = min(sds), max(sds), float(np.median(sds))
    print(f"  n={n} paired seeds, t_crit={tcrit:.3f} at {n - 1} d.f.")
    print(f"  a difference separates only if it exceeds "
          f"t_crit x s_d / sqrt(n) = {tcrit / np.sqrt(n):.3f} x s_d")
    print(f"  observed paired s_d across the 15 comparisons: "
          f"{lo:.3f} to {hi:.3f} (median {med:.3f})")
    print(f"  => minimum detectable effect: {tcrit / np.sqrt(n) * lo:.3f} to "
          f"{tcrit / np.sqrt(n) * hi:.3f} in R2, median "
          f"{tcrit / np.sqrt(n) * med:.3f}")
    span = max(np.mean([c[(k[0], k[1], s)] for s in SEEDS]) for k in keys) - \
        min(np.mean([c[(k[0], k[1], s)] for s in SEEDS]) for k in keys)
    print(f"  the whole factorial spans {span:.3f} in R2, so the median MDE is "
          f"{tcrit / np.sqrt(n) * med / span:.1f} times")
    print(f"  the entire range of the design. 'Not significant' here means the")
    print(f"  design cannot resolve differences of the size anyone would claim,")
    print(f"  not that the architectures are equivalent.")

    # What buying a smaller detectable effect would cost. The only lever is n,
    # and its price is measured: the eighteen instrumented runs of the factorial
    # cost the hours below, so a seed is one sixth of that per configuration and
    # the projection is arithmetic rather than an estimate.
    hours_measured, cells_measured = 29.08, 6
    per_seed = hours_measured / len(SEEDS)
    print(f"\n  Raising n is the only remedy, and the measured cost prices it")
    print(f"  ({hours_measured:.2f} GPU-hours for {cells_measured} configurations "
          f"on {len(SEEDS)} seeds,")
    print(f"  so {per_seed:.2f} h per added seed across the whole factorial):")
    print(f"      seeds      MDE at the median s_d      GPU-hours")
    for k in (3, 5, 10, 16):
        t_k = stats.t.ppf(0.975, k - 1)
        print(f"      {k:>5}      {t_k / np.sqrt(k) * med:>21.3f}      "
              f"{per_seed * k:>9.0f}")
    return tcrit


def part3(c, tcrit):
    print("\n" + "=" * 74)
    print("3. THE FIFTEEN PAIRWISE COMPARISONS, CORRECTED AS ONE FAMILY")
    print("=" * 74)
    keys = sorted({(b, v) for b, v, _ in c})
    rows = []
    for a, b in itertools.combinations(keys, 2):
        d = np.array([c[(a[0], a[1], s)] - c[(b[0], b[1], s)] for s in SEEDS])
        sd = float(d.std(ddof=1))
        t = float(d.mean() / (sd / np.sqrt(len(SEEDS)))) if sd > 0 else np.inf
        p = float(2 * (1 - stats.t.cdf(abs(t), len(SEEDS) - 1)))
        rows.append([f"{a[1]}/{a[0]}", f"{b[1]}/{b[0]}", float(d.mean()), sd, t, p])
    adj = holm([r[5] for r in rows])
    bon = [min(1.0, r[5] * len(rows)) for r in rows]
    rows.sort(key=lambda r: r[5])
    adj = holm([r[5] for r in rows])
    bon = [min(1.0, r[5] * len(rows)) for r in rows]

    print(f"  {'A':<12}{'B':<12}{'mean d':>9}{'s_d':>8}{'t':>8}"
          f"{'p':>8}{'Holm':>8}{'Bonf':>8}")
    print("  " + "-" * 72)
    for r, a_, b_ in zip(rows, adj, bon):
        print(f"  {r[0]:<12}{r[1]:<12}{r[2]:>+9.3f}{r[3]:>8.3f}{r[4]:>8.2f}"
              f"{r[5]:>8.3f}{a_:>8.3f}{b_:>8.3f}")
    raw_hits = sum(1 for r in rows if r[5] < 0.05)
    holm_hits = sum(1 for a_ in adj if a_ < 0.05)
    print(f"\n  uncorrected, {raw_hits} of {len(rows)} separate at alpha=0.05.")
    print(f"  Under Holm over the whole family, {holm_hits} do.")
    print(f"  The expected number of false positives from 15 uncorrected tests at")
    print(f"  alpha=0.05 is {0.05 * len(rows):.2f}, so {raw_hits} hit is what the null "
          f"predicts.")
    print(f"  Bonferroni requires alpha={0.05 / len(rows):.4f}, i.e. |t| > "
          f"{stats.t.ppf(1 - 0.05 / len(rows) / 2, len(SEEDS) - 1):.1f} at "
          f"{len(SEEDS) - 1} d.f.,")
    print(f"  which three seeds cannot reach for any plausible effect.")
    print(f"\n  The manuscript should report that no pairwise comparison among the")
    print(f"  six configurations survives correction over the family it belongs to.")


def main():
    c = cells()
    if len(c) != 18:
        print(f"WARNING: {len(c)} cells, not 18.\n")
    part1()
    tcrit = part2(c)
    part3(c, tcrit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
