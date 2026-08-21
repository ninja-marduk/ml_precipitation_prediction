"""Summarise the corrected factorial as its own design says it must be read.

The p30 tree is the feature-by-architecture factorial retrained on one protocol:
two bundles by three graph operators by three seeds. This reads whatever cells
exist, reports them the way Component 2 of the protocol requires, and refuses to
present as resolved anything the seeds do not resolve.

Three things it does that a table of means does not:

  paired differences   The same seeds run every cell, so the design is crossed and
                       the uncertainty of a difference is the dispersion of the
                       per-seed difference, not the marginal spread of either arm.
  inflation            What quoting a configuration's best seed would add to its
                       expectation, which is the quantity the manuscript reports
                       as the cost of single-run reporting.
  the archived contrast Every cell against the same cell in the pre-correction
                       factorial, so the pipeline effect is visible per cell
                       rather than as one summary number.

It prints how many seeds each cell has and marks the whole report incomplete while
any cell has fewer than three, because two orderings that disagree are not a
factorial.

Usage: python models/scripts/analysis/factorial_p30_summary.py
"""
from __future__ import annotations

import glob
import itertools
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
P30 = ROOT / "models" / "output" / "V4_GNN_TAT_Models_p30"
ARCHIVE = (ROOT / "models" / "output" / "V4_GNN_TAT_Models" /
           "_archive_2026-04_leaked_graph" / "metrics_factorial_consolidated.csv")
SEEDS = (42, 123, 456)
VARIANTS = ("GAT", "GCN", "SAGE")
BUNDLES = ("BASIC", "PAFC")


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def r2(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    p, t = pred[m], tgt[m]
    return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))


def read_cells():
    """cells[(variant, bundle)][seed] = dict(mean=..., peak=...)"""
    cells = {}
    for d in sorted(glob.glob(str(P30 / "SEED*" / "map_exports" / "H12" / "*" /
                                  "GNN_TAT_*"))):
        parts = d.replace(os.sep, "/").split("/")
        seed = int([p for p in parts if p.startswith("SEED")][0][4:])
        bundle, variant = parts[-2], parts[-1].replace("GNN_TAT_", "")
        pred, tgt = load(d), load(d, "targets")
        per = [r2(pred[:, h], tgt[:, h]) for h in range(pred.shape[1])]
        cells.setdefault((variant, bundle), {})[seed] = dict(
            mean=float(np.mean(per)), peak=float(max(per)))
    return cells


def archived():
    if not ARCHIVE.exists():
        return {}
    import csv
    acc = {}
    with ARCHIVE.open(encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            if r["feat"] not in BUNDLES:
                continue
            acc.setdefault((r["variant"], r["feat"]), {}).setdefault(
                int(r["seed"]), []).append(float(r["R^2"]))
    return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in acc.items()}


def paired_t(diffs):
    d = np.asarray(diffs, float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return d.mean(), 0.0, np.nan
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return d.mean(), d.std(ddof=1), t


# critical |t| at alpha=0.05, two-sided
TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def main():
    cells = read_cells()
    if not cells:
        raise SystemExit(f"no cells under {P30}")
    arch = archived()
    complete = [k for k, v in cells.items() if len(v) >= 3]
    n_expected = len(VARIANTS) * len(BUNDLES)

    print("=" * 78)
    print("CORRECTED FACTORIAL, mean R2 over the twelve horizons")
    print("=" * 78)
    hdr = f"{'cell':<13}" + "".join(f"{'seed ' + str(s):>11}" for s in SEEDS)
    print(hdr + f"{'mean':>9}{'s.d.':>8}{'infl.':>8}")
    print("-" * len(hdr + " " * 25))
    summary = {}
    for v, b in itertools.product(VARIANTS, BUNDLES):
        k = (v, b)
        got = cells.get(k, {})
        vals = [got[s]["mean"] for s in SEEDS if s in got]
        peaks = [got[s]["peak"] for s in SEEDS if s in got]
        row = f"{v + '/' + b:<13}"
        for s in SEEDS:
            row += f"{got[s]['mean']:>11.4f}" if s in got else f"{'-':>11}"
        if len(vals) >= 2:
            sd = float(np.std(vals, ddof=1))
            infl = max(peaks) - float(np.mean(peaks))
            summary[k] = dict(mean=float(np.mean(vals)), sd=sd, n=len(vals))
            row += f"{np.mean(vals):>9.4f}{sd:>8.4f}{infl:>8.4f}"
        else:
            row += f"{'-':>9}{'-':>8}{'-':>8}"
        print(row)

    print()
    print("ranking by expectation over the seeds available:")
    for i, (k, d) in enumerate(sorted(summary.items(), key=lambda kv: -kv[1]["mean"]), 1):
        print(f"  {i}. {k[0] + '/' + k[1]:<12} {d['mean']:.4f} "
              f"+/- {d['sd']:.4f}  (n={d['n']})")

    print()
    print("=" * 78)
    print("PAIRED COMPARISONS  (the dispersion of the difference, not of either arm)")
    print("=" * 78)
    keys = [k for k in summary if len(cells[k]) >= 2]
    common = sorted(set.intersection(*[set(cells[k]) for k in keys])) if keys else []
    print(f"seeds common to every cell: {common}")
    if len(common) < 2:
        print("  too few shared seeds for a paired test")
    else:
        crit = TCRIT.get(len(common))
        print(f"{'comparison':<28}{'paired':>9}{'sd':>8}{'marginal':>10}{'t':>8}   verdict")
        print("-" * 78)
        rows = []
        for a, b in itertools.combinations(keys, 2):
            d = [cells[a][s]["mean"] - cells[b][s]["mean"] for s in common]
            m, sd, t = paired_t(d)
            marg = float(np.hypot(summary[a]["sd"], summary[b]["sd"]))
            rows.append((abs(m), a, b, m, sd, marg, t))
        for _, a, b, m, sd, marg, t in sorted(rows, reverse=True):
            name = f"{a[0]}/{a[1]} vs {b[0]}/{b[1]}"
            ok = crit is not None and abs(t) > crit
            verdict = "resolved" if ok else "unresolved"
            print(f"{name:<28}{m:>+9.4f}{sd:>8.4f}{marg:>10.4f}{t:>8.2f}   {verdict}")
        print()
        print(f"  |t| > {crit} is required at alpha=0.05 with n={len(common)}. "
              f"With {len(common)} seeds that bar is very high, which is the point:")
        print("  a crossed design with few seeds resolves few differences, and the")
        print("  protocol asks that the rest be reported as unresolved rather than ranked.")

    if arch:
        print()
        print("=" * 78)
        print("AGAINST THE PRE-CORRECTION FACTORIAL, per cell and per seed")
        print("=" * 78)
        print(f"{'cell':<13}{'seed':>6}{'archived':>11}{'corrected':>11}{'change':>10}")
        print("-" * 51)
        for v, b in itertools.product(VARIANTS, BUNDLES):
            k = (v, b)
            for s in SEEDS:
                if k in cells and s in cells[k] and k in arch and s in arch[k]:
                    c, a = cells[k][s]["mean"], arch[k][s]
                    print(f"{v + '/' + b:<13}{s:>6}{a:>11.4f}{c:>11.4f}{c - a:>+10.4f}")

    print()
    print("=" * 78)
    missing = n_expected * len(SEEDS) - sum(len(v) for v in cells.values())
    if missing:
        print(f"INCOMPLETE: {missing} of {n_expected * len(SEEDS)} cells not yet run.")
        print("Orderings from a subset of the seeds disagree with each other and none")
        print("of them is the factorial. Report them as disagreeing, not as a result.")
    else:
        print(f"COMPLETE: {n_expected} cells on {len(SEEDS)} seeds, one protocol.")
    print("=" * 78)
    return 0 if not missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
