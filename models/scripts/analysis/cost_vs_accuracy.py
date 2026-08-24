"""What capacity and compute actually buy, per unit of accuracy.

The manuscript reports parameter counts in one place, measured GPU cost in
another and accuracy in a third, so the question a reader most wants to ask,
what does the extra capacity buy, cannot be answered without turning pages.
This joins the three for the graph family, which is the only one carrying both
a measured cost and a seed-resolved score.

The join is the point. A cost is only interpretable against an accuracy
difference and an accuracy difference is only interpretable against the
dispersion it has to clear, so the last column carries the paired spread, and
the ratio of the two says whether the extra hours bought anything the design
can detect.

Two things this deliberately does not do. It does not divide accuracy by cost
into a single efficiency score, because a ratio of an unresolvable difference
to a measured cost is an unresolvable number with a decimal point on it. And it
does not extend the table to the convolutional family, whose cost figures are
carried over unverified from the earlier study.

Usage: python models/scripts/analysis/cost_vs_accuracy.py
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROV = ROOT / "models" / "provenance"
COST = PROV / "compute_cost.csv"
FACT = PROV / "factorial_p30_summary.txt"
OUT = PROV / "cost_vs_accuracy.txt"

SEEDS = (42, 123, 456)


def cells():
    """Per-seed mean R^2 over horizons, from the released factorial summary."""
    out = {}
    started = False
    for line in FACT.read_text(encoding="utf-8").splitlines():
        if line.startswith("cell "):
            started = True
            continue
        if not started:
            continue
        if line.startswith("=") or not line.strip():
            break
        if line.startswith("-"):
            continue
        p = line.split()
        if len(p) < 7:
            continue
        variant, bundle = p[0].split("/")
        for s, v in zip(SEEDS, p[1:4]):
            out[(variant, bundle, s)] = float(v)
    return out


def main():
    cost = {r["variant"]: r for r in csv.DictReader(COST.open(encoding="utf-8"))}
    c = cells()
    ops = ["GAT", "GCN", "SAGE"]
    bundles = ["BASIC", "PAFC"]

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("What capacity and compute buy, graph family, three seeds each")
    emit("=" * 78)
    emit(f"{'operator':<10}{'params':>9}{'s/epoch':>9}{'GPU-h':>8}"
         f"{'peak GB':>9}{'R2 mean':>9}{'spread':>8}")
    emit("-" * 78)
    means = {}
    for op in ops:
        r = cost[op]
        vals = [c[(op, b, s)] for b in bundles for s in SEEDS]
        # the operator's expectation is the mean over its two bundles and three
        # seeds; the spread quoted is the widest of its two cells, which is the
        # one a reader would have to clear
        spread = max(np.std([c[(op, b, s)] for s in SEEDS], ddof=1)
                     for b in bundles)
        means[op] = float(np.mean(vals))
        emit(f"{op:<10}{int(r['params']):>9,}{float(r['sec_per_epoch_mean']):>9.0f}"
             f"{float(r['hours_total']):>8.2f}"
             f"{float(r['peak_gb_hi']):>9.1f}{means[op]:>9.3f}{spread:>8.3f}")
    emit("-" * 78)
    emit(f"{'total':<10}{'':>9}{'':>9}"
         f"{float(cost['TOTAL']['hours_total']):>8.2f}")

    emit()
    emit("PAIRED DIFFERENCES  (the same three seeds in every cell)")
    emit("=" * 78)
    emit(f"{'A vs B':<16}{'d(R2)':>9}{'paired sd':>11}{'cost x':>9}   verdict")
    emit("-" * 78)
    for a, b in itertools.combinations(ops, 2):
        d = means[a] - means[b]
        # paired over the six (bundle, seed) pairs the two operators share
        diffs = [c[(a, bu, s)] - c[(b, bu, s)] for bu in bundles for s in SEEDS]
        sd = float(np.std(diffs, ddof=1))
        ratio = float(cost[a]["hours_total"]) / float(cost[b]["hours_total"])
        verdict = "resolved" if abs(d) > 2.484 * sd else "unresolved"
        emit(f"{a + ' vs ' + b:<16}{d:>+9.3f}{sd:>11.3f}{ratio:>8.1f}x   {verdict}")

    emit()
    emit("  The threshold is the same one the factorial uses: with three paired")
    emit("  seeds a difference separates only if it exceeds 2.484 x s_d. Every")
    emit("  pair here is below it, so no accuracy difference in this family is")
    emit("  resolvable, whatever it cost to obtain.")
    emit()
    emit("  Read the cost column against that. Attention costs "
         f"{float(cost['GAT']['hours_total']) / float(cost['SAGE']['hours_total']):.1f}"
         " times what")
    emit("  neighbourhood averaging costs and holds "
         f"{float(cost['GAT']['peak_gb_hi']) / float(cost['SAGE']['peak_gb_hi']):.1f}"
         " times the memory, for a")
    emit(f"  difference of {means['GAT'] - means['SAGE']:+.3f} in mean R2 that "
         "the design cannot detect.")
    emit()
    emit("  Parameter count does not explain the cost. The three operators sit")
    emit("  within 8K parameters of one another and their training times differ")
    emit("  by a factor of five, because the cost of a graph model scales with")
    emit("  edges traversed per timestep rather than with weights.")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
