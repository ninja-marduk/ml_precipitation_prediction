"""Emit the manuscript's GNN-TAT factorial table from the corrected runs.

The table in the manuscript was built from the archived, pre-correction factorial
with one corrected row appended under a dagger. That table cannot be read as a
whole: six of its rows come from a pipeline the retrain shows moves results by more
than the differences being reported. This produces the same columns from the
eighteen corrected cells, and writes a CSV so the audit can bind to it.

Usage: python models/scripts/analysis/factorial_p30_table.py
"""
from __future__ import annotations

import csv
import glob
import itertools
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
P30 = ROOT / "models" / "output" / "V4_GNN_TAT_Models_p30"
OUT = ROOT / "models" / "provenance" / "benchmark_p30.csv"
SEEDS = (42, 123, 456)
VARIANTS = ("GAT", "GCN", "SAGE")
BUNDLES = ("BASIC", "PAFC")
PARAMS = {"GAT": "98K", "GCN": "98K", "SAGE": "106K"}


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def r2(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    p, t = pred[m], tgt[m]
    return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))


def rmse(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    return float(np.sqrt(np.mean((tgt[m] - pred[m]) ** 2)))


def bias(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    return float(np.mean(pred[m] - tgt[m]))


def main():
    cells = {}
    for d in sorted(glob.glob(str(P30 / "SEED*" / "map_exports" / "H12" / "*" /
                                  "GNN_TAT_*"))):
        parts = d.replace(os.sep, "/").split("/")
        seed = int([p for p in parts if p.startswith("SEED")][0][4:])
        bundle, variant = parts[-2], parts[-1].replace("GNN_TAT_", "")
        pred, tgt = load(d), load(d, "targets")
        per = [r2(pred[:, h], tgt[:, h]) for h in range(pred.shape[1])]
        cells.setdefault((variant, bundle), {})[seed] = dict(
            mean=float(np.mean(per)), peak=float(max(per)),
            rmse=rmse(pred, tgt), bias=bias(pred, tgt))

    rows = []
    for v, b in itertools.product(VARIANTS, BUNDLES):
        d = cells.get((v, b))
        if not d or len(d) < len(SEEDS):
            print(f"  [incomplete] {v}/{b}: {sorted(d) if d else 'none'}")
            continue
        g = lambda k: np.array([d[s][k] for s in SEEDS])
        rows.append(dict(
            variant=v, features=b, params=PARAMS[v],
            r2_mean=g("mean").mean(), r2_mean_sd=g("mean").std(ddof=1),
            r2_peak=g("peak").mean(), r2_peak_sd=g("peak").std(ddof=1),
            rmse=g("rmse").mean(), rmse_sd=g("rmse").std(ddof=1),
            bias=g("bias").mean(),
            inflation=g("peak").max() - g("peak").mean()))
    rows.sort(key=lambda r: -r["r2_peak"])

    print("=" * 84)
    print("GNN-TAT FACTORIAL, corrected pipeline, three seeds")
    print("=" * 84)
    print(f"{'Variant':<7}{'Feat.':<7}{'Params':>7}{'R2_mean':>17}{'R2_peak':>17}"
          f"{'RMSE (mm)':>15}{'Bias':>9}{'Infl.':>8}")
    for r in rows:
        print(f"{r['variant']:<7}{r['features']:<7}{r['params']:>7}"
              f"{r['r2_mean']:>10.3f} +- {r['r2_mean_sd']:<4.3f}"
              f"{r['r2_peak']:>10.3f} +- {r['r2_peak_sd']:<4.3f}"
              f"{r['rmse']:>8.1f} +- {r['rmse_sd']:<4.1f}"
              f"{r['bias']:>9.1f}{r['inflation']:>8.3f}")

    med_infl = float(np.median([r["inflation"] for r in rows]))
    med_sd = float(np.median([r["r2_peak_sd"] for r in rows]))
    print()
    print(f"median inflation over the {len(rows)} configurations : {med_infl:.4f}")
    print(f"median seed spread on R2_peak                : {med_sd:.4f}")
    worst = max(rows, key=lambda r: r["inflation"])
    print(f"worst inflation: {worst['variant']}/{worst['features']} "
          f"at {worst['inflation']:.4f}")

    print()
    print("LaTeX rows for the manuscript table:")
    for r in rows:
        cols = [
            r["variant"], r["features"], r["params"],
            rf'{r["r2_mean"]:.3f} $\pm$ {r["r2_mean_sd"]:.3f}',
            rf'{r["r2_peak"]:.3f} $\pm$ {r["r2_peak_sd"]:.3f}',
            rf'{r["rmse"]:.1f} $\pm$ {r["rmse_sd"]:.1f}',
            f'$-${abs(r["bias"]):.1f}' if r["bias"] < 0 else f'{r["bias"]:.1f}',
            f'{r["inflation"]:.3f}',
        ]
        if r is rows[0]:                      # best expectation, bolded as before
            cols = [r"\textbf{" + c + "}" for c in cols]
        print("  " + " & ".join(cols) + r" \\")

    with OUT.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
