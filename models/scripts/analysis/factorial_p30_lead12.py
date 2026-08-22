"""Score the corrected factorial at the twelfth lead, for the supplement table.

The manuscript's factorial table reports the mean and the peak over the twelve
leads of each run. The supplement reports the last lead on its own, which is the
hardest one and the one the grouped bar figure draws, so it needs its own numbers
rather than a re-labelling of the horizon mean.

Everything is recomputed from the prediction arrays, not read from the released
metric CSVs, for the same reason the rest of this analysis is: those CSVs are what
the duplicate-array defect was found in. The CSV values are loaded anyway and
printed beside the recomputed ones, so a disagreement is visible rather than
silent.

Usage: python models/scripts/analysis/factorial_p30_lead12.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
P30 = ROOT / "models" / "output" / "V4_GNN_TAT_Models_p30"
LEAD = 12
SEEDS = (42, 123, 456)
ORDER = (("BASIC", "GAT"), ("BASIC", "GCN"), ("BASIC", "SAGE"),
         ("PAFC", "GAT"), ("PAFC", "GCN"), ("PAFC", "SAGE"))
LABEL = {"GAT": "GAT", "GCN": "GCN", "SAGE": "GraphSAGE"}


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def score(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    p, t = pred[m], tgt[m]
    r2 = float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
    return (r2, float(np.sqrt(np.mean((t - p) ** 2))),
            float(np.mean(np.abs(t - p))), float(np.mean(p - t)))


def from_arrays():
    out = {}
    for d in sorted(glob.glob(str(P30 / "SEED*" / "map_exports" / "H12" / "*" /
                                  "GNN_TAT_*"))):
        parts = d.replace(os.sep, "/").split("/")
        seed = int([p for p in parts if p.startswith("SEED")][0][4:])
        bundle, variant = parts[-2], parts[-1].replace("GNN_TAT_", "")
        pred, tgt = load(d), load(d, "targets")
        out[(bundle, variant, seed)] = score(pred[:, LEAD - 1], tgt[:, LEAD - 1])
    return out


def from_csv():
    out = {}
    for seed in SEEDS:
        p = P30 / f"SEED{seed}" / "metrics_spatial_v4_gnn_tat_h12.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["H"] == LEAD]
        for _, r in df.iterrows():
            key = (r["Experiment"], r["Model"].replace("GNN_TAT_", ""), seed)
            out[key] = (float(r["R^2"]), float(r["RMSE"]),
                        float(r["MAE"]), float(r["mean_bias_mm"]))
    return out


def main():
    arr, csv = from_arrays(), from_csv()
    if len(arr) != 18:
        print(f"WARNING: {len(arr)} cells found, not 18.")

    worst = 0.0
    print(f"lead H={LEAD}, recomputed from the prediction arrays "
          f"(CSV disagreement in brackets)\n")
    print(f"{'bundle':<7}{'variant':<11}{'R2 mean':>10}{'sd':>8}"
          f"{'RMSE mean':>11}{'sd':>8}")
    print("-" * 55)
    rows, agg = [], []
    for bundle, variant in ORDER:
        per_seed = []
        for seed in SEEDS:
            k = (bundle, variant, seed)
            if k not in arr:
                continue
            per_seed.append(arr[k])
            if k in csv:
                worst = max(worst, abs(csv[k][0] - arr[k][0]))
        if not per_seed:
            continue
        m = np.array(per_seed)
        mu, sd = m.mean(axis=0), m.std(axis=0, ddof=1)
        rows.append((bundle, LABEL[variant], mu[0], sd[0], mu[1], sd[1],
                     float(m[:, 0].min()), float(m[:, 0].max())))
        agg.append(dict(feat=bundle, variant=variant,
                        **{f"{n}_{s}": v for n, i in
                           (("R^2", 0), ("RMSE", 1), ("MAE", 2), ("Bias", 3))
                           for s, v in (("mean", mu[i]), ("std", sd[i]),
                                        ("count", len(per_seed)))}))
        print(f"{bundle:<7}{LABEL[variant]:<11}{mu[0]:>10.4f}{sd[0]:>8.4f}"
              f"{mu[1]:>11.2f}{sd[1]:>8.2f}")

    best_r2 = max(r[2] for r in rows)
    best_rmse = min(r[4] for r in rows)
    print(f"\nlargest CSV-vs-array disagreement in R2: {worst:.4f}")

    frame = pd.DataFrame(agg)
    dst = ROOT / ".docs" / "papers" / "5" / "data" / "factorial_feat_variant_p30.csv"
    if dst.parent.exists():
        frame.to_csv(dst, index=False)
        print(f"wrote {dst}")
    prov = ROOT / "models" / "provenance" / "factorial_p30_lead12.csv"
    frame.to_csv(prov, index=False)
    print(f"wrote {prov}")

    print("\nsupplement table body:")
    for bundle, variant, r2m, r2sd, rm, rs, lo, hi in rows:
        f_r2 = (f"$\\mathbf{{{r2m:.3f} \\pm {r2sd:.3f}}}$" if r2m == best_r2
                else f"${r2m:.3f} \\pm {r2sd:.3f}$")
        f_rm = (f"$\\mathbf{{{rm:.2f} \\pm {rs:.2f}}}$" if rm == best_rmse
                else f"${rm:.2f} \\pm {rs:.2f}$")
        print(f"{bundle:<6}& {variant:<11}& {f_r2:<32}& {f_rm} \\\\")

    print("\nper-cell seed range at this lead:")
    for bundle, variant, _, sd, _, _, lo, hi in rows:
        print(f"  {bundle:<6}{variant:<11}sd={sd:.3f}  range {lo:.3f}-{hi:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
