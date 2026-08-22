"""Does the anchor score better in the months CHIRPS had fewest gauges?

The manuscript concedes that scoring a per-cell monthly climatology against
CHIRPS is "partly circular", because CHIRPS is built on CHPclim, a climatology,
and blends station data on top of it. Where the station network is dense the
product is pulled towards the gauges; where it is sparse the product falls back
on its climatological backbone. If that is what drives the 0.730, the anchor
should score higher in gauge-sparse months than in gauge-dense ones.

CHIRPS publishes which stations it used every month, so the split is observable.
Within the 44 scored months the domain sees 0 to 191 blended gauges, four months
at 10 or fewer and the rest at a median of 176. That is a natural experiment the
study can run on data it already has, with no new observations.

It is a small one, and the script says so: four months against thirty-nine, on a
record whose months are not independent. What it can do is bound the effect. A
large difference would mean the concession understates the problem; a difference
inside the month-to-month scatter would mean the anchor is not simply reading
back the product's own backbone.

Usage: python models/scripts/analysis/skill_vs_gauge_density.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
NC = (ROOT / "notebooks" / "data" / "output" /
      "complete_dataset_with_features_with_clusters_elevation_windows_imfs"
      "_with_onehot_elevation_clean.nc")
OVERLAP = ROOT / "models" / "provenance" / "chirps_station_overlap.csv"
OUT = ROOT / "models" / "provenance" / "skill_vs_gauge_density.txt"
LF = ROOT / "models" / "output" / "V10_Late_Fusion" / "SEED42"
MODELS = {
    "ConvLSTM-Bidir": ROOT / "models/output/V2_Enhanced_Models/map_exports/H12/"
                             "BASIC/ConvLSTM_Bidirectional",
    "GNN-TAT-GAT": ROOT / "models/output/V4_GNN_TAT_Models_p30/SEED42/"
                          "map_exports/H12/BASIC/GNN_TAT_GAT",
    "Late Fusion": LF,
}
SPLIT = 414
FIRST_SCORED = 474          # absolute month index of window 0, lead 1
SPARSE_AT = 20              # gauges in domain at or below which we call it sparse


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def rmse(pred, tgt):
    """RMSE, not R^2.

    An R^2 computed across cells within one month divides by that month's
    spatial variance, which is small and unstable, so the resulting series
    swings between +0.75 and -1.40 and its month-to-month standard deviation
    (1.53) is five times any effect worth looking for. RMSE has no such
    denominator and is comparable across months.
    """
    m = np.isfinite(pred) & np.isfinite(tgt)
    if m.sum() < 10:
        return np.nan
    return float(np.sqrt(np.mean((tgt[m] - pred[m]) ** 2)))


def main():
    if not OVERLAP.exists():
        print(f"missing {OVERLAP}; run chirps_station_overlap.py --all-months")
        return 1
    gauges = {}
    for r in csv.DictReader(OVERLAP.open(encoding="utf-8")):
        if r["period"] == "scored":
            gauges[(int(r["year"]), int(r["month"]))] = int(r["in_domain"])

    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)
    months = (ds["month"].values.astype(int) if "month" in ds else
              np.array([int(str(x)[5:7]) for x in ds["time"].values]))
    if months.ndim > 1:
        months = months[:, 0, 0]
    ds.close()

    clim = np.full((13,) + P.shape[1:], np.nan)
    for m in range(1, 13):
        sel = (np.arange(P.shape[0]) < SPLIT) & (months == m)
        if sel.any():
            clim[m] = np.nanmean(P[sel], axis=0)

    tgt = load(LF, "targets")
    S, H = tgt.shape[0], tgt.shape[1]
    preds = {k: load(v) for k, v in MODELS.items() if (Path(v) / "predictions.npy").exists()}

    # absolute month index -> the (window, lead) pairs that target it
    bymonth = {}
    for s in range(S):
        for h in range(H):
            bymonth.setdefault(FIRST_SCORED + s + h, []).append((s, h))

    import pandas as pd
    stamps = pd.date_range("1982-01-01", periods=P.shape[0], freq="MS")

    rows = []
    for idx in sorted(bymonth):
        ym = (stamps[idx].year, stamps[idx].month)
        if ym not in gauges:
            continue
        pairs = bymonth[idx]
        t = np.concatenate([tgt[s, h].ravel() for s, h in pairs])
        cal = stamps[idx].month
        c = np.tile(clim[cal].ravel(), len(pairs))
        row = dict(month=f"{ym[0]}-{ym[1]:02d}", gauges=gauges[ym],
                   n_pairs=len(pairs), clim=rmse(c, t))
        for name, arr in preds.items():
            row[name] = rmse(np.concatenate([arr[s, h].ravel() for s, h in pairs]),
                             t)
        # The question is not whether every error is smaller in a given month,
        # which tracks how wet the month was, but whether the anchor does
        # relatively better against the models. That ratio is dimensionless and
        # is what the circularity argument predicts should move.
        for name in preds:
            if np.isfinite(row[name]) and row[name] > 0:
                row[f"ratio:{name}"] = row["clim"] / row[name]
        rows.append(row)

    if not rows:
        print("no months matched")
        return 1

    hdr = (["month", "gauges", "n_pairs", "clim"] + list(preds)
           + [f"ratio:{k}" for k in preds])
    print(f"{len(rows)} scored months with a gauge count\n")
    print("".join(f"{h:>16}" if h not in ("month", "gauges", "n_pairs") else
                  f"{h:>9}" for h in hdr))
    print("-" * 96)
    for r in rows:
        print(f"{r['month']:>9}{r['gauges']:>9}{r['n_pairs']:>9}" +
              "".join(f"{r[k]:>16.3f}" if np.isfinite(r[k]) else f"{'-':>16}"
                      for k in ["clim"] + list(preds)))

    sparse = [r for r in rows if r["gauges"] <= SPARSE_AT]
    dense = [r for r in rows if r["gauges"] > SPARSE_AT]
    print()
    print("=" * 96)
    print(f"gauge-sparse months (<= {SPARSE_AT} in domain): "
          f"{len(sparse)}  |  gauge-dense: {len(dense)}")
    print(f"\n{'series':<18}{'sparse':>10}{'dense':>10}{'difference':>13}")
    print("-" * 55)
    for k in ["clim"] + list(preds) + [f"ratio:{x}" for x in preds]:
        a = np.nanmean([r[k] for r in sparse]) if sparse else np.nan
        b = np.nanmean([r[k] for r in dense]) if dense else np.nan
        print(f"{k:<18}{a:>10.3f}{b:>10.3f}{a - b:>+13.3f}")

    sd = np.nanstd([r["clim"] for r in dense], ddof=1) if len(dense) > 1 else np.nan
    print(f"\nmonth-to-month s.d. of the anchor in dense months: {sd:.3f}")
    print(f"{len(sparse)} sparse months is too few to test this; the numbers "
          f"above describe\nthe split and bound the effect, and no p-value is "
          f"quoted because none\nwould mean anything at this sample size.")

    with OUT.open("w", encoding="utf-8") as fh:
        fh.write(f"{len(rows)} scored months\n")
        w = csv.DictWriter(fh, fieldnames=hdr, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
