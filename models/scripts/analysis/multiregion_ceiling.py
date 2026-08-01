"""Does the predictability ceiling generalise beyond one region?

The central finding of the case study is that a per-cell monthly climatology is a
baseline no deep architecture in the benchmark exceeds. That could be a property of
the Colombian Andes rather than of the problem. This script tests it by applying the
same zero-cost baselines to eight regions chosen for contrasting precipitation
regimes, using the global CHIRPS v2.0 monthly product read remotely (no full
download): bimodal tropical, unimodal tropical, two monsoon systems, subtropical,
semi-arid, equatorial maritime, and Mediterranean winter-rainfall.

For every region the protocol is identical to the case study: estimate a per-cell
monthly climatology on the training period only, then score it, persistence and a
seasonal-naive predictor on the held-out period, pooled over all cells and months.

Usage: python models/scripts/analysis/multiregion_ceiling.py [--out results.csv]
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

CHIRPS_URL = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/"
              "netcdf/chirps-v2.0.monthly.nc")
TRAIN_FRAC = 0.8

# name, lat_min, lat_max, lon_min, lon_max, regime
REGIONS = [
    ("Boyaca (Colombian Andes)",   4.375,  7.375, -74.925, -71.725, "bimodal tropical"),
    ("Mantaro (Peruvian Andes)", -13.50, -10.50,  -76.50,  -73.50, "unimodal tropical"),
    ("Ethiopian Highlands",        8.00,  11.00,   37.00,   40.00, "monsoonal (Kiremt)"),
    ("Nepal Himalaya",            27.00,  29.00,   83.00,   86.00, "South Asian monsoon"),
    ("Southeast Brazil",         -22.00, -19.00,  -47.00,  -44.00, "subtropical"),
    ("Central Sahel",             12.00,  15.00,   -2.00,    1.00, "semi-arid, single wet season"),
    ("Java (Indonesia)",          -8.00,  -6.00,  106.00,  109.00, "equatorial maritime"),
    ("High Atlas (Morocco)",      30.50,  33.00,   -8.00,   -5.00, "Mediterranean winter"),
]


def r2(pred, obs):
    m = np.isfinite(pred) & np.isfinite(obs)
    p, o = pred[m], obs[m]
    if o.size == 0:
        return np.nan
    ss_tot = np.sum((o - o.mean()) ** 2)
    return float(1 - np.sum((o - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan


def evaluate_region(P, months):
    """P: (T, ny, nx) precipitation; months: (T,) calendar month. Returns dict."""
    T = P.shape[0]
    split = int(T * TRAIN_FRAC)

    clim = np.full((13,) + P.shape[1:], np.nan)
    for m in range(1, 13):
        sel = (np.arange(T) < split) & (months == m)
        if sel.any():
            clim[m] = np.nanmean(P[sel], axis=0)

    idx = np.arange(split, T)
    obs = P[idx]
    pred_clim = np.stack([clim[months[i]] for i in idx])
    pred_pers = P[idx - 1]
    keep = idx - 12 >= 0
    out = {
        "n_test_months": len(idx),
        "n_cells": int(np.isfinite(P[0]).sum()),
        "mean_mm": float(np.nanmean(obs)),
        "R2_climatology": r2(pred_clim, obs),
        "R2_persistence": r2(pred_pers, obs),
        "R2_seasonal_naive": r2(P[idx[keep] - 12], obs[keep]),
    }
    # how much of the variance is the seasonal cycle: R2 of climatology on TRAIN
    tr = np.arange(split)
    out["R2_clim_train"] = r2(np.stack([clim[months[i]] for i in tr]), P[tr])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="optional CSV output path")
    args = ap.parse_args()

    import fsspec
    import xarray as xr

    print(f"opening remote CHIRPS (lazy): {CHIRPS_URL}")
    fh = fsspec.open(CHIRPS_URL, mode="rb", block_size=2 ** 22).open()
    ds = xr.open_dataset(fh, engine="h5netcdf", chunks={})
    var = "precip" if "precip" in ds.data_vars else list(ds.data_vars)[0]
    lat = [c for c in ds.coords if "lat" in c.lower()][0]
    lon = [c for c in ds.coords if "lon" in c.lower()][0]
    tname = [c for c in ds.coords if "time" in c.lower()][0]
    times = ds[tname].values
    months_all = np.array([int(str(x)[5:7]) for x in times])
    print(f"grid {ds.sizes[lat]}x{ds.sizes[lon]}, {len(times)} months "
          f"({str(times[0])[:7]} to {str(times[-1])[:7]})\n")

    rows = []
    hdr = (f"{'region':<28}{'regime':<30}{'cells':>7}{'clim':>8}{'seas':>8}{'pers':>8}")
    print(hdr); print("-" * len(hdr))
    for name, la0, la1, lo0, lo1, regime in REGIONS:
        try:
            sub = ds[var].sel({lat: slice(la0, la1), lon: slice(lo0, lo1)}).load()
            P = sub.values.astype(np.float64)
            P[P < -100] = np.nan                      # CHIRPS missing sentinel
            if np.isfinite(P).sum() == 0:
                print(f"{name:<28}{regime:<30}   no valid cells"); continue
            res = evaluate_region(P, months_all)
            res.update(region=name, regime=regime,
                       lat=f"{la0},{la1}", lon=f"{lo0},{lo1}")
            rows.append(res)
            print(f"{name:<28}{regime:<30}{res['n_cells']:>7}"
                  f"{res['R2_climatology']:>8.3f}{res['R2_seasonal_naive']:>8.3f}"
                  f"{res['R2_persistence']:>8.3f}")
        except Exception as exc:
            print(f"{name:<28}{regime:<30}   ERROR {type(exc).__name__}: {str(exc)[:60]}")

    if rows:
        vals = [r["R2_climatology"] for r in rows if np.isfinite(r["R2_climatology"])]
        print("\nclimatology R2 across regions: "
              f"min {min(vals):.3f}, median {float(np.median(vals)):.3f}, max {max(vals):.3f}")
        print("Interpretation: a per-cell monthly climatology is a strong baseline wherever the")
        print("seasonal cycle dominates the variance, which is most of the CHIRPS domain. Any")
        print("reported skill for a learned model must be read against this number, not against")
        print("other learned models.")

    if args.out and rows:
        import csv
        keys = ["region", "regime", "lat", "lon", "n_cells", "n_test_months", "mean_mm",
                "R2_climatology", "R2_seasonal_naive", "R2_persistence", "R2_clim_train"]
        with open(args.out, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
