"""One anchoring engine, applied identically to the case study and to every region.

Two incompatible sets of reference-baseline numbers reached the manuscript because
two scripts computed them on different objects. `naive_baselines.py` scored
forecast targets at leads 1 to 12, where persistence carries the last observed
month across the whole window; `multiregion_ceiling.py` scored the monthly series,
where persistence is always one month stale. Persistence therefore appeared as
-0.510 in one section and +0.297 in another, for the same region and product, under
a sentence claiming the protocol had been applied unchanged.

This module defines the anchoring step once. Everything that anchors calls it.

It also reports two metrics rather than one, because they disagree and the
disagreement is the point:

  pooled R2    references every cell and lead to a single global mean. It is
               dominated by between-cell and seasonal variance, so a climatology
               scores high on it almost anywhere. Reviewers are right that a
               ceiling defined on this quantity is close to a restatement of how
               the variance partitions.
  per-cell NSE Nash-Sutcliffe at each cell against that cell's own temporal mean,
               averaged over cells. Climatology cannot flatter itself here: it
               scores near zero by construction once the cell mean is removed.

Reporting both, and saying which supports which claim, is what the protocol should
have done from the start.

Usage:
    python models/scripts/analysis/anchoring_protocol.py --case-study
    python models/scripts/analysis/anchoring_protocol.py --regions [--csv out.csv]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
# The working tree holds two byte-identical copies of this file. Every other
# analysis script reads the one under notebooks/, so this one does too, and the
# deposit ships a single copy at that path.
NC = ROOT / "notebooks" / "data" / "output" / (
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_"
    "with_onehot_elevation_clean.nc"
)
CHIRPS_URL = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/"
              "netcdf/chirps-v2.0.monthly.nc")

HORIZON = 12
TRAIN_FRAC = 0.8

# name, lat_min, lat_max, lon_min, lon_max, regime
REGIONS = [
    ("Boyaca (Colombian Andes)",   4.375,  7.375, -74.925, -71.725, "bimodal tropical"),
    ("Mantaro (Peruvian Andes)", -13.50, -10.50,  -76.50,  -73.50, "unimodal tropical"),
    ("Ethiopian Highlands",        8.00,  11.00,   37.00,   40.00, "monsoonal (Kiremt)"),
    ("Nepal Himalaya",            27.00,  29.00,   83.00,   86.00, "South Asian monsoon"),
    ("Southeast Brazil",         -22.00, -19.00,  -47.00,  -44.00, "subtropical"),
    ("Central Sahel",             12.00,  15.00,   -2.00,    1.00, "semi-arid"),
    ("Java (Indonesia)",          -8.00,  -6.00,  106.00,  109.00, "equatorial maritime"),
    ("High Atlas (Morocco)",      30.50,  33.00,   -8.00,   -5.00, "Mediterranean winter"),
]


def pooled_r2(pred, obs):
    """R2 against one global mean over cells, leads and origins."""
    m = np.isfinite(pred) & np.isfinite(obs)
    p, o = pred[m], obs[m]
    if o.size == 0:
        return np.nan
    ss = np.sum((o - o.mean()) ** 2)
    return float(1 - np.sum((o - p) ** 2) / ss) if ss > 0 else np.nan


def per_cell_nse(pred, obs):
    """Mean over cells of the Nash-Sutcliffe efficiency against each cell's own mean.

    pred, obs: (n_origins, horizon, n_cells). Climatology scores near zero here by
    construction, which is why this is the metric a ceiling claim must survive.
    """
    p = pred.reshape(-1, pred.shape[-1])
    o = obs.reshape(-1, obs.shape[-1])
    good = np.isfinite(o).all(0) & np.isfinite(p).all(0)
    if not good.any():
        return np.nan
    p, o = p[:, good], o[:, good]
    ss_res = ((o - p) ** 2).sum(0)
    ss_tot = ((o - o.mean(0)) ** 2).sum(0)
    ok = ss_tot > 0
    return float(np.mean(1 - ss_res[ok] / ss_tot[ok])) if ok.any() else np.nan


def anchor(P, months, horizon=HORIZON, train_frac=TRAIN_FRAC):
    """Score the three zero-cost references on forecast targets at leads 1..horizon.

    P       : (T, ny, nx) monthly precipitation
    months  : (T,) calendar month of each step
    returns : dict of metrics, plus the shapes the caller may want to report

    Forecast convention, identical to the case study: from each origin t in the
    test period, predict months t+1 .. t+horizon.
      persistence     y_hat(t+h) = y(t)                  (the last observed month)
      seasonal-naive  y_hat(t+h) = y(t+h-12)
      climatology     y_hat(t+h) = mean of month(t+h) over training years only
    """
    T = P.shape[0]
    split = int(T * train_frac)
    ncell = P.shape[1] * P.shape[2]
    flat = P.reshape(T, ncell)

    clim = np.full((13, ncell), np.nan)
    for m in range(1, 13):
        sel = (np.arange(T) < split) & (months == m)
        if sel.any():
            clim[m] = np.nanmean(flat[sel], axis=0)

    origins = [t for t in range(split, T - horizon)
               if t - 0 >= 0 and (t + horizon) < T and (t + 1 - 12) >= 0]
    if not origins:
        return None

    obs = np.empty((len(origins), horizon, ncell))
    pers = np.empty_like(obs)
    seas = np.empty_like(obs)
    clm = np.empty_like(obs)
    for i, t in enumerate(origins):
        for h in range(1, horizon + 1):
            obs[i, h - 1] = flat[t + h]
            pers[i, h - 1] = flat[t]
            seas[i, h - 1] = flat[t + h - 12]
            clm[i, h - 1] = clim[months[t + h]]

    out = {"n_origins": len(origins), "n_cells": int(np.isfinite(flat[0]).sum()),
           "horizon": horizon, "mean_mm": float(np.nanmean(obs))}
    for name, pred in (("climatology", clm), ("seasonal_naive", seas), ("persistence", pers)):
        out[f"pooled_{name}"] = pooled_r2(pred, obs)
        out[f"percell_{name}"] = per_cell_nse(pred, obs)
    return out


def _report(rows, title):
    print(f"\n{title}")
    hdr = (f"{'region':<26}{'regime':<22}"
           f"{'clim':>8}{'seas':>8}{'pers':>8}   |"
           f"{'clim':>8}{'seas':>8}{'pers':>8}")
    print(f"{'':<48}{'--- pooled R2 ---':^24}   |{'--- per-cell NSE ---':^24}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['region'][:25]:<26}{r.get('regime','')[:21]:<22}"
              f"{r['pooled_climatology']:>8.3f}{r['pooled_seasonal_naive']:>8.3f}"
              f"{r['pooled_persistence']:>8.3f}   |"
              f"{r['percell_climatology']:>8.3f}{r['percell_seasonal_naive']:>8.3f}"
              f"{r['percell_persistence']:>8.3f}")


def case_study():
    import xarray as xr
    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)
    months = (ds["month"].values.astype(int) if "month" in ds
              else np.array([int(str(x)[5:7]) for x in ds["time"].values]))
    if months.ndim > 1:
        months = months[:, 0, 0]
    ds.close()
    r = anchor(P, months)
    r.update(region="Boyaca (case study)", regime="bimodal tropical")
    _report([r], "CASE STUDY, forecast targets at leads 1..12")
    print(f"\norigins={r['n_origins']}  cells={r['n_cells']}  mean={r['mean_mm']:.1f} mm")
    print("\nThe manuscript reports climatology 0.730, seasonal-naive 0.431 and")
    print("persistence -0.510 from naive_baselines.py, which scores the 33 released")
    print("forecast windows. Any residual difference from the pooled column above is")
    print("the window set, not the definition; report one of them, from one script.")
    return [r]


def regions(csv_path=None, only=None, merge=False):
    """Score the references over the listed regions.

    `only` restricts the run to regions whose name contains that substring, and
    `merge` updates those rows in an existing CSV instead of rewriting the file.
    The global CHIRPS product is 7.7 GB and is read remotely, so re-running all
    eight to repair one row is an hour of transfer for nothing.
    """
    import fsspec
    import xarray as xr
    print(f"opening remote CHIRPS (lazy): {CHIRPS_URL}", flush=True)
    fh = fsspec.open(CHIRPS_URL, mode="rb", block_size=2 ** 22).open()
    ds = xr.open_dataset(fh, engine="h5netcdf", chunks={})
    var = "precip" if "precip" in ds.data_vars else list(ds.data_vars)[0]
    lat = [c for c in ds.coords if "lat" in c.lower()][0]
    lon = [c for c in ds.coords if "lon" in c.lower()][0]
    tname = [c for c in ds.coords if "time" in c.lower()][0]
    times = ds[tname].values
    months_all = np.array([int(str(x)[5:7]) for x in times])
    print(f"{len(times)} months ({str(times[0])[:7]} to {str(times[-1])[:7]})", flush=True)

    todo = [r for r in REGIONS if only is None or only.lower() in r[0].lower()]
    if only is not None:
        print(f"restricted to {len(todo)} of {len(REGIONS)} regions", flush=True)
    rows = []
    for name, la0, la1, lo0, lo1, regime in todo:
        try:
            sub = ds[var].sel({lat: slice(la0, la1), lon: slice(lo0, lo1)}).load()
            P = sub.values.astype(np.float64)
            P[P < -100] = np.nan
            r = anchor(P, months_all)
            if r is None:
                print(f"  {name}: too few origins, skipped"); continue
            r.update(region=name, regime=regime, lat=f"{la0},{la1}", lon=f"{lo0},{lo1}")
            rows.append(r)
            print(f"  {name:<28} done", flush=True)
        except Exception as exc:
            print(f"  {name:<28} ERROR {type(exc).__name__}: {str(exc)[:56]}", flush=True)

    _report(rows, "EIGHT REGIMES, same protocol as the case study")

    if rows:
        pc = np.array([r["percell_climatology"] for r in rows])
        po = np.array([r["pooled_climatology"] for r in rows])
        print(f"\nclimatology, pooled   : {po.min():.3f} to {po.max():.3f}  "
              f"(median {np.median(po):.3f})")
        print(f"climatology, per-cell : {pc.min():.3f} to {pc.max():.3f}  "
              f"(median {np.median(pc):.3f})")
        print("\nIf the pooled spread is wide and the per-cell spread collapses toward")
        print("zero, the pooled result is a statement about how variance partitions,")
        print("not about a ceiling, and the section should say so.")

    if csv_path and rows:
        import csv
        keys = ["region", "regime", "lat", "lon", "n_origins", "n_cells", "mean_mm",
                "pooled_climatology", "pooled_seasonal_naive", "pooled_persistence",
                "percell_climatology", "percell_seasonal_naive", "percell_persistence"]
        out = {r["region"]: {k: r.get(k) for k in keys} for r in rows}
        if merge and Path(csv_path).exists():
            existing = list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))
            seen = {e["region"] for e in existing}
            final = [out.get(e["region"], e) for e in existing]
            final += [v for k, v in out.items() if k not in seen]
        else:
            if only is not None and Path(csv_path).exists():
                raise SystemExit(
                    f"refusing to overwrite {csv_path} with a partial run: "
                    f"--region was given without --merge, which would drop every "
                    f"region not in this run")
            final = list(out.values())
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=keys)
            w.writeheader()
            for r in final:
                w.writerow({k: r.get(k) for k in keys})
        print(f"\nwrote {csv_path} ({len(final)} rows, {len(rows)} recomputed)",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-study", action="store_true")
    ap.add_argument("--regions", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--region", default=None,
                    help="substring: run only the regions whose name contains it")
    ap.add_argument("--merge", action="store_true",
                    help="update those rows in --csv instead of rewriting the file")
    a = ap.parse_args()
    if not (a.case_study or a.regions):
        a.case_study = True
    if a.case_study:
        case_study()
    if a.regions:
        regions(a.csv, only=a.region, merge=a.merge)


if __name__ == "__main__":
    main()
