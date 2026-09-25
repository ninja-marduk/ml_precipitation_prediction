"""Skill against the climatology by lead, by window set and inside Boyaca only.

Extends purged_skill_vs_climatology.py in three directions that the verdict
depends on:

  1. by lead (H=1..12): MSE skill score against the per-cell climatology
     (MSESS = 1 - MSE / MSE_clim) and anomaly correlation (ACC, anomalies taken
     against the same climatology), for the trained base learners on all 33
     validation windows and on the purged block (windows 21-32), and for the
     Ridge fusion refitted on the purged fitting windows (0-9);
  2. per cell: the median over cells of the temporal anomaly correlation, so the
     ACC is not dominated by between-cell contrasts;
  3. inside the department: the same headline scores restricted to the 757 grid
     cells whose centres fall inside Boyaca (data/input/MGN_Departamento.shp).

The base learners' early stopping used all 33 windows, so every number here
leans in the models' favour. Seeds 42, 123 and 456; mean +/- s.d. over seeds.

Usage: python models/scripts/analysis/skill_by_lead_and_domain.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "output"
DATA = ROOT / "data" / "output" / (
    "complete_dataset_with_features_with_clusters_elevation_windows_"
    "imfs_with_onehot_elevation_clean.nc"
)
SHAPE = ROOT / "data" / "input" / "MGN_Departamento.shp"
SEEDS = (42, 123, 456)
CUTOFF = 414
S = 33
TEST = np.arange(21, 33)
TRAIN = np.arange(0, 10)
ALL = np.arange(S)


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def msess(p, y, c):
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(c)
    return 1 - ((y[ok] - p[ok]) ** 2).mean() / ((y[ok] - c[ok]) ** 2).mean()


def acc(p, y, c):
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(c)
    return np.corrcoef(p[ok] - c[ok], y[ok] - c[ok])[0, 1]


def r2(p, y):
    ok = np.isfinite(p) & np.isfinite(y)
    return 1 - ((y[ok] - p[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()


def cell_acc_median(p, y, c):
    """Median over cells of the correlation between predicted and observed anomalies."""
    a = (p - c).reshape(-1, p.shape[-2] * p.shape[-1])
    b = (y - c).reshape(-1, p.shape[-2] * p.shape[-1])
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    r = np.where(den > 0, (a * b).sum(0) / np.where(den > 0, den, 1), np.nan)
    return np.nanmedian(r)


def main():
    ds = xr.open_dataset(DATA)
    obs = ds["total_precipitation"].values.astype("f8")
    month = ds["time"].dt.month.values
    y = load(OUT / "V10_Late_Fusion/SEED42", "targets")
    t0 = int(np.argmin([np.nanmean(np.abs(obs[t] - y[0, 0])) for t in range(len(obs))]))
    clim_m = np.stack([np.nanmean(obs[:CUTOFF][month[:CUTOFF] == m], axis=0) for m in range(1, 13)])
    c = np.stack([[clim_m[month[t0 + w + h] - 1] for h in range(12)] for w in range(S)])

    b = gpd.read_file(SHAPE).geometry.iloc[0]
    lat, lon = np.meshgrid(ds.latitude.values, ds.longitude.values, indexing="ij")
    inside = np.array([b.contains(Point(x, yy)) for x, yy in zip(lon.ravel(), lat.ravel())])
    inside = inside.reshape(lat.shape)
    print(f"Cells inside Boyaca: {inside.sum()} of {inside.size}")

    arms = {"Enhanced ConvLSTM": [], "GNN-TAT": [], "Ridge fusion (purged fit)": []}
    for seed in SEEDS:
        yy = load(OUT / f"V10_Late_Fusion/SEED{seed}", "targets")
        assert np.nanmax(np.abs(yy - y)) < 1e-3
        p2 = load(OUT / f"V2_Enhanced_Models/SEED{seed}/map_exports/H12/BASIC/ConvLSTM_Bidirectional")
        p4 = load(OUT / f"V4_GNN_TAT_Models/SEED{seed}/map_exports/H12/BASIC/GNN_TAT_GAT")
        x = np.column_stack([p2[TRAIN].ravel(), p4[TRAIN].ravel(), np.ones(p2[TRAIN].size)])
        t = y[TRAIN].ravel()
        m = np.isfinite(t) & np.isfinite(x).all(1)
        coef, *_ = np.linalg.lstsq(x[m], t[m], rcond=None)
        arms["Enhanced ConvLSTM"].append(p2)
        arms["GNN-TAT"].append(p4)
        arms["Ridge fusion (purged fit)"].append(coef[0] * p2 + coef[1] * p4 + coef[2])

    def by_lead(fn, windows, name):
        vals = np.array([[fn(p[windows][:, h], y[windows][:, h], c[windows][:, h]) for h in range(12)]
                         for p in arms[name]])
        return vals.mean(0), vals.std(0, ddof=1)

    for label, wins in (("All 33 validation windows", ALL), ("Purged block, windows 21-32", TEST)):
        print(f"\n{label}: climatology R2 = {r2(c[wins], y[wins]):.3f}")
        for metric, fn in (("MSESS vs climatology", msess), ("Anomaly correlation", acc)):
            print(f"  {metric} by lead (H=1..12), mean over seeds:")
            for name in arms:
                if name.startswith("Ridge") and label.startswith("All"):
                    continue          # the purged fit is only out of sample on 21-32
                mu, sd = by_lead(fn, wins, name)
                print(f"    {name:<27}" + " ".join(f"{v:+.2f}" for v in mu)
                      + f"   (max s.d. {sd.max():.2f})")
        print("  Median over cells of the temporal anomaly correlation (all leads pooled):")
        for name in arms:
            if name.startswith("Ridge") and label.startswith("All"):
                continue
            v = np.array([cell_acc_median(p[wins], y[wins], c[wins]) for p in arms[name]])
            print(f"    {name:<27}{v.mean():+.3f} +/- {v.std(ddof=1):.3f}")

    print("\nInside Boyaca only (757 cells):")
    mask = np.broadcast_to(inside, y.shape)
    for label, wins in (("All 33 windows", ALL), ("Purged block", TEST)):
        yy, cc = y[wins][mask[wins]], c[wins][mask[wins]]
        print(f"  {label}: climatology R2 = {r2(cc, yy):.3f}")
        for name in arms:
            if name.startswith("Ridge") and label.startswith("All"):
                continue
            rs = np.array([r2(p[wins][mask[wins]], yy) for p in arms[name]])
            ms = np.array([msess(p[wins][mask[wins]], yy, cc) for p in arms[name]])
            print(f"    {name:<27}R2 {rs.mean():.3f} +/- {rs.std(ddof=1):.3f}   "
                  f"MSESS {ms.mean():+.3f} +/- {ms.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
