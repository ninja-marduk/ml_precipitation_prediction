"""Skill relative to the per-cell climatology on the purged holdout itself.

The purged fusion estimate is scored on the last 12 of the 33 validation windows
(windows 21-32, after an 11-window embargo), while the climatology's headline R2
is quoted over all 33. This scores every arm on the same 12 windows and 12 leads:

  - R2 and MSE skill score against the climatology, MSESS = 1 - MSE / MSE_clim;
  - anomaly correlation coefficient (ACC), with anomalies taken against the same
    climatology, which is the skill that remains once the annual cycle is removed;
  - a per-window paired comparison of squared error against the climatology,
    reported with the caveat that consecutive windows overlap.

Arms: Enhanced ConvLSTM and GNN-TAT as trained, and the Ridge fusion fitted on
the purged training windows (0-9), for seeds 42, 123 and 456.

Usage: python models/scripts/analysis/purged_skill_vs_climatology.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "output"
DATA = ROOT / "data" / "output" / (
    "complete_dataset_with_features_with_clusters_elevation_windows_"
    "imfs_with_onehot_elevation_clean.nc"
)
SEEDS = (42, 123, 456)
CUTOFF = 414
EMBARGO = 11
S = 33
TEST = np.arange(S // 2 + EMBARGO // 2, S)                      # 21..32
TRAIN = np.array([w for w in range(S // 2 - EMBARGO // 2) if w + 11 < TEST.min()])


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def climatology(shape_windows):
    ds = xr.open_dataset(DATA)
    obs = ds["total_precipitation"].values.astype("f8")
    month = np.asarray(ds["month"].values if "month" in ds else np.arange(len(obs)) % 12 + 1)
    month = (month[:, 0, 0] if month.ndim == 3 else month).astype(int)
    y = shape_windows
    t0 = int(np.argmin([np.nanmean(np.abs(obs[t] - y[0, 0])) for t in range(len(obs))]))
    clim = np.stack([np.nanmean(obs[[t for t in range(CUTOFF) if month[t] == m]], axis=0)
                     for m in range(1, 13)])
    return np.stack([[clim[month[t0 + w + h] - 1] for h in range(12)] for w in range(S)]), t0, ds


def scores(p, y, c):
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(c)
    p, y, c = p[ok], y[ok], c[ok]
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    msess = 1 - ((y - p) ** 2).mean() / ((y - c) ** 2).mean()
    acc = np.corrcoef(p - c, y - c)[0, 1]
    return r2, msess, acc


def main():
    y42 = load(OUT / "V10_Late_Fusion/SEED42", "targets")
    clim, t0, ds = climatology(y42)
    times = ds["time"].values
    print(f"Purged holdout windows {TEST[0]}-{TEST[-1]} (first target months "
          f"{str(times[t0 + TEST[0]])[:7]} to {str(times[t0 + TEST[-1]])[:7]}); "
          f"fusion fitted on windows {TRAIN[0]}-{TRAIN[-1]}.")
    yt, ct = y42[TEST], clim[TEST]
    ok = np.isfinite(yt) & np.isfinite(ct)
    r2c = 1 - ((yt[ok] - ct[ok]) ** 2).sum() / ((yt[ok] - yt[ok].mean()) ** 2).sum()
    print(f"\nClimatology on the purged windows: R2 = {r2c:.3f}\n")

    rows = {"Enhanced ConvLSTM": [], "GNN-TAT": [], "Ridge fusion": []}
    win_diff = {k: [] for k in rows}
    for seed in SEEDS:
        y = load(OUT / f"V10_Late_Fusion/SEED{seed}", "targets")
        assert np.nanmax(np.abs(y - y42)) < 1e-3
        p2 = load(OUT / f"V2_Enhanced_Models/SEED{seed}/map_exports/H12/BASIC/ConvLSTM_Bidirectional")
        p4 = load(OUT / f"V4_GNN_TAT_Models/SEED{seed}/map_exports/H12/BASIC/GNN_TAT_GAT")
        xtr = np.column_stack([p2[TRAIN].ravel(), p4[TRAIN].ravel(), np.ones(p2[TRAIN].size)])
        ytr = y[TRAIN].ravel()
        m = np.isfinite(ytr) & np.isfinite(xtr).all(1)
        coef, *_ = np.linalg.lstsq(xtr[m], ytr[m], rcond=None)
        pf = coef[0] * p2 + coef[1] * p4 + coef[2]
        for name, p in (("Enhanced ConvLSTM", p2), ("GNN-TAT", p4), ("Ridge fusion", pf)):
            rows[name].append(scores(p[TEST], y[TEST], clim[TEST]))
            e_m = np.nanmean(((p[TEST] - y[TEST]) ** 2).reshape(len(TEST), -1), axis=1)
            e_c = np.nanmean(((clim[TEST] - y[TEST]) ** 2).reshape(len(TEST), -1), axis=1)
            win_diff[name].append(e_m - e_c)

    print(f"{'arm':<20}{'R2':>16}{'MSESS vs clim':>18}{'ACC (anomaly)':>18}")
    for name, v in rows.items():
        a = np.array(v)
        cell = lambda i: f"{a[:, i].mean():.3f} +/- {a[:, i].std(ddof=1):.3f}"
        print(f"{name:<20}{cell(0):>16}{cell(1):>18}{cell(2):>18}")

    print("\nPer-window squared-error difference (model minus climatology), seed mean:")
    for name, d in win_diff.items():
        d = np.mean(d, axis=0)
        w = stats.wilcoxon(d)
        print(f"  {name:<18} worse than climatology in {int((d > 0).sum())} of {len(d)} windows;"
              f" Wilcoxon p = {w.pvalue:.4f} (windows overlap, so this p is optimistic)")


if __name__ == "__main__":
    main()
