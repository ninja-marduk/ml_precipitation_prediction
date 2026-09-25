"""Score the per-cell monthly climatology under every reading convention the models use.

The climatology is the calendar-month mean of each cell over the training months
(t < 414, the 80 % chronological cutoff). It is scored here on the same 33
validation windows and 12 leads as the trained models, three ways:

  - pooled over every (window, lead, cell) value, the convention of its headline R2;
  - averaged over per-window R2 values, the convention of the multi-seed tables;
  - per lead, to set against the per-horizon tables.

Late Fusion's released-scheme predictions are scored the same way for reference.

Usage: python models/scripts/analysis/climatology_reading_conventions.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "output" / (
    "complete_dataset_with_features_with_clusters_elevation_windows_"
    "imfs_with_onehot_elevation_clean.nc"
)
LF = ROOT / "models" / "output" / "V10_Late_Fusion"
CUTOFF = 414


def r2(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ds = xr.open_dataset(DATA)
    obs = ds["total_precipitation"].values.astype("f8")
    month = np.asarray(ds["month"].values if "month" in ds else np.arange(len(obs)) % 12 + 1)
    month = (month[:, 0, 0] if month.ndim == 3 else month).astype(int)

    y = np.load(LF / "targets.npy").astype("f8").reshape(33, 12, *obs.shape[1:])
    t0 = int(np.argmin([np.nanmean(np.abs(obs[t] - y[0, 0])) for t in range(len(obs))]))
    assert np.nanmax(np.abs(obs[t0 + 32 + 11] - y[32, 11])) < 1e-3, "window alignment failed"

    clim = np.stack([np.nanmean(obs[[t for t in range(CUTOFF) if month[t] == mm]], axis=0)
                     for mm in range(1, 13)])
    pc = np.stack([[clim[month[t0 + w + h] - 1] for h in range(12)] for w in range(33)])
    pl = np.load(LF / "predictions.npy").astype("f8").reshape(y.shape)

    print(f"Validation windows start at month index {t0}; climatology fit on t < {CUTOFF}.")
    for name, p in (("Climatology", pc), ("Late Fusion (released)", pl)):
        per_w = np.array([r2(y[w], p[w]) for w in range(33)])
        print(f"\n{name}")
        print(f"  pooled R2            {r2(y, p):.3f}")
        print(f"  per-window mean R2   {per_w.mean():.3f} +/- {per_w.std(ddof=1):.3f}")
        print("  per lead R2          " + " ".join(f"{r2(y[:, h], p[:, h]):.3f}" for h in range(12)))


if __name__ == "__main__":
    main()
