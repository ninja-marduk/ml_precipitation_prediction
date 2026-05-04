"""Sensitivity of degradation % to choice of upper elevation threshold.

Sweeps the High band lower-bound from 2400 to 3000 m in 100 m steps.
Reports R² per band per threshold + degradation % Low→High.

Output: scripts/benchmark/output/threshold_sensitivity.csv
"""
from pathlib import Path
import sys
import numpy as np
import xarray as xr
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "scripts" / "benchmark" / "output"
OUT.mkdir(parents=True, exist_ok=True)

V2 = ROOT / "models/output/V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM_Bidirectional"
V4 = ROOT / "models/output/V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT"
V10 = ROOT / "models/output/V10_Late_Fusion"
DATA = ROOT / "data/output/complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"

LOW_BOUND = 1500   # fixed
UPPER_SWEEP = [2400, 2500, 2600, 2700, 2800, 2900, 3000]


def load(d):
    p = np.load(d / "predictions.npy")
    t = np.load(d / "targets.npy")
    if p.ndim == 5:
        p = p[..., 0]; t = t[..., 0]
    return p, t


def per_cell_r2(pred, targ):
    n, h, nlat, nlon = pred.shape
    pred = pred.reshape(-1, nlat, nlon); targ = targ.reshape(-1, nlat, nlon)
    ss_res = ((targ - pred) ** 2).sum(axis=0)
    ss_tot = ((targ - targ.mean(axis=0)[None]) ** 2).sum(axis=0)
    return 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)


def main():
    p2, t = load(V2); p4, _ = load(V4); p10, _ = load(V10)
    ds = xr.open_dataset(DATA)
    elev = np.asarray(ds["elevation"].values, dtype=float)
    if elev.ndim >= 3:
        elev = elev[0] if elev.ndim == 3 else elev[0, :, :, 0]

    r2 = {"V2": per_cell_r2(p2, t),
          "V4": per_cell_r2(p4, t),
          "V10": per_cell_r2(p10, t)}

    rows = []
    for upper in UPPER_SWEEP:
        low_mask = (elev < LOW_BOUND)
        high_mask = (elev >= upper)
        med_mask = ~low_mask & ~high_mask
        for arch in ["V2", "V4", "V10"]:
            r_low = float(np.nanmean(r2[arch][low_mask]))
            r_med = float(np.nanmean(r2[arch][med_mask]))
            r_high = float(np.nanmean(r2[arch][high_mask]))
            deg = (r_low - r_high) / r_low * 100 if r_low > 0 else np.nan
            rows.append({
                "upper_threshold_m": upper,
                "architecture": arch,
                "n_low": int(low_mask.sum()),
                "n_med": int(med_mask.sum()),
                "n_high": int(high_mask.sum()),
                "R2_low": round(r_low, 4),
                "R2_med": round(r_med, 4),
                "R2_high": round(r_high, 4),
                "degradation_pct": round(deg, 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "threshold_sensitivity.csv", index=False)

    print("=" * 90)
    print("Threshold sensitivity sweep (Low<1500 fixed, sweeping High lower bound)")
    print("=" * 90)
    print(df.to_string(index=False))

    # Per-architecture summary
    print("\nDegradation Low→High range per architecture:")
    for arch in ["V2", "V4", "V10"]:
        sub = df[df["architecture"] == arch]["degradation_pct"]
        print(f"  {arch:5s}: {sub.min():.1f}% – {sub.max():.1f}%  "
              f"(at thresholds {UPPER_SWEEP})")

    print(f"\nSaved: {OUT / 'threshold_sensitivity.csv'}")


if __name__ == "__main__":
    main()
