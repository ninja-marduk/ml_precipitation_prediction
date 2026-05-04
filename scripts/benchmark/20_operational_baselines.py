"""Operational baselines: persistence + climatology + comparison with Late Fusion.

Computes monthly precipitation R² for two zero-cost baselines:
  - Persistence: ŷ(t+H) = y(t)
  - Climatology: ŷ(t+H) = mean(y[train_years][month(t+H)])

Both computed on the same test set as Late Fusion (2016-2022) at H=1, 3, 6, 12.

Output: scripts/benchmark/output/operational_baselines.csv
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

DATA = ROOT / "data/output/complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"

TRAIN_END_YEAR = 2015  # train: 1981-2015
TEST_START_YEAR = 2016  # test:  2016-2022
HORIZONS = [1, 3, 6, 12]


def aggregate_r2(obs, pred):
    """Aggregate R² over all valid (obs, pred) pairs."""
    mask = np.isfinite(obs) & np.isfinite(pred)
    o = obs[mask]; p = pred[mask]
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def aggregate_rmse(obs, pred):
    mask = np.isfinite(obs) & np.isfinite(pred)
    return float(np.sqrt(np.mean((obs[mask] - pred[mask]) ** 2)))


def main():
    print("Loading dataset...")
    ds = xr.open_dataset(DATA)
    var = "total_precipitation"
    da = ds[var]  # (time, lat, lon)
    times = pd.to_datetime(da["time"].values)
    arr = da.values  # (T, lat, lon)
    T, lat, lon = arr.shape
    print(f"  Shape: {arr.shape}, range: {times[0]} to {times[-1]}")

    # Train mask
    years = pd.DatetimeIndex(times).year
    months = pd.DatetimeIndex(times).month
    train_mask = years <= TRAIN_END_YEAR
    test_mask = years >= TEST_START_YEAR

    # Climatology: mean per calendar month, computed on training years only
    climatology = np.zeros((12, lat, lon))
    for m in range(1, 13):
        sel = train_mask & (months == m)
        climatology[m - 1] = arr[sel].mean(axis=0)

    rows = []
    for H in HORIZONS:
        # For each test month index, evaluate at (t, t+H)
        # Need both: source month t (for persistence) and target month t+H
        # Test pairs: t in test set, t+H also in test set (or train)
        # For consistency with paper: we evaluate H-month-ahead predictions on test years
        target_idx = np.where(test_mask)[0]
        # Limit to indices where t+H < T
        target_idx = target_idx[target_idx + 0 < T]  # target month is the test month itself
        # Source is t-H (from H months earlier)
        source_idx = target_idx - H
        valid = source_idx >= 0
        target_idx = target_idx[valid]
        source_idx = source_idx[valid]

        obs = arr[target_idx]               # actual precipitation at t (target)
        # Persistence: predict t-H value as t (i.e., last observed value)
        pred_pers = arr[source_idx]
        # Climatology: predict mean for the calendar month of t
        target_months = months[target_idx]
        pred_clim = np.stack([climatology[m - 1] for m in target_months])

        r2_pers = aggregate_r2(obs.flatten(), pred_pers.flatten())
        r2_clim = aggregate_r2(obs.flatten(), pred_clim.flatten())
        rmse_pers = aggregate_rmse(obs.flatten(), pred_pers.flatten())
        rmse_clim = aggregate_rmse(obs.flatten(), pred_clim.flatten())

        rows.append({"H": H, "baseline": "Persistence", "R2_agg": round(r2_pers, 4),
                     "RMSE": round(rmse_pers, 2), "n_target_months": len(target_idx)})
        rows.append({"H": H, "baseline": "Climatology", "R2_agg": round(r2_clim, 4),
                     "RMSE": round(rmse_clim, 2), "n_target_months": len(target_idx)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "operational_baselines.csv", index=False)

    print("\n=== Operational baselines on test set (2016-2022) ===")
    print(df.to_string(index=False))

    # Compare with Paper 5 headline R² = 0.672 (V10 Late Fusion seed 42, aggregate, H=12)
    pers_h12 = df[(df["H"] == 12) & (df["baseline"] == "Persistence")]["R2_agg"].iloc[0]
    clim_h12 = df[(df["H"] == 12) & (df["baseline"] == "Climatology")]["R2_agg"].iloc[0]
    v10_h12 = 0.672
    print(f"\n=== Comparison at H=12 ===")
    print(f"  Persistence:    R²={pers_h12:.4f}")
    print(f"  Climatology:    R²={clim_h12:.4f}")
    print(f"  Late Fusion:    R²={v10_h12:.4f}")
    best_baseline = max(pers_h12, clim_h12)
    delta_abs = v10_h12 - best_baseline
    delta_rel = (v10_h12 - best_baseline) / best_baseline * 100
    print(f"  Late Fusion vs best baseline: +{delta_abs:.3f} absolute (+{delta_rel:.1f}% relative)")

    print(f"\nSaved: {OUT / 'operational_baselines.csv'}")


if __name__ == "__main__":
    main()
