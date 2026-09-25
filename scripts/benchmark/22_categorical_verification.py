"""Categorical verification of the Late Fusion forecast against a climatological reference.

RMSE, MAE and R^2 measure average closeness; they say nothing about whether a
forecast signals the anomalies a planner acts on. This script computes the
standard contingency scores (POD, FAR, CSI, frequency bias) for two events
defined per cell and per calendar month on the training period:

    dry  event:  observed total below the lower tercile
    wet  event:  observed total above the upper tercile

Terciles are estimated on 1981-2015 only, so the event definition never sees
the evaluation months. The untrained per-cell monthly climatology is scored
under exactly the same definition, which is the comparison that matters: it
carries the highest R^2 in this thesis while being, by construction, nearly
incapable of signalling a tercile anomaly.

Output: models/provenance/categorical_verification.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/output/complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"
V10 = ROOT / "models/output/V10_Late_Fusion"
OUT = ROOT / "models/provenance/categorical_verification.txt"

SEEDS = ["SEED42", "SEED123", "SEED456"]
TRAIN_END_YEAR = 2015
FIRST_TARGET_IDX = 474  # validation target months span idx 474..517 (naive_baselines.txt)


def contingency(pred_event, obs_event):
    """POD, FAR, CSI and frequency bias from a boolean forecast/observation pair."""
    hits = np.count_nonzero(pred_event & obs_event)
    misses = np.count_nonzero(~pred_event & obs_event)
    false_alarms = np.count_nonzero(pred_event & ~obs_event)
    pod = hits / (hits + misses) if hits + misses else np.nan
    far = false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan
    csi = hits / (hits + misses + false_alarms) if hits + misses + false_alarms else np.nan
    bias = (hits + false_alarms) / (hits + misses) if hits + misses else np.nan
    return dict(POD=pod, FAR=far, CSI=csi, BIAS=bias,
                hits=hits, misses=misses, fa=false_alarms)


def main():
    ds = xr.open_dataset(DATA)
    arr = ds["total_precipitation"].values                 # (T, lat, lon)
    times = pd.to_datetime(ds["time"].values)
    months = np.asarray(pd.DatetimeIndex(times).month)
    years = pd.DatetimeIndex(times).year
    train = years <= TRAIN_END_YEAR

    # per-cell, per-calendar-month climatology and terciles, training period only
    clim = np.zeros((12,) + arr.shape[1:], dtype=np.float64)
    lo = np.zeros_like(clim)
    hi = np.zeros_like(clim)
    for m in range(1, 13):
        sel = train & (months == m)
        block = arr[sel]
        clim[m - 1] = block.mean(axis=0)
        lo[m - 1] = np.quantile(block, 1 / 3, axis=0)
        hi[m - 1] = np.quantile(block, 2 / 3, axis=0)

    targets = np.load(V10 / SEEDS[0] / "targets.npy")[..., 0]   # (S, H, lat, lon)
    S, H = targets.shape[:2]

    # recover the time index of every (window, lead) pair and verify it
    idx = np.array([[FIRST_TARGET_IDX + s + h for h in range(H)] for s in range(S)])
    recovered = arr[idx]                                        # (S, H, lat, lon)
    mse = float(np.mean((recovered - targets) ** 2))
    if mse > 1e-6:
        raise SystemExit(f"index recovery failed: mean match MSE = {mse:.3e}")

    tmonth = months[idx]                                        # (S, H)
    lo_g = lo[tmonth - 1]                                       # (S, H, lat, lon)
    hi_g = hi[tmonth - 1]
    clim_g = clim[tmonth - 1]

    obs_dry = targets < lo_g
    obs_wet = targets > hi_g

    lines = []
    w = lines.append
    w("Categorical verification of the Late Fusion forecast, released 33-window set.")
    w("Events are per-cell, per-calendar-month terciles estimated on 1981-2015 only.")
    w(f"index recovery: first target idx {FIRST_TARGET_IDX}, mean match MSE = {mse:.3e}")
    w(f"grid {arr.shape[1]}x{arr.shape[2]}, {S} windows x {H} leads = {S*H*arr.shape[1]*arr.shape[2]:,} forecast cells")
    w(f"observed base rate  dry {obs_dry.mean():.3f}   wet {obs_wet.mean():.3f}")
    w("")

    def block(name, pred_stack):
        """pred_stack: list of (S,H,lat,lon) forecasts, one per seed (or one entry)."""
        w(f"########## {name}")
        w(f"{'event':<6}{'POD':>9}{'FAR':>9}{'CSI':>9}{'bias':>9}")
        for label, obs in (("dry", obs_dry), ("wet", obs_wet)):
            scores = []
            for pr in pred_stack:
                pe = (pr < lo_g) if label == "dry" else (pr > hi_g)
                scores.append(contingency(pe, obs))
            def agg(k):
                v = np.array([sc[k] for sc in scores], dtype=float)
                return (v.mean(), v.std(ddof=0) if len(v) > 1 else 0.0)
            row = f"{label:<6}"
            for k in ("POD", "FAR", "CSI", "BIAS"):
                m_, s_ = agg(k)
                row += f"{m_:>9.3f}" if len(scores) == 1 else f"{m_:>6.3f}±{s_:.3f}"
            w(row)
        w("")

    preds = [np.load(V10 / s / "predictions.npy")[..., 0] for s in SEEDS]
    block(f"Late Fusion, {len(SEEDS)} seeds (mean +/- s.d.)", preds)
    block("Untrained per-cell monthly climatology", [clim_g])

    w("--- reading ---")
    w("POD is the fraction of observed tercile events the forecast caught; FAR the")
    w("fraction of forecast events that did not occur; CSI the hits over hits plus")
    w("misses plus false alarms; bias the forecast event count over the observed one.")
    w("A bias far below 1 means the forecast almost never issues the event.")
    w("The climatological mean is a single value per cell and month, so whether it")
    w("falls inside a tercile is fixed in advance and does not respond to the year")
    w("being forecast. Its scores are therefore a property of the climatology's")
    w("position in the distribution, not forecast skill, which is the point of")
    w("scoring it here: the reference that wins on R^2 cannot signal an anomaly.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
