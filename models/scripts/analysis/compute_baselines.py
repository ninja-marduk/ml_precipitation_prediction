"""
Traditional baselines for the Main Hypothesis (T-P0-1)
======================================================
Computes non-deep-learning reference models on the SAME chronological
validation split used by the deep models, so the thesis can claim "hybrid
deep learning outperforms traditional models" against actual evidence
(not only ML-vs-ML).

Baselines:
  1. Climatology      : calendar-month mean from the training period.
  2. Persistence (l12): y_hat_t = y_{t-12} (same month, previous year).
  3. MLR              : multivariate linear regression on BASIC features.
  4. SARIMA           : seasonal ARIMA, fitted per elevation zone (cheap proxy
                        for per-cell SARIMA; full per-cell is ~4k fits).

Evaluation: pooled spatial-temporal R2/RMSE/MAE over all valid (cell, month)
targets in the validation period, matching the models' aggregate skill.

Split: 80% chronological cutoff at time index 414 (training < 414, validation
>= 414), consistent with thesis Section "Data splitting and cross-validation".

Output:
  .docs/thesis/reviews/baseline_metrics.csv  + console summary

Usage:
  python models/scripts/analysis/compute_baselines.py
"""
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "output" / (
    "complete_dataset_with_features_with_clusters_elevation_windows_"
    "imfs_with_onehot_elevation_clean.nc"
)
OUT_CSV = ROOT / ".docs" / "thesis" / "reviews" / "baseline_metrics.csv"

CUTOFF = 414                 # 80% chronological split (training < 414)
TARGET = "total_precipitation"
BASIC_FEATURES = [
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "elevation", "slope", "aspect",
    "total_precipitation_lag1", "total_precipitation_lag2",
    "total_precipitation_lag12",
]


def metrics(y_true, y_pred):
    """Pooled R2/RMSE/MAE over finite (true,pred) pairs."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    if len(yt) < 2:
        return dict(R2=np.nan, RMSE=np.nan, MAE=np.nan, n=len(yt))
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    return dict(R2=round(r2, 4), RMSE=round(rmse, 2), MAE=round(mae, 2), n=int(len(yt)))


def main():
    print("=" * 60)
    print("  Traditional baselines (T-P0-1)")
    print("=" * 60)
    ds = xr.open_dataset(DATA)
    y = ds[TARGET]  # (time, lat, lon)
    n_time = y.sizes["time"]
    months = ds["month"].values if "month" in ds else (np.arange(n_time) % 12) + 1
    # month per time-step (collapse spatial if month is 3-D)
    m = np.asarray(months)
    if m.ndim == 3:
        m = m[:, 0, 0]
    m = m.astype(int)

    yv = y.values  # (T, lat, lon)
    val_idx = np.arange(CUTOFF, n_time)
    results = []

    # --- 1. Climatology (calendar-month mean from training) ---
    clim = np.full((12,) + yv.shape[1:], np.nan)
    for mm in range(1, 13):
        tr = [t for t in range(CUTOFF) if m[t] == mm]
        if tr:
            clim[mm - 1] = np.nanmean(yv[tr], axis=0)
    yhat_clim = np.stack([clim[m[t] - 1] for t in val_idx])
    yt_val = yv[val_idx]
    results.append({"Baseline": "Climatology (calendar-month mean)", **metrics(yt_val.ravel(), yhat_clim.ravel())})

    # --- 2. Persistence lag-12 ---
    yhat_pers = yv[val_idx - 12]
    results.append({"Baseline": "Persistence (lag-12)", **metrics(yt_val.ravel(), yhat_pers.ravel())})

    # --- 3. MLR on BASIC features ---
    feats = [f for f in BASIC_FEATURES if f in ds]
    F = np.stack([ds[f].values for f in feats], axis=-1)  # (T, lat, lon, k) or (T,1,1,k)
    # broadcast static features (elevation/slope/aspect) to full grid if needed
    Fb = []
    for i, f in enumerate(feats):
        a = ds[f].values
        if a.ndim == 1:          # time-only
            a = a[:, None, None] * np.ones((1,) + yv.shape[1:])
        Fb.append(a)
    F = np.stack(Fb, axis=-1)    # (T, lat, lon, k)
    from sklearn.linear_model import LinearRegression
    tr_idx = np.arange(CUTOFF)
    Xtr = F[tr_idx].reshape(-1, len(feats)); ytr = yv[tr_idx].ravel()
    Xva = F[val_idx].reshape(-1, len(feats)); yva = yt_val.ravel()
    mask_tr = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    lr = LinearRegression().fit(Xtr[mask_tr], ytr[mask_tr])
    yhat_mlr = np.full_like(yva, np.nan)
    mask_va = np.isfinite(Xva).all(1)
    yhat_mlr[mask_va] = lr.predict(Xva[mask_va])
    results.append({"Baseline": f"MLR (BASIC, {len(feats)} feats)", **metrics(yva, yhat_mlr)})

    # --- 4. SARIMA per elevation zone (proxy) ---
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        zone = ds["cluster_elevation"].values if "cluster_elevation" in ds else None
        # spatial-mean series per zone -> forecast -> broadcast (coarse proxy)
        sarima_pred = np.full_like(yt_val, np.nan)
        if zone is not None and zone.ndim >= 2:
            z2d = zone if zone.ndim == 2 else zone[0]
            # map string zones to ids
            uniq = list(np.unique(z2d))
            for zid in uniq:
                cells = (z2d == zid)
                if not cells.any():
                    continue
                series = np.nanmean(yv[:, cells], axis=1)  # (T,)
                tr = series[:CUTOFF]
                try:
                    fit = SARIMAX(tr, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    fc = fit.forecast(steps=len(val_idx))
                    for k, t in enumerate(range(len(val_idx))):
                        sarima_pred[t][cells] = fc[k]
                except Exception:
                    pass
            results.append({"Baseline": "SARIMA(1,0,1)(1,0,1)12 per zone", **metrics(yt_val.ravel(), sarima_pred.ravel())})
        else:
            results.append({"Baseline": "SARIMA (skipped: no zone var)", "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": 0})
    except ImportError:
        results.append({"Baseline": "SARIMA (skipped: statsmodels missing)", "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": 0})

    df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  Validation period: time index {CUTOFF}..{n_time - 1} ({len(val_idx)} months)")
    print(f"  Grid: {yv.shape[1]}x{yv.shape[2]} cells\n")
    print(df.to_string(index=False))
    print(f"\n  Saved: {OUT_CSV.relative_to(ROOT)}")
    print("\n  Reference (deep models, mean R2): ConvLSTM~0.60, GNN-TAT~0.60, Late Fusion 0.672")


if __name__ == "__main__":
    main()
