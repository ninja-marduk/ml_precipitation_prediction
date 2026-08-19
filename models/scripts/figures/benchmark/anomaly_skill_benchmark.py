"""Deseasonalized anomaly-skill benchmark. SUPERSEDED, kept for the record.

Do not quote this script's output. It pools cells and time into one correlation,
which conflates a static spatial pattern with forecast skill, and it reads the
pre-correction graph array. `beyond_aggregate_corrected.py` replaces it on both
counts and is the source of every anomaly figure in the manuscript. This file is
retained because the manuscript states that the pooled definition inflates the
result, and this is the script that produced the inflated numbers.

Climatology inflates pooled R^2 because it captures the strong seasonal cycle.
The genuine forecasting target is the interannual anomaly (precip minus the
per-cell monthly climatology), on which climatology scores zero by construction.
This script deseasonalizes targets and model predictions and reports anomaly
skill (anomaly R^2 and the anomaly correlation coefficient, ACC) for each model,
pooled and per horizon. Reuses the leakage-safe target-index recovery of
`naive_baselines.py` (targets matched to the CHIRPS series by minimum MSE).

Usage: python models/scripts/figures/benchmark/anomaly_skill_benchmark.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[4]
NC = ROOT / "notebooks" / "data" / "output" / \
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"
CANON = ROOT / "models" / "output" / "V10_Late_Fusion" / "SEED42" / "targets.npy"
SPLIT = 414  # chronological 80% cutoff (train indices < SPLIT)


def _finite(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return [a[m] for a in arrs]


def r2(pred, tgt, tot_ref=None):
    p, t = _finite(pred.ravel(), tgt.ravel())
    ss_res = np.sum((t - p) ** 2)
    ref = t if tot_ref is None else _finite(tot_ref.ravel())[0]
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def acc(a_pred, a_true):
    p, t = _finite(a_pred.ravel(), a_true.ravel())
    if p.std() == 0 or t.std() == 0:
        return np.nan
    return float(np.corrcoef(p, t)[0, 1])


def rmse(pred, tgt):
    p, t = _finite(pred.ravel(), tgt.ravel())
    return float(np.sqrt(np.mean((p - t) ** 2)))


def main():
    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)          # (T, 61, 65)
    months = ds["month"].values.astype(int) if "month" in ds else \
        np.array([int(str(x)[5:7]) for x in ds["time"].values])
    if months.ndim > 1:
        months = months[:, 0, 0]
    ds.close()
    T = P.shape[0]
    Pflat = P.reshape(T, -1)

    tgt = np.load(CANON).astype(np.float64)
    if tgt.ndim == 5:
        tgt = tgt[..., 0]
    S, H = tgt.shape[0], tgt.shape[1]
    print(f"targets: S={S} windows, H={H} horizons, grid={tgt.shape[2]}x{tgt.shape[3]}")

    # recover absolute time index of each (window, horizon) target
    idx = np.full((S, H), -1, int)
    for s in range(S):
        for h in range(H):
            tf = tgt[s, h].reshape(-1)
            mm = np.isfinite(tf)
            d = np.mean((Pflat[:, mm] - tf[mm]) ** 2, axis=1)
            idx[s, h] = int(np.argmin(d))
    print(f"index recovery span idx {idx.min()}..{idx.max()} (train < {SPLIT})")

    # per-cell monthly climatology (train years only)
    clim = np.full((13, P.shape[1], P.shape[2]), np.nan)
    for m in range(1, 13):
        sel = (np.arange(T) < SPLIT) & (months == m)
        if sel.any():
            clim[m] = np.nanmean(P[sel], axis=0)

    pred_clim = np.empty_like(tgt)   # per (s,h) climatology field
    for s in range(S):
        for h in range(H):
            pred_clim[s, h] = clim[months[idx[s, h]]]

    tgt_anom = tgt - pred_clim       # true interannual anomaly

    # models sharing the same validation targets
    OUT = ROOT / "models" / "output"
    model_dirs = {
        "ConvLSTM-Bidir": OUT / "V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM_Bidirectional",
        "GNN-TAT-GAT":    OUT / "V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT",
        "Late Fusion":    OUT / "V10_Late_Fusion/SEED42",
    }

    def load(d):
        d = Path(d)
        p = np.load(d / "predictions.npy").astype(np.float64)
        if p.ndim == 5:
            p = p[..., 0]
        return p

    horizons = [0, 2, 5, 8, 11]  # H=1,3,6,9,12
    print("\n================ ANOMALY-SKILL BENCHMARK ================")
    print("Deseasonalized (precip - per-cell monthly climatology).")
    print(f"{'Model':<16}{'rawR2':>7}{'anomR2':>8}{'ACC':>7}{'anomRMSE':>10}   per-H ACC (1/3/6/9/12)")
    # climatology reference row
    print(f"{'Climatology':<16}{r2(pred_clim, tgt):>7.3f}{0.0:>8.3f}{0.0:>7.3f}"
          f"{rmse(pred_clim, tgt):>10.1f}   (0 by construction)")

    results = {}
    for name, d in model_dirs.items():
        try:
            p = load(d)
        except Exception as e:
            print(f"{name:<16} load error: {e}")
            continue
        if p.shape != tgt.shape:
            print(f"{name:<16} shape {p.shape} != {tgt.shape} (skip)")
            continue
        a_pred = p - pred_clim
        rawr2 = r2(p, tgt)
        aR2 = r2(a_pred, tgt_anom, tot_ref=tgt_anom)
        aACC = acc(a_pred, tgt_anom)
        aRMSE = rmse(a_pred, tgt_anom)
        perH = [acc(a_pred[:, h], tgt_anom[:, h]) for h in horizons]
        results[name] = dict(rawR2=rawr2, anomR2=aR2, ACC=aACC, anomRMSE=aRMSE, perH=perH)
        print(f"{name:<16}{rawr2:>7.3f}{aR2:>8.3f}{aACC:>7.3f}{aRMSE:>10.1f}   "
              + " ".join(f"{x:+.2f}" for x in perH))

    # climatology-residual framing: does ML add skill on the residual?
    print("\nInterpretation: anomR2>0 or ACC>0 means the model captures genuine")
    print("interannual variability that climatology cannot (climatology anomR2=ACC=0).")
    return results


if __name__ == "__main__":
    main()
