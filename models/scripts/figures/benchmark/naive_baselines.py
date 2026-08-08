"""Naive reference baselines (climatology, persistence, seasonal-naive) for the
precipitation benchmark, evaluated on the SAME validation windows/targets that
the deep-learning models were scored against.

Addresses the reviewer gap: "climatology achieves high aggregate skill" — we now
report the naive baselines explicitly so the predictability-ceiling claim is
rigorous. No model re-training required (targets recovered from saved arrays).

Usage: python models/scripts/figures/benchmark/naive_baselines.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[4]
NC = ROOT / "notebooks" / "data" / "output" / \
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"
TGT = ROOT / "models" / "output" / "V10_Late_Fusion" / "SEED42" / "targets.npy"
SPLIT = 414  # chronological 80% cutoff (train = indices < SPLIT)


def r2(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    p, t = pred[m], tgt[m]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def per_cell_nse(pred, tgt):
    """NSE against each cell's own temporal mean, then averaged over cells.

    The pooled R2 above uses one global mean, so it is credited with the
    between-cell variance that a per-cell climatology also captures. This is the
    metric that does not give that credit.
    """
    p = pred.reshape(-1, pred.shape[-2] * pred.shape[-1])
    g = tgt.reshape(-1, tgt.shape[-2] * tgt.shape[-1])
    out = []
    for c in range(g.shape[1]):
        m = np.isfinite(p[:, c]) & np.isfinite(g[:, c])
        if m.sum() < 3:
            continue
        gt = g[m, c]
        ss_tot = np.sum((gt - gt.mean()) ** 2)
        if ss_tot <= 0:
            continue
        out.append(1 - np.sum((gt - p[m, c]) ** 2) / ss_tot)
    return float(np.mean(out)) if out else np.nan


def rmse(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    return float(np.sqrt(np.mean((pred[m] - tgt[m]) ** 2)))


def mae(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    return float(np.mean(np.abs(pred[m] - tgt[m])))


def main():
    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)      # (518, 61, 65)
    months = ds["month"].values.astype(int) if "month" in ds else \
        (np.array([int(str(t)[5:7]) for t in ds["time"].values]))
    if months.ndim > 1:
        months = months[:, 0, 0]
    ds.close()
    T = P.shape[0]

    tgt = np.load(TGT)
    if tgt.ndim == 5:
        tgt = tgt[..., 0]
    S, H = tgt.shape[0], tgt.shape[1]
    print(f"targets: S={S} windows, H={H} horizons, grid={tgt.shape[2]}x{tgt.shape[3]}")

    # --- recover absolute time index of each (window, horizon) target ----------
    # match each target field against the full series by minimum MSE
    Pflat = P.reshape(T, -1)
    idx = np.full((S, H), -1, int)
    for s in range(S):
        for h in range(H):
            tf = tgt[s, h].reshape(-1)
            mm = np.isfinite(tf)
            d = np.mean((Pflat[:, mm] - tf[mm]) ** 2, axis=1)
            idx[s, h] = int(np.argmin(d))
    # sanity: horizons consecutive, windows stride-1
    consec = np.all(np.diff(idx, axis=1) == 1)
    stride = np.all(np.diff(idx[:, 0]) == 1)
    resid = np.mean([np.mean((P[idx[s, h]] - tgt[s, h]) ** 2)
                     for s in range(S) for h in range(H)])
    print(f"index recovery: horizons_consecutive={consec} window_stride1={stride} "
          f"mean_match_MSE={resid:.3e}")
    print(f"validation target months span idx {idx.min()}..{idx.max()} "
          f"(split at {SPLIT})")

    # --- climatology: per-cell training mean for the target calendar month -----
    clim = np.full((13, P.shape[1], P.shape[2]), np.nan)
    for m in range(1, 13):
        sel = (np.arange(T) < SPLIT) & (months == m)
        if sel.any():
            clim[m] = np.nanmean(P[sel], axis=0)

    pred_clim = np.empty_like(tgt)
    pred_pers = np.empty_like(tgt)   # naive persistence: value at forecast origin
    pred_seas = np.empty_like(tgt)   # seasonal-naive: value 12 months earlier
    for s in range(S):
        origin = idx[s, 0] - 1       # month before first target (forecast origin)
        for h in range(H):
            t = idx[s, h]
            pred_clim[s, h] = clim[months[t]]
            pred_pers[s, h] = P[origin]
            pred_seas[s, h] = P[t - 12] if t - 12 >= 0 else np.nan

    # --- report: pooled over all horizons, and per-horizon --------------------
    def block(name, pred):
        print(f"\n[{name}]")
        print(f"  POOLED (all H): R2={r2(pred, tgt):+.3f}  RMSE={rmse(pred, tgt):5.1f}  "
              f"MAE={mae(pred, tgt):5.1f}  per-cell NSE={per_cell_nse(pred, tgt):+.3f}")
        for h in [0, 4, 11]:  # H=1, H=5, H=12
            print(f"  H={h+1:2d}: R2={r2(pred[:, h], tgt[:, h]):+.3f}  "
                  f"RMSE={rmse(pred[:, h], tgt[:, h]):5.1f}  "
                  f"MAE={mae(pred[:, h], tgt[:, h]):5.1f}")

    block("Climatology (monthly, per-cell)", pred_clim)
    block("Persistence (naive, origin value)", pred_pers)
    block("Seasonal-naive (t-12)", pred_seas)

    # --- reconcile: DL models pooled with the SAME R2 definition ---------------
    def load(d):
        d = Path(d)
        if not (d / "predictions.npy").exists() and (d / "SEED42").exists():
            d = d / "SEED42"
        p = np.load(d / "predictions.npy").astype(np.float64)
        t = np.load(d / "targets.npy").astype(np.float64)
        if p.ndim == 5:
            p, t = p[..., 0], t[..., 0]
        return p, t

    OUT = ROOT / "models" / "output"
    models = {
        "ConvLSTM-Bidir (V2)": OUT / "V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM_Bidirectional",
        # Two arrays exist for this model. The root map_exports/ is what the
        # released pipeline wrote; SEED42/map_exports/ is the corrected rerun.
        # Both are scored so the difference is visible rather than path-dependent.
        "GNN-TAT-GAT (V4, as released)":
            OUT / "V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT",
        "GNN-TAT-GAT (V4, corrected rerun)":
            OUT / "V4_GNN_TAT_Models/SEED42/map_exports/H12/BASIC/GNN_TAT_GAT",
        "Late Fusion (V10)": OUT / "V10_Late_Fusion",
    }
    print("\n================ RECONCILED COMPARISON (same pooled R²) ================")
    print(f"{'Method':<36}{'R2(pooled)':>11}{'R2(H=5)':>9}{'R2(H=12)':>10}"
          f"{'RMSE':>7}{'perCellNSE':>12}")
    rows = [("Climatology", pred_clim, tgt), ("Seasonal-naive", pred_seas, tgt),
            ("Persistence", pred_pers, tgt)]
    for name, d in models.items():
        try:
            p, t = load(d)
            if p.shape != tgt.shape:
                print(f"{name:<26}  shape {p.shape} != targets {tgt.shape} (skip pooled cmp)")
                continue
            rows.append((name, p, t))
        except Exception as e:
            print(f"{name:<26}  load error: {e}")
    for name, p, t in rows:
        print(f"{name:<36}{r2(p, t):>+11.3f}{r2(p[:, 4], t[:, 4]):>+9.3f}"
              f"{r2(p[:, 11], t[:, 11]):>+10.3f}{rmse(p, t):>7.1f}"
              f"{per_cell_nse(p, t):>+12.3f}")


if __name__ == "__main__":
    main()
