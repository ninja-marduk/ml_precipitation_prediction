"""Corrected 'beyond aggregate skill' diagnostics.

This supersedes three earlier analyses whose definitions inflated their results:

  (A) ANOMALY SKILL. The anomaly correlation must be computed per cell over time
      (the standard ACC of the forecast-verification literature), or per time step
      over space, and reported separately. Pooling both axes into a single Pearson
      correlation conflates a time-invariant spatial anomaly pattern with genuine
      forecast skill and inflates the result.
  (C) OROGRAPHIC CONNECTIVITY. Inter-cell correlations must be computed on
      deseasonalized series. On raw precipitation every pair is dominated by the
      shared annual cycle, which manufactures spurious 'long-range links'.
  (D) CONFORMAL INTERVALS. The exchangeable unit is the validation window, not the
      individual grid-cell scalar; intervals must be truncated at zero for a
      non-negative variable; coverage must be reported conditionally as well as
      marginally; and the procedure must be compared against the same interval built
      on a zero-cost climatology.

Usage: python models/scripts/figures/benchmark/beyond_aggregate_corrected.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[4]
NC = ROOT / "notebooks" / "data" / "output" / \
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"
OUT = ROOT / "models" / "output"
SPLIT = 414
ALPHA = 0.10
RNG = np.random.default_rng(42)


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def main():
    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)
    elev = ds["elevation"].values.astype(np.float64)
    elev = elev[0] if elev.ndim == 3 else elev
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    months = ds["month"].values.astype(int) if "month" in ds else \
        np.array([int(str(x)[5:7]) for x in ds["time"].values])
    if months.ndim > 1:
        months = months[:, 0, 0]
    ds.close()
    T = P.shape[0]
    Pflat = P.reshape(T, -1)

    LF = OUT / "V10_Late_Fusion/SEED42"
    tgt = load(LF, "targets")
    S, H = tgt.shape[0], tgt.shape[1]

    idx = np.full((S, H), -1, int)
    for s in range(S):
        for h in range(H):
            tf = tgt[s, h].reshape(-1)
            mm = np.isfinite(tf)
            idx[s, h] = int(np.argmin(np.mean((Pflat[:, mm] - tf[mm]) ** 2, axis=1)))

    clim = np.full((13,) + P.shape[1:], np.nan)
    for m in range(1, 13):
        sel = (np.arange(T) < SPLIT) & (months == m)
        if sel.any():
            clim[m] = np.nanmean(P[sel], axis=0)
    pred_clim = np.empty_like(tgt)
    for s in range(S):
        for h in range(H):
            pred_clim[s, h] = clim[months[idx[s, h]]]
    tgt_anom = tgt - pred_clim

    models = {
        "ConvLSTM-Bidir": OUT / "V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM_Bidirectional",
        "GNN-TAT-GAT":    OUT / "V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT",
        "Late Fusion":    LF,
    }

    # ================= (A) ANOMALY SKILL, correct definitions =================
    print("=" * 74)
    print("(A) ANOMALY SKILL  [per-cell temporal ACC = standard definition]")
    print("=" * 74)
    print(f"{'model':<16}{'H':>3}{'ACC_cell':>10}{'%cells>0':>10}{'ACC_pattern':>13}{'ACC_pooled':>12}")
    for name, d in models.items():
        p = load(d)
        if p.shape != tgt.shape:
            print(f"{name}: shape mismatch, skipped"); continue
        ap = p - pred_clim
        for h in [0, 11]:
            A, Tn = ap[:, h], tgt_anom[:, h]              # (S, ny, nx)
            # per-cell temporal ACC
            a2 = A.reshape(S, -1); t2 = Tn.reshape(S, -1)
            a2c = a2 - a2.mean(0); t2c = t2 - t2.mean(0)
            num = (a2c * t2c).sum(0)
            den = np.sqrt((a2c ** 2).sum(0) * (t2c ** 2).sum(0))
            good = den > 0
            acc_cell = num[good] / den[good]
            # per-window spatial pattern ACC
            a3 = a2 - a2.mean(1, keepdims=True); t3 = t2 - t2.mean(1, keepdims=True)
            n2 = (a3 * t3).sum(1); d2 = np.sqrt((a3 ** 2).sum(1) * (t3 ** 2).sum(1))
            acc_pat = n2[d2 > 0] / d2[d2 > 0]
            # pooled (the inflated definition previously used)
            fa, ft = A.ravel(), Tn.ravel()
            m = np.isfinite(fa) & np.isfinite(ft)
            pooled = np.corrcoef(fa[m], ft[m])[0, 1]
            print(f"{name:<16}{h+1:>3}{np.nanmean(acc_cell):>10.3f}"
                  f"{100*np.mean(acc_cell > 0):>9.1f}%{np.nanmean(acc_pat):>13.3f}{pooled:>12.3f}")

    # ================= (C) OROGRAPHIC CONNECTIVITY, deseasonalized =================
    print()
    print("=" * 74)
    print("(C) OROGRAPHIC CONNECTIVITY  [raw vs deseasonalized]")
    print("=" * 74)
    Ptr = P[:SPLIT]
    mtr = months[:SPLIT]
    des = Ptr.copy()
    for m in range(1, 13):
        sel = mtr == m
        if sel.any():
            des[sel] -= np.nanmean(Ptr[sel], axis=0)
    LON, LAT = np.meshgrid(lon, lat) if (lat.ndim == 1 and lon.ndim == 1) else (lon, lat)
    ev = elev.reshape(-1); la = LAT.reshape(-1); lo = LON.reshape(-1)
    flat_raw = Ptr.reshape(SPLIT, -1); flat_des = des.reshape(SPLIT, -1)
    valid = np.isfinite(flat_raw).all(0) & np.isfinite(ev) & np.isfinite(la) & np.isfinite(lo)
    cc = np.where(valid)[0]
    cc = RNG.choice(cc, 1400, replace=False) if cc.size > 1400 else cc
    ev, la, lo = ev[cc], la[cc], lo[cc]
    iu, ju = np.triu_indices(cc.size, 1)
    R = 6371.0
    dist = 2 * R * np.arcsin(np.sqrt(
        np.sin(np.radians(la[ju] - la[iu]) / 2) ** 2 +
        np.cos(np.radians(la[iu])) * np.cos(np.radians(la[ju])) *
        np.sin(np.radians(lo[ju] - lo[iu]) / 2) ** 2))
    dele = np.abs(ev[iu] - ev[ju])
    for label, flat in [("RAW", flat_raw), ("DESEASONALIZED", flat_des)]:
        X = flat[:, cc]
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        C = (Xs.T @ Xs) / SPLIT
        rho = C[iu, ju]
        far = dist > 50
        hi = far & (rho > 0.8)
        oro = hi & (dele < 200)
        A = np.column_stack([np.ones_like(dist),
                             (dist - dist.mean()) / dist.std(),
                             (dele - dele.mean()) / dele.std()])
        beta, *_ = np.linalg.lstsq(A, rho, rcond=None)
        pred = A @ beta
        r2reg = 1 - np.sum((rho - pred) ** 2) / np.sum((rho - rho.mean()) ** 2)
        print(f"\n[{label}]")
        print(f"  mean rho <10 km = {rho[dist < 10].mean():+.3f} | >100 km = {rho[dist > 100].mean():+.3f}")
        print(f"  far pairs (>50 km) with rho>0.8 : {hi.sum():,} ({100*hi.sum()/far.sum():.1f}% of far pairs)")
        print(f"    of those, |dElev|<200 m       : {oro.sum():,}")
        print(f"  partial regression: beta_dist={beta[1]:+.4f}  beta_|dElev|={beta[2]:+.4f}  "
              f"R2={r2reg:.3f}  ratio |elev/dist|={abs(beta[2]/beta[1]):.2f}")

    # ================= (D) CONFORMAL, window-exchangeable =================
    print()
    print("=" * 74)
    print("(D) CONFORMAL INTERVALS  [window-level exchangeability, truncated at 0]")
    print("=" * 74)
    plf = load(LF)
    ncal = S // 2
    n_units = S - ncal
    lvl = np.ceil((ncal + 1) * (1 - ALPHA)) / ncal
    print(f"exchangeable units: {ncal} calibration windows, {n_units} test windows")
    print(f"required conformal level at n={ncal}: {min(lvl,1.0):.3f}"
          + ("  (=1.0: the guarantee requires the maximum residual)" if lvl >= 1 else ""))

    def conformal(pred):
        widths, covs, negfrac = [], [], []
        for h in range(H):
            rcal = np.abs(tgt[:ncal, h] - pred[:ncal, h])
            # window-level score: one scalar per calibration window (max abs residual is
            # too conservative; we use the per-window 90th percentile of |residual|)
            per_win = np.nanpercentile(rcal.reshape(ncal, -1), 90, axis=1)
            k = int(np.ceil((ncal + 1) * (1 - ALPHA))) - 1
            k = min(max(k, 0), ncal - 1)
            q = np.sort(per_win)[k]
            err = np.abs(tgt[ncal:, h] - pred[ncal:, h])
            covs.append(np.nanmean(err <= q))
            lo_b = pred[ncal:, h] - q
            widths.append(np.nanmean(np.minimum(pred[ncal:, h] + q, np.inf) - np.maximum(lo_b, 0)))
            negfrac.append(np.nanmean(lo_b < 0))
        return np.array(widths), np.array(covs), np.array(negfrac)

    w_lf, c_lf, neg_lf = conformal(plf)
    w_cl, c_cl, _ = conformal(pred_clim)
    print(f"\nLate Fusion : coverage {c_lf.mean():.3f} | mean truncated width {w_lf.mean():.1f} mm"
          f" | width H1->H12 {w_lf[0]:.0f}->{w_lf[-1]:.0f}")
    print(f"Climatology : coverage {c_cl.mean():.3f} | mean truncated width {w_cl.mean():.1f} mm")
    print(f"=> zero-cost climatology intervals are "
          f"{'NARROWER' if w_cl.mean() < w_lf.mean() else 'wider'} by {abs(w_cl.mean()-w_lf.mean()):.1f} mm")
    print(f"untruncated lower bound below zero in {100*neg_lf.mean():.1f}% of cells "
          f"(intervals are truncated at 0 above)")

    # conditional coverage by elevation band
    ev2 = elev
    bands = [("Low <1500", ev2 < 1500), ("Med 1500-2800", (ev2 >= 1500) & (ev2 < 2800)), ("High >2800", ev2 >= 2800)]
    print("\nconditional coverage by elevation (pooled over horizons):")
    for nm, mask in bands:
        cs = []
        for h in range(H):
            rcal = np.abs(tgt[:ncal, h] - plf[:ncal, h])
            per_win = np.nanpercentile(rcal.reshape(ncal, -1), 90, axis=1)
            k = min(max(int(np.ceil((ncal + 1) * (1 - ALPHA))) - 1, 0), ncal - 1)
            q = np.sort(per_win)[k]
            err = np.abs(tgt[ncal:, h] - plf[ncal:, h])[:, mask]
            cs.append(np.nanmean(err <= q))
        print(f"  {nm:<15} n={mask.sum():>5}  coverage={np.mean(cs):.3f}")


if __name__ == "__main__":
    main()
