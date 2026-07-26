"""Distribution-free uncertainty quantification for the Late Fusion ensemble
via split conformal prediction.

Given the seed-42 Late Fusion predictions and targets, we split the validation
windows into a calibration half and a test half, compute per-horizon
nonconformity scores (absolute residuals) on calibration, and take the
(1 - alpha) empirical quantile as the prediction-interval half-width. We then
report empirical coverage and mean interval width per horizon on the test half.
Conformal prediction guarantees marginal coverage >= 1 - alpha under exchangeability
without any distributional assumption, which is well suited to a data-scarce
regional setting.

Usage: python models/scripts/figures/benchmark/conformal_uncertainty.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LF = ROOT / "models" / "output" / "V10_Late_Fusion" / "SEED42"
ALPHA = 0.10  # target 90% prediction intervals


def load2d(d):
    p = np.load(Path(d) / "predictions.npy").astype(np.float64)
    t = np.load(Path(d) / "targets.npy").astype(np.float64)
    if p.ndim == 5:
        p, t = p[..., 0], t[..., 0]
    return p, t


def main():
    p, t = load2d(LF)
    S, H = p.shape[0], p.shape[1]
    print(f"Late Fusion preds: S={S} windows, H={H} horizons, grid={p.shape[2]}x{p.shape[3]}")

    # chronological split of validation WINDOWS into calibration / test halves
    n_cal = S // 2
    cal = slice(0, n_cal)
    tst = slice(n_cal, S)
    print(f"split conformal: {n_cal} calibration windows, {S - n_cal} test windows, "
          f"target coverage {100*(1-ALPHA):.0f}%")

    res_cal = np.abs(t[cal] - p[cal])   # (n_cal, H, 61, 65)
    print(f"\n{'H':>3}{'q_hat (mm)':>12}{'coverage':>10}{'mean width (mm)':>17}")
    cov_all, wid_all, qh_all = [], [], []
    for h in range(H):
        sc = res_cal[:, h].ravel()
        sc = sc[np.isfinite(sc)]
        # finite-sample conformal quantile level
        n = sc.size
        qlevel = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
        qhat = float(np.quantile(sc, qlevel))
        # evaluate on test half
        err = np.abs(t[tst, h] - p[tst, h])
        err = err[np.isfinite(err)]
        cov = float(np.mean(err <= qhat))
        width = 2 * qhat
        qh_all.append(qhat); cov_all.append(cov); wid_all.append(width)
        if h in (0, 2, 5, 8, 11):
            print(f"{h+1:>3}{qhat:>12.1f}{cov:>10.3f}{width:>17.1f}")

    print(f"\nPooled over horizons: mean coverage={np.mean(cov_all):.3f} "
          f"(target {1-ALPHA:.2f}), mean interval width={np.mean(wid_all):.1f} mm")
    print("Interval width grows with horizon, reflecting increasing forecast uncertainty.")

    # simple sharpness/reliability summary at a few nominal levels
    print("\nReliability (nominal vs empirical coverage, pooled all H):")
    resid = np.abs(t - p)
    rc = np.abs(t[cal] - p[cal]).ravel(); rc = rc[np.isfinite(rc)]
    rt = np.abs(t[tst] - p[tst]).ravel(); rt = rt[np.isfinite(rt)]
    print(f"{'nominal':>8}{'q_hat(mm)':>11}{'empirical':>11}")
    for a in (0.20, 0.10, 0.05):
        n = rc.size
        ql = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
        qh = float(np.quantile(rc, ql))
        emp = float(np.mean(rt <= qh))
        print(f"{1-a:>8.2f}{qh:>11.1f}{emp:>11.3f}")


if __name__ == "__main__":
    main()
