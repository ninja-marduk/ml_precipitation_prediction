"""Decompose the Late Fusion gain into calibration and combination components.

The Late Fusion ensemble improves on its base learners, but the improvement has two
distinct sources that the headline number conflates:

  (1) CALIBRATION. Both base learners are amplitude-deflated and mean-biased, so a
      two-parameter affine rescaling y = a*p + b of a SINGLE model already recovers
      much of the gap.
  (2) COMBINATION. Genuinely exploiting two different architectures, over and above
      what recalibrating the better one achieves.

This script fits every variant with the SAME out-of-fold protocol used for the Ridge
meta-learner, so the comparison is like-for-like, and reports the incremental value of
combination against the paper's own inter-seed dispersion.

Usage: python models/scripts/figures/benchmark/fusion_decomposition.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "models" / "output"
V2 = OUT / "V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM_Bidirectional"
V4 = OUT / "V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT"
LF = OUT / "V10_Late_Fusion/SEED42"
N_FOLDS = 5
SEED = 42


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def r2(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    p, t = pred[m], tgt[m]
    return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))


def oof_linear(X, y, groups, n_folds=N_FOLDS):
    """Out-of-fold least-squares with intercept, folds blocked by group (window)."""
    oof = np.full(y.shape, np.nan)
    ug = np.unique(groups)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(ug)
    chunks = np.array_split(order, n_folds)
    for held in chunks:
        te = np.isin(groups, held)
        tr = ~te
        A = np.column_stack([X[tr], np.ones(tr.sum())])
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        oof[te] = np.column_stack([X[te], np.ones(te.sum())]) @ coef
    # full-data coefficients for reporting
    A = np.column_stack([X, np.ones(len(y))])
    full, *_ = np.linalg.lstsq(A, y, rcond=None)
    return oof, full


def main():
    tgt = load(LF, "targets")
    p2, p4, plf = load(V2), load(V4), load(LF)
    assert p2.shape == p4.shape == tgt.shape, (p2.shape, p4.shape, tgt.shape)
    S, H = tgt.shape[0], tgt.shape[1]

    # flatten, keeping window id as the grouping unit for blocked folds
    win = np.broadcast_to(np.arange(S)[:, None, None, None], tgt.shape).ravel()
    y = tgt.ravel()
    x2, x4, xlf = p2.ravel(), p4.ravel(), plf.ravel()
    m = np.isfinite(y) & np.isfinite(x2) & np.isfinite(x4)
    y, x2, x4, xlf, win = y[m], x2[m], x4[m], xlf[m], win[m]

    print(f"evaluated scalars: {y.size:,}  windows: {S}  horizons: {H}")
    print(f"target mean {y.mean():.1f} mm, sd {y.std():.1f} mm\n")

    rows = []
    rows.append(("ConvLSTM-Bidir (raw)", r2(x2, y), ""))
    rows.append(("GNN-TAT-GAT (raw)", r2(x4, y), ""))
    rows.append(("Simple average (50/50)", r2((x2 + x4) / 2, y), "combination only, no calibration"))

    oof2, c2 = oof_linear(x2[:, None], y, win)
    rows.append(("ConvLSTM + affine recal.", r2(oof2, y), f"a={c2[0]:.3f}, b={c2[1]:+.1f} mm"))
    oof4, c4 = oof_linear(x4[:, None], y, win)
    rows.append(("GNN-TAT + affine recal.", r2(oof4, y), f"a={c4[0]:.3f}, b={c4[1]:+.1f} mm"))

    oofR, cR = oof_linear(np.column_stack([x2, x4]), y, win)
    rows.append(("Ridge over both (refit)", r2(oofR, y),
                 f"w2={cR[0]:.3f}, w4={cR[1]:.3f}, b={cR[2]:+.1f} mm"))
    rows.append(("Late Fusion (as published)", r2(xlf, y), "seed-42 released predictions"))

    print(f"{'variant':<28}{'R2':>8}   detail")
    print("-" * 78)
    for n, v, d in rows:
        print(f"{n:<28}{v:>8.4f}   {d}")

    best_raw = max(r2(x2, y), r2(x4, y))
    best_recal = max(r2(oof2, y), r2(oof4, y))
    avg = r2((x2 + x4) / 2, y)
    ridge = r2(oofR, y)
    print("\n--- decomposition of the fusion gain ---")
    print(f"best single raw model                  : {best_raw:.4f}")
    print(f"  + combination only (simple average)  : {avg:.4f}   (delta {avg-best_raw:+.4f})")
    print(f"  + calibration only (best single)     : {best_recal:.4f}   (delta {best_recal-best_raw:+.4f})")
    print(f"  + both (Ridge)                       : {ridge:.4f}   (delta {ridge-best_raw:+.4f})")
    print(f"\nincremental value of COMBINING two architectures,")
    print(f"over simply recalibrating the better one: {ridge-best_recal:+.4f} R2")
    print(f"paper's inter-seed s.d. of Late Fusion  :  0.018")
    verdict = "SMALLER than seed noise" if (ridge - best_recal) < 0.018 else "larger than seed noise"
    print(f"=> the combination term is {verdict}.")


if __name__ == "__main__":
    main()
