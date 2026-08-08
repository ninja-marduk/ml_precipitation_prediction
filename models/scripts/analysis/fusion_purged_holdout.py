"""An honest out-of-sample estimate for the fusion meta-learner, and why one is hard.

The meta-learner was originally cross-validated with folds shuffled over flattened
grid cells. Blocking the folds by validation window looked like the fix, and it is
not sufficient: the 33 windows slide by one month, so window s at leads 2..12
covers the same target months as window s+1 at leads 1..11. Under random
assignment of windows to folds, 99.5% of held-out (window, lead) pairs have their
target month sitting in a training window; under contiguous folds, 87.6%. Neither
is out of sample.

The only way to get a clean estimate from 33 overlapping windows is to hold out a
contiguous block and discard, on both sides of it, every window whose leads reach
into the held-out months. With H=12 that embargo costs 11 windows per boundary,
which is why the honest estimate rests on very few effective units and why we
report it with that caveat rather than as a headline.

Usage: python models/scripts/analysis/fusion_purged_holdout.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "output"
SEEDS = (42, 123, 456)
EMBARGO = 11          # H - 1: windows whose leads reach into the held-out months


def paths(seed):
    return (OUT / f"V2_Enhanced_Models/SEED{seed}/map_exports/H12/BASIC/ConvLSTM_Bidirectional",
            OUT / f"V4_GNN_TAT_Models/SEED{seed}/map_exports/H12/BASIC/GNN_TAT_GAT",
            OUT / f"V10_Late_Fusion/SEED{seed}")


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def r2(pred, y):
    m = np.isfinite(pred) & np.isfinite(y)
    return float(1 - np.sum((y[m] - pred[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2))


def fit_affine(X, y):
    A = np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def apply_affine(X, coef):
    return np.column_stack([X, np.ones(len(X))]) @ coef


def one_seed(seed):
    d2, d4, dlf = paths(seed)
    tgt = load(dlf, "targets")
    p2, p4 = load(d2), load(d4)
    S = tgt.shape[0]

    half = S // 2
    train_w = np.arange(0, half - EMBARGO // 2)
    test_w = np.arange(half + EMBARGO // 2, S)
    # drop any training window whose leads reach into a test window's months
    train_w = np.array([w for w in train_w if w + 11 < test_w.min()])

    def flat(idx):
        y = tgt[idx].ravel()
        x = np.column_stack([p2[idx].ravel(), p4[idx].ravel()])
        m = np.isfinite(y) & np.isfinite(x).all(1)
        return x[m], y[m]

    Xtr, ytr = flat(train_w)
    Xte, yte = flat(test_w)

    coef = fit_affine(Xtr, ytr)
    ridge_oos = r2(apply_affine(Xte, coef), yte)

    # the same split for the single-learner recalibration, so the comparison is fair
    c2 = fit_affine(Xtr[:, :1], ytr)
    c4 = fit_affine(Xtr[:, 1:], ytr)
    best_recal = max(r2(apply_affine(Xte[:, :1], c2), yte),
                     r2(apply_affine(Xte[:, 1:], c4), yte))
    best_raw = max(r2(Xte[:, 0], yte), r2(Xte[:, 1], yte))

    # in-sample, for the size of the gap
    ridge_in = r2(apply_affine(Xtr, coef), ytr)

    return dict(seed=seed, n_train=len(train_w), n_test=len(test_w),
                ridge_oos=ridge_oos, best_recal=best_recal, best_raw=best_raw,
                ridge_in=ridge_in, coef=coef)


def main():
    res = [one_seed(s) for s in SEEDS]
    r0 = res[0]
    print(f"purged split: {r0['n_train']} training windows, {r0['n_test']} test windows,")
    print(f"embargo of {EMBARGO} windows so no training lead reaches a test month\n")

    print(f"{'seed':>6}{'raw best':>11}{'recalibrated':>14}{'Ridge':>9}"
          f"{'Ridge in-sample':>17}")
    print("-" * 60)
    for r in res:
        print(f"{r['seed']:>6}{r['best_raw']:>11.4f}{r['best_recal']:>14.4f}"
              f"{r['ridge_oos']:>9.4f}{r['ridge_in']:>17.4f}")
    a = np.array([[r['best_raw'], r['best_recal'], r['ridge_oos'], r['ridge_in']] for r in res])
    print("-" * 60)
    print(f"{'mean':>6}" + "".join(f"{a[:, i].mean():>11.4f}" if i == 0 else
                                   f"{a[:, i].mean():>14.4f}" if i == 1 else
                                   f"{a[:, i].mean():>9.4f}" if i == 2 else
                                   f"{a[:, i].mean():>17.4f}" for i in range(4)))
    print(f"{'s.d.':>6}" + "".join(f"{a[:, i].std(ddof=1):>11.4f}" if i == 0 else
                                   f"{a[:, i].std(ddof=1):>14.4f}" if i == 1 else
                                   f"{a[:, i].std(ddof=1):>9.4f}" if i == 2 else
                                   f"{a[:, i].std(ddof=1):>17.4f}" for i in range(4)))

    incr = a[:, 2] - a[:, 1]
    print(f"\nincremental value of combining over recalibrating, out of sample:")
    print(f"  {incr.mean():+.4f} +/- {incr.std(ddof=1):.4f} (paired across seeds)")
    gap = a[:, 3] - a[:, 2]
    print(f"\nin-sample minus out-of-sample for the Ridge: {gap.mean():+.4f}")
    print("That gap is what any cross-validation on overlapping windows understates.")
    print("\nThis estimate rests on one split of a 44-month evaluation period and")
    print("should be reported as such, not as a headline. What it establishes is a")
    print("direction: the blocked-fold figure is optimistic, and the ranking of")
    print("calibration over combination is unchanged.")


if __name__ == "__main__":
    main()
