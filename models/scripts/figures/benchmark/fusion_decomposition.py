"""Decompose the Late Fusion gain into calibration and combination components.

The Late Fusion ensemble improves on its base learners, but the improvement has two
distinct sources that the headline number conflates:

  (1) CALIBRATION. Both base learners are amplitude-deflated and mean-biased, so a
      two-parameter affine rescaling y = a*p + b of a SINGLE model already recovers
      much of the gap.
  (2) COMBINATION. Genuinely exploiting two different architectures, over and above
      what recalibrating the better one achieves.

Every variant is fitted with out-of-fold least squares whose folds are blocked by
validation window, so no held-out scalar has a near-duplicate neighbour in training.
This matters: the released meta-learner (`ridge_fusion_oof` in the V10 notebook)
flattens windows, horizons and cells into one vector and calls KFold(shuffle=True),
which leaves each held-out cell surrounded by its own neighbours in the training
fold. Its reported skill is therefore not out-of-sample, and this script exists to
replace it.

The design is fully crossed: the same three seeds run every architecture and the
fusion. Comparisons are consequently PAIRED, and the uncertainty of a difference is
the standard deviation of the per-seed difference, not the marginal spread of each
arm. The distinction decides whether the combination term separates from noise.

Usage: python models/scripts/figures/benchmark/fusion_decomposition.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "models" / "output"
SEEDS = (42, 123, 456)
N_FOLDS = 5


def paths(seed):
    return (OUT / f"V2_Enhanced_Models/SEED{seed}/map_exports/H12/BASIC/ConvLSTM_Bidirectional",
            OUT / f"V4_GNN_TAT_Models/SEED{seed}/map_exports/H12/BASIC/GNN_TAT_GAT",
            OUT / f"V10_Late_Fusion/SEED{seed}")


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def r2(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    p, t = pred[m], tgt[m]
    return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))


def oof_linear(X, y, groups, seed, n_folds=N_FOLDS):
    """Out-of-fold least squares with intercept, folds blocked by group (window)."""
    oof = np.full(y.shape, np.nan)
    ug = np.unique(groups)
    order = np.random.default_rng(seed).permutation(ug)
    for held in np.array_split(order, n_folds):
        te = np.isin(groups, held)
        tr = ~te
        A = np.column_stack([X[tr], np.ones(tr.sum())])
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        oof[te] = np.column_stack([X[te], np.ones(te.sum())]) @ coef
    A = np.column_stack([X, np.ones(len(y))])
    full, *_ = np.linalg.lstsq(A, y, rcond=None)
    return oof, full


def one_seed(seed):
    d2, d4, dlf = paths(seed)
    tgt = load(dlf, "targets")
    p2, p4, plf = load(d2), load(d4), load(dlf)
    if not (p2.shape == p4.shape == tgt.shape):
        raise SystemExit(f"seed {seed}: shape mismatch {p2.shape} {p4.shape} {tgt.shape}")
    S = tgt.shape[0]

    win = np.broadcast_to(np.arange(S)[:, None, None, None], tgt.shape).ravel()
    y, x2, x4, xlf = tgt.ravel(), p2.ravel(), p4.ravel(), plf.ravel()
    m = np.isfinite(y) & np.isfinite(x2) & np.isfinite(x4)
    y, x2, x4, xlf, win = y[m], x2[m], x4[m], xlf[m], win[m]

    oof2, c2 = oof_linear(x2[:, None], y, win, seed)
    oof4, c4 = oof_linear(x4[:, None], y, win, seed)
    oofR, cR = oof_linear(np.column_stack([x2, x4]), y, win, seed)

    return {
        "seed": seed, "n": y.size, "windows": S,
        "conv_raw": r2(x2, y), "gnn_raw": r2(x4, y),
        "average": r2((x2 + x4) / 2, y),
        "conv_recal": r2(oof2, y), "gnn_recal": r2(oof4, y),
        "ridge": r2(oofR, y), "published": r2(xlf, y),
        "coef": (cR[0], cR[1], cR[2]),
    }


def main():
    res = [one_seed(s) for s in SEEDS]

    print(f"window-blocked out-of-fold, {N_FOLDS} folds, {len(SEEDS)} seeds")
    print(f"evaluated scalars per seed: {res[0]['n']:,} over {res[0]['windows']} windows\n")

    cols = [("conv_raw", "ConvLSTM raw"), ("gnn_raw", "GNN-TAT raw"),
            ("average", "simple average"), ("conv_recal", "ConvLSTM recalibrated"),
            ("gnn_recal", "GNN-TAT recalibrated"), ("ridge", "Ridge over both"),
            ("published", "Late Fusion as published")]
    print(f"{'variant':<26}" + "".join(f"{'seed '+str(s):>11}" for s in SEEDS)
          + f"{'mean':>10}{'s.d.':>8}")
    print("-" * 89)
    for k, label in cols:
        v = np.array([r[k] for r in res])
        print(f"{label:<26}" + "".join(f"{x:>11.4f}" for x in v)
              + f"{v.mean():>10.4f}{v.std(ddof=1):>8.4f}")

    # ---- paired differences, computed within seed then aggregated
    print("\n--- decomposition, paired within seed ---")
    print(f"{'term':<46}{'mean':>9}{'s.d.':>8}   per-seed")
    print("-" * 89)

    def paired(f, label):
        d = np.array([f(r) for r in res])
        print(f"{label:<46}{d.mean():>+9.4f}{d.std(ddof=1):>8.4f}   "
              + " ".join(f"{x:+.4f}" for x in d))
        return d

    best_raw = lambda r: max(r["conv_raw"], r["gnn_raw"])
    best_recal = lambda r: max(r["conv_recal"], r["gnn_recal"])
    d_comb_only = paired(lambda r: r["average"] - best_raw(r),
                         "combination only (average) over best raw")
    d_cal_only = paired(lambda r: best_recal(r) - best_raw(r),
                        "calibration only (recalibrate best) over best raw")
    d_both = paired(lambda r: r["ridge"] - best_raw(r),
                    "both (Ridge) over best raw")
    d_incr = paired(lambda r: r["ridge"] - best_recal(r),
                    "INCREMENTAL value of combining, over recalibrating")

    share = d_cal_only.mean() / d_both.mean() if d_both.mean() != 0 else np.nan
    print(f"\ncalibration accounts for {100*share:.0f}% of the total Ridge gain")

    # ---- the verdict, on the paired standard deviation
    t = d_incr.mean() / (d_incr.std(ddof=1) / np.sqrt(len(d_incr))) if d_incr.std(ddof=1) > 0 else np.inf
    marginal = np.array([r["ridge"] for r in res]).std(ddof=1)
    print("\n--- does the combination term separate from noise? ---")
    print(f"paired mean                     : {d_incr.mean():+.4f}")
    print(f"paired s.d. across seeds        : {d_incr.std(ddof=1):.4f}")
    print(f"marginal s.d. of Ridge R2       : {marginal:.4f}   <- what the manuscript compared against")
    print(f"paired t (n={len(SEEDS)}, 2 d.f.)          : {t:.2f}   (|t|>4.30 for p<0.05)")
    if abs(t) > 4.303:
        print("=> the combination term DOES separate from noise under the paired test,")
        print("   although n=3 makes this a weak inference; the manuscript's claim that it")
        print("   is smaller than seed noise used the marginal spread and must be revised.")
    else:
        print("=> the combination term does NOT separate from noise, and the conclusion")
        print("   survives the correction to a paired test.")

    print("\n--- published meta-learner versus a correctly blocked refit ---")
    pub = np.array([r["published"] for r in res])
    rid = np.array([r["ridge"] for r in res])
    dd = pub - rid
    print(f"published (KFold shuffled over cells) : {pub.mean():.4f} +/- {pub.std(ddof=1):.4f}")
    print(f"refit (folds blocked by window)       : {rid.mean():.4f} +/- {rid.std(ddof=1):.4f}")
    print(f"paired difference                     : {dd.mean():+.4f} +/- {dd.std(ddof=1):.4f}")
    print("The published figure is the one to replace throughout the manuscript.")


if __name__ == "__main__":
    main()
