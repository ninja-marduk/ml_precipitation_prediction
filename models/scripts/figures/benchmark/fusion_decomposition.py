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


def fit_full(X, y):
    A = np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def oof_shuffled(X, y, seed, n_folds=N_FOLDS):
    """The same estimator under folds shuffled over flattened scalars.

    This exists to isolate one thing. The published figure differs from the blocked
    refit in two respects at once: the fold scheme, and the estimator itself, since
    the released meta-learner is a regularised fit made inside the training notebook.
    Attributing the whole difference to the fold scheme, as an earlier version of this
    analysis did, credits the fold defect with a cost it does not carry alone. Running
    the identical estimator under both schemes splits the difference into a part that
    is the scheme and a remainder that is not.
    """
    oof = np.full(y.shape, np.nan)
    idx = np.random.default_rng(seed).permutation(len(y))
    for held in np.array_split(idx, n_folds):
        te = np.zeros(len(y), bool)
        te[held] = True
        tr = ~te
        A = np.column_stack([X[tr], np.ones(tr.sum())])
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        oof[te] = np.column_stack([X[te], np.ones(te.sum())]) @ coef
    return oof


def rmse(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    return float(np.sqrt(np.mean((tgt[m] - pred[m]) ** 2)))


def bias(pred, tgt):
    m = np.isfinite(pred) & np.isfinite(tgt)
    return float(np.mean(pred[m] - tgt[m]))


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
    oofS = oof_shuffled(np.column_stack([x2, x4]), y, seed)

    return {
        "seed": seed, "n": y.size, "windows": S,
        "conv_raw": r2(x2, y), "gnn_raw": r2(x4, y),
        "average": r2((x2 + x4) / 2, y),
        "conv_recal": r2(oof2, y), "gnn_recal": r2(oof4, y),
        "ridge": r2(oofR, y), "ridge_shuffled": r2(oofS, y), "published": r2(xlf, y),
        "ridge_rmse": rmse(oofR, y), "ridge_bias": bias(oofR, y),
        "ridge_mae": float(np.mean(np.abs(oofR - y))),
        "pub_rmse": rmse(xlf, y), "pub_bias": bias(xlf, y),
        "conv_rmse": rmse(x2, y), "conv_mae": float(np.mean(np.abs(x2 - y))),
        "conv_bias": bias(x2, y),
        "gnn_rmse": rmse(x4, y), "gnn_mae": float(np.mean(np.abs(x4 - y))),
        "gnn_bias": bias(x4, y),
        "conv_slope": float(c2[0]), "conv_intercept": float(c2[1]),
        "gnn_slope": float(c4[0]), "gnn_intercept": float(c4[1]),
        "coef": (cR[0], cR[1], cR[2]),
    }


def main():
    res = [one_seed(s) for s in SEEDS]

    print(f"window-blocked out-of-fold, {N_FOLDS} folds, {len(SEEDS)} seeds")
    print(f"evaluated scalars per seed: {res[0]['n']:,} over {res[0]['windows']} windows\n")

    cols = [("conv_raw", "ConvLSTM raw"), ("gnn_raw", "GNN-TAT raw"),
            ("average", "simple average"), ("conv_recal", "ConvLSTM recalibrated"),
            ("gnn_recal", "GNN-TAT recalibrated"), ("ridge", "Ridge over both"),
            ("ridge_shuffled", "Ridge, folds shuffled"),
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

    # ---- the three rows of Table 'late-fusion-results', from one set of arrays
    r0 = res[0]
    print("\n--- seed 42: what Table 'late-fusion-results' must contain ---")
    print("all three rows scored on the same targets, from the same prediction arrays")
    print(f"{'row':<32}{'R2':>8}{'RMSE':>9}{'MAE':>8}{'Bias':>9}")
    print("-" * 66)
    for lbl, k in (("ConvLSTM-Bidir (base learner)", "conv"),
                   ("GNN-TAT-GAT (base learner)", "gnn")):
        print(f"{lbl:<32}{r0[k + '_raw']:>8.3f}{r0[k + '_rmse']:>9.2f}"
              f"{r0[k + '_mae']:>8.2f}{r0[k + '_bias']:>+9.2f}")
    print(f"{'Late Fusion, blocked refit':<32}{r0['ridge']:>8.3f}{r0['ridge_rmse']:>9.2f}"
          f"{r0['ridge_mae']:>8.2f}{r0['ridge_bias']:>+9.2f}")
    print(f"{'Late Fusion, as released':<32}{r0['published']:>8.3f}{r0['pub_rmse']:>9.2f}"
          f"{'-':>8}{r0['pub_bias']:>+9.2f}")
    print("The graph row is the corrected array. An earlier version of this table gave")
    print("0.597 for it, which is the pre-correction array, next to a fusion score from")
    print("the corrected one. Mixing the two is what this block exists to prevent.")

    # ---- per-seed detail the manuscript tabulates, emitted here so it traces
    print("\n--- per seed, blocked refit: what Table 'multiseed-ridge' must contain ---")
    print(f"{'seed':>6}{'R2':>9}{'RMSE (mm)':>11}{'w_Conv':>9}{'w_GNN':>8}"
          f"{'b (mm)':>9}{'bias (mm)':>11}")
    print("-" * 63)
    for r in res:
        print(f"{r['seed']:>6}{r['ridge']:>9.4f}{r['ridge_rmse']:>11.2f}"
              f"{r['coef'][0]:>9.3f}{r['coef'][1]:>8.3f}{r['coef'][2]:>9.2f}"
              f"{r['ridge_bias']:>11.3f}")
    arr = lambda k: np.array([r[k] for r in res])
    wc, wg = arr("ridge_rmse"), None
    print("-" * 63)
    print(f"{'mean':>6}{arr('ridge').mean():>9.4f}{arr('ridge_rmse').mean():>11.2f}"
          f"{np.array([r['coef'][0] for r in res]).mean():>9.3f}"
          f"{np.array([r['coef'][1] for r in res]).mean():>8.3f}"
          f"{np.array([r['coef'][2] for r in res]).mean():>9.2f}"
          f"{arr('ridge_bias').mean():>11.3f}")
    print(f"{'s.d.':>6}{arr('ridge').std(ddof=1):>9.4f}{arr('ridge_rmse').std(ddof=1):>11.2f}"
          f"{np.array([r['coef'][0] for r in res]).std(ddof=1):>9.3f}"
          f"{np.array([r['coef'][1] for r in res]).std(ddof=1):>8.3f}"
          f"{np.array([r['coef'][2] for r in res]).std(ddof=1):>9.2f}"
          f"{arr('ridge_bias').std(ddof=1):>11.3f}")
    c0 = np.array([r["coef"][0] for r in res]).mean()
    c1 = np.array([r["coef"][1] for r in res]).mean()
    print(f"\nconvolutional share of the mean weight: {100*c0/(c0+c1):.1f}%"
          f"  (graph {100*c1/(c0+c1):.1f}%)")
    ratio = [r["coef"][1] / r["coef"][0] for r in res]
    print("w_GNN / w_Conv per seed: " + ", ".join(f"{x:.2f}" for x in ratio))
    print("base-learner calibration, fitted on the full sample:")
    for r in res:
        print(f"  seed {r['seed']}: ConvLSTM slope {r['conv_slope']:.3f} "
              f"intercept {r['conv_intercept']:+.2f} mm | GNN-TAT slope "
              f"{r['gnn_slope']:.3f} intercept {r['gnn_intercept']:+.2f} mm")

    print("\n--- published meta-learner versus a correctly blocked refit ---")
    pub = arr("published")
    rid = arr("ridge")
    shf = arr("ridge_shuffled")
    dd = pub - rid
    print(f"published (released meta-learner, shuffled) : {pub.mean():.4f} +/- {pub.std(ddof=1):.4f}")
    print(f"same estimator, folds shuffled over scalars : {shf.mean():.4f} +/- {shf.std(ddof=1):.4f}")
    print(f"same estimator, folds blocked by window     : {rid.mean():.4f} +/- {rid.std(ddof=1):.4f}")
    print(f"total, published minus blocked refit        : {dd.mean():+.4f} +/- {dd.std(ddof=1):.4f}")
    d_fold = shf - rid
    d_est = pub - shf
    print(f"  of which the fold scheme                  : {d_fold.mean():+.4f} "
          f"+/- {d_fold.std(ddof=1):.4f}")
    print(f"  of which not the folds                    : {d_est.mean():+.4f} "
          f"+/- {d_est.std(ddof=1):.4f}")
    print("Quote the fold-scheme term as the cost of the fold defect. The total is not")
    print("that cost. The remainder carries two things at once that cannot be separated")
    print("with what is in the archive: the released meta-learner is a regularised fit,")
    print("not this least-squares refit, and it was fitted on the pre-correction graph")
    print("predictions. Its per-seed base-learner scores in v10_summary.json are")
    print("0.5974, 0.4642 and 0.3866 for the graph model, against 0.3893, 0.4970 and")
    print("0.4556 for the corrected arrays used here. The pre-correction per-seed")
    print("arrays were not archived, so the remainder is reported as unattributed.")

    # what can still be measured: seed 42 keeps its pre-correction array at the
    # root of map_exports, so for that seed the input change is quantifiable.
    leaked = OUT / "V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT"
    if (leaked / "predictions.npy").exists():
        d2, _, dlf = paths(42)
        tgt = load(dlf, "targets")
        p2, p4l = load(d2), load(leaked)
        if p4l.shape == tgt.shape:
            S42 = tgt.shape[0]
            win = np.broadcast_to(np.arange(S42)[:, None, None, None], tgt.shape).ravel()
            y, x2, x4l = tgt.ravel(), p2.ravel(), p4l.ravel()
            m = np.isfinite(y) & np.isfinite(x2) & np.isfinite(x4l)
            oof, _ = oof_linear(np.column_stack([x2[m], x4l[m]]), y[m], win[m], 42)
            v = r2(oof, y[m])
            print(f"\nseed 42 only, blocked refit on the pre-correction graph array: "
                  f"{v:.4f}")
            print(f"  against {res[0]['ridge']:.4f} on the corrected array, "
                  f"a difference of {v - res[0]['ridge']:+.4f} from the inputs alone.")


if __name__ == "__main__":
    main()
