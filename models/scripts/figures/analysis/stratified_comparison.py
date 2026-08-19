"""Stratified ConvLSTM vs GNN-TAT vs Late Fusion comparison, recomputed from the
released validation predictions.

The manuscript previously carried two stratified tables that disagreed on which
architecture wins each elevation band. This script recomputes the comparison from a
single source of truth (the seed-42 validation predictions) under one stated
convention, so that the stratified analysis and the elevation-stratified table agree.

Convention: per-cell R^2 (Nash-Sutcliffe against the cell's temporal mean within the
stratum), averaged over the cells of the stratum, pooled over all horizons unless a
horizon stratum is specified.

Usage: python models/scripts/figures/analysis/stratified_comparison.py
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

# Two arrays exist for the seed-42 graph model. The root map_exports/ holds the
# pre-correction run, whose graph encoded the evaluation months; SEED42/map_exports/
# holds the retrain on the corrected graph. Both are scored here, side by side, so a
# table built from this output cannot silently mix them, which is what happened when
# the stratified tables were built from one and the fusion table from the other.
MODELS = {
    "ConvLSTM": OUT / "V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM_Bidirectional",
    "GNN-TAT": OUT / "V4_GNN_TAT_Models/SEED42/map_exports/H12/BASIC/GNN_TAT_GAT",
    "GNN-pre": OUT / "V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT",
    "LateFusion": OUT / "V10_Late_Fusion/SEED42",
}
SHOW = ["ConvLSTM", "GNN-TAT", "GNN-pre", "LateFusion"]


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


def percell_r2(pred, tgt, cellmask=None, tsel=None):
    """Mean per-cell R^2 over selected timesteps and cells. pred/tgt: (N, ny, nx)."""
    p, t = pred, tgt
    if tsel is not None:
        p, t = p[tsel], t[tsel]
    N = p.shape[0]
    p2 = p.reshape(N, -1)
    t2 = t.reshape(N, -1)
    if cellmask is not None:
        cm = cellmask.reshape(-1)
        p2, t2 = p2[:, cm], t2[:, cm]
    mu = np.nanmean(t2, axis=0, keepdims=True)
    ss_res = np.nansum((t2 - p2) ** 2, axis=0)
    ss_tot = np.nansum((t2 - mu) ** 2, axis=0)
    good = ss_tot > 0
    return float(np.nanmean(1 - ss_res[good] / ss_tot[good]))


def main():
    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)
    elev = ds["elevation"].values.astype(np.float64)
    elev = elev[0] if elev.ndim == 3 else elev
    months = ds["month"].values.astype(int) if "month" in ds else \
        np.array([int(str(x)[5:7]) for x in ds["time"].values])
    if months.ndim > 1:
        months = months[:, 0, 0]
    ds.close()
    T = P.shape[0]
    Pflat = P.reshape(T, -1)

    tgt = load(MODELS["LateFusion"], "targets")
    S, H = tgt.shape[0], tgt.shape[1]
    preds = {k: load(v) for k, v in MODELS.items()}
    for k, v in preds.items():
        assert v.shape == tgt.shape, (k, v.shape, tgt.shape)

    # recover the calendar month of each (window, horizon) target for seasonal strata
    idx = np.full((S, H), -1, int)
    for s in range(S):
        for h in range(H):
            tf = tgt[s, h].reshape(-1)
            mm = np.isfinite(tf)
            idx[s, h] = int(np.argmin(np.mean((Pflat[:, mm] - tf[mm]) ** 2, axis=1)))
    tmonth = months[idx]                       # (S, H)

    # collapse (window, horizon) into one sample axis
    flat = {k: v.reshape(S * H, *v.shape[2:]) for k, v in preds.items()}
    ftgt = tgt.reshape(S * H, *tgt.shape[2:])
    fmon = tmonth.reshape(S * H)
    fhor = np.broadcast_to(np.arange(H)[None, :], (S, H)).reshape(S * H)

    def row(label, cellmask=None, tsel=None, n=None):
        vals = {k: percell_r2(flat[k], ftgt, cellmask, tsel) for k in flat}
        # the winner is decided among the current arrays only; the pre-correction
        # column is shown for continuity and is not a candidate
        win = max((k for k in vals if k != "GNN-pre"), key=lambda k: vals[k])
        extra = f"n={n}" if n is not None else ""
        print(f"{label:<22}" + "".join(f"{vals[k]:>12.3f}" for k in SHOW)
              + f"   winner: {win:<11}{extra}")
        return vals

    print("Per-cell R^2 by stratum (seed 42, pooled over horizons unless stated)\n")
    print(f"{'stratum':<22}" + "".join(f"{k:>12}" for k in SHOW))
    print("-" * 86)

    allv = row("ALL", None, None, n=int(np.isfinite(elev).sum()))

    print("\n[elevation]")
    bands = [("Low <1500 m", elev < 1500), ("Medium 1500-2800 m", (elev >= 1500) & (elev < 2800)),
             ("High >2800 m", elev >= 2800)]
    elev_rows = {}
    for nm, mk in bands:
        elev_rows[nm] = row(nm, cellmask=mk, n=int(mk.sum()))

    print("\n[season]")
    seasons = [("DJF", np.isin(fmon, [12, 1, 2])), ("MAM", np.isin(fmon, [3, 4, 5])),
               ("JJA", np.isin(fmon, [6, 7, 8])), ("SON", np.isin(fmon, [9, 10, 11]))]
    for nm, sel in seasons:
        if sel.sum() > 2:
            row(nm, tsel=sel, n=int(sel.sum()))

    print("\n[horizon group]")
    for nm, sel in [("Short H1-4", fhor < 4), ("Medium H5-8", (fhor >= 4) & (fhor < 8)),
                    ("Long H9-12", fhor >= 8)]:
        row(nm, tsel=sel, n=int(sel.sum()))

    print("\n[elevation degradation, Low to High]")
    lo, hi = elev_rows["Low <1500 m"], elev_rows["High >2800 m"]
    for k in SHOW:
        print(f"  {k:<12} {lo[k]:.3f} -> {hi[k]:.3f}   {100*(1-hi[k]/lo[k]):+.1f}%")

    print("\n[sensitivity of that degradation to the upper threshold]")
    print("  the 2,800 m boundary is an ecological convention, so the claim that the")
    print("  degradation is a property of the terrain rather than of the cut is only")
    print("  worth making if it survives moving the cut")
    degr = []
    for thr in range(2400, 3001, 100):
        m_lo, m_hi = elev < 1500, elev >= thr
        if m_hi.sum() < 50:
            continue
        v_lo = percell_r2(flat["LateFusion"], ftgt, m_lo)
        v_hi = percell_r2(flat["LateFusion"], ftgt, m_hi)
        d = 100 * (1 - v_hi / v_lo)
        degr.append(d)
        print(f"  upper cut {thr:>5} m  n={int(m_hi.sum()):>5}  "
              f"Low {v_lo:.3f} -> High {v_hi:.3f}   {d:+.1f}%")
    if degr:
        print(f"  Late Fusion degradation over the sweep: "
              f"{min(degr):.1f}% to {max(degr):.1f}%")

    print("\n[per-cell complementarity]")
    print("  a claim that two models fail in complementary places is a claim about")
    print("  cells, not about zone means, so it is counted here")

    def percell_vector(pred, tgt):
        N = pred.shape[0]
        p2, t2 = pred.reshape(N, -1), tgt.reshape(N, -1)
        mu = np.nanmean(t2, axis=0, keepdims=True)
        ss_res = np.nansum((t2 - p2) ** 2, axis=0)
        ss_tot = np.nansum((t2 - mu) ** 2, axis=0)
        out = np.full(ss_tot.shape, np.nan)
        ok = ss_tot > 0
        out[ok] = 1 - ss_res[ok] / ss_tot[ok]
        return out

    vc = percell_vector(flat["ConvLSTM"], ftgt)
    vg = percell_vector(flat["GNN-TAT"], ftgt)
    vf = percell_vector(flat["LateFusion"], ftgt)
    ok = np.isfinite(vc) & np.isfinite(vg) & np.isfinite(vf)
    n = int(ok.sum())
    print(f"  cells where the graph model beats the convolutional one: "
          f"{int((vg[ok] > vc[ok]).sum())} of {n} "
          f"({100*(vg[ok] > vc[ok]).mean():.1f}%)")
    rescue = ok & (np.minimum(vc, vg) < 0.2) & (vf >= 0.5)
    print(f"  cells where either base learner is below 0.2 and the fusion reaches 0.5: "
          f"{int(rescue.sum())} of {n} ({100*rescue.sum()/n:.1f}%)")
    both_low = ok & (np.maximum(vc, vg) < 0.2)
    print(f"  cells where both base learners are below 0.2: {int(both_low.sum())} "
          f"({100*both_low.sum()/n:.1f}%), of which the fusion reaches 0.5 in "
          f"{int((both_low & (vf >= 0.5)).sum())}")

    # complementarity verdict
    print("\n--- complementarity check ---")
    wins = {"ConvLSTM": 0, "GNN-TAT": 0}
    for nm, mk in bands:
        v = elev_rows[nm]
        wins["ConvLSTM" if v["ConvLSTM"] > v["GNN-TAT"] else "GNN-TAT"] += 1
    print(f"elevation bands won: ConvLSTM {wins['ConvLSTM']}/3, GNN-TAT {wins['GNN-TAT']}/3")
    print("If one architecture wins every stratum, a stratified (switching) ensemble")
    print("degenerates to selecting that single model, which is the H7 outcome.")


if __name__ == "__main__":
    main()
