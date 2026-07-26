"""Audit the temporal leakage in the GNN-TAT graph construction.

The V4 graph builder derives part of each edge weight from the inter-cell precipitation
correlation. In the released pipeline that correlation is computed over the FULL record,
including the held-out months, so the fixed spatial prior of the graph encodes
information from the evaluation period. This script quantifies how much the adjacency
changes when the correlation is instead estimated on the training period only.

Edge-weight recipe (from the V4 configuration):
    w_corr = max(0, rho - edge_threshold) * correlation_weight
with edge_threshold = 0.3 and correlation_weight = 0.5.

Usage: python models/scripts/analysis/graph_leakage_audit.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
NC = ROOT / "notebooks" / "data" / "output" / \
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"
SPLIT = 414             # training period is P[:SPLIT]
EDGE_THRESHOLD = 0.3
CORRELATION_WEIGHT = 0.5
MIN_EDGE_WEIGHT = 0.01
N_CELLS = 1500
RNG = np.random.default_rng(42)


def corr_matrix(X):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return (Xs.T @ Xs) / X.shape[0]


def main():
    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)
    ds.close()
    T = P.shape[0]
    flat = P.reshape(T, -1)
    valid = np.isfinite(flat).all(0)
    cells = np.where(valid)[0]
    if cells.size > N_CELLS:
        cells = RNG.choice(cells, N_CELLS, replace=False)
    Xfull = flat[:, cells]
    Xtrain = flat[:SPLIT, cells]
    print(f"cells sampled: {cells.size} | full record T={T} | training T={SPLIT} "
          f"| held-out months in the full estimate: {T - SPLIT}")

    Cf = corr_matrix(Xfull)
    Ct = corr_matrix(Xtrain)
    iu, ju = np.triu_indices(cells.size, 1)
    rf, rt = Cf[iu, ju], Ct[iu, ju]

    wf = np.maximum(0.0, rf - EDGE_THRESHOLD) * CORRELATION_WEIGHT
    wt = np.maximum(0.0, rt - EDGE_THRESHOLD) * CORRELATION_WEIGHT

    print("\n--- correlation estimates ---")
    print(f"mean rho  full={rf.mean():+.4f}   train-only={rt.mean():+.4f}   "
          f"mean shift={np.mean(rf - rt):+.4f}")
    print(f"Pearson r between the two edge sets: {np.corrcoef(rf, rt)[0,1]:.4f}")
    print(f"mean |delta rho| = {np.mean(np.abs(rf - rt)):.4f}   "
          f"max |delta rho| = {np.max(np.abs(rf - rt)):.4f}")

    print("\n--- correlation edge weights ---")
    print(f"mean |delta w| = {np.mean(np.abs(wf - wt)):.5f}  "
          f"(mean w_full = {wf.mean():.5f})")
    rel = np.mean(np.abs(wf - wt)) / max(wf.mean(), 1e-12)
    print(f"relative perturbation of the correlation term: {100*rel:.2f}%")

    on_f = wf > MIN_EDGE_WEIGHT
    on_t = wt > MIN_EDGE_WEIGHT
    appear = (~on_t) & on_f
    vanish = on_t & (~on_f)
    print(f"\nedges retained (w > {MIN_EDGE_WEIGHT}):  full={on_f.sum():,}   train-only={on_t.sum():,}")
    print(f"edges that exist ONLY because held-out months were used: {appear.sum():,} "
          f"({100*appear.sum()/max(on_f.sum(),1):.2f}% of the full-record edge set)")
    print(f"edges that disappear when leakage is removed:            {vanish.sum():,}")
    changed = appear.sum() + vanish.sum()
    print(f"total edge-set disagreement: {changed:,} of {on_f.sum():,} "
          f"({100*changed/max(on_f.sum(),1):.2f}%)")

    print("\nInterpretation: the correlation term is a fixed, data-estimated component of the")
    print("model's spatial prior. Any non-zero disagreement means the released GNN-TAT graph")
    print("encodes information from the evaluation period that the convolutional baselines")
    print("never saw, so the architecture comparison is not clean until the graph is rebuilt")
    print("on P[:SPLIT] and every affected result is regenerated.")


if __name__ == "__main__":
    main()
