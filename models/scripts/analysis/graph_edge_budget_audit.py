"""Reproduce the GNN-TAT adjacency on CPU and report what the model actually saw.

Two questions the training logs alone cannot settle:

  1. How many edges does the thresholded graph contain, before any budget? The
     notebook prints only the post-budget count, so the pre-budget size, and
     therefore whether the budget binds at all, was never recorded.
  2. Does the leakage fix change the size of the graph, or only its contents? If
     the budget binds in both cases the model sees the same number of edges
     either way, and any performance difference is attributable to the leakage
     fix rather than to a change in graph capacity.

The adjacency is rebuilt here exactly as `SpatialGraphBuilder.build_adjacency_matrix`
does it: k-nearest-neighbour distance similarity, plus a dense elevation
similarity term, plus a thresholded correlation term, normalised by the maximum,
symmetrised, and cut at min_edge_weight.

Usage: python models/scripts/analysis/graph_edge_budget_audit.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
NC = ROOT / "data" / "output" / (
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_"
    "with_onehot_elevation_clean.nc"
)

# Values from CONFIG['gnn_config'] in base_models_gnn_tat_v4.ipynb
MAX_NEIGHBORS = 8
DISTANCE_SCALE_KM = 10.0
ELEVATION_SCALE = 0.2
ELEVATION_WEIGHT = 0.3
CORRELATION_WEIGHT = 0.5
EDGE_THRESHOLD = 0.3
MIN_EDGE_WEIGHT = 0.01
MAX_EDGES = 500_000
OLD_TRIGGER = 1_000_000          # the guard the budget used to sit behind
TRAIN_VAL_SPLIT = 0.8


def haversine_km(lat, lon):
    """Pairwise great-circle distance, matching compute_distance_matrix."""
    la = np.radians(lat)[:, None]
    lo = np.radians(lon)[:, None]
    dlat = la - la.T
    dlon = lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def correlation_matrix(P):
    """Matches compute_correlation_matrix: zero-filled, centred, /T, clipped."""
    T = P.shape[0]
    flat = np.nan_to_num(P.reshape(T, -1), nan=0.0)
    centred = flat - flat.mean(axis=0, keepdims=True)
    norm = centred / (flat.std(axis=0, keepdims=True) + 1e-8)
    return np.clip((norm.T @ norm) / T, -1, 1)


def build_adjacency(base, corr):
    """base: distance + elevation contribution. corr: correlation matrix."""
    adj = base + np.maximum(corr - EDGE_THRESHOLD, 0.0) * CORRELATION_WEIGHT
    np.fill_diagonal(adj, 0.0)
    m = adj.max()
    if m > 0:
        adj /= m
    adj = (adj + adj.T) / 2
    return adj


def edge_set(adj):
    rows, cols = np.nonzero(adj > MIN_EDGE_WEIGHT)
    return rows, cols, adj[rows, cols]


def apply_budget(rows, cols, w, n_nodes):
    if len(w) <= MAX_EDGES:
        return set(zip(rows.tolist(), cols.tolist()))
    keep = np.argsort(w)[-MAX_EDGES:]
    return set(zip(rows[keep].tolist(), cols[keep].tolist()))


def main():
    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)
    lat_c = ds["latitude"].values
    lon_c = ds["longitude"].values
    elev_name = "elevation" if "elevation" in ds.data_vars else None
    elev = ds[elev_name].values if elev_name else np.zeros(P.shape[1:])
    if elev.ndim == 3:
        elev = elev[0]
    ds.close()

    T, n_lat, n_lon = P.shape
    n_nodes = n_lat * n_lon
    split = int(T * TRAIN_VAL_SPLIT)
    print(f"grid {n_lat}x{n_lon} = {n_nodes:,} nodes | T={T} | train split at {split}")
    print(f"complete graph would have {n_nodes * (n_nodes - 1):,} directed edges\n")

    lat_g, lon_g = np.meshgrid(lat_c, lon_c, indexing="ij")
    dist = haversine_km(lat_g.ravel(), lon_g.ravel())

    # --- distance term: k-NN only
    base = np.zeros((n_nodes, n_nodes))
    d = dist.copy()
    d[d == 0] = 1e-6
    sim = 1.0 / (1.0 + d / DISTANCE_SCALE_KM)
    order = np.argsort(d, axis=1)[:, : MAX_NEIGHBORS + 1]
    for i in range(n_nodes):
        nb = order[i][order[i] != i][:MAX_NEIGHBORS]
        base[i, nb] += sim[i, nb]
    n_knn = int((base > 0).sum())
    print(f"distance term alone:  {n_knn:,} non-zero entries "
          f"({n_knn / n_nodes:.1f} per node)")

    # --- elevation term: dense by construction
    e = elev.ravel().astype(np.float64)[:, None]
    rng = e.max() - e.min() + 1e-6
    elev_sim = np.exp(-np.abs(e - e.T) / (rng * ELEVATION_SCALE))
    print(f"elevation term:       dense, min similarity {elev_sim.min():.4g}, "
          f"weighted floor {elev_sim.min() * ELEVATION_WEIGHT:.4g}")
    base += elev_sim * ELEVATION_WEIGHT
    del elev_sim

    # --- the two correlation variants
    results = {}
    for label, series in (("full record (leaked)", P), ("training only (fixed)", P[:split])):
        adj = build_adjacency(base, correlation_matrix(series))
        rows, cols, w = edge_set(adj)
        results[label] = (len(w), apply_budget(rows, cols, w, n_nodes))
        del adj
        n = results[label][0]
        print(f"\n{label}")
        print(f"  edges above min_edge_weight: {n:,}  ({n / n_nodes:.0f} per node)")
        print(f"  fraction of the complete graph: {100 * n / (n_nodes * (n_nodes - 1)):.2f}%")
        print(f"  above the old {OLD_TRIGGER:,} trigger? {'YES' if n > OLD_TRIGGER else 'NO'}")
        print(f"  budget of {MAX_EDGES:,} binds?          "
              f"{'YES' if n > MAX_EDGES else 'NO'}")

    (nf, sf), (nt, st) = results["full record (leaked)"], results["training only (fixed)"]
    inter = len(sf & st)
    print("\n--- the graph the model was actually trained on ---")
    print(f"leaked run:    {len(sf):,} edges after budget")
    print(f"corrected run: {len(st):,} edges after budget")
    print(f"shared:        {inter:,} ({100 * inter / max(len(sf), 1):.2f}% of the leaked set)")
    print(f"changed:       {len(sf) - inter:,} edges swapped by removing the leakage")

    if nf > MAX_EDGES and nt > MAX_EDGES:
        print("\nBoth runs hit the budget, so both trained on exactly "
              f"{MAX_EDGES:,} edges. Graph size is not a confounder: the "
              "difference between the runs is the leakage fix and seed variation.")
    else:
        print("\nThe budget does not bind in both cases; graph size differs "
              "between the runs and remains a confounder.")


if __name__ == "__main__":
    main()
