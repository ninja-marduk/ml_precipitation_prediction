"""What does the GNN-TAT graph actually connect?

The architecture is presented as a non-Euclidean spatial prior whose edges encode
topographic relations. That claim is about the graph the model receives, which is
not the thresholded adjacency but the 500,000 highest-weight edges the budget
keeps out of 15.7 million. Which edges survive, and what they are made of, has
never been measured.

This script rebuilds the adjacency exactly as the notebook does, applies the
budget, and characterises the surviving edge set:

  - how each weight decomposes into the distance, elevation and correlation terms
  - the geographic span of the retained edges, against the span of all pairs
  - the elevation contrast they connect
  - the degree distribution, which the average of 126 per node conceals

The question it answers is whether the retained graph is a local neighbourhood
graph, in which case it is doing what a convolution does, or whether it links
distant cells with similar terrain, which is the orographic claim.

Usage: python models/scripts/analysis/graph_structure_diagnostic.py
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

MAX_NEIGHBORS = 8
DISTANCE_SCALE_KM = 10.0
ELEVATION_SCALE = 0.2
ELEVATION_WEIGHT = 0.3
CORRELATION_WEIGHT = 0.5
EDGE_THRESHOLD = 0.3
MIN_EDGE_WEIGHT = 0.01
MAX_EDGES = 500_000
TRAIN_VAL_SPLIT = 0.8


def haversine_km(lat, lon):
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    a = (np.sin((la - la.T) / 2) ** 2
         + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def correlation_matrix(P):
    T = P.shape[0]
    flat = np.nan_to_num(P.reshape(T, -1), nan=0.0)
    centred = flat - flat.mean(axis=0, keepdims=True)
    norm = centred / (flat.std(axis=0, keepdims=True) + 1e-8)
    return np.clip((norm.T @ norm) / T, -1, 1)


def pct(x, q):
    return np.percentile(x, q)


def main():
    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)
    lat_c, lon_c = ds["latitude"].values, ds["longitude"].values
    elev = ds["elevation"].values
    ds.close()
    if elev.ndim == 3:
        elev = elev[0]

    T, n_lat, n_lon = P.shape
    n_nodes = n_lat * n_lon
    split = int(T * TRAIN_VAL_SPLIT)

    lat_g, lon_g = np.meshgrid(lat_c, lon_c, indexing="ij")
    dist = haversine_km(lat_g.ravel(), lon_g.ravel())
    e = elev.ravel().astype(np.float64)
    d_elev = np.abs(e[:, None] - e[None, :])

    # --- the three terms, kept separately so the weight can be decomposed
    d = dist.copy()
    d[d == 0] = 1e-6
    sim = 1.0 / (1.0 + d / DISTANCE_SCALE_KM)
    W_dist = np.zeros((n_nodes, n_nodes))
    order = np.argsort(d, axis=1)[:, : MAX_NEIGHBORS + 1]
    for i in range(n_nodes):
        nb = order[i][order[i] != i][:MAX_NEIGHBORS]
        W_dist[i, nb] += sim[i, nb]

    rng_e = e.max() - e.min() + 1e-6
    W_elev = np.exp(-d_elev / (rng_e * ELEVATION_SCALE)) * ELEVATION_WEIGHT
    W_corr = np.maximum(correlation_matrix(P[:split]) - EDGE_THRESHOLD, 0.0) * CORRELATION_WEIGHT

    adj = W_dist + W_elev + W_corr
    np.fill_diagonal(adj, 0.0)
    scale = adj.max()
    adj /= scale
    adj = (adj + adj.T) / 2

    rows, cols = np.nonzero(adj > MIN_EDGE_WEIGHT)
    w = adj[rows, cols]
    n_thresh = len(w)
    keep = np.argsort(w)[-MAX_EDGES:]
    r, c = rows[keep], cols[keep]

    print(f"grid {n_lat}x{n_lon} = {n_nodes:,} nodes, elevation {e.min():.0f} to {e.max():.0f} m")
    print(f"thresholded graph: {n_thresh:,} edges ({100*n_thresh/(n_nodes*(n_nodes-1)):.2f}% complete)")
    print(f"after the {MAX_EDGES:,} budget: {len(r):,} edges, "
          f"{len(r)/n_nodes:.0f} per node on average\n")

    # ---------------------------------------------------------- weight composition
    # symmetrisation halves each directed contribution, so compare like with like
    td = (W_dist[r, c] + W_dist[c, r]) / 2 / scale
    te = (W_elev[r, c] + W_elev[c, r]) / 2 / scale
    tc = (W_corr[r, c] + W_corr[c, r]) / 2 / scale
    tot = td + te + tc
    print("--- what the retained weights are made of ---")
    print(f"{'term':<14}{'share of total weight':>24}{'edges where it dominates':>28}")
    for name, t in (("distance kNN", td), ("elevation", te), ("correlation", tc)):
        dom = (t >= np.maximum.reduce([td, te, tc]))
        print(f"{name:<14}{100*t.sum()/tot.sum():>23.1f}%{100*dom.mean():>27.1f}%")

    kept_knn = int((W_dist[r, c] > 0).sum())
    print(f"\nof the {n_nodes*MAX_NEIGHBORS:,} k-NN edges, {kept_knn:,} survive the budget "
          f"({100*kept_knn/(n_nodes*MAX_NEIGHBORS):.1f}%)")
    print(f"they are {100*kept_knn/len(r):.2f}% of the retained graph")

    # ---------------------------------------------------------- geography
    dk = dist[r, c]
    iu, ju = np.triu_indices(n_nodes, 1)
    sample = np.random.default_rng(0).choice(len(iu), 2_000_000, replace=False)
    all_d = dist[iu[sample], ju[sample]]
    print("\n--- geographic span (km) ---")
    print(f"{'':<18}{'p05':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p95':>9}{'max':>9}")
    for name, x in (("retained edges", dk), ("all pairs", all_d)):
        print(f"{name:<18}" + "".join(f"{pct(x, q):>9.1f}" for q in (5, 25, 50, 75, 95, 100)))
    for thr in (10, 25, 50):
        print(f"retained edges shorter than {thr:>3} km: {100*(dk < thr).mean():>5.1f}%   "
              f"(all pairs: {100*(all_d < thr).mean():.1f}%)")

    # ---------------------------------------------------------- elevation
    de = d_elev[r, c]
    all_e = d_elev[iu[sample], ju[sample]]
    print("\n--- elevation contrast connected (m) ---")
    print(f"{'':<18}{'p05':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p95':>9}{'max':>9}")
    for name, x in (("retained edges", de), ("all pairs", all_e)):
        print(f"{name:<18}" + "".join(f"{pct(x, q):>9.0f}" for q in (5, 25, 50, 75, 95, 100)))

    far = dk > 50
    print(f"\nedges longer than 50 km: {far.sum():,} ({100*far.mean():.1f}%)")
    if far.any():
        print(f"  their median elevation contrast: {np.median(de[far]):.0f} m "
              f"(vs {np.median(de):.0f} m over all retained edges)")

    # ---------------------------------------------------------- degree
    deg = np.bincount(r, minlength=n_nodes)
    print("\n--- degree distribution (the 126 per node average conceals this) ---")
    print(f"min {deg.min()}  p05 {pct(deg,5):.0f}  median {pct(deg,50):.0f}  "
          f"p95 {pct(deg,95):.0f}  max {deg.max()}")
    print(f"isolated nodes (degree 0): {(deg == 0).sum():,} of {n_nodes:,} "
          f"({100*(deg == 0).mean():.1f}%)")
    top = np.sort(deg)[::-1]
    print(f"the busiest 10% of nodes hold {100*top[:n_nodes//10].sum()/deg.sum():.1f}% of all edges")

    # ---------------------------------------------------------- verdict inputs
    print("\n--- read this against the architectural claim ---")
    print(f"share of retained edges that are also k-NN neighbours: {100*kept_knn/len(r):.2f}%")
    print(f"share of retained weight contributed by elevation:     {100*te.sum()/tot.sum():.1f}%")
    print(f"share of retained weight contributed by correlation:   {100*tc.sum()/tot.sum():.1f}%")
    print(f"median edge length {np.median(dk):.1f} km against a median pair distance of "
          f"{np.median(all_d):.1f} km")


if __name__ == "__main__":
    main()
