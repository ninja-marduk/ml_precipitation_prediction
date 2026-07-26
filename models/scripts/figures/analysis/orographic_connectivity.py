"""Empirical orographic connectivity: why a topography-aware, non-Euclidean graph
helps for precipitation.

Interpretability for the GNN-TAT inductive bias without needing the trained model.
Using only the CHIRPS training series and SRTM elevation, we quantify how the
inter-cell precipitation correlation depends on elevation similarity *beyond*
geographic distance. If elevation similarity carries correlation that geographic
proximity does not, then long-range, orographically-aligned links exist that a
grid-local CNN cannot see but a topography-weighted graph can, explaining the
parameter efficiency of GNN-TAT.

Usage: python models/scripts/figures/analysis/orographic_connectivity.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[4]
NC = ROOT / "notebooks" / "data" / "output" / \
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"
SPLIT = 414
RNG = np.random.default_rng(42)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    ds = xr.open_dataset(NC)
    P = ds["total_precipitation"].values.astype(np.float64)      # (T,61,65)
    elev = ds["elevation"].values.astype(np.float64)             # (61,65) or (T,61,65)
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    ds.close()
    if elev.ndim == 3:
        elev = elev[0]
    T, ny, nx = P.shape

    # build 2D lat/lon grids
    if lat.ndim == 1 and lon.ndim == 1:
        LON, LAT = np.meshgrid(lon, lat)
    else:
        LAT, LON = lat, lon

    Ptr = P[:SPLIT].reshape(SPLIT, -1)          # train series per cell
    ev = elev.reshape(-1)
    la = LAT.reshape(-1); lo = LON.reshape(-1)
    valid = np.isfinite(Ptr).all(0) & np.isfinite(ev) & np.isfinite(la) & np.isfinite(lo)
    cells = np.where(valid)[0]
    # subsample cells for the pairwise analysis
    if cells.size > 1400:
        cells = RNG.choice(cells, 1400, replace=False)
    X = Ptr[:, cells]                            # (SPLIT, n)
    ev, la, lo = ev[cells], la[cells], lo[cells]
    n = cells.size
    print(f"cells used: {n} (train timesteps={SPLIT})")

    # standardize and correlation matrix
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    C = (Xs.T @ Xs) / SPLIT                       # (n,n) correlation

    iu, ju = np.triu_indices(n, k=1)
    rho = C[iu, ju]
    dist = haversine(la[iu], lo[iu], la[ju], lo[ju])
    dele = np.abs(ev[iu] - ev[ju])
    print(f"pairs: {rho.size}")

    # 1) correlation-distance decay
    print("\n[1] correlation vs geographic distance (binned mean rho):")
    for lo_d, hi_d in [(0, 10), (10, 25), (25, 50), (50, 100), (100, 1e9)]:
        m = (dist >= lo_d) & (dist < hi_d)
        if m.any():
            print(f"  {lo_d:>3}-{hi_d if hi_d < 1e9 else 'inf':>4} km: "
                  f"mean rho={rho[m].mean():+.3f}  (n={m.sum()})")

    # 2) partial effect: does elevation similarity add correlation beyond distance?
    #    regress rho ~ 1 + dist + |dElev|  (standardized predictors)
    A = np.column_stack([np.ones_like(dist),
                         (dist - dist.mean()) / dist.std(),
                         (dele - dele.mean()) / dele.std()])
    beta, *_ = np.linalg.lstsq(A, rho, rcond=None)
    print("\n[2] rho ~ dist + |dElev|  (standardized coefficients):")
    print(f"  beta_distance  = {beta[1]:+.4f}  (correlation falls with distance)")
    print(f"  beta_|dElev|   = {beta[2]:+.4f}  (negative => similar-elevation cells "
          f"correlate more, independent of distance)")

    # 3) long-range orographic links (the non-Euclidean advantage)
    far = dist > 50
    farhi = far & (rho > 0.8)
    oro = farhi & (dele < 200)
    print("\n[3] long-range orographic links:")
    print(f"  pairs >50 km apart: {far.sum()}")
    print(f"  of those, highly correlated (rho>0.8): {farhi.sum()} "
          f"({100*farhi.sum()/max(far.sum(),1):.1f}%)")
    print(f"  of the highly-correlated far pairs, elevation-similar (<200 m): "
          f"{oro.sum()} ({100*oro.sum()/max(farhi.sum(),1):.1f}%)")
    print("  -> such links are invisible to a grid-local CNN but native to a "
          "topography-weighted graph.")


if __name__ == "__main__":
    main()
