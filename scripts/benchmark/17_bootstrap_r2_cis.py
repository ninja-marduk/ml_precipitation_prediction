"""Paired bootstrap 95% confidence intervals on R² differences.

Computes per-cell R² for V2 ConvLSTM, V4 GNN-TAT, V10 Late Fusion (full grid,
H=12, BASIC features), then runs paired-cell bootstrap (1000 resamples) on:
  - R²(V10) - R²(V2): overall, Low, Medium, High bands
  - R²(V10) - R²(V4): same
  - R²(V10) Medium - R²(V10) High (the "0.001 difference" claim)

Output: scripts/benchmark/output/bootstrap_r2_cis.json
        scripts/benchmark/output/bootstrap_r2_cis.csv
"""
from pathlib import Path
import json
import sys

import numpy as np
import xarray as xr

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "scripts" / "benchmark" / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Predictions (full grid 61×65, H=12)
V2 = ROOT / "models/output/V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM_Bidirectional"
V4 = ROOT / "models/output/V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT"
V10 = ROOT / "models/output/V10_Late_Fusion"
DATA = ROOT / "data/output/complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"

BANDS = {
    "Low":    (0, 1500),
    "Medium": (1500, 2800),
    "High":   (2800, 6000),
}
N_BOOT = 1000
RNG = np.random.default_rng(42)


def load(d):
    p = np.load(d / "predictions.npy")
    t = np.load(d / "targets.npy")
    if p.ndim == 5:
        p = p[..., 0]
        t = t[..., 0]
    return p, t


def per_cell_r2(pred, targ):
    """Per-cell R² via SS_res/SS_tot. Returns (lat, lon)."""
    n, h, nlat, nlon = pred.shape
    pred = pred.reshape(-1, nlat, nlon)
    targ = targ.reshape(-1, nlat, nlon)
    ss_res = ((targ - pred) ** 2).sum(axis=0)
    targ_mean = targ.mean(axis=0)
    ss_tot = ((targ - targ_mean[None]) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    return r2


def paired_bootstrap_diff(r2_a, r2_b, mask, n_boot=N_BOOT):
    cells = np.argwhere(mask)
    n = len(cells)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        sample = cells[idx]
        delta = np.nanmean(
            r2_a[sample[:, 0], sample[:, 1]] - r2_b[sample[:, 0], sample[:, 1]]
        )
        deltas[b] = delta
    point = np.nanmean(r2_a[cells[:, 0], cells[:, 1]] - r2_b[cells[:, 0], cells[:, 1]])
    return {
        "point": float(point),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "n_cells": int(n),
        "crosses_zero": bool(np.percentile(deltas, 2.5) < 0 < np.percentile(deltas, 97.5)),
    }


def band_r2(r2_map, mask):
    cells = r2_map[mask]
    cells = cells[~np.isnan(cells)]
    return {
        "point": float(cells.mean()),
        "n_cells": int(len(cells)),
    }


def main():
    print("Loading predictions...")
    p2, t = load(V2)
    p4, _ = load(V4)
    p10, _ = load(V10)

    print("Loading elevation grid...")
    ds = xr.open_dataset(DATA)
    elev = np.asarray(ds["elevation"].values, dtype=float)
    if elev.ndim == 3:
        elev = elev[0]
    elif elev.ndim == 4:
        elev = elev[0, :, :, 0]

    print("Computing per-cell R²...")
    r2_v2 = per_cell_r2(p2, t)
    r2_v4 = per_cell_r2(p4, t)
    r2_v10 = per_cell_r2(p10, t)

    masks = {name: (elev >= lo) & (elev < hi) for name, (lo, hi) in BANDS.items()}
    masks["Overall"] = elev >= 0  # all cells

    results = {"per_band_r2": {}, "diffs": {}}

    print("\n=== Per-band R² (point estimates) ===")
    for arch_name, r2 in [("V2_ConvLSTM", r2_v2), ("V4_GNN_TAT", r2_v4),
                          ("V10_Late_Fusion", r2_v10)]:
        results["per_band_r2"][arch_name] = {}
        for band, mask in masks.items():
            d = band_r2(r2, mask)
            results["per_band_r2"][arch_name][band] = d
            print(f"  {arch_name:18s} {band:10s}: R²={d['point']:.4f} (n={d['n_cells']})")

    print("\n=== Paired bootstrap 95% CIs on R² differences ===")
    diffs_to_compute = [
        ("V10_minus_V2",  r2_v10, r2_v2),
        ("V10_minus_V4",  r2_v10, r2_v4),
        ("V2_minus_V4",   r2_v2,  r2_v4),
    ]

    for diff_name, a, b in diffs_to_compute:
        results["diffs"][diff_name] = {}
        for band, mask in masks.items():
            ci = paired_bootstrap_diff(a, b, mask)
            results["diffs"][diff_name][band] = ci
            zero_str = " (CROSSES ZERO)" if ci["crosses_zero"] else ""
            print(f"  {diff_name:18s} {band:10s}: Δ={ci['point']:+.4f} "
                  f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]{zero_str}")

    # The critical "Med vs High for V10" claim
    print("\n=== Critical claim: V10 Medium vs High ===")
    med_mask = masks["Medium"]
    high_mask = masks["High"]
    # Pool both bands then compute paired bootstrap on a vs b within combined cells
    # But these are different cells, so use a different approach:
    # - Compute R² in Medium cells minus mean R² in High cells per resample
    med_cells = np.argwhere(med_mask)
    high_cells = np.argwhere(high_mask)
    deltas_mvh = np.empty(N_BOOT)
    for bidx in range(N_BOOT):
        m_idx = RNG.integers(0, len(med_cells), size=len(med_cells))
        h_idx = RNG.integers(0, len(high_cells), size=len(high_cells))
        m_r2 = r2_v10[med_cells[m_idx, 0], med_cells[m_idx, 1]]
        h_r2 = r2_v10[high_cells[h_idx, 0], high_cells[h_idx, 1]]
        deltas_mvh[bidx] = np.nanmean(m_r2) - np.nanmean(h_r2)
    point_mvh = np.nanmean(r2_v10[med_mask]) - np.nanmean(r2_v10[high_mask])
    ci_mvh = {
        "point": float(point_mvh),
        "ci_low": float(np.percentile(deltas_mvh, 2.5)),
        "ci_high": float(np.percentile(deltas_mvh, 97.5)),
        "crosses_zero": bool(np.percentile(deltas_mvh, 2.5) < 0 < np.percentile(deltas_mvh, 97.5)),
    }
    print(f"  V10 Medium − High: Δ={ci_mvh['point']:+.4f} "
          f"[{ci_mvh['ci_low']:+.4f}, {ci_mvh['ci_high']:+.4f}] "
          f"{'CROSSES ZERO' if ci_mvh['crosses_zero'] else 'DOES NOT CROSS ZERO'}")
    results["v10_medium_minus_high"] = ci_mvh

    out_path = OUT / "bootstrap_r2_cis.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
