"""Moran's I on per-cell residuals — spatial autocorrelation justification.

Computes Moran's I for the per-cell mean residuals of V2, V4, V10 using
rook adjacency on the 61×65 grid. Implementation is manual (no pysal
dependency) for portability.

A weak Moran's I (|I| < 0.30) on residuals justifies the temporal-only
train/test split (no need for spatial block CV).

Output: scripts/benchmark/output/morans_i.json
"""
from pathlib import Path
import sys
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "scripts" / "benchmark" / "output"
OUT.mkdir(parents=True, exist_ok=True)

V2 = ROOT / "models/output/V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM_Bidirectional"
V4 = ROOT / "models/output/V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT"
V10 = ROOT / "models/output/V10_Late_Fusion"


def load(d):
    p = np.load(d / "predictions.npy"); t = np.load(d / "targets.npy")
    if p.ndim == 5: p = p[..., 0]; t = t[..., 0]
    return p, t


def per_cell_mean_residual(pred, targ):
    """Mean residual per cell over all (sample × horizon) combinations."""
    n, h, nlat, nlon = pred.shape
    res = (targ - pred).reshape(-1, nlat, nlon)
    return res.mean(axis=0)  # (lat, lon)


def morans_i_rook(values):
    """Compute Moran's I with rook adjacency on a 2D grid.

    Args: values: (lat, lon) float array
    Returns: dict with I, expected_I (= -1/(N-1)), z-score, p-value (one-sided)
    """
    from scipy.stats import norm

    nlat, nlon = values.shape
    valid = np.isfinite(values)
    flat_valid = valid.flatten()
    flat_values = values.flatten()
    n = int(flat_valid.sum())
    mean = flat_values[flat_valid].mean()
    deviations = flat_values - mean
    deviations[~flat_valid] = 0.0

    # Build rook adjacency: each cell paired with N/E/S/W neighbours
    numerator = 0.0
    sum_w = 0.0
    for i in range(nlat):
        for j in range(nlon):
            if not valid[i, j]:
                continue
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < nlat and 0 <= nj < nlon and valid[ni, nj]:
                    numerator += deviations[i * nlon + j] * deviations[ni * nlon + nj]
                    sum_w += 1.0  # binary rook weight

    denominator = (deviations[flat_valid] ** 2).sum()

    if sum_w == 0 or denominator == 0:
        return {"I": float("nan"), "expected": float("nan"),
                "p_normal": float("nan")}

    I = (n / sum_w) * (numerator / denominator)
    expected_I = -1.0 / (n - 1)

    # Approximate variance and z-score using normality assumption
    # Var(I) ≈ (n^2 * S1 - n * S2 + 3 * sum_w^2) / ((n-1)^2 * (n+1) * sum_w^2) - expected_I^2
    # For binary rook weights: S1 = 2*sum_w, S2 = sum of (row_sum + col_sum)^2 ≈ 4*4*n
    # Simplified for large n:
    var_I = ((n - 1) * sum_w * 4 * 2 - sum_w ** 2 * (n - 1)) / ((n - 1) ** 2 * (n + 1) * sum_w ** 2)
    var_I = max(abs(var_I), 1e-10)
    z = (I - expected_I) / np.sqrt(var_I)
    p_one = 1 - norm.cdf(z) if z > 0 else norm.cdf(z)

    return {"I": float(I), "expected": float(expected_I),
            "z_score": float(z), "p_normal": float(p_one),
            "n": n, "sum_w": float(sum_w)}


def main():
    print("Loading predictions and computing residuals...")
    p2, t = load(V2); p4, _ = load(V4); p10, _ = load(V10)

    results = {}
    for arch_name, pred in [("V2_ConvLSTM", p2),
                            ("V4_GNN_TAT", p4),
                            ("V10_Late_Fusion", p10)]:
        residuals = per_cell_mean_residual(pred, t)
        moran = morans_i_rook(residuals)
        moran["mean_residual"] = float(np.nanmean(residuals))
        moran["std_residual"] = float(np.nanstd(residuals))
        results[arch_name] = moran
        print(f"\n  {arch_name}:")
        print(f"    Moran's I = {moran['I']:+.4f} (expected {moran['expected']:+.4f})")
        print(f"    z = {moran['z_score']:+.2f}, p (one-sided) = {moran['p_normal']:.4g}")
        print(f"    Mean residual = {moran['mean_residual']:+.2f} mm,"
              f" std = {moran['std_residual']:.2f} mm")

    out_path = OUT / "morans_i.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Justification verdict
    max_I = max(abs(r["I"]) for r in results.values())
    if max_I < 0.30:
        print(f"\nVERDICT: max |I| = {max_I:.3f} < 0.30 — temporal-only split JUSTIFIED")
    else:
        print(f"\nVERDICT: max |I| = {max_I:.3f} >= 0.30 — consider spatial block CV")


if __name__ == "__main__":
    main()
