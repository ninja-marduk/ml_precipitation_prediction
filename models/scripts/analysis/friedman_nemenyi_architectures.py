"""Friedman test and Nemenyi critical difference across the five architectures.

Blocks are the 33 validation windows at H=12; treatments are the released-scheme
single-run prediction arrays of Enhanced ConvLSTM (bidirectional, BASIC), GNN-TAT
(GAT, BASIC), the GNN-ConvLSTM stacking ensemble, GNN-BiMamba and Late Fusion.
Each window is scored by RMSE over its 12 leads and 3,965 cells; ranks are taken
within each window (1 = lowest RMSE).

The 33 windows slide by one month, so they are not independent. The test is also
run on the three non-overlapping windows (0, 12, 24) and the critical difference
is reported at both N=33 and N=3; only differences that clear the N=3 value are
robust to the overlap.

Usage: python models/scripts/analysis/friedman_nemenyi_architectures.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "output"
MODELS = {
    "Enhanced ConvLSTM": OUT / "V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM_Bidirectional",
    "GNN-TAT": OUT / "V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT",
    "Stacking": OUT / "V5_GNN_ConvLSTM_Stacking/map_exports/H12/BASIC_KCE/V5_STACKING",
    "GNN-BiMamba": OUT / "V9_GNN_BiMamba/data",
    "Late Fusion": OUT / "V10_Late_Fusion",
}
# BiMamba stores (windows, nodes, leads) over 91 windows; the last 33 are the
# validation windows shared with the other models (checked against the targets).
BIMAMBA_VAL = slice(58, 91)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


def load(name, path, kind):
    a = np.load(path / f"{kind}.npy").astype("f8")
    if name == "GNN-BiMamba":
        a = a[BIMAMBA_VAL].transpose(0, 2, 1)
    return a.reshape(33, 12, -1)


def nemenyi_cd(k, n):
    return Q05[k] * np.sqrt(k * (k + 1) / (6 * n))


def report(rmse, names, rows, label):
    x = rmse[rows]
    chi2, p = stats.friedmanchisquare(*x.T)
    ranks = stats.rankdata(x, axis=1).mean(0)
    k, n = x.shape[1], x.shape[0]
    cd = nemenyi_cd(k, n)
    print(f"\n{label}: N={n}, k={k}, chi2={chi2:.2f}, df={k - 1}, p={p:.2e}, CD={cd:.2f}")
    for name, r, m in zip(names, ranks, x.mean(0)):
        print(f"  {name:18s} mean rank {r:.2f}   mean RMSE {m:7.2f} mm")
    for a, b in itertools.combinations(range(k), 2):
        d = abs(ranks[a] - ranks[b])
        print(f"  {names[a]:18s} vs {names[b]:18s} |dR|={d:.2f}  {'> CD' if d > cd else 'within CD'}")


def main():
    names = list(MODELS)
    y = load("Enhanced ConvLSTM", MODELS["Enhanced ConvLSTM"], "targets")
    cols = []
    for name, path in MODELS.items():
        t = load(name, path, "targets")
        assert np.abs(t - y).max() < 1e-3, f"{name}: targets do not match"
        p = load(name, path, "predictions")
        cols.append(np.sqrt(((p - y) ** 2).reshape(33, -1).mean(1)))
    rmse = np.column_stack(cols)
    report(rmse, names, slice(None), "All 33 sliding windows")
    report(rmse, names, [0, 12, 24], "Three non-overlapping windows")


if __name__ == "__main__":
    main()
