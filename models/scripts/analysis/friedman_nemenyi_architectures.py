"""Friedman test and Nemenyi critical difference: trained architectures against the anchor.

Blocks are the 33 validation windows at H=12; treatments are Enhanced ConvLSTM
(bidirectional, BASIC) and GNN-TAT (GAT, BASIC) on the corrected pipeline, each
as the mean over seeds 42, 123 and 456 of its per-window RMSE; the GNN-ConvLSTM
stacking ensemble and GNN-BiMamba (single runs, the only arrays that exist); and
the untrained per-cell monthly climatology fitted on months t < 414. Each window's
RMSE over its 12 leads and 3,965 cells is ranked within the window (1 = lowest).

Late Fusion is left out on purpose: its combiner is fitted on these same windows,
so ranking it here would score it in sample.

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
import xarray as xr
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "output"
DATA = ROOT / "data" / "output" / (
    "complete_dataset_with_features_with_clusters_elevation_windows_"
    "imfs_with_onehot_elevation_clean.nc"
)
SEEDS = (42, 123, 456)
CUTOFF = 414
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


def load(path, what="predictions"):
    a = np.load(Path(path) / f"{what}.npy").astype("f8")
    return a[..., 0] if a.ndim == 5 else a


def main():
    y = load(OUT / "V10_Late_Fusion/SEED42", "targets")

    def rmse(p):
        return np.sqrt(np.nanmean(((p.reshape(y.shape) - y) ** 2).reshape(33, -1), axis=1))

    cols = {
        "Enhanced ConvLSTM": np.mean([rmse(load(
            OUT / f"V2_Enhanced_Models/SEED{s}/map_exports/H12/BASIC/ConvLSTM_Bidirectional"))
            for s in SEEDS], axis=0),
        "GNN-TAT": np.mean([rmse(load(
            OUT / f"V4_GNN_TAT_Models/SEED{s}/map_exports/H12/BASIC/GNN_TAT_GAT"))
            for s in SEEDS], axis=0),
        "Stacking": rmse(load(OUT / "V5_GNN_ConvLSTM_Stacking/map_exports/H12/BASIC_KCE/V5_STACKING")),
        # BiMamba stores (windows, nodes, leads) over 91 windows; the last 33 are the
        # validation windows shared with the other models.
        "GNN-BiMamba": rmse(load(OUT / "V9_GNN_BiMamba/data")[58:91].transpose(0, 2, 1)),
    }

    ds = xr.open_dataset(DATA)
    obs = ds["total_precipitation"].values.astype("f8")
    month = ds["time"].dt.month.values
    t0 = int(np.argmin([np.nanmean(np.abs(obs[t] - y[0, 0])) for t in range(len(obs))]))
    clim = np.stack([np.nanmean(obs[:CUTOFF][month[:CUTOFF] == m], axis=0) for m in range(1, 13)])
    cols["Climatology"] = rmse(np.stack([[clim[month[t0 + w + h] - 1] for h in range(12)]
                                         for w in range(33)]))

    names = list(cols)
    x_all = np.column_stack([cols[n] for n in names])
    for label, rows in (("All 33 sliding windows", slice(None)),
                        ("Three non-overlapping windows", [0, 12, 24])):
        x = x_all[rows]
        chi2, p = stats.friedmanchisquare(*x.T)
        ranks = stats.rankdata(x, axis=1).mean(0)
        k, n = x.shape[1], x.shape[0]
        cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
        print(f"\n{label}: N={n}, k={k}, chi2={chi2:.2f}, df={k - 1}, p={p:.2e}, CD={cd:.2f}")
        for name, r, m in zip(names, ranks, x.mean(0)):
            print(f"  {name:18s} mean rank {r:.2f}   mean RMSE {m:7.2f} mm")
        for a, b in itertools.combinations(range(k), 2):
            d = abs(ranks[a] - ranks[b])
            print(f"  {names[a]:18s} vs {names[b]:18s} |dR|={d:.2f}  {'> CD' if d > cd else 'within CD'}")
    first = (x_all.argmin(axis=1) == names.index("Climatology")).sum()
    print(f"\nClimatology has the lowest RMSE in {first} of 33 windows.")


if __name__ == "__main__":
    main()
