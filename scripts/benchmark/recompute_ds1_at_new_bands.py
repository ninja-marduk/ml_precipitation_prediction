"""DS1 Pearson r per KCE-aligned elevation band.

Reproduces the bidirectional analysis (DS1: precipitation conditioned on
elevation) at the new threshold system 1,500 / 2,800 m for Spec 03.

Output: prints r and p per band; saves CSV to scripts/benchmark/output/.
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent.parent
DS = ROOT / "data" / "output" / (
    "complete_dataset_with_features_with_clusters_elevation_windows_"
    "imfs_with_onehot_elevation_clean.nc"
)
OUT = ROOT / "scripts" / "benchmark" / "output" / "ds1_kce_aligned.csv"


def main() -> None:
    ds = xr.open_dataset(DS)

    elev = np.asarray(ds["elevation"].values, dtype=float)
    if elev.ndim == 3:
        elev = elev[0]
    elif elev.ndim == 4:
        elev = elev[0, :, :, 0]
    elev = elev.astype(float)

    var = "total_precipitation"
    precip_mean = ds[var].mean(dim="time").values

    bands = [
        ("Low",    (0, 1500)),
        ("Medium", (1500, 2800)),
        ("High",   (2800, 6000)),
    ]

    rows = []
    print("Band     | n cells | Pearson r | Spearman r | p (Pearson)")
    print("-" * 60)
    for name, (lo, hi) in bands:
        mask = (elev >= lo) & (elev < hi)
        e = elev[mask].ravel()
        p = precip_mean[mask].ravel()
        valid = np.isfinite(e) & np.isfinite(p)
        e_v, p_v = e[valid], p[valid]
        if e_v.size < 2:
            print(f"{name:8s} | {e_v.size:7d} | --        | --         | --")
            continue
        r_p, p_p = pearsonr(e_v, p_v)
        r_s, p_s = spearmanr(e_v, p_v)
        print(f"{name:8s} | {e_v.size:7d} | r={r_p:+.4f} | r={r_s:+.4f}  | p={p_p:.4g}")
        rows.append({
            "band": name,
            "lo": lo, "hi": hi,
            "n_cells": int(e_v.size),
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "spearman_r": float(r_s),
            "spearman_p": float(p_s),
        })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
