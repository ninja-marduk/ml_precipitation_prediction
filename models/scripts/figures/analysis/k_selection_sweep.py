"""
K-selection sweep for thesis Chapter 5 (Feature engineering).

Runs K-means with k=2..9 on:
  (a) Elevation field        -> justifies k=3 in Figure 5.2 (KCE regimes)
  (b) Per-cell precipitation climatology (12 month-of-year vector)
                              -> justifies k=4 in Figure 5.3 (climatology regimes)

For each clustering, reports inertia (elbow) and silhouette score, and
produces a 2x2 figure in English ready for the thesis.

Output:
  .docs/thesis/figures/k_selection_sweep.png
  models/scripts/output/k_selection_sweep_metrics.csv
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import sys
FIGURES_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIGURES_ROOT))
from _config import setup_paper_style, COLORS, OUTPUT_DPI  # noqa: E402


PROJECT_ROOT = FIGURES_ROOT.parents[2]
DATA_NC = PROJECT_ROOT / "data" / "output" / "complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"
FIG_OUT = PROJECT_ROOT / ".docs" / "thesis" / "figures" / "k_selection_sweep.png"
CSV_OUT = PROJECT_ROOT / "models" / "scripts" / "output" / "k_selection_sweep_metrics.csv"

K_RANGE = list(range(2, 10))
SEED = 42
K_ELEV_SELECTED = 3
K_CLIM_SELECTED = 4


def kmeans_sweep(X, k_values, seed=SEED):
    """Run K-means for each k; return dict {k: (inertia, silhouette)}."""
    out = {}
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else float("nan")
        out[k] = (km.inertia_, sil)
        print(f"  k={k:>2d}  inertia={km.inertia_:>14.2f}  silhouette={sil:.4f}")
    return out


def main():
    print(f"Loading dataset: {DATA_NC.name}")
    ds = xr.open_dataset(DATA_NC)

    # ---- (a) Elevation: 1-D feature ----
    elev = ds["elevation"].isel(time=0).values  # (61, 65)
    elev_flat = elev.ravel().reshape(-1, 1)
    valid_e = np.isfinite(elev_flat.ravel())
    Xe = elev_flat[valid_e]
    print(f"\nElevation clustering on {Xe.shape[0]} cells (1-D feature):")
    elev_res = kmeans_sweep(Xe, K_RANGE)

    # ---- (b) Climatology: 12-D per-cell month-of-year mean ----
    print("\nComputing per-cell monthly climatology...")
    clim = ds["total_precipitation"].groupby("time.month").mean("time")  # (12, lat, lon)
    clim_arr = clim.values  # (12, 61, 65)
    n_lat, n_lon = clim_arr.shape[1], clim_arr.shape[2]
    Xp_full = clim_arr.reshape(12, -1).T  # (n_cells, 12)
    valid_p = np.all(np.isfinite(Xp_full), axis=1)
    Xp = Xp_full[valid_p]
    print(f"Climatology clustering on {Xp.shape[0]} cells (12-D vector):")
    clim_res = kmeans_sweep(Xp, K_RANGE)

    # ---- Save metrics CSV ----
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in K_RANGE:
        rows.append({"feature": "elevation", "k": k,
                     "inertia": elev_res[k][0], "silhouette": elev_res[k][1]})
        rows.append({"feature": "climatology", "k": k,
                     "inertia": clim_res[k][0], "silhouette": clim_res[k][1]})
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"\nMetrics saved: {CSV_OUT.relative_to(PROJECT_ROOT)}")

    # ---- Figure: 2x2 (rows = feature, cols = inertia/silhouette) ----
    setup_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    panels = [
        ("(a) Elevation - inertia (elbow)",      "Inertia",         elev_res, K_ELEV_SELECTED, axes[0, 0]),
        ("(b) Elevation - silhouette",            "Silhouette score", elev_res, K_ELEV_SELECTED, axes[0, 1]),
        ("(c) Climatology - inertia (elbow)",    "Inertia",         clim_res, K_CLIM_SELECTED, axes[1, 0]),
        ("(d) Climatology - silhouette",          "Silhouette score", clim_res, K_CLIM_SELECTED, axes[1, 1]),
    ]
    for title, ylabel, res, k_sel, ax in panels:
        ks = np.array(K_RANGE)
        if "inertia" in title:
            ys = np.array([res[k][0] for k in K_RANGE])
            color = COLORS["v2"]
        else:
            ys = np.array([res[k][1] for k in K_RANGE])
            color = COLORS["v10"]
        ax.plot(ks, ys, "o-", color=color, markersize=7, linewidth=1.8)
        # Highlight selected k
        ax.axvline(k_sel, color="#999999", linestyle="--", linewidth=1.0, alpha=0.8)
        idx_sel = K_RANGE.index(k_sel)
        ax.scatter([k_sel], [ys[idx_sel]], s=160, facecolor="none",
                   edgecolor="#D55E00", linewidth=2.0, zorder=5,
                   label=f"selected k = {k_sel}")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(K_RANGE)
        ax.legend(loc="best", frameon=True)
        if ax in (axes[1, 0], axes[1, 1]):
            ax.set_xlabel("Number of clusters (k)")

    fig.suptitle(
        "K-means cluster-number selection for elevation (k=3) and "
        "monthly precipitation climatology (k=4) over the Boyaca study area",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=OUTPUT_DPI)
    print(f"Figure saved : {FIG_OUT.relative_to(PROJECT_ROOT)}")

    # ---- Plain-text justification block ----
    sil_max_e = max(K_RANGE, key=lambda k: elev_res[k][1])
    sil_max_c = max(K_RANGE, key=lambda k: clim_res[k][1])
    print("\n=== Justification summary ===")
    print(f"Elevation  : silhouette max at k={sil_max_e} "
          f"(score={elev_res[sil_max_e][1]:.4f}); selected k=3 (score={elev_res[3][1]:.4f}).")
    print(f"Climatology: silhouette max at k={sil_max_c} "
          f"(score={clim_res[sil_max_c][1]:.4f}); selected k=4 (score={clim_res[4][1]:.4f}).")


if __name__ == "__main__":
    main()
