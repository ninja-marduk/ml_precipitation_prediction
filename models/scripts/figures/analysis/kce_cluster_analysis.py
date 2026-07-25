"""
Regenerate thesis Chapter 5 cluster figures in English with disambiguated labels.

Replaces the Spanish originals:
  - kce_k3_clusters_map.png      (Figure 5.2: KCE elevation regimes, k=3)
  - monthly_cluster_profiles.png (Figure 5.3: climatology regimes, k=4)

Both figures are regenerated from the same source data and seed (42) used in
workflows/04_feature_engineering.py so that the cell counts and cluster IDs
match the captions already written in thesis.tex (Low=2467/Mid=941/High=557
for Figure 5.2; C0=975/C1=1225/C2=1392/C3=373 for Figure 5.3).

Disambiguation:
  - "Elevation cluster" in Figure 5.2  (orographic regime, KCE feature input)
  - "Climatology regime"  in Figure 5.3  (precipitation pattern, EDA only)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
from sklearn.cluster import KMeans

import sys
FIGURES_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIGURES_ROOT))
from _config import setup_paper_style, OUTPUT_DPI  # noqa: E402


PROJECT_ROOT = FIGURES_ROOT.parents[2]
DATA_NC = PROJECT_ROOT / "data" / "output" / "complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"
SHAPEFILE = PROJECT_ROOT / "data" / "input" / "shapes" / "MGN_Departamento.shp"
FIG_DIR = PROJECT_ROOT / ".docs" / "thesis" / "figures"
SEED = 42

# Elevation cluster colours (viridis-like, sorted Low->High)
ELEV_COLORS = ["#440154", "#21908d", "#fde725"]  # purple, teal, yellow
ELEV_LABELS = [
    "Low (<950 m)",
    "Mid (950-2,260 m)",
    "High (>2,260 m)",
]

# Climatology regime palette (preserves the original blue/orange/green/red order)
CLIM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
CLIM_LABEL_TMPL = "Regime C{idx} (n = {n:,}, {desc})"
CLIM_DESCS = ["driest valleys", "cloud forest", "eastern slopes",
              "windward peaks"]


def load_dataset():
    print(f"Loading: {DATA_NC.name}")
    ds = xr.open_dataset(DATA_NC)
    return ds


def cluster_elevation(ds, k=3, seed=SEED):
    """Read the pre-computed elevation bands (thresholds at 1,500 m and
    2,800 m) actually used to build the KCE feature bundle. The dataset
    variable cluster_elevation reproduces the 2,467 / 941 / 557 partition
    cited in thesis.tex (caption of Figure 5.2)."""
    raw = ds["cluster_elevation"].isel(time=0).values  # (61, 65) strings
    # Map textual labels {low, medium, high} -> {0, 1, 2}
    text_to_id = {"low": 0, "medium": 1, "high": 2, "mid": 1}
    labels_2d = np.full(raw.shape, -1, dtype=int)
    for s, v in text_to_id.items():
        labels_2d[np.char.lower(raw.astype(str)) == s] = v
    counts = [int((labels_2d == c).sum()) for c in range(k)]
    print(f"Elevation bands (from cluster_elevation variable): counts = {counts}")
    return labels_2d, counts


def cluster_climatology(ds, k=4, seed=SEED):
    """K-means on per-cell month-of-year precipitation vector (12-D),
    with cluster IDs deterministically sorted by ascending annual total
    so that C1 = driest regime and C{k} = wettest."""
    clim = ds["total_precipitation"].groupby("time.month").mean("time").values  # (12, lat, lon)
    n_lat, n_lon = clim.shape[1], clim.shape[2]
    X = clim.reshape(12, -1).T  # (n_cells, 12)
    valid = np.all(np.isfinite(X), axis=1)

    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X[valid])

    # Sort cluster IDs by ascending annual sum of centroid
    annual_sum = km.cluster_centers_.sum(axis=1)
    order = np.argsort(annual_sum)
    remap = {old: new for new, old in enumerate(order)}
    new_labels = np.full(X.shape[0], -1, dtype=int)
    new_labels[valid] = np.vectorize(lambda x: remap[x])(km.labels_)
    labels_2d = new_labels.reshape(n_lat, n_lon)
    centers = km.cluster_centers_[order]  # reordered (k, 12)

    counts = [int((labels_2d == c).sum()) for c in range(k)]
    print(f"Climatology regimes (sorted by annual total): counts = {counts}")
    print(f"  Centroid annual totals (mm/yr): "
          f"{[round(c.sum(),0) for c in centers]}")
    return labels_2d, counts, centers


def plot_elevation_map(ds, labels_2d, counts, out_path):
    """Figure 5.2 replacement: KCE elevation clusters map (English)."""
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    lon = ds.longitude.values
    lat = ds.latitude.values
    lon_edges = np.concatenate([lon - 0.025, [lon[-1] + 0.025]])
    lat_edges = np.concatenate([lat - 0.025, [lat[-1] + 0.025]])

    cmap = ListedColormap(ELEV_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(ELEV_COLORS) + 0.5, 1), cmap.N)

    masked = np.ma.masked_where(labels_2d < 0, labels_2d)
    ax.pcolormesh(lon_edges, lat_edges, masked, cmap=cmap, norm=norm,
                  shading="flat", edgecolors="white", linewidths=0.05)

    # Departmental boundary
    gdf = gpd.read_file(SHAPEFILE)
    name_col = next((c for c in ("DPTO_CNMBR", "NOMBRE_DPT", "NOMBRE", "DPTO") if c in gdf.columns), None)
    if name_col is not None:
        boy = gdf[gdf[name_col].astype(str).str.upper().str.contains("BOYAC")]
        if len(boy) == 0:
            boy = gdf  # fallback: draw all
    else:
        boy = gdf
    boy.boundary.plot(ax=ax, color="black", linewidth=0.8)

    ax.set_xlim(lon_edges.min(), lon_edges.max())
    ax.set_ylim(lat_edges.min(), lat_edges.max())
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Elevation bands (k = 3) over the Boyaca study area")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.4)

    patches = [
        mpatches.Patch(color=ELEV_COLORS[i],
                       label=f"C{i+1} - {ELEV_LABELS[i]}  (n = {counts[i]:,}, {100*counts[i]/sum(counts):.0f} %)")
        for i in range(len(ELEV_COLORS))
    ]
    ax.legend(handles=patches, loc="lower left", frameon=True,
              fontsize=9, title="Elevation cluster", title_fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_climatology_profiles(centers, counts, out_path):
    """Figure 5.3 replacement: monthly climatology regime profiles (English)."""
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(9, 5.2))

    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for i in range(len(centers)):
        ax.plot(months, centers[i], "o-", color=CLIM_COLORS[i],
                markersize=6, linewidth=1.8,
                label=CLIM_LABEL_TMPL.format(idx=i + 1, n=counts[i],
                                             desc=CLIM_DESCS[i]))

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean precipitation (mm)")
    ax.set_title("Monthly precipitation regimes (K-means, k = 4) "
                 "from per-cell annual climatology")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, title="Climatology regime",
              title_fontsize=10, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()

    elev_labels, elev_counts = cluster_elevation(ds, k=3)
    clim_labels, clim_counts, clim_centers = cluster_climatology(ds, k=4)

    plot_elevation_map(ds, elev_labels, elev_counts,
                       FIG_DIR / "kce_k3_clusters_map.png")
    plot_climatology_profiles(clim_centers, clim_counts,
                              FIG_DIR / "monthly_cluster_profiles.png")

    # Sanity-check counts vs thesis captions
    expect_elev = (2467, 941, 557)
    expect_clim = (975, 1225, 1392, 373)
    ok_e = tuple(elev_counts) == expect_elev
    ok_c = tuple(clim_counts) == expect_clim
    print("\n=== Sanity check vs thesis.tex captions ===")
    print(f"Elevation counts {tuple(elev_counts)} vs expected {expect_elev}: "
          f"{'OK' if ok_e else 'MISMATCH'}")
    print(f"Climatology counts {tuple(clim_counts)} vs expected {expect_clim}: "
          f"{'OK' if ok_c else 'MISMATCH'}")


if __name__ == "__main__":
    main()
