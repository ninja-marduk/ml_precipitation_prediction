"""
Paper 4 Reviewer-Requested Figures
====================================
Generates figures addressing Reviewer 2 feedback (MDPI Hydrology):
  1. DEM elevation map with topographic information (replaces colombia_map.png)
  2. Spatial R² map (per-grid-cell performance)
  3. Observed vs predicted scatter plot with density
  4. Elevation-stratified analysis (R²/RMSE by elevation band)
  5. Time series at representative grid cells

Uses Q1 journal standards from figure_config.py.

Usage:
    python models/scripts/generate_paper4_reviewer_figures.py
"""

import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Add scripts directory to path for figure_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_config import COLORS, setup_style, add_panel_label, OUTPUT_DPI

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_NC = PROJECT_ROOT / "data" / "output" / \
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"
SHP_PATH = PROJECT_ROOT / "data" / "input" / "MGN_Departamento.shp"

V2_DIR = PROJECT_ROOT / "models" / "output" / "V2_Enhanced_Models" / \
    "map_exports" / "H12" / "BASIC" / "ConvLSTM_Bidirectional"
V4_DIR = PROJECT_ROOT / "models" / "output" / "V4_GNN_TAT_Models" / \
    "map_exports" / "H12" / "BASIC" / "GNN_TAT_GAT"

OUT_DIR = PROJECT_ROOT / "docs" / "papers" / "4" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
def load_data():
    """Load NetCDF dataset, predictions, and targets."""
    print("Loading NetCDF dataset...")
    ds = xr.open_dataset(DATA_NC)
    lats = ds.latitude.values
    lons = ds.longitude.values

    # Elevation: first timestep, squeeze extra dims
    elev_raw = ds["elevation"].values
    if elev_raw.ndim == 3:
        elev = elev_raw[0]
    elif elev_raw.ndim == 4:
        elev = elev_raw[0, :, :, 0]
    else:
        elev = elev_raw
    elev = np.asarray(elev, dtype=np.float64)

    print(f"  Grid: {len(lats)}x{len(lons)} = {len(lats)*len(lons)} cells")
    print(f"  Elevation range: {np.nanmin(elev):.0f} - {np.nanmax(elev):.0f} m")

    # Load model predictions and targets
    def _load(d):
        p = np.load(d / "predictions.npy")
        t = np.load(d / "targets.npy")
        # shape: (samples, horizons, lat, lon, 1) -> squeeze last dim
        if p.ndim == 5:
            p = p[..., 0]
            t = t[..., 0]
        return p, t

    pred_v2, tgt_v2 = _load(V2_DIR)
    pred_v4, tgt_v4 = _load(V4_DIR)
    print(f"  V2 ConvLSTM predictions: {pred_v2.shape}")
    print(f"  V4 GNN-TAT predictions:  {pred_v4.shape}")

    ds.close()
    return lats, lons, elev, pred_v2, tgt_v2, pred_v4, tgt_v4


def load_shapefile():
    """Load Boyaca shapefile if geopandas available."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(SHP_PATH)
        print(f"  Shapefile loaded: {len(gdf)} features")
        return gdf
    except Exception as e:
        print(f"  Shapefile not available: {e}")
        return None


# ── Metric helpers ─────────────────────────────────────────────────────
def r2_per_cell(pred, tgt):
    """Compute R² at each grid cell across samples and horizons."""
    # Flatten sample and horizon dims → (N, lat, lon)
    s, h, nlat, nlon = pred.shape
    p = pred.reshape(s * h, nlat, nlon)
    t = tgt.reshape(s * h, nlat, nlon)
    ss_res = np.nansum((t - p) ** 2, axis=0)
    ss_tot = np.nansum((t - np.nanmean(t, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)
    return r2


def rmse_per_cell(pred, tgt):
    s, h, nlat, nlon = pred.shape
    p = pred.reshape(s * h, nlat, nlon)
    t = tgt.reshape(s * h, nlat, nlon)
    return np.sqrt(np.nanmean((t - p) ** 2, axis=0))


# ── Figure 1: DEM Elevation Map ───────────────────────────────────────
def figure_dem_elevation(lats, lons, elev, gdf):
    """DEM elevation map with hillshade and Boyaca boundary."""
    fig, ax = plt.subplots(figsize=(7, 8))

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Hillshade
    dy, dx = np.gradient(elev)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(315), np.radians(45)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)

    # Elevation filled contours
    levels = np.linspace(np.nanpercentile(elev, 1), np.nanpercentile(elev, 99), 60)
    im = ax.contourf(lon_grid, lat_grid, elev, levels=levels, cmap="terrain", extend="both")
    ax.contourf(lon_grid, lat_grid, hs, levels=50, cmap="gray", alpha=0.25, vmin=-1, vmax=1)

    # Contour lines every 500 m
    c_levels = np.arange(0, 5001, 500)
    cs = ax.contour(lon_grid, lat_grid, elev, levels=c_levels, colors="k",
                    linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=5, fmt="%d m")

    # Boyaca boundary
    if gdf is not None:
        gdf.boundary.plot(ax=ax, color="k", linewidth=1.5, zorder=5)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(x):.1f}\u00b0W"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}\u00b0N"))

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Elevation (m)")

    path = OUT_DIR / "dem_elevation.png"
    fig.savefig(path, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Figure 2: Spatial R² Maps ─────────────────────────────────────────
def figure_spatial_r2(lats, lons, elev, pred_v2, tgt_v2, pred_v4, tgt_v4, gdf):
    """Side-by-side spatial R² maps for V2 and V4."""
    r2_v2 = r2_per_cell(pred_v2, tgt_v2)
    r2_v4 = r2_per_cell(pred_v4, tgt_v4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    cmap = plt.cm.RdYlGn
    vmin, vmax = -0.2, 0.8
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for ax, r2, title, label in [
        (axes[0], r2_v2, "V2 ConvLSTM", "a"),
        (axes[1], r2_v4, "V4 GNN-TAT", "b"),
    ]:
        im = ax.pcolormesh(lon_grid, lat_grid, r2, cmap=cmap, norm=norm, shading="auto")
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color="k", linewidth=1.0, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(x):.1f}\u00b0W"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}\u00b0N"))
        add_panel_label(ax, label)

    axes[0].set_ylabel("Latitude")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("$R^{2}$ (NSE)")

    path = OUT_DIR / "spatial_r2_map.png"
    fig.savefig(path, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")
    return r2_v2, r2_v4


# ── Figure 3: Scatter Observed vs Predicted ────────────────────────────
def figure_scatter(pred_v2, tgt_v2, pred_v4, tgt_v4):
    """Density scatter plots: observed vs predicted for V2 and V4."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, pred, tgt, title, color, label in [
        (axes[0], pred_v2, tgt_v2, "V2 ConvLSTM", COLORS["v2"], "a"),
        (axes[1], pred_v4, tgt_v4, "V4 GNN-TAT", COLORS["v4"], "b"),
    ]:
        obs = tgt.ravel()
        prd = pred.ravel()
        # Remove NaN
        mask = np.isfinite(obs) & np.isfinite(prd)
        obs, prd = obs[mask], prd[mask]

        # 2D histogram for density
        h, xedges, yedges = np.histogram2d(obs, prd, bins=80,
                                           range=[[0, obs.max()], [0, obs.max()]])
        h = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xedges, yedges, h.T, cmap="viridis",
                           norm=mcolors.LogNorm(vmin=1), shading="auto")

        # 1:1 line
        lim = max(obs.max(), prd.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.6, label="1:1 line")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Observed precipitation (mm)")
        ax.set_ylabel("Predicted precipitation (mm)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", framealpha=0.8)

        # R² annotation
        r2_val = 1 - np.sum((obs - prd) ** 2) / np.sum((obs - obs.mean()) ** 2)
        rmse_val = np.sqrt(np.mean((obs - prd) ** 2))
        ax.text(0.97, 0.05, f"$R^{{2}}$ = {r2_val:.3f}\nRMSE = {rmse_val:.1f} mm",
                transform=ax.transAxes, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Sample count")
        add_panel_label(ax, label)

    fig.tight_layout()
    path = OUT_DIR / "scatter_obs_vs_pred.png"
    fig.savefig(path, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Figure 4: Elevation-Stratified Analysis ────────────────────────────
def figure_elevation_stratified(lats, lons, elev, pred_v2, tgt_v2, pred_v4, tgt_v4):
    """R² and RMSE by elevation band for V2 and V4."""
    bands = [
        (0, 1000, "<1000 m"),
        (1000, 2000, "1000\u20132000 m"),
        (2000, 3000, "2000\u20133000 m"),
        (3000, 5000, ">3000 m"),
    ]

    results = {"V2 ConvLSTM": {"r2": [], "rmse": [], "n": []},
               "V4 GNN-TAT": {"r2": [], "rmse": [], "n": []}}

    for lo, hi, _ in bands:
        mask = (elev >= lo) & (elev < hi)
        n_cells = np.sum(mask)

        for name, pred, tgt in [
            ("V2 ConvLSTM", pred_v2, tgt_v2),
            ("V4 GNN-TAT", pred_v4, tgt_v4),
        ]:
            s, h, nlat, nlon = pred.shape
            p_flat = pred.reshape(s * h, nlat, nlon)[:, mask]
            t_flat = tgt.reshape(s * h, nlat, nlon)[:, mask]
            obs_all = t_flat.ravel()
            prd_all = p_flat.ravel()
            valid = np.isfinite(obs_all) & np.isfinite(prd_all)
            obs_all, prd_all = obs_all[valid], prd_all[valid]

            ss_res = np.sum((obs_all - prd_all) ** 2)
            ss_tot = np.sum((obs_all - obs_all.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = np.sqrt(np.mean((obs_all - prd_all) ** 2))

            results[name]["r2"].append(r2)
            results[name]["rmse"].append(rmse)
            results[name]["n"].append(int(n_cells))

    band_labels = [b[2] for b in bands]
    x = np.arange(len(band_labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # (a) R² by elevation
    ax = axes[0]
    ax.bar(x - w / 2, results["V2 ConvLSTM"]["r2"], w, color=COLORS["v2"],
           label="V2 ConvLSTM", edgecolor="none")
    ax.bar(x + w / 2, results["V4 GNN-TAT"]["r2"], w, color=COLORS["v4"],
           label="V4 GNN-TAT", edgecolor="none")
    for i, (r2_v2, r2_v4) in enumerate(
        zip(results["V2 ConvLSTM"]["r2"], results["V4 GNN-TAT"]["r2"])
    ):
        ax.text(i - w / 2, r2_v2 + 0.01, f"{r2_v2:.3f}", ha="center", va="bottom", fontsize=6)
        ax.text(i + w / 2, r2_v4 + 0.01, f"{r2_v4:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.set_ylabel("$R^{2}$ (NSE)")
    ax.set_xlabel("Elevation band")
    ax.legend(loc="upper right", framealpha=0.8)
    add_panel_label(ax, "a")

    # (b) RMSE by elevation
    ax = axes[1]
    ax.bar(x - w / 2, results["V2 ConvLSTM"]["rmse"], w, color=COLORS["v2"],
           label="V2 ConvLSTM", edgecolor="none")
    ax.bar(x + w / 2, results["V4 GNN-TAT"]["rmse"], w, color=COLORS["v4"],
           label="V4 GNN-TAT", edgecolor="none")
    for i, (r_v2, r_v4) in enumerate(
        zip(results["V2 ConvLSTM"]["rmse"], results["V4 GNN-TAT"]["rmse"])
    ):
        ax.text(i - w / 2, r_v2 + 1, f"{r_v2:.1f}", ha="center", va="bottom", fontsize=6)
        ax.text(i + w / 2, r_v4 + 1, f"{r_v4:.1f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.set_ylabel("RMSE (mm)")
    ax.set_xlabel("Elevation band")
    ax.legend(loc="upper right", framealpha=0.8)
    add_panel_label(ax, "b")

    # Cell count annotation
    for i, n in enumerate(results["V2 ConvLSTM"]["n"]):
        axes[0].text(i, -0.03, f"n={n}", ha="center", va="top",
                     fontsize=5, transform=axes[0].get_xaxis_transform())

    fig.tight_layout()
    path = OUT_DIR / "elevation_stratified_analysis.png"
    fig.savefig(path, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")

    # Print table for LaTeX
    print("\n  Elevation-Stratified Results:")
    print(f"  {'Band':<14} {'V2 R²':>8} {'V4 R²':>8} {'V2 RMSE':>9} {'V4 RMSE':>9} {'Cells':>6}")
    for i, (_, _, lbl) in enumerate(bands):
        print(f"  {lbl:<14} {results['V2 ConvLSTM']['r2'][i]:8.3f} "
              f"{results['V4 GNN-TAT']['r2'][i]:8.3f} "
              f"{results['V2 ConvLSTM']['rmse'][i]:9.1f} "
              f"{results['V4 GNN-TAT']['rmse'][i]:9.1f} "
              f"{results['V2 ConvLSTM']['n'][i]:6d}")


# ── Figure 5: Time Series at Representative Grid Cells ────────────────
def figure_timeseries(lats, lons, elev, pred_v2, tgt_v2, pred_v4, tgt_v4):
    """Time series comparison at 3 elevation-representative grid cells."""
    # Pick representative cells: low, mid, high elevation
    flat_elev = elev.copy()
    targets = [
        ("Valley (<1500 m)", 800),
        ("Mid-slope (1500-2500 m)", 2000),
        ("Highland (>2500 m)", 3200),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for idx, (ax, (zone_name, target_elev)) in enumerate(zip(axes, targets)):
        # Find the grid cell closest to target elevation with valid data
        diff = np.abs(flat_elev - target_elev)
        # Mask NaN
        diff[np.isnan(diff)] = 1e9
        iy, ix = np.unravel_index(np.argmin(diff), diff.shape)
        actual_elev = flat_elev[iy, ix]

        # Extract time series at this cell - average across samples, keep horizons
        # Use horizon=12 (last horizon index=11 for the furthest forecast)
        # Average predictions across all samples for this cell
        obs_ts = tgt_v2[:, :, iy, ix].mean(axis=0)  # (12,)
        v2_ts = pred_v2[:, :, iy, ix].mean(axis=0)
        v4_ts = pred_v4[:, :, iy, ix].mean(axis=0)

        horizons = np.arange(1, len(obs_ts) + 1)
        ax.plot(horizons, obs_ts, "k-o", markersize=3, linewidth=1.2, label="Observed")
        ax.plot(horizons, v2_ts, "-s", markersize=3, linewidth=1.0,
                color=COLORS["v2"], label="V2 ConvLSTM")
        ax.plot(horizons, v4_ts, "-^", markersize=3, linewidth=1.0,
                color=COLORS["v4"], label="V4 GNN-TAT")

        ax.set_ylabel("Precipitation (mm)")
        ax.set_title(f"{zone_name} \u2014 cell ({iy},{ix}), elev={actual_elev:.0f} m",
                     fontsize=8)
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.8)
        add_panel_label(ax, chr(ord("a") + idx))

    axes[-1].set_xlabel("Forecast horizon (months)")
    axes[-1].set_xticks(np.arange(1, 13))

    fig.tight_layout()
    path = OUT_DIR / "timeseries_representative.png"
    fig.savefig(path, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    setup_style()
    print("=" * 60)
    print("  Paper 4 Reviewer Figures Generator")
    print("=" * 60)

    lats, lons, elev, pred_v2, tgt_v2, pred_v4, tgt_v4 = load_data()
    gdf = load_shapefile()

    print("\nGenerating figures...")

    print("\n[1/5] DEM Elevation Map")
    figure_dem_elevation(lats, lons, elev, gdf)

    print("\n[2/5] Spatial R² Maps")
    figure_spatial_r2(lats, lons, elev, pred_v2, tgt_v2, pred_v4, tgt_v4, gdf)

    print("\n[3/5] Scatter Observed vs Predicted")
    figure_scatter(pred_v2, tgt_v2, pred_v4, tgt_v4)

    print("\n[4/5] Elevation-Stratified Analysis")
    figure_elevation_stratified(lats, lons, elev, pred_v2, tgt_v2, pred_v4, tgt_v4)

    print("\n[5/5] Time Series at Representative Cells")
    figure_timeseries(lats, lons, elev, pred_v2, tgt_v2, pred_v4, tgt_v4)

    print("\n" + "=" * 60)
    print("  All 5 figures saved to:")
    print(f"  {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
