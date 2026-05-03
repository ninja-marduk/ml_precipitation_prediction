"""
Paper 5 Study Area Map - High-Quality Regeneration
====================================================
Generates boyaca.png at 800 DPI for Paper 5.
Adapted from Paper 4's figure_dem_elevation().

Usage:
    python models/scripts/generate_paper5_study_area.py
"""

import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# Add scripts directory to path for figure_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_config import setup_style, OUTPUT_DPI

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_NC = PROJECT_ROOT / "notebooks" / "data" / "output" / \
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"
SHP_PATH = PROJECT_ROOT / "data" / "input" / "MGN_Departamento.shp"
OUT_PATH = PROJECT_ROOT / ".docs" / "papers" / "5" / "figures" / "boyaca.png"


def generate_study_area_map():
    """Generate high-quality DEM elevation map with hillshade and boundary."""
    setup_style()

    print("Loading elevation data...")
    ds = xr.open_dataset(DATA_NC)
    lats = ds.latitude.values
    lons = ds.longitude.values

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

    # Load shapefile (already filtered to Boyaca - single geometry, no attribute columns)
    gdf = None
    try:
        import geopandas as gpd
        if SHP_PATH.exists():
            gdf = gpd.read_file(SHP_PATH)
            if "DPTO_CNMBR" in gdf.columns:
                gdf = gdf[gdf["DPTO_CNMBR"].str.contains("Boyac", case=False, na=False)]
            print(f"  Loaded Boyaca boundary ({len(gdf)} feature(s))")
    except ImportError:
        print("  Warning: geopandas not available, skipping boundary")

    # Create figure
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

    fig.savefig(OUT_PATH, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # Verify
    from PIL import Image
    img = Image.open(OUT_PATH)
    dpi_info = img.info.get("dpi", "N/A")
    print(f"\n  Saved: {OUT_PATH.name}")
    print(f"  Size: {img.size[0]}x{img.size[1]} px")
    print(f"  DPI: {dpi_info}")
    print(f"  File: {OUT_PATH.stat().st_size // 1024} KB")


if __name__ == "__main__":
    generate_study_area_map()
