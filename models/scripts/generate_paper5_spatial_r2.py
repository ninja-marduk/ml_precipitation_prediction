"""Paper 5 Spatial R^2 3-panel map (ConvLSTM / GNN-TAT / Late Fusion) at H=12.

Adapts the poster figure (`generate_poster_figures.py::poster_spatial_r2_3panel`)
to the paper typography (14/11/10 hierarchy) and embedded width (0.95 textwidth).

Output: `.docs/papers/5/figures/spatial_r2_map_3panel.png` at 800 DPI.

Usage:
    python models/scripts/generate_paper5_spatial_r2.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# figure_config (single source of truth for fonts + colors)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_config import setup_paper_style, OUTPUT_DPI  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_PRED = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / \
    'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM_Bidirectional'
V4_PRED = PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models' / \
    'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT'
V10_DIR = PROJECT_ROOT / 'models' / 'output' / 'V10_Late_Fusion'
DATA_NC = PROJECT_ROOT / 'notebooks' / 'data' / 'output' / \
    'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
SHP_PATH = PROJECT_ROOT / 'data' / 'input' / 'MGN_Departamento.shp'

OUT_PATH = PROJECT_ROOT / '.docs' / 'papers' / '5' / 'figures' / 'spatial_r2_map_3panel.png'
OUT_PATH_DELIVERY = PROJECT_ROOT / '.docs' / 'papers' / '5' / 'delivery' / 'figures' / 'spatial_r2_map_3panel.png'


def _load(d: Path):
    p = np.load(d / 'predictions.npy')
    t = np.load(d / 'targets.npy')
    if p.ndim == 5:
        p = p[..., 0]
        t = t[..., 0]
    return p, t


def _r2_per_cell(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """NSE-style coefficient of determination per grid cell."""
    s, h, nlat, nlon = pred.shape
    p = pred.reshape(s * h, nlat, nlon)
    t = tgt.reshape(s * h, nlat, nlon)
    ss_res = np.nansum((t - p) ** 2, axis=0)
    ss_tot = np.nansum((t - np.nanmean(t, axis=0, keepdims=True)) ** 2, axis=0)
    return 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)


def generate_spatial_r2_3panel() -> int:
    """Generate the 3-panel spatial R^2 map at paper typography."""
    setup_paper_style()
    # Embedded at ~0.95 textwidth → no font bump needed (PAPER_RC defaults are fine).

    # Lat/lon grid
    try:
        import xarray as xr
        ds = xr.open_dataset(DATA_NC)
        lats = ds.latitude.values
        lons = ds.longitude.values
        ds.close()
    except Exception as e:
        print(f'  WARN: NetCDF load failed ({e}); falling back to indices')
        lats = np.arange(61)
        lons = np.arange(65)

    # Department boundary shapefile
    try:
        import geopandas as gpd
        gdf = gpd.read_file(SHP_PATH)
    except Exception as e:
        print(f'  WARN: shapefile load failed: {e}')
        gdf = None

    # Per-cell R^2 from saved predictions
    p2, t2 = _load(V2_PRED)
    p4, t4 = _load(V4_PRED)
    p10, t10 = _load(V10_DIR)

    r2_v2 = _r2_per_cell(p2, t2)
    r2_v4 = _r2_per_cell(p4, t4)
    r2_v10 = _r2_per_cell(p10, t10)

    # Title R^2 values are the AGGREGATE NSE (global SS_res / SS_tot across
    # all 3965 cells × all sample-horizon pairs), matching Paper 5 Table 18
    # BASIC row: ConvLSTM=0.629, GNN-TAT=0.597, Late Fusion=0.666. (Different
    # from cell-mean R^2 reported in Table 16 overall row.)
    def _agg_r2(pred, tgt):
        valid = np.isfinite(pred) & np.isfinite(tgt)
        p, t = pred[valid].ravel(), tgt[valid].ravel()
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    r2_agg_v2  = _agg_r2(p2, t2)
    r2_agg_v4  = _agg_r2(p4, t4)
    r2_agg_v10 = _agg_r2(p10, t10)
    print(f'  aggregate R^2: ConvLSTM={r2_agg_v2:.3f}  GNN-TAT={r2_agg_v4:.3f}  Late Fusion={r2_agg_v10:.3f}')

    # Compact 3-panel layout for paper width.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=-0.2, vmax=0.8)

    panels = [
        (axes[0], r2_v2,  rf'ConvLSTM ($R^{{2}}$ = {r2_agg_v2:.3f})',           'a'),
        (axes[1], r2_v4,  rf'GNN-TAT ($R^{{2}}$ = {r2_agg_v4:.3f})',            'b'),
        (axes[2], r2_v10, rf'Late Fusion (Ridge) ($R^{{2}}$ = {r2_agg_v10:.3f})', 'c'),
    ]

    im = None
    for ax, r2, title, label in panels:
        im = ax.pcolormesh(lon_grid, lat_grid, r2, cmap=cmap, norm=norm, shading='auto')
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color='k', linewidth=0.7, zorder=5)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=6)
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_aspect('equal')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.1f}°W'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}°N'))
        ax.tick_params(labelsize=10)
        ax.text(0.03, 0.97, f'({label})', transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.25'))

    axes[0].set_ylabel('Latitude', fontsize=11)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, aspect=28)
    cbar.set_label(r'$R^{2}$ (NSE)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')

    # Mirror to delivery for ZIP bundle.
    OUT_PATH_DELIVERY.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH_DELIVERY, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'  wrote: {OUT_PATH.relative_to(PROJECT_ROOT)}  ({OUT_PATH.stat().st_size/1024:.1f} KB)')
    print(f'  wrote: {OUT_PATH_DELIVERY.relative_to(PROJECT_ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(generate_spatial_r2_3panel())
