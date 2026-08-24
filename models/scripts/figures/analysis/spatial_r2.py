"""Spatial skill over Boyaca, six panels, at H=12.

Adapts the poster figure (`generate_poster_figures.py::poster_spatial_r2_3panel`)
to the paper typography (14/11/10 hierarchy) and embedded width (0.95 textwidth).

Output: `.docs/papers/5/figures/spatial_r2_map_3panel.png` at 800 DPI.

Usage:
    python models/scripts/figures/analysis/spatial_r2.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Bootstrap _config from figures/
FIGURES_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = FIGURES_ROOT.parents[2]
sys.path.insert(0, str(FIGURES_ROOT))
from _config import setup_paper_style, save_figure, OUTPUT_DPI  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────
V2_PRED = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / \
    'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM_Bidirectional'
# Seed 42 explicitly, for both. The parent directories also hold older arrays: the
# V4 root is the pre-correction graph run and the V10 root predates the seed-42
# rerun, and reading them here while the tables read SEED42 is what made the figure
# and the elevation table disagree.
V4_PRED = PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models' / 'SEED42' / \
    'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT'
V10_DIR = PROJECT_ROOT / 'models' / 'output' / 'V10_Late_Fusion' / 'SEED42'
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

    # Elevation, so the reader can see the terrain the paper argues about.
    try:
        import xarray as xr
        ds = xr.open_dataset(DATA_NC)
        elev = ds['elevation'].values.astype(float)
        elev = elev[0] if elev.ndim == 3 else elev
        ds.close()
    except Exception as e:                                        # noqa: BLE001
        print(f'  WARN: elevation load failed ({e}); panel (d) skipped')
        elev = np.full_like(r2_v2, np.nan)

    # Two-row layout. Row 1 is per-cell skill for the three models; row 2 is the
    # terrain and the two spatial claims the manuscript otherwise states only as
    # counts.
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), sharex=True, sharey=True)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    # RdYlGn was the wrong choice and Copernicus says so explicitly: it asks that
    # colour schemes in maps and charts be readable with a colour vision
    # deficiency, and a red-to-green ramp is the one that is not.
    #
    # The skill panels are sequential rather than diverging, which is a second
    # decision worth recording. A diverging map pivots on a meaningful centre,
    # and for skill the only candidate centre is zero, which sits a fifth of the
    # way up a range running from -0.2 to 0.8: nearly every cell is on one side
    # of it, so a diverging ramp would spend half its colour on values that
    # barely occur. Panel (e) is a signed difference, where zero is meaningful,
    # and it is the one panel that takes a diverging map.
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=-0.2, vmax=0.8)

    def frame(ax, label, title):
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color='k', linewidth=0.7, zorder=5)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=5)
        ax.set_aspect('equal')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{abs(x):.1f}\u00b0W'))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.1f}\u00b0N'))
        ax.tick_params(labelsize=10)
        ax.text(0.03, 0.97, f'({label})', transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='gray',
                          boxstyle='round,pad=0.25'))

    # ---- row 1: per-cell skill -------------------------------------------
    skill = [
        (axes[0, 0], r2_v2,  rf'ConvLSTM ($R^{{2}}$ = {r2_agg_v2:.3f})',  'a'),
        (axes[0, 1], r2_v4,  rf'GNN-TAT ($R^{{2}}$ = {r2_agg_v4:.3f})',   'b'),
        (axes[0, 2], r2_v10, rf'Late Fusion ($R^{{2}}$ = {r2_agg_v10:.3f})', 'c'),
    ]
    im = None
    for ax, r2, title, label in skill:
        # rasterized: each panel of ~3,965 quads becomes thousands of vector
        # paths in a PDF for no gain, since a continuous field carries no text
        # to keep selectable. Everything around it stays vector.
        im = ax.pcolormesh(lon_grid, lat_grid, r2, cmap=cmap, norm=norm,
                           shading='auto', rasterized=True)
        frame(ax, label, title)
    cb1 = fig.colorbar(im, ax=axes[0, :], shrink=0.88, pad=0.015, aspect=24)
    cb1.set_label(r'$R^{2}$ (NSE)', fontsize=11)
    cb1.ax.tick_params(labelsize=10)

    # ---- (d) the terrain --------------------------------------------------
    ax = axes[1, 0]
    imd = ax.pcolormesh(lon_grid, lat_grid, elev, cmap=plt.cm.cividis,
                        shading='auto', rasterized=True)
    if np.isfinite(elev).any():
        # the two ecological cuts the evaluation strata use
        ax.contour(lon_grid, lat_grid, elev, levels=[1500, 2800],
                   colors='white', linewidths=0.8, zorder=4)
    frame(ax, 'd', 'Elevation and the two band cuts')
    cbd = fig.colorbar(imd, ax=ax, shrink=0.82, pad=0.02, aspect=16)
    cbd.set_label('m a.s.l.', fontsize=10)
    cbd.ax.tick_params(labelsize=9)

    # ---- (e) what the fusion adds ----------------------------------------
    best_base = np.fmax(r2_v2, r2_v4)
    gain = r2_v10 - best_base
    ax = axes[1, 1]
    lim = float(np.nanpercentile(np.abs(gain), 98))
    ime = ax.pcolormesh(lon_grid, lat_grid, gain, cmap=plt.cm.RdBu,
                        norm=mcolors.Normalize(vmin=-lim, vmax=lim),
                        shading='auto', rasterized=True)
    frame(ax, 'e', 'Fusion minus the better base learner')
    cbe = fig.colorbar(ime, ax=ax, shrink=0.82, pad=0.02, aspect=16)
    cbe.set_label(r'$\Delta R^{2}$', fontsize=10)
    cbe.ax.tick_params(labelsize=9)

    # ---- (f) where the evidence runs out ---------------------------------
    # Three states, in the manuscript's own terms: cells the fusion lifts over
    # 0.5 from a base learner below 0.2, cells where both base learners are
    # below 0.2, and the twenty cells where the graph model is the better of
    # the two.
    both_low = (r2_v2 < 0.2) & (r2_v4 < 0.2)
    rescued = (~both_low) & ((r2_v2 < 0.2) | (r2_v4 < 0.2)) & (r2_v10 >= 0.5)
    gnn_wins = r2_v4 > r2_v2
    state = np.full(r2_v2.shape, np.nan)
    state[np.isfinite(r2_v2)] = 0.0
    state[rescued] = 1.0
    state[both_low] = 2.0
    state[gnn_wins] = 3.0
    ax = axes[1, 2]
    cats = mcolors.ListedColormap(['#DDDDDD', '#4477AA', '#CC6677', '#DDCC77'])
    ax.pcolormesh(lon_grid, lat_grid, state, cmap=cats,
                  norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4),
                  shading='auto', rasterized=True)
    frame(ax, 'f', 'Where the two branches differ')
    handles = [
        mpatches.Patch(color='#4477AA',
                       label=f'fusion lifts over 0.5 ({int(rescued.sum()):,})'),
        mpatches.Patch(color='#CC6677',
                       label=f'both base learners below 0.2 '
                             f'({int(both_low.sum()):,})'),
        mpatches.Patch(color='#DDCC77',
                       label=f'graph model the better one '
                             f'({int(gnn_wins.sum()):,})'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=8.5, frameon=True,
              framealpha=0.92, borderpad=0.35, handlelength=1.2)

    for ax in axes[1, :]:
        ax.set_xlabel('Longitude', fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel('Latitude', fontsize=11)

    save_figure(fig, OUT_PATH, dpi=OUTPUT_DPI, mirror=OUT_PATH_DELIVERY,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'  panel (f): rescued={int(rescued.sum())}, '
          f'both_low={int(both_low.sum())}, gnn_wins={int(gnn_wins.sum())}')
    print(f'  wrote: {OUT_PATH.relative_to(PROJECT_ROOT)}  '
          f'({OUT_PATH.stat().st_size/1024:.1f} KB)')
    print(f'  wrote: {OUT_PATH_DELIVERY.relative_to(PROJECT_ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(generate_spatial_r2_3panel())
