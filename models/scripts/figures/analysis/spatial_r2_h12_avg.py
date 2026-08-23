"""Spatial R^2 (NSE) maps: (a) at H=12 and (b) averaged over H=1..12.

Exploratory. Its output is not used by the manuscript, and it still reads the
pre-correction graph array; `spatial_r2.py` is the generator of the spatial figure
that is used. Point this at the SEED42 paths before quoting anything from it.

Two rows (a: hardest horizon H=12; b: mean across all horizons) x three columns
(ConvLSTM / GNN-TAT / Late Fusion). Same NSE-per-cell computation, RdYlGn palette
and Boyaca boundary as the single-row figure. Separating H=12 from the average
avoids the mislabelling in the previous version (which pooled all horizons but was
titled H=12).

Usage:
    python models/scripts/figures/analysis/spatial_r2_h12_avg.py [out.png]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FIGURES_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = FIGURES_ROOT.parents[2]
sys.path.insert(0, str(FIGURES_ROOT))
try:
    from _config import setup_paper_style, OUTPUT_DPI  # noqa
except Exception:
    def setup_paper_style():
        pass
    OUTPUT_DPI = 300

OUT = PROJECT_ROOT / 'models' / 'output'
V2 = OUT / 'V2_Enhanced_Models' / 'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM_Bidirectional'
V4 = OUT / 'V4_GNN_TAT_Models' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT'
V10 = OUT / 'V10_Late_Fusion'
DATA_NC = PROJECT_ROOT / 'notebooks' / 'data' / 'output' / \
    'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
SHP = PROJECT_ROOT / 'data' / 'input' / 'MGN_Departamento.shp'


def _load(d: Path):
    if not (d / 'predictions.npy').exists() and (d / 'SEED42' / 'predictions.npy').exists():
        d = d / 'SEED42'
    p = np.load(d / 'predictions.npy'); t = np.load(d / 'targets.npy')
    if p.ndim == 5:
        p, t = p[..., 0], t[..., 0]
    return p, t  # (s, h, nlat, nlon)


def _cell_h12(p, t):
    a, b = p[:, -1], t[:, -1]
    ss_res = np.nansum((b - a) ** 2, axis=0)
    ss_tot = np.nansum((b - np.nanmean(b, axis=0, keepdims=True)) ** 2, axis=0)
    return 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)


def _cell_pooled(p, t):
    s, h, nl, no = p.shape
    a = p.reshape(s * h, nl, no); b = t.reshape(s * h, nl, no)
    ss_res = np.nansum((b - a) ** 2, axis=0)
    ss_tot = np.nansum((b - np.nanmean(b, axis=0, keepdims=True)) ** 2, axis=0)
    return 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)


def _agg(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    x, y = a[m].ravel(), b[m].ravel()
    sr = np.sum((y - x) ** 2); st = np.sum((y - y.mean()) ** 2)
    return float(1 - sr / st) if st > 0 else float('nan')


def main():
    setup_paper_style()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (PROJECT_ROOT / '.docs' / 'thesis' / 'spatial_r2_map_h12_avg.png')
    try:
        import xarray as xr
        ds = xr.open_dataset(DATA_NC); lats = ds.latitude.values; lons = ds.longitude.values; ds.close()
    except Exception as e:
        print('  WARN latlon:', e); lats = np.arange(61); lons = np.arange(65)
    try:
        import geopandas as gpd; gdf = gpd.read_file(SHP)
    except Exception as e:
        print('  WARN shp:', e); gdf = None

    models = [('ConvLSTM', *_load(V2)), ('GNN-TAT', *_load(V4)), ('Late Fusion (Ridge)', *_load(V10))]
    lon_g, lat_g = np.meshgrid(lons, lats)
    cmap = plt.cm.viridis; norm = mcolors.Normalize(vmin=-0.2, vmax=0.8)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), sharex=True, sharey=True)
    row_defs = [
        ('(a)  $H = 12$', lambda p, t: _cell_h12(p, t), lambda p, t: _agg(p[:, -1], t[:, -1])),
        ('(b)  Mean over $H = 1$--$12$', lambda p, t: _cell_pooled(p, t), lambda p, t: _agg(p, t)),
    ]
    im = None
    for r, (rlabel, cell_fn, agg_fn) in enumerate(row_defs):
        for c, (name, p, t) in enumerate(models):
            ax = axes[r, c]
            im = ax.pcolormesh(lon_g, lat_g, cell_fn(p, t), cmap=cmap, norm=norm, shading='auto')
            if gdf is not None:
                gdf.boundary.plot(ax=ax, color='k', linewidth=0.7, zorder=5)
            ax.set_aspect('equal'); ax.tick_params(labelsize=9)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.1f}°W'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}°N'))
            ax.text(0.035, 0.965, rf'$R^{{2}} = {agg_fn(p, t):.3f}$', transform=ax.transAxes,
                    fontsize=10, va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.25'))
            if r == 0:
                ax.set_title(name, fontsize=13, fontweight='bold', pad=6)
            if c == 0:
                ax.set_ylabel(rlabel + '\nLatitude', fontsize=11)
        axes[r, 0]
    for c in range(3):
        axes[1, c].set_xlabel('Longitude', fontsize=11)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, aspect=30)
    cbar.set_label(r'$R^{2}$ (NSE)', fontsize=11); cbar.ax.tick_params(labelsize=10)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote: {out}  ({out.stat().st_size/1024:.0f} KB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
