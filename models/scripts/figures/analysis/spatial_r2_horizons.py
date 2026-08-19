"""Spatial R^2 (NSE) atlas: 3 models x 12 horizons.

Exploratory. Its output is not used by the manuscript, and it still reads the
pre-correction graph array; `spatial_r2.py` is the generator of the spatial figure
that is used. Point this at the SEED42 paths before quoting anything from it.

Rows = ConvLSTM / GNN-TAT / Late Fusion; columns = H=1..12. Each cell is the
per-grid-cell NSE at that horizon (over the 33 validation windows), sharing the
palette and Boyaca boundary of the 3-panel figure (spatial_r2.py). Complements
the H=12 summary by showing how spatial skill degrades across horizons.

Usage:
    python models/scripts/figures/analysis/spatial_r2_horizons.py [out.png]
"""
from __future__ import annotations
import os
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
    OUTPUT_DPI = 200

OUT = PROJECT_ROOT / 'models' / 'output'
V2 = OUT / 'V2_Enhanced_Models' / 'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM_Bidirectional'
V4 = OUT / 'V4_GNN_TAT_Models' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT'
V10 = OUT / 'V10_Late_Fusion'
DATA_NC = PROJECT_ROOT / 'notebooks' / 'data' / 'output' / \
    'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
SHP = PROJECT_ROOT / 'data' / 'input' / 'MGN_Departamento.shp'


def _load(d: Path):
    if not (d / 'predictions.npy').exists():
        alt = d / 'SEED42'
        if (alt / 'predictions.npy').exists():
            d = alt
    p = np.load(d / 'predictions.npy'); t = np.load(d / 'targets.npy')
    if p.ndim == 5:
        p, t = p[..., 0], t[..., 0]
    return p, t  # (s, h, nlat, nlon)


def _r2_cell_h(pred, tgt, h):
    p, t = pred[:, h], tgt[:, h]  # (s, nlat, nlon)
    ss_res = np.nansum((t - p) ** 2, axis=0)
    ss_tot = np.nansum((t - np.nanmean(t, axis=0, keepdims=True)) ** 2, axis=0)
    return 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)


def _agg_r2_h(pred, tgt, h):
    p, t = pred[:, h], tgt[:, h]
    m = np.isfinite(p) & np.isfinite(t)
    pp, tt = p[m].ravel(), t[m].ravel()
    ss_res = np.sum((tt - pp) ** 2); ss_tot = np.sum((tt - tt.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def main():
    setup_paper_style()
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (PROJECT_ROOT / '.docs' / 'papers' / '5' / 'figures' / 'spatial_r2_atlas_12h.png')

    try:
        import xarray as xr
        ds = xr.open_dataset(DATA_NC); lats = ds.latitude.values; lons = ds.longitude.values; ds.close()
    except Exception as e:
        print('  WARN latlon:', e); lats = np.arange(61); lons = np.arange(65)
    try:
        import geopandas as gpd; gdf = gpd.read_file(SHP)
    except Exception as e:
        print('  WARN shp:', e); gdf = None

    models = [('ConvLSTM', *_load(V2)), ('GNN-TAT', *_load(V4)), ('Late Fusion', *_load(V10))]
    H = models[0][1].shape[1]
    lon_g, lat_g = np.meshgrid(lons, lats)
    cmap = getattr(plt.cm, os.environ.get('ATLAS_CMAP', 'RdYlGn'))
    norm = mcolors.Normalize(vmin=-0.2, vmax=0.8)

    nmod = len(models)
    fig, axes = plt.subplots(nmod, H, figsize=(1.5 * H + 1.2, 1.85 * nmod + 0.8),
                             sharex=True, sharey=True)
    im = None
    for i, (name, p, t) in enumerate(models):
        for h in range(H):
            ax = axes[i, h]
            r2 = _r2_cell_h(p, t, h)
            im = ax.pcolormesh(lon_g, lat_g, r2, cmap=cmap, norm=norm, shading='auto')
            if gdf is not None:
                gdf.boundary.plot(ax=ax, color='k', linewidth=0.45, zorder=5)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            ax.text(0.5, -0.05, f'{_agg_r2_h(p, t, h):.2f}', transform=ax.transAxes,
                    fontsize=8, ha='center', va='top', color='#333333')
            if i == 0:
                ax.set_title(f'H={h + 1}', fontsize=11, fontweight='bold', pad=5)
        axes[i, 0].set_ylabel(name, fontsize=13, fontweight='bold', labelpad=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.008, aspect=28)
    cbar.set_label(r'$R^{2}$ (NSE)', fontsize=12); cbar.ax.tick_params(labelsize=10)
    if os.environ.get('ATLAS_LANG', 'en') == 'es':
        _sup = ('$R^{2}$ por celda (NSE) a lo largo de los horizontes de pronóstico '
                '($H=1$ a $12$). El número bajo cada mapa es el $R^{2}$ agregado del horizonte.')
    else:
        _sup = ('Per-cell $R^{2}$ (NSE) across forecast horizons ($H=1$ to $12$). '
                'Number under each map = horizon-aggregate $R^{2}$.')
    fig.suptitle(_sup, fontsize=13, y=0.995)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote: {out_path}  ({out_path.stat().st_size/1024:.0f} KB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
