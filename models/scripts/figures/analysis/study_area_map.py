"""The study area, drawn so that the section's numbers can be seen.

Section 3.3 states three things and had no figure that showed any of them: the
domain is a cordillera, precipitation *decreases* with elevation there, and the
decrease is not uniform, being weak in the lowlands, strong at mid elevation
and slightly positive above the cloud-forest transition. The earlier map showed
a flat elevation raster and nothing else.

Three panels, each carrying one of those:

  (a) elevation, hillshaded, so the cordillera reads as terrain rather than as
      a colour ramp, with the two band cuts the evaluation strata use drawn on
      it. Flat shading makes a mountain range look like a blob; a light source
      makes the valleys and the two flanks visible, which is what the rain
      shadow argument needs the reader to see.
  (b) mean annual precipitation on the same grid, so the reader can put the two
      fields side by side and see the sign of the gradient without being told.
  (c) the relation itself, cell by cell, split at the two cuts, with the
      correlation inside each band. This is where r=-0.700 and its
      non-uniformity stop being three numbers in a sentence.

Every quantity printed here is recomputed from the released NetCDF, so the
figure cannot disagree with the section, and the ones the section quotes are
printed to stdout for checking.

Legends and annotations sit outside the plotting areas, per the package rule:
inside the axes belongs to the data.

Usage: python models/scripts/figures/analysis/study_area_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.colors as mcolors      # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

FIGURES_ROOT = Path(__file__).resolve().parent.parent
ROOT = FIGURES_ROOT.parents[2]
sys.path.insert(0, str(FIGURES_ROOT))
from _config import setup_paper_style, save_figure, OUTPUT_DPI  # noqa: E402

NC = ROOT / "notebooks" / "data" / "output" / \
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_" \
    "with_onehot_elevation_clean.nc"
SHP = ROOT / "data" / "input" / "MGN_Departamento.shp"
OUT = ROOT / ".docs" / "papers" / "5" / "figures" / "study_area.png"
OUT_DELIVERY = ROOT / ".docs" / "papers" / "5" / "delivery" / "figures" / \
    "study_area.png"

LOW, HIGH = 1500.0, 2800.0
# Plain text: these are matplotlib labels, not LaTeX, so a thousands brace
# from the manuscript would print as literal braces on the figure.
BANDS = (("Low, below 1,500 m", "#4477AA"),
         ("Medium, 1,500 to 2,800 m", "#DDAA33"),
         ("High, above 2,800 m", "#BB5566"))


def main() -> int:
    import xarray as xr
    ds = xr.open_dataset(NC)
    lats = ds.latitude.values
    lons = ds.longitude.values
    elev = ds["elevation"].values.astype(float)
    elev = elev[0] if elev.ndim == 3 else elev
    # monthly totals; twelve of them make the annual figure the section quotes
    annual = np.nanmean(ds["total_precipitation"].values.astype(float),
                        axis=0) * 12.0
    ds.close()

    try:
        import geopandas as gpd
        gdf = gpd.read_file(SHP)
    except Exception as e:                                     # noqa: BLE001
        print(f"  WARN: shapefile load failed ({e}); boundary omitted")
        gdf = None

    lon_g, lat_g = np.meshgrid(lons, lats)
    setup_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})

    def frame(ax, label, title):
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color="k", linewidth=0.8, zorder=6)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude", fontsize=11)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{abs(x):.1f}°W"))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.1f}°N"))
        ax.tick_params(labelsize=10)
        ax.text(0.03, 0.97, f"({label})", transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", zorder=7,
                bbox=dict(facecolor="white", edgecolor="gray",
                          boxstyle="round,pad=0.25"))

    # ---- (a) hillshaded terrain ----------------------------------------
    ax = axes[0]
    ls = mcolors.LightSource(azdeg=315, altdeg=45)
    # vert_exag is in the units of the data over the units of the axes, and the
    # axes here are degrees, so the exaggeration is large by construction: one
    # degree is about 111 km and the relief is a few kilometres.
    shaded = ls.shade(np.nan_to_num(elev, nan=float(np.nanmin(elev))),
                      cmap=plt.cm.cividis, blend_mode="soft",
                      vert_exag=2.5e-4, dx=1.0, dy=1.0)
    ax.imshow(shaded, extent=[lons.min(), lons.max(), lats.min(), lats.max()],
              origin="lower", interpolation="bilinear", zorder=1)
    cs = ax.contour(lon_g, lat_g, elev, levels=[LOW, HIGH],
                    colors=["#FFFFFF", "#FFFFFF"], linewidths=[0.8, 1.3],
                    zorder=5)
    frame(ax, "a", "Terrain, shaded")
    ax.set_ylabel("Latitude", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                               norm=mcolors.Normalize(vmin=float(np.nanmin(elev)),
                                                      vmax=float(np.nanmax(elev))))
    cb = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02, aspect=18)
    cb.set_label("m a.s.l.", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # ---- (b) mean annual precipitation ----------------------------------
    ax = axes[1]
    im = ax.pcolormesh(lon_g, lat_g, annual, cmap=plt.cm.YlGnBu,
                       shading="auto", rasterized=True)
    ax.contour(lon_g, lat_g, elev, levels=[HIGH], colors=["0.25"],
               linewidths=0.8, zorder=5)
    frame(ax, "b", "Mean annual precipitation")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02, aspect=18)
    cb.set_label(r"mm yr$^{-1}$", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # ---- (c) the relation, band by band ---------------------------------
    ax = axes[2]
    ok = np.isfinite(elev) & np.isfinite(annual)
    e, a = elev[ok], annual[ok]
    masks = (e < LOW, (e >= LOW) & (e < HIGH), e >= HIGH)
    rs = []
    for (name, colour), m in zip(BANDS, masks):
        ax.scatter(e[m], a[m], s=3.5, c=colour, alpha=0.35, edgecolors="none",
                   rasterized=True)
        r = float(np.corrcoef(e[m], a[m])[0, 1])
        rs.append((name, colour, r, int(m.sum())))
    for cut in (LOW, HIGH):
        ax.axvline(cut, color="0.35", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Elevation (m a.s.l.)", fontsize=11)
    ax.set_ylabel(r"Mean annual precipitation (mm yr$^{-1}$)", fontsize=11)
    ax.set_title("The gradient, cell by cell", fontsize=12, fontweight="bold",
                 pad=6)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.03, 0.97, "(c)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="gray",
                      boxstyle="round,pad=0.25"))
    # legend under the axes: the rule is that nothing but data goes inside one
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c,
                              markersize=5,
                              label=f"{n}  ($r$={r:+.3f}, $n$={k:,})")
                       for n, c, r, k in rs],
              loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1,
              fontsize=9, frameon=False, handletextpad=0.5, labelspacing=0.3)

    plt.tight_layout()
    save_figure(fig, OUT, dpi=OUTPUT_DPI, mirror=OUT_DELIVERY,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    r_all = float(np.corrcoef(e, a)[0, 1])
    print(f"  elevation {np.nanmin(elev):.0f} to {np.nanmax(elev):.0f} m")
    print(f"  annual precipitation {np.nanmin(annual):.0f} to "
          f"{np.nanmax(annual):.0f} mm/yr")
    print(f"  overall r = {r_all:.3f}")
    for n, _, r, k in rs:
        print(f"  {n:<30} r={r:+.3f}  n={k}")
    print(f"  wrote: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
