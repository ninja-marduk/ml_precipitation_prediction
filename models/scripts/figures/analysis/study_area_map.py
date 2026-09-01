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
from matplotlib.patches import Rectangle, ConnectionPatch  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

FIGURES_ROOT = Path(__file__).resolve().parent.parent
ROOT = FIGURES_ROOT.parents[2]
sys.path.insert(0, str(FIGURES_ROOT))
from _config import setup_paper_style, save_figure, OUTPUT_DPI  # noqa: E402

NC = ROOT / "notebooks" / "data" / "output" / \
    "complete_dataset_with_features_with_clusters_elevation_windows_imfs_" \
    "with_onehot_elevation_clean.nc"
SHP = ROOT / "data" / "input" / "MGN_Departamento.shp"
# Natural Earth 1:110m, six countries, extracted into the repository so the
# figure does not read geometry out of a dependency's test fixtures.
COUNTRIES = ROOT / "data" / "input" / "shapes" / "locator_countries_110m.geojson"
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
    # Drawn at 11.3 in for a 6.99 in text width: a reduction to 0.62, which
    # is the one the type in _config.py is calibrated against. Four panels in
    # a single row would need 16 in and print at 0.43, with 5 pt titles.
    fig = plt.figure(figsize=(11.3, 8.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.72, 1, 1],
                          height_ratios=[1.0, 0.95], hspace=0.32, wspace=0.55)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0:2])]

    def frame(ax, label, title):
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color="k", linewidth=0.8, zorder=6)
        ax.set_title(f"({label})  {title}", fontsize=12, fontweight="bold",
                     pad=6, loc="left")
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude", fontsize=11)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{abs(x):.1f}°W"))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.1f}°N"))
        ax.tick_params(labelsize=10)

    # ---- (a) where in Colombia this is ----------------------------------
    ax = axes[0]
    dom = (float(lons.min()), float(lons.max()),
           float(lats.min()), float(lats.max()))
    try:
        import geopandas as gpd
        countries = gpd.read_file(COUNTRIES)
    except Exception as e:                                     # noqa: BLE001
        print(f"  WARN: locator geometry failed ({e}); locator omitted")
        countries = None
    if countries is not None:
        # sea first: everything not covered by a polygon is ocean
        ax.set_facecolor("#D6E8F2")
        neighbours = countries[countries["name"] != "Colombia"]
        colombia = countries[countries["name"] == "Colombia"]
        neighbours.plot(ax=ax, facecolor="0.93", edgecolor="0.75",
                        linewidth=0.5, zorder=1)
        colombia.plot(ax=ax, facecolor="0.82", edgecolor="0.35",
                      linewidth=0.8, zorder=2)
        # the study window's own SRTM raster inside the box: the terrain the
        # top row zooms into, from the released feature set. The rest of the
        # country stays flat because the deposit carries no national DEM.
        ax.imshow(np.nan_to_num(elev, nan=float(np.nanmin(elev))),
                  extent=[dom[0], dom[1], dom[2], dom[3]], origin="lower",
                  cmap=plt.cm.cividis, interpolation="bilinear", zorder=3)
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color="#BB5566", linewidth=1.0, zorder=4)
        ax.add_patch(Rectangle((dom[0], dom[2]), dom[1] - dom[0],
                               dom[3] - dom[2], facecolor="none",
                               edgecolor="k", linewidth=1.1, zorder=5))
        ax.set_xlim(-80.5, -65.5)
        ax.set_ylim(-5.0, 13.5)
        # the two seas, named where they are
        ax.text(-78.9, 3.2, "Pacific\nOcean", fontsize=8, style="italic",
                color="#4A7C99", ha="center", va="center", zorder=2)
        ax.text(-75.8, 12.6, "Caribbean Sea", fontsize=8, style="italic",
                color="#4A7C99", ha="center", va="center", zorder=2)
        # north arrow, top-left, outside the land
        ax.annotate("N", xy=(-79.6, 12.4), xytext=(-79.6, 10.4),
                    arrowprops=dict(arrowstyle="-|>", color="0.25",
                                    linewidth=1.2),
                    fontsize=11, fontweight="bold", color="0.25",
                    ha="center", va="bottom", zorder=6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("0.7")
        ax.spines[side].set_linewidth(0.6)
    ax.set_title("(a)  Location in Colombia", fontsize=12,
                 fontweight="bold", pad=6, loc="left")
    locator_ax, locator_dom = ax, dom

    # ---- (b) hillshaded terrain ----------------------------------------
    ax = axes[1]
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
    frame(ax, "b", "Terrain, shaded")
    ax.set_ylabel("Latitude", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                               norm=mcolors.Normalize(vmin=float(np.nanmin(elev)),
                                                      vmax=float(np.nanmax(elev))))
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.10)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("m a.s.l.", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # ---- (c) mean annual precipitation ----------------------------------
    ax = axes[2]
    im = ax.pcolormesh(lon_g, lat_g, annual, cmap=plt.cm.YlGnBu,
                       shading="auto", rasterized=True)
    ax.contour(lon_g, lat_g, elev, levels=[HIGH], colors=["0.25"],
               linewidths=0.8, zorder=5)
    frame(ax, "c", "Mean annual precipitation")
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.10)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"mm yr$^{-1}$", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # ---- (d) the relation, band by band ---------------------------------
    ax = axes[3]
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
    ax.set_title("(d)  The gradient, cell by cell", fontsize=12,
                 fontweight="bold", pad=6, loc="left")
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # legend under the axes: the rule is that nothing but data goes inside one
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c,
                              markersize=5,
                              label=f"{n}  ($r$={r:+.3f}, $n$={k:,})")
                       for n, c, r, k in rs],
              loc="center left", bbox_to_anchor=(1.03, 0.5), ncol=1,
              fontsize=9, frameon=False, handletextpad=0.5, labelspacing=0.5)

    # Leaders from the domain box to the frame it opens into, so the two read
    # as one map at two scales. Drawn after tight_layout, when both axes have
    # the positions the leaders have to connect.
    if countries is not None:
        for corner, frac in ((locator_dom[3], 1.0), (locator_dom[2], 0.0)):
            fig.add_artist(ConnectionPatch(
                xyA=(locator_dom[1], corner), coordsA=locator_ax.transData,
                xyB=(0.0, frac), coordsB=axes[1].transAxes,
                color="0.45", linewidth=0.7, linestyle=(0, (4, 2)),
                zorder=0))

    save_figure(fig, OUT, dpi=OUTPUT_DPI, mirror=OUT_DELIVERY, pdf_dpi=300,
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
