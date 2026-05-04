"""Generate bimodal seasonal-cycle climograph for EGU26 poster.

Renders monthly mean precipitation across Boyaca with peak-month markers,
using the Okabe-Ito palette and project figure_config defaults.

Output: .docs/conferences/EGU26/poster/figures/bimodal_cycle.png  (600 DPI)
"""
from pathlib import Path
import sys

import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "models" / "scripts"))
from figure_config import setup_paper_style, COLORS  # noqa: E402

DATA = ROOT / "data" / "output" / (
    "complete_dataset_with_features_with_clusters_elevation_windows_"
    "imfs_with_onehot_elevation_clean.nc"
)
OUT = (
    ROOT / ".docs" / "conferences" / "EGU26" / "poster"
    / "figures" / "bimodal_cycle.png"
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main() -> None:
    setup_paper_style()

    ds = xr.open_dataset(DATA)
    var = "total_precipitation"
    da = ds[var]

    spatial_dims = [d for d in da.dims if d != "time"]
    spatial_mean = da.mean(dim=spatial_dims)
    monthly = spatial_mean.groupby("time.month").mean().values

    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    ax.plot(
        MONTHS, monthly,
        marker="o", color=COLORS["v2"],
        linewidth=1.8, markersize=5, zorder=2,
    )

    peak_indices = {4: "Peak May", 9: "Peak Oct"}
    for idx, label in peak_indices.items():
        ax.scatter(
            [MONTHS[idx]], [monthly[idx]],
            marker="^", s=80, color=COLORS["v5"],
            edgecolor="black", linewidth=0.6, zorder=3,
        )
        ax.annotate(
            label,
            xy=(MONTHS[idx], monthly[idx]),
            xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=8,
        )

    ax.set_xlabel("Month", fontsize=8)
    ax.set_ylabel("Mean precipitation (mm/month)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=600, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {OUT}")
    print(f"Peak May: {monthly[4]:.1f} mm/month")
    print(f"Peak Oct: {monthly[9]:.1f} mm/month")
    print(f"Min:      {monthly.min():.1f} mm/month")


if __name__ == "__main__":
    main()
