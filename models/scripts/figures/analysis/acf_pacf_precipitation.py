"""ACF and PACF of the spatially averaged CHIRPS-2.0 monthly precipitation series.

The figure this replaces had no generator in the repository, so its numbers
could not be checked against the data and its annotations could not be moved.
Two defects are fixed here: the PACF lag-1 annotation collided with the panel
title, and both titles carried a raw double hyphen.

Output: .docs/thesis/figures/pacf_acf_chirps.png
"""
from pathlib import Path
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

FIGURES_ROOT = Path(__file__).resolve().parent.parent
ROOT = FIGURES_ROOT.parents[2]
sys.path.insert(0, str(FIGURES_ROOT))
from _config import setup_paper_style, OUTPUT_DPI  # noqa: E402

DATA = ROOT / "data" / "output" / (
    "complete_dataset_with_features_with_clusters_elevation_windows_"
    "imfs_with_onehot_elevation_clean.nc"
)
OUT = ROOT / ".docs" / "thesis" / "figures" / "pacf_acf_chirps.png"

NLAGS = 36
ANNOTATE = (1, 2, 12, 24)


def acf(x, nlags):
    """Sample autocorrelation, the same normalisation statsmodels uses."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    denom = np.dot(x, x)
    return np.array([1.0 if k == 0 else np.dot(x[k:], x[:-k]) / denom
                     for k in range(nlags + 1)])


def pacf_durbin_levinson(r, nlags):
    """Partial autocorrelation by the Durbin-Levinson recursion on the ACF."""
    phi = np.zeros((nlags + 1, nlags + 1))
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    phi[1, 1] = r[1]
    out[1] = phi[1, 1]
    for k in range(2, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        out[k] = phi[k, k]
    return out


def stem_panel(ax, lags, vals, colour, ylabel, title, ci, annotate_at, lag0=0):
    ax.axhspan(-ci, ci, color="0.85", zorder=0,
               label=f"95\\% CI (±{ci:.3f})".replace("\\", ""))
    ax.axhline(0, color="0.3", linewidth=0.8, zorder=1)
    ax.vlines(lags, 0, vals, color=colour, linewidth=1.8, zorder=2)
    ax.plot(lags, vals, "o", color=colour, markersize=5, zorder=3)

    span = vals.max() - vals.min()
    for k in annotate_at:
        i = k - lag0
        if not 0 <= i < len(vals):
            continue
        v = vals[i]
        # annotate away from the panel edge: above a positive spike, below a
        # negative one, which is what put the lag-1 label into the title before
        above = v >= 0
        ax.annotate(f"lag {k}\n{v:.2f}",
                    xy=(k, v), xytext=(0, 16 if above else -30),
                    textcoords="offset points", ha="center",
                    va="bottom" if above else "top", fontsize=8,
                    color="0.15")

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(vals.min() - 0.22 * span, vals.max() + 0.22 * span)
    ax.set_xlim(lag0 - 1, lag0 + len(vals))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)


def main():
    setup_paper_style()

    ds = xr.open_dataset(DATA)
    da = ds["total_precipitation"]
    series = da.mean(dim=[d for d in da.dims if d != "time"]).values
    n = series.size
    ci = 1.96 / np.sqrt(n)

    r = acf(series, NLAGS)
    p = pacf_durbin_levinson(r, NLAGS)
    lags = np.arange(NLAGS + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    area = "CHIRPS-2.0 monthly precipitation, Boyacá study area"
    stem_panel(ax1, lags, r, "#1f77b4", "ACF",
               f"Autocorrelation function (ACF): {area}", ci, ANNOTATE)
    stem_panel(ax2, lags[1:], p[1:], "#ff7f0e", "PACF",
               f"Partial autocorrelation function (PACF): {area}", ci, ANNOTATE,
               lag0=1)
    ax2.set_xlabel("Lag (months)")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT.relative_to(ROOT)}")
    print(f"series length n={n}, 95% CI = +/-{ci:.4f}")
    for k in ANNOTATE:
        print(f"  lag {k:>2}: ACF={r[k]:+.3f}  PACF={p[k]:+.3f}")


if __name__ == "__main__":
    main()
