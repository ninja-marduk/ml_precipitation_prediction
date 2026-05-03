"""
Paper 1 Data Chart Generation Script (v3 - PDF vector output)
==============================================================
Generates data chart figures as vector PDF for Paper 1.
Tree diagrams and PRISMA are now TikZ (see generate_paper1_tikz.py).

Figures:
  image10.pdf - Metric frequency bar chart
  image11.pdf - Research trends stacked bar chart
  image12.pdf - R2 ranking horizontal bar chart
  image13.pdf - Boxplots + histogram (R2 by category)

Usage:
  python models/scripts/generate_paper1_figures.py
"""
import sys
import io
import warnings

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "docs" / "papers" / "1" / "data"
FIG_DIR = PROJECT_ROOT / "docs" / "papers" / "1" / "latex" / "figures"

# --- Page geometry (IWA template) ---
TEXT_WIDTH = 6.30   # inches (A4 with 2.5cm margins each side)

# --- Okabe-Ito colorblind-safe palette ---
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY = "#999999"
C_WINE = "#882255"

# --- Global style ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def _save(fig, path):
    """Save figure as vector PDF."""
    fig.savefig(path, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# =============================================================================
# MANUAL CATEGORY MAPPING (from paper Table 2 "Hybrid Category" column)
# Maps study reference title -> primary category for boxplots
# =============================================================================
STUDY_CATEGORIES = {
    "A Stacking Ensemble Learning Model for Monthly Rainfall Prediction in the Taihu Basin, China":
        "Component Combination",
    "Improved Monthly and Seasonal Multi-Model Ensemble Precipitation Forecasts in Southwest Asia Using Machine Learning Algorithms":
        "Component Combination",
    "A multiscale long short-term memory model with attention mechanism for improving monthly precipitation prediction":
        "Deep Hybrid",
    "Linking Singular Spectrum Analysis and Machine Learning for Monthly Rainfall Forecasting":
        "Data Preprocessing",
    "Forecasting of monthly precipitation based on ensemble empirical mode decomposition and Bayesian model averaging":
        "Data Preprocessing",
    "Machine learning model combined with CEEMDAN algorithm for monthly precipitation prediction":
        "Data Preprocessing",
    "A comparative study of extensive machine learning models for predicting long-term monthly rainfall with an ensemble of climatic and meteorological predictors":
        "Component Combination",
    "Stacking machine learning models versus a locally weighted linear model to generate high-resolution monthly precipitation over a topographically complex area":
        "Component Combination",
    "A Comparative Assessment of Metaheuristic Optimized Extreme Learning Machine and Deep Neural Network in Multi-Step-Ahead Long-term Rainfall Prediction for All-Indian Regions":
        "Parameter Optimization",
    "Combining time varying filtering based empirical mode decomposition and machine learning to predict precipitation from nonlinear series":
        "Data Preprocessing",
    "Artificial intelligent systems optimized by metaheuristic algorithms and teleconnection indices for rainfall modeling: The case of a humid region in the mediterranean basin":
        "Parameter Optimization",
    "Deep BLSTM-GRU Model for Monthly Rainfall Prediction: A Case Study of Simtokha, Bhutan":
        "Deep Hybrid",
    "Prediction of the standardized precipitation index based on the long short-term memory and empirical mode decomposition-extreme learning machine models: The Case of Sakarya, Turkiye":
        "Data Preprocessing",
    "A novel integrated learning model for rainfall prediction CEEMD-FCMSE -Stacking":
        "Component Combination",
    "Enhancing monthly precipitation prediction using a hybrid model of neural network and grey wolf optimizer":
        "Parameter Optimization",
    "Development and assessment of a monthly water-energy balance model with different precipitation inputs and spatial resolutions":
        "Other Hybrid",
    "Stacking-based ensemble learning with multi-model forecasting for monthly precipitation prediction":
        "Component Combination",
    "Spatiotemporal analysis of monthly precipitation using hybrid ML models with optimization":
        "Parameter Optimization",
    "Gaussian mutation\u2013orca predation algorithm\u2013deep residual shrinkage network (DRSN)\u2013temporal convolutional network (TCN)\u2013 random forest model: an advanced machine learning model for predicting monthly rainfall and filtering irrelevant data":
        "Postprocessing",
    "Precipitation prediction based on variational mode decomposition combined with the crested porcupine optimization algorithm for long short-term memory model":
        "Data Preprocessing",
    "The Development of a Hybrid Wavelet-ARIMA-LSTM Model for Precipitation Amounts and Drought Analysis":
        "Data Preprocessing",
    "Semi-empirical prediction method for monthly precipitation prediction based on environmental factors and comparison with stochastic and machine learning models":
        "Data Preprocessing",
    "Developing an innovative machine learning model for rainfall prediction in a semi-arid region":
        "Data Preprocessing",
    "Comparative analysis of data-driven models and signal processing techniques in the monthly maximum daily precipitation prediction of El Kerma station Northeast of Algeria":
        "Data Preprocessing",
    "Monthly precipitation estimation using the PERSIANN-CDR dataset":
        "Other Hybrid",
    "Application of wavelet neural network for monthly precipitation prediction":
        "Data Preprocessing",
    "Evaluation of random forests for short-term daily streamflow forecast in rainfall- and snowmelt-driven watersheds":
        "Component Combination",
    "A hybrid wavelet-based adaptive model for precipitation prediction":
        "Data Preprocessing",
    "SMOTE-based resampling combined with K-means and XGBoost for monthly precipitation":
        "Component Combination",
    "Rainfall forecasting model using machine learning methods: Case study Terengganu, Malaysia":
        "Component Combination",
    # --- Added from PDF verification (Phase 28) ---
    "Development of Monthly Scale Precipitation-Forecasting Model for Indian Subcontinent using Wavelet-Based Deep Learning Approach":
        "Data Preprocessing",
    "Evaluation of a novel hybrid lion swarm optimization \u2013 AdaBoostRegressor model for forecasting monthly precipitation":
        "Parameter Optimization",
    "Comparative Evaluation of Hybrid SARIMA and Machine Learning Techniques Based on Time Varying and Decomposition of Precipitation Time Serie":
        "Data Preprocessing",
}


# =============================================================================
# FIGURE 10: METRIC FREQUENCY BAR CHART
# =============================================================================
UNIT_DEPENDENT_METRICS = {"RMSE", "MAE", "MSE", "MW"}


def generate_metric_frequency(output_path):
    from matplotlib.patches import Patch

    df = pd.read_csv(DATA_DIR / "metric_frequency.csv")
    top = df.head(12).copy()

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.7))

    colors = [C_GRAY if m in UNIT_DEPENDENT_METRICS else C_BLUE
              for m in top["metric"]]

    bars = ax.bar(range(len(top)), top["frequency"], color=colors,
                  edgecolor="white", linewidth=0.5, width=0.72)

    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["metric"], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Number of Studies", fontsize=8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                str(int(h)), ha="center", va="bottom", fontsize=7,
                fontweight="bold")

    ax.set_ylim(0, top["frequency"].max() + 3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    legend_elements = [
        Patch(facecolor=C_BLUE, edgecolor="white", label="Dimensionless"),
        Patch(facecolor=C_GRAY, edgecolor="white", label="Unit-dependent"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right",
              framealpha=0.9, edgecolor="none")

    plt.tight_layout()
    _save(fig, output_path)
    print(f"  image10.pdf (Metrics frequency): saved")


# =============================================================================
# FIGURE 11: RESEARCH TRENDS STACKED BAR
# =============================================================================
def generate_research_trends(output_path):
    df = pd.read_csv(DATA_DIR / "research_trends.csv")
    df["total"] = df["sciencedirect"] + df["scopus"] + df["ieee_xplore"]
    df = df[df["total"] > 0].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.7))

    x = np.arange(len(df))
    w = 0.72

    ax.bar(x, df["sciencedirect"], width=w, label="ScienceDirect",
           color=C_BLUE, edgecolor="white", linewidth=0.5)
    ax.bar(x, df["scopus"], width=w, bottom=df["sciencedirect"],
           label="Scopus", color=C_ORANGE, edgecolor="white", linewidth=0.5)
    ax.bar(x, df["ieee_xplore"], width=w,
           bottom=df["sciencedirect"] + df["scopus"],
           label="IEEE Xplore", color=C_GREEN, edgecolor="white", linewidth=0.5)

    for i, row in df.iterrows():
        t = int(row["total"])
        if t > 0:
            ax.text(i, t + 0.2, str(t), ha="center", va="bottom",
                    fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    labels = [str(int(y)) for y in df["year"]]
    if labels[-1] == "2025":
        labels[-1] = "2025*"
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Number of Publications", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9,
              edgecolor="none", fancybox=True)
    ax.set_ylim(0, df["total"].max() + 3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    _save(fig, output_path)
    print(f"  image11.pdf (Research trends): saved")


# =============================================================================
# FIGURE 12: R2 RANKING HORIZONTAL BARS
# =============================================================================
def generate_r2_ranking(output_path):
    df = pd.read_csv(DATA_DIR / "metrics_results.csv")
    r2 = df[df["metric"].str.strip() == "R^2"].copy()
    r2["result"] = pd.to_numeric(r2["result"], errors="coerce")
    r2 = r2.dropna(subset=["result"])
    r2["model"] = r2["model"].str.strip()

    # Sort ascending (worst at bottom, best at top)
    r2 = r2.sort_values("result", ascending=True).reset_index(drop=True)

    n = len(r2)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, max(2.5, n * 0.24)))

    # Color by R2 tier
    colors = []
    for v in r2["result"]:
        if v >= 0.90:
            colors.append(C_BLUE)
        elif v >= 0.70:
            colors.append(C_ORANGE)
        else:
            colors.append(C_RED)

    bars = ax.barh(range(n), r2["result"], color=colors,
                   edgecolor="white", linewidth=0.5, height=0.72)

    ax.set_yticks(range(n))
    ax.set_yticklabels(r2["model"], fontsize=7)
    ax.set_xlabel("$R^{2}$", fontsize=8)
    ax.set_xlim(0, 1.08)

    for i, (bar, val) in enumerate(zip(bars, r2["result"])):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)

    # Threshold lines
    ax.axvline(x=0.90, color=C_GREEN, ls="--", lw=0.8, alpha=0.5)
    ax.axvline(x=0.70, color=C_ORANGE, ls="--", lw=0.8, alpha=0.5)
    ax.text(0.905, -0.8, "High", fontsize=6, color=C_GREEN, ha="left", va="top")
    ax.text(0.705, -0.8, "Mid", fontsize=6, color=C_ORANGE, ha="left", va="top")

    plt.tight_layout()
    _save(fig, output_path)
    print(f"  image12.pdf (R2 ranking): saved")


# =============================================================================
# FIGURE 13: BOXPLOTS + HISTOGRAM (with corrected category mapping)
# =============================================================================
def generate_boxplots_histogram(output_path):
    metrics = pd.read_csv(DATA_DIR / "metrics_results.csv")
    r2 = metrics[metrics["metric"].str.strip() == "R^2"].copy()
    r2["result"] = pd.to_numeric(r2["result"], errors="coerce")
    r2 = r2.dropna(subset=["result"])

    # Use manual category mapping from Table 2
    def get_category(ref_title):
        ref = str(ref_title).strip()
        if ref in STUDY_CATEGORIES:
            return STUDY_CATEGORIES[ref]
        for key, cat in STUDY_CATEGORIES.items():
            if ref[:50] == key[:50]:
                return cat
        return "Other Hybrid"

    r2["category"] = r2["ref"].apply(get_category)

    cat_order = [
        "Parameter Optimization",
        "Data Preprocessing",
        "Deep Hybrid",
        "Other Hybrid",
        "Component Combination",
        "Postprocessing",
    ]
    cat_colors = {
        "Parameter Optimization": C_BLUE,
        "Data Preprocessing": C_ORANGE,
        "Deep Hybrid": C_WINE,
        "Other Hybrid": C_GRAY,
        "Component Combination": C_GREEN,
        "Postprocessing": C_PURPLE,
    }
    cat_short = {
        "Parameter Optimization": "Param.\nOptim.",
        "Data Preprocessing": "Data\nPreproc.",
        "Deep Hybrid": "Deep\nHybrid",
        "Other Hybrid": "Other\nHybrid",
        "Component Combination": "Comp.\nComb.",
        "Postprocessing": "Post-\nproc.",
    }

    available = [c for c in cat_order if c in r2["category"].values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8),
                                    gridspec_kw={"width_ratios": [1, 1]})

    # (a) Boxplots
    data = [r2[r2["category"] == c]["result"].values for c in available]
    bp = ax1.boxplot(
        data, vert=True, patch_artist=True, widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker="o", markersize=4, markerfacecolor=C_GRAY,
                        alpha=0.6),
    )
    for patch, cat in zip(bp["boxes"], available):
        patch.set_facecolor(cat_colors[cat])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    # Mean markers (white diamond)
    for i, (cat, vals) in enumerate(zip(available, data)):
        if len(vals) > 0:
            ax1.scatter(i + 1, np.mean(vals), marker="D", s=30,
                        color="white", edgecolors="black", linewidth=1.0,
                        zorder=6)

    # Overlay individual points
    for i, (cat, vals) in enumerate(zip(available, data)):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax1.scatter(np.full(len(vals), i + 1) + jitter, vals,
                    color=cat_colors[cat], s=18, alpha=0.5, zorder=5,
                    edgecolors="white", linewidth=0.3)

    ax1.set_xticklabels([cat_short[c] for c in available], fontsize=6.5)
    ax1.set_ylabel("$R^{2}$", fontsize=8)
    ax1.set_ylim(0.3, 1.05)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2, linewidth=0.5)

    for i, cat in enumerate(available):
        n = len(r2[r2["category"] == cat])
        ax1.text(i + 1, 0.35, f"n={n}", ha="center", fontsize=6,
                 fontweight="bold", color="#666666")

    # (b) Histogram
    all_r2 = r2["result"].values
    ax2.hist(all_r2, bins=10, color=C_BLUE, edgecolor="white",
             linewidth=0.5, alpha=0.75)

    median_r2 = np.median(all_r2)
    ax2.axvline(x=median_r2, color=C_RED, ls="--", lw=1.5,
                label=f"Median = {median_r2:.3f}")

    ax2.set_xlabel("$R^{2}$", fontsize=8)
    ax2.set_ylabel("Count", fontsize=8)
    ax2.legend(fontsize=7, framealpha=0.9, edgecolor="none")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2, linewidth=0.5)

    # Panel labels
    ax1.text(-0.06, 1.04, "(a)", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", va="top", ha="right")
    ax2.text(-0.06, 1.04, "(b)", transform=ax2.transAxes,
             fontsize=9, fontweight="bold", va="top", ha="right")

    plt.tight_layout(w_pad=1.5)
    plt.subplots_adjust(bottom=0.18, top=0.95, left=0.08, right=0.97)
    _save(fig, output_path)
    print(f"  image13.pdf (Boxplots + histogram): saved")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 55)
    print("  Paper 1 Data Chart Generation (v3 - PDF vector)")
    print("=" * 55)
    print(f"  Output: {FIG_DIR}")
    print(f"  Width: {TEXT_WIDTH} in ({TEXT_WIDTH * 2.54:.1f} cm)")
    print()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("[Data Charts - PDF vector output]")
    generate_metric_frequency(FIG_DIR / "image10.pdf")
    generate_research_trends(FIG_DIR / "image11.pdf")
    generate_r2_ranking(FIG_DIR / "image12.pdf")
    generate_boxplots_histogram(FIG_DIR / "image13.pdf")
    print()

    print("[Verification]")
    for i in range(10, 14):
        p = FIG_DIR / f"image{i}.pdf"
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  image{i}.pdf: {size_kb:.0f} KB (vector)")
        else:
            print(f"  image{i}.pdf: MISSING!")

    print()
    print("NOTE: Tree diagrams and PRISMA are now TikZ files.")
    print("      Run generate_paper1_tikz.py to regenerate them.")
    print()
    print("=" * 55)
    print("  Done")
    print("=" * 55)


if __name__ == "__main__":
    main()
