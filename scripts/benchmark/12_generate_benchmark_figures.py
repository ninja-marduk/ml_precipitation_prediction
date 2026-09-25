"""
Benchmark Analysis Script 12: Generate Publication-Quality Figures

Generates doctoral-level benchmark visualizations following GraphCast, Pangu-Weather,
FourCastNet, and WeatherBench 2 standards.

Includes ALL three model families:
- ConvLSTM (Baselines)
- FNO (Physics-Informed)
- GNN-TAT (Hybrid)

Figures:
1. Horizon Degradation Curves (H=1 to H=12)
2. Feature Set Heatmap (BASIC vs KCE vs PAFC)
3. Multi-Metric Radar Chart comparing model families
4. Parameter Efficiency Plot (Pareto frontier)
5. Model Ranking Bar Plot (Top performers)
6. Training Dynamics Comparison
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure matplotlib for publication quality.
# Same sans-serif 14/11/10 pt hierarchy as models/scripts/figures/_config.py
# (PAPER_RC), so these figures read as one set with the rest of the thesis.
# Titles are NOT drawn inside the images: the LaTeX caption carries them.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 700,
    'savefig.dpi': 700,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Embedded width of each figure in the thesis (fraction of \textwidth) and the
# thesis \textwidth in inches (A4, 30/25 mm margins = 155 mm).
TEXTWIDTH_IN = 155 / 25.4
EMBED_WIDTH = {
    'horizon_degradation': 0.95,
    'feature_heatmap': 0.72,
    'radar_chart': 0.68,
    'parameter_efficiency': 0.72,
    'model_ranking': 0.72,
    'training_dynamics': 0.90,
    'model_comparison_v4_gnn_tat': 0.95,
}
# On-page scale target: a 10 pt source font renders at ~8 pt on the page,
# matching the other thesis figures (ticks ~7.5-8 pt, axis labels ~8.5-9 pt).
PAGE_SCALE = 0.80


def figsize_for(name: str, aspect: float) -> Tuple[float, float]:
    """Figure size (in) whose width, embedded at EMBED_WIDTH[name], gives PAGE_SCALE."""
    w = EMBED_WIDTH[name] * TEXTWIDTH_IN / PAGE_SCALE
    return (w, w * aspect)


# Dark variant of the Okabe-Ito yellow for lines and markers: #F0E442 has a
# contrast of ~1.3:1 on white. Filled areas keep #F0E442 with a dark edge.
FNO_LINE = '#B89B00'
FNO_EDGE = '#6B5A00'

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
V2_DIR = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models'
V3_DIR = PROJECT_ROOT / 'models' / 'output' / 'V3_FNO_Models'
V4_DIR = PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'papers' / '4' / 'figures'

# Okabe-Ito colorblind-safe palette (consistent with models/scripts/figures/_config.py)
COLORS = {
    # Model families
    'ConvLSTM': '#0072B2',   # Blue (Okabe-Ito) - Baselines & Enhanced
    'FNO': '#F0E442',        # Yellow (Okabe-Ito) - Physics-Informed
    'GNN-TAT': '#E69F00',    # Orange (Okabe-Ito) - Hybrid
    # Feature sets
    'BASIC': '#009E73',      # Bluish green (Okabe-Ito)
    'KCE': '#CC79A7',        # Reddish purple (Okabe-Ito)
    'PAFC': '#56B4E9',       # Sky blue (Okabe-Ito)
}
# Variant colors for individual models within families
VARIANT_COLORS = {
    'ConvLSTM_Bidir': '#56B4E9',    # Sky blue (lighter than main blue)
    'ConvLSTM_Residual': '#999999',  # Gray (Okabe-Ito baseline)
    'FNO_Pure': '#CC79A7',           # Reddish purple (distinct from yellow)
    'GNN_TAT_alt': '#D55E00',        # Vermillion (darker than orange)
}

# Model parameter counts (all families)
MODEL_PARAMS = {
    # ConvLSTM Models (Baselines)
    'ConvLSTM': 78_000,
    'ConvRNN': 45_000,
    'ConvLSTM_Enhanced': 156_000,
    'ConvRNN_Enhanced': 89_000,
    'ConvLSTM_Bidirectional': 148_000,
    'ConvLSTM_Residual': 153_000,
    'ConvLSTM_Attention': 206_000,
    'ConvLSTM_MeteoAttention': 198_000,
    'ConvLSTM_EfficientBidir': 60_000,
    'Transformer_Baseline': 41_800_000,
    # FNO Models (Physics-Informed)
    'FNO_ConvLSTM_Hybrid': 106_000,
    'FNO_Pure': 85_000,
    # GNN-TAT Models (Hybrid)
    'GNN_TAT_GAT': 98_000,
    'GNN_TAT_SAGE': 106_000,
    'GNN_TAT_GCN': 98_000,
}


# Single-run benchmark as published (V2 ConvLSTM family + V3 FNO, 39-configuration
# suite). The per-version CSVs under models/output/ were later overwritten by the
# seed-resolved retrains, so they no longer hold the published single-run values.
UNIFIED_BENCHMARK_CSV = PROJECT_ROOT / 'docs' / 'models' / 'comparative' / 'data' / 'unified_metrics_h12.csv'


def _load_unified(version: str, family: str) -> pd.DataFrame:
    df = pd.read_csv(UNIFIED_BENCHMARK_CSV)
    df = df[df['Version'] == version].copy()
    df['Family'] = family
    if 'mean_bias_mm' not in df.columns and 'Mean_Pred_mm' in df.columns:
        df['mean_bias_mm'] = df['Mean_Pred_mm'] - df['Mean_True_mm']
    logger.info(f"Loaded {family} metrics from {UNIFIED_BENCHMARK_CSV.name}: {len(df)} records")
    return df


def load_convlstm_metrics() -> pd.DataFrame:
    """Load ConvLSTM baseline metrics from CSV."""
    if UNIFIED_BENCHMARK_CSV.exists():
        return _load_unified('V2', 'ConvLSTM')
    metrics_path = V2_DIR / 'metrics_spatial_v2_refactored_h12.csv'
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df['Family'] = 'ConvLSTM'
        # Calculate bias if not present
        if 'mean_bias_mm' not in df.columns and 'Mean_Pred_mm' in df.columns and 'Mean_True_mm' in df.columns:
            df['mean_bias_mm'] = df['Mean_Pred_mm'] - df['Mean_True_mm']
        logger.info(f"Loaded ConvLSTM metrics: {len(df)} records")
        return df
    return pd.DataFrame()


def load_fno_metrics() -> pd.DataFrame:
    """Load FNO metrics from CSV."""
    if UNIFIED_BENCHMARK_CSV.exists():
        return _load_unified('V3', 'FNO')
    metrics_path = V3_DIR / 'metrics_spatial_v2_refactored_h12.csv'
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df['Family'] = 'FNO'
        # Calculate bias if not present
        if 'mean_bias_mm' not in df.columns and 'Mean_Pred_mm' in df.columns and 'Mean_True_mm' in df.columns:
            df['mean_bias_mm'] = df['Mean_Pred_mm'] - df['Mean_True_mm']
        logger.info(f"Loaded FNO metrics: {len(df)} records")
        return df
    return pd.DataFrame()


def load_gnn_tat_metrics() -> pd.DataFrame:
    """Load GNN-TAT metrics from CSV."""
    metrics_path = V4_DIR / 'metrics_spatial_v4_gnn_tat_h12.csv'
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df['Family'] = 'GNN-TAT'
        # Calculate bias if not present
        if 'mean_bias_mm' not in df.columns and 'Mean_Pred_mm' in df.columns and 'Mean_True_mm' in df.columns:
            df['mean_bias_mm'] = df['Mean_Pred_mm'] - df['Mean_True_mm']
        logger.info(f"Loaded GNN-TAT metrics: {len(df)} records")
        return df
    return pd.DataFrame()


def load_training_logs() -> Dict[str, pd.DataFrame]:
    """Load training logs from experiments."""
    logs = {}
    training_dir = V2_DIR / 'h12'

    for exp in ['BASIC', 'KCE', 'PAFC']:
        metrics_dir = training_dir / exp / 'training_metrics'
        if metrics_dir.exists():
            for log_file in metrics_dir.glob('*_training_log_h12.csv'):
                model_name = log_file.stem.replace('_training_log_h12', '')
                key = f"{exp}_{model_name}"
                try:
                    logs[key] = pd.read_csv(log_file)
                except Exception as e:
                    logger.warning(f"Could not load {log_file}: {e}")

    logger.info(f"Loaded {len(logs)} training logs")
    return logs


def create_horizon_degradation_plot(combined_df: pd.DataFrame) -> plt.Figure:
    """
    Create horizon degradation curves showing R2 from H=1 to H=12.
    Includes ALL three model families: ConvLSTM, FNO, GNN-TAT.
    Following GraphCast visualization style.
    """
    logger.info("Creating horizon degradation plot (all families)...")

    fig, ax = plt.subplots(figsize=figsize_for('horizon_degradation', 0.55))

    # Select representative models from EACH family
    models_to_plot = [
        # ConvLSTM (Baselines)
        ('ConvLSTM', 'BASIC', 'ConvLSTM', COLORS['ConvLSTM'], '-', 'o'),
        ('ConvLSTM_Bidirectional', 'BASIC', 'ConvLSTM', VARIANT_COLORS['ConvLSTM_Bidir'], '--', 's'),
        ('ConvLSTM_Residual', 'BASIC', 'ConvLSTM', VARIANT_COLORS['ConvLSTM_Residual'], '-.', '^'),
        # FNO (Physics-Informed)
        ('FNO_ConvLSTM_Hybrid', 'BASIC', 'FNO', FNO_LINE, '-', 'D'),
        ('FNO_Pure', 'BASIC', 'FNO', VARIANT_COLORS['FNO_Pure'], '--', 'v'),
        # GNN-TAT (Hybrid)
        ('GNN_TAT_GAT', 'BASIC', 'GNN-TAT', COLORS['GNN-TAT'], '-', 'p'),
        ('GNN_TAT_GCN', 'PAFC', 'GNN-TAT', VARIANT_COLORS['GNN_TAT_alt'], '--', 'h'),
    ]

    horizons = range(1, 13)

    for model, exp, family, color, linestyle, marker in models_to_plot:
        model_data = combined_df[(combined_df['Model'] == model) &
                                  (combined_df['Experiment'] == exp) &
                                  (combined_df['Family'] == family)]

        if model_data.empty:
            logger.warning(f"No data for {model} ({exp}) in {family}")
            continue

        r2_by_horizon = []
        for h in horizons:
            h_data = model_data[model_data['H'] == h]
            if not h_data.empty:
                r2_by_horizon.append(h_data['R^2'].mean())
            else:
                r2_by_horizon.append(np.nan)

        # Professional label without version numbers
        label = f"{model.replace('_', ' ')} ({exp})"
        ax.plot(horizons, r2_by_horizon, color=color, linestyle=linestyle,
                marker=marker, markersize=5, linewidth=1.6, label=label,
                markeredgecolor=FNO_EDGE if family == 'FNO' and color == FNO_LINE else color)

    ax.set_xlabel('Forecast horizon (months)')
    ax.set_ylabel('$R^2$')

    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.0, 0.75)
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend below the plot, outside the axes
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False,
              ncol=3, columnspacing=1.2, handlelength=2.6)

    plt.tight_layout()
    return fig


def create_feature_heatmap(combined_df: pd.DataFrame) -> plt.Figure:
    """
    Create heatmap showing R2 by model and feature set.
    Includes ALL three model families.
    """
    logger.info("Creating feature set heatmap (all families)...")

    h12_data = combined_df[combined_df['H'] == 12]

    # Pivot table for heatmap
    pivot = h12_data.pivot_table(
        values='R^2',
        index='Model',
        columns='Experiment',
        aggfunc='mean'
    )

    # Get models sorted by max R2, include top performers from each family
    model_order = pivot.max(axis=1).sort_values(ascending=False).head(15).index
    pivot = pivot.loc[model_order]

    # Reorder columns
    cols_to_use = [c for c in ['BASIC', 'KCE', 'PAFC'] if c in pivot.columns]
    pivot = pivot[cols_to_use]

    fig, ax = plt.subplots(figsize=figsize_for('feature_heatmap', 1.20))

    # Create annotation array with proper minus signs (U+2212) per MDPI requirement
    annot_array = pivot.map(lambda v: f'{v:.3f}'.replace('-', '\u2212') if pd.notna(v) else '')

    # Sequential colour-blind-safe ramp (cividis) instead of the red-green
    # RdYlGn: an R^2 skill score has no meaningful centre. Values below 0 are
    # clipped to the darkest tone; the printed number keeps the exact value.
    sns.heatmap(pivot, annot=annot_array, fmt='', cmap='cividis',
                vmin=0.0, vmax=0.7,
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'fontsize': 9},
                cbar_kws={'label': '$R^2$ (H=12)', 'shrink': 0.8})

    ax.set_xlabel('Feature bundle')
    ax.set_ylabel('')

    # Rotate x labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    return fig


def create_radar_chart(combined_df: pd.DataFrame) -> plt.Figure:
    """
    Create multi-metric radar chart comparing ALL three model families.
    Following FourCastNet visualization style.
    """
    logger.info("Creating radar chart (all families)...")

    h12_data = combined_df[combined_df['H'] == 12]

    # Define metric order explicitly (must match categories order)
    metric_keys = ['R2', 'RMSE_inv', 'MAE_inv', 'Bias_inv', 'Param_eff', 'Stability']
    categories = ['$R^2$', 'Inverse\nRMSE', 'Inverse\nMAE', 'Inverse\nbias', 'Parameter\nefficiency', 'Training\nstability']
    N = len(categories)

    # Define parameter counts for each family (from paper)
    PARAM_COUNTS = {
        'ConvLSTM': 148_000,  # ~148K (best variant: Bidirectional)
        'FNO': 106_000,       # ~106K (FNO-Hybrid)
        'GNN-TAT': 98_000     # ~98K parameters
    }

    # Define RMSE standard deviations for each family (calculated from data)
    RMSE_SD = {
        'ConvLSTM': 27.43,
        'FNO': 23.60,
        'GNN-TAT': 6.94
    }

    # Calculate aggregate metrics for each family (using OrderedDict to ensure order)
    def get_family_metrics(df, family):
        family_data = df[df['Family'] == family]
        if family_data.empty:
            return None

        max_rmse = df['RMSE'].max() if 'RMSE' in df.columns else 200
        max_mae = df['MAE'].max() if 'MAE' in df.columns else 150

        # Use BEST model performance (highest R²) instead of mean
        # This shows the potential of each family at its best
        best_idx = family_data['R^2'].idxmax()
        best_model = family_data.loc[best_idx]

        # Calculate R² from best model
        r2_mean = best_model['R^2']

        # Calculate parameter efficiency using min-max normalization
        # This ensures all values are in [0,1] range without zeros
        min_params = min(PARAM_COUNTS.values())
        max_params = max(PARAM_COUNTS.values())
        # Normalize parameters: lower is better, so invert the scale
        param_normalized = (max_params - PARAM_COUNTS[family]) / (max_params - min_params)
        # Weight by R² to account for performance
        param_eff = param_normalized * r2_mean

        # Calculate training stability using min-max normalization
        # This ensures all values are in [0,1] range without zeros
        min_sd = min(RMSE_SD.values())
        max_sd = max(RMSE_SD.values())
        # Normalize SD: lower is better, so invert the scale
        stability = (max_sd - RMSE_SD[family]) / (max_sd - min_sd)

        # Return values in the exact order needed (using best model metrics)
        return [
            r2_mean,  # R² (best)
            1 - (best_model['RMSE'] / max_rmse) if 'RMSE' in best_model else 0.5,  # RMSE_inv (best)
            1 - (best_model['MAE'] / max_mae) if 'MAE' in best_model else 0.5,  # MAE_inv (best)
            1 - abs(best_model.get('mean_bias_mm', 0)) / 50,  # Bias_inv (best)
            param_eff,  # Parameter efficiency (min-max normalized, weighted by R²)
            stability,  # Training stability (min-max normalized)
        ]

    convlstm_values = get_family_metrics(h12_data, 'ConvLSTM')
    fno_values = get_family_metrics(h12_data, 'FNO')
    gnn_values = get_family_metrics(h12_data, 'GNN-TAT')

    # Angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    for name, vals in (('ConvLSTM', convlstm_values), ('FNO', fno_values), ('GNN-TAT', gnn_values)):
        logger.info(f"  radar {name}: " + ', '.join(f'{k}={v:.3f}' for k, v in zip(metric_keys, vals or [])))

    fig, ax = plt.subplots(figsize=figsize_for('radar_chart', 0.95), subplot_kw=dict(projection='polar'))

    # Plot ConvLSTM (Baselines)
    if convlstm_values is not None:
        values = convlstm_values + [convlstm_values[0]]  # Close the polygon
        ax.plot(angles, values, 'o-', linewidth=1.6, color=COLORS['ConvLSTM'], label='ConvLSTM', markersize=5)
        ax.fill(angles, values, alpha=0.15, color=COLORS['ConvLSTM'])

    # Plot FNO (Physics-Informed): dark yellow line, light yellow fill
    if fno_values is not None:
        values = fno_values + [fno_values[0]]  # Close the polygon
        ax.plot(angles, values, 's-', linewidth=1.6, color=FNO_LINE, markeredgecolor=FNO_EDGE,
                label='FNO', markersize=5)
        ax.fill(angles, values, alpha=0.25, color=COLORS['FNO'])

    # Plot GNN-TAT (Hybrid)
    if gnn_values is not None:
        values = gnn_values + [gnn_values[0]]  # Close the polygon
        ax.plot(angles, values, 'D-', linewidth=1.6, color=COLORS['GNN-TAT'], label='GNN-TAT', markersize=5)
        ax.fill(angles, values, alpha=0.15, color=COLORS['GNN-TAT'])

    # Set category labels - pushed outward so they don't overlap polygons
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.tick_params(axis='x', pad=10)

    # Set radial labels
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='#555555')
    ax.set_rlabel_position(90)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)

    plt.tight_layout()
    return fig


def create_parameter_efficiency_plot(combined_df: pd.DataFrame) -> plt.Figure:
    """
    Create parameter efficiency scatter plot showing Pareto frontier.
    Includes ALL three model families.
    """
    logger.info("Creating parameter efficiency plot (all families)...")

    h12_data = combined_df[combined_df['H'] == 12]

    # Aggregate by model
    model_stats = h12_data.groupby(['Model', 'Family']).agg({
        'R^2': 'max',
        'RMSE': 'min',
    }).reset_index()

    # Add parameter counts
    model_stats['Params'] = model_stats['Model'].map(MODEL_PARAMS)
    model_stats = model_stats.dropna(subset=['Params'])

    fig, ax = plt.subplots(figsize=figsize_for('parameter_efficiency', 0.72))

    # Plot ConvLSTM models
    convlstm_data = model_stats[model_stats['Family'] == 'ConvLSTM']
    ax.scatter(convlstm_data['Params'], convlstm_data['R^2'],
               s=40, c=COLORS['ConvLSTM'], alpha=0.85, label='ConvLSTM', marker='o',
               edgecolors='white', linewidths=0.5, zorder=3)

    # Plot FNO models (yellow needs dark edge for visibility on white)
    fno_data = model_stats[model_stats['Family'] == 'FNO']
    ax.scatter(fno_data['Params'], fno_data['R^2'],
               s=55, c=COLORS['FNO'], edgecolors=FNO_EDGE, linewidths=0.9,
               alpha=1.0, label='FNO', marker='^', zorder=3)

    # Plot GNN-TAT models
    gnn_data = model_stats[model_stats['Family'] == 'GNN-TAT']
    ax.scatter(gnn_data['Params'], gnn_data['R^2'],
               s=50, c=COLORS['GNN-TAT'], alpha=0.9, label='GNN-TAT', marker='s',
               edgecolors='white', linewidths=0.5, zorder=3)

    # Label representative models only; offsets in points, chosen so no label
    # sits on a marker or on another label.
    important_models = {
        'ConvLSTM': (-8, 10),
        'ConvLSTM_Bidirectional': (18, 6),
        'ConvRNN': (10, 12),
        'FNO_Pure': (-10, -12),
        'FNO_ConvLSTM_Hybrid': (0, 24),
        'GNN_TAT_GCN': (14, 0),
        'GNN_TAT_SAGE': (14, 0),
        'Transformer_Baseline': (-20, 12),
    }
    pretty = {
        'ConvLSTM': 'ConvLSTM',
        'ConvLSTM_Residual': 'Residual', 'ConvLSTM_Bidirectional': 'Bidirectional, Residual',
        'ConvRNN': 'ConvRNN', 'FNO_Pure': 'FNO (pure)', 'FNO_ConvLSTM_Hybrid': 'FNO-ConvLSTM',
        'GNN_TAT_GAT': 'GAT', 'GNN_TAT_GCN': 'GAT, GCN', 'GNN_TAT_SAGE': 'SAGE',
        'Transformer_Baseline': 'Transformer',
    }

    for _, row in model_stats.iterrows():
        if row['Model'] in important_models:
            offset = important_models[row['Model']]
            ax.annotate(pretty[row['Model']], (row['Params'], row['R^2']),
                        textcoords='offset points', xytext=offset,
                        fontsize=9, color='#333333',
                        ha='left' if offset[0] > 0 else ('right' if offset[0] < 0 else 'center'), va='center',
                        arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5,
                                        shrinkA=0, shrinkB=3))
        logger.info(f"  params {row['Model']}: {row['Params']:.0f}, R2={row['R^2']:.3f}")

    # Find and highlight Pareto frontier
    def is_pareto_efficient(costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient

    # Pareto: minimize params, maximize R2 -> minimize params, minimize -R2
    costs = np.column_stack([model_stats['Params'], -model_stats['R^2']])
    pareto_mask = is_pareto_efficient(costs)

    pareto_models = model_stats[pareto_mask].sort_values('Params')
    if len(pareto_models) > 1:
        ax.plot(pareto_models['Params'], pareto_models['R^2'],
                '--', color='#009E73', alpha=0.8, linewidth=1.4, label='Pareto frontier', zorder=2)

    ax.set_xscale('log')
    ax.set_xlabel('Parameters (log scale)')
    ax.set_ylabel('Best $R^2$ (H=12)')

    ax.set_xlim(2e4, 1e8)
    ax.set_ylim(0.0, 0.75)
    ax.grid(True, alpha=0.3, linestyle='--', which='major')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False,
              columnspacing=1.0, handletextpad=0.3)

    plt.tight_layout()
    return fig


def create_model_ranking_barplot(combined_df: pd.DataFrame) -> plt.Figure:
    """
    Create horizontal bar plot showing model ranking by R2.
    Includes ALL three model families.
    """
    logger.info("Creating model ranking bar plot (all families)...")

    h12_data = combined_df[combined_df['H'] == 12]

    # Get best R2 per model across all experiments
    model_best = h12_data.groupby(['Model', 'Family']).agg({
        'R^2': 'max',
        'Experiment': 'first',
    }).reset_index()

    # Sort by R2 and get top 15
    model_best = model_best.sort_values('R^2', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=figsize_for('model_ranking', 0.85))

    # Colors based on family; the Transformer baseline is not a ConvLSTM
    colors, edges = [], []
    for model, family in zip(model_best['Model'], model_best['Family']):
        if model.startswith('Transformer'):
            colors.append('#999999'); edges.append('#999999')
        elif family == 'ConvLSTM':
            colors.append(COLORS['ConvLSTM']); edges.append(COLORS['ConvLSTM'])
        elif family == 'FNO':
            colors.append(COLORS['FNO']); edges.append(FNO_EDGE)
        else:
            colors.append(COLORS['GNN-TAT']); edges.append(COLORS['GNN-TAT'])

    # Create horizontal bars
    y_pos = range(len(model_best))
    bars = ax.barh(y_pos, model_best['R^2'], color=colors, edgecolor=edges,
                   linewidth=0.8, height=0.7)

    # Add value labels
    for i, (bar, r2) in enumerate(zip(bars, model_best['R^2'])):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
                f'{r2:.3f}', va='center', fontsize=9)
        logger.info(f"  ranking {model_best.iloc[i]['Model']}: {r2:.3f}")

    # Y-axis labels: model name and the feature bundle of its best H=12 score
    best_exp = []
    for _, row in model_best.iterrows():
        sub = h12_data[(h12_data['Model'] == row['Model']) & (h12_data['Family'] == row['Family'])]
        best_exp.append(sub.loc[sub['R^2'].idxmax(), 'Experiment'])
    labels = [f"{m.replace('_', ' ')} ({e})" for m, e in zip(model_best['Model'], best_exp)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel('Best $R^2$ at H=12')

    ax.set_xlim(0, 0.72)

    # Custom legend with ALL families
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['ConvLSTM'], label='ConvLSTM'),
        mpatches.Patch(facecolor=COLORS['FNO'], edgecolor=FNO_EDGE, label='FNO'),
        mpatches.Patch(facecolor=COLORS['GNN-TAT'], label='GNN-TAT'),
        mpatches.Patch(facecolor='#999999', label='Transformer'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.4, -0.12),
              ncol=4, frameon=False, columnspacing=1.0, handletextpad=0.4)

    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def create_training_dynamics_plot(training_logs: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Create training dynamics comparison showing loss curves.
    """
    logger.info("Creating training dynamics plot...")

    fig, axes = plt.subplots(1, 2, figsize=figsize_for('training_dynamics', 0.40))

    # Select representative models; line style as well as colour tells them apart
    models_to_plot = [
        ('BASIC_ConvLSTM', 'ConvLSTM', COLORS['ConvLSTM'], '-'),
        ('BASIC_ConvLSTM_Bidirectional', 'ConvLSTM Bidirectional', VARIANT_COLORS['ConvLSTM_Bidir'], '--'),
        ('BASIC_ConvLSTM_Residual', 'ConvLSTM Residual', '#555555', ':'),
    ]

    # Training loss (left)
    ax1 = axes[0]
    for key, label, color, ls in models_to_plot:
        if key in training_logs:
            df = training_logs[key]
            if 'loss' in df.columns:
                epochs = range(len(df))
                ax1.plot(epochs, df['loss'], color=color, linestyle=ls, linewidth=1.5, label=label)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training loss')
    ax1.set_title('(a)', loc='left', fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 50)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Validation loss (right)
    ax2 = axes[1]
    for key, label, color, ls in models_to_plot:
        if key in training_logs:
            df = training_logs[key]
            if 'val_loss' in df.columns:
                epochs = range(len(df))
                ax2.plot(epochs, df['val_loss'], color=color, linestyle=ls, linewidth=1.5, label=label)
                logger.info(f"  {label}: final val_loss={df['val_loss'].iloc[-1]:.4f}, "
                            f"min={df['val_loss'].min():.4f} at epoch {int(df['val_loss'].idxmin())}")

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation loss')
    ax2.set_title('(b)', loc='left', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 50)
    ax2.grid(True, alpha=0.3, linestyle='--')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3,
               frameon=False, handlelength=2.6)

    plt.tight_layout()
    return fig


def create_gnn_tat_comparison_plot(combined_df: pd.DataFrame) -> plt.Figure:
    """
    GNN-TAT comparison, three panels in one row.
    (a) best RMSE and (b) best R^2 by GNN operator and feature bundle, over the
    twelve horizons; (c) mean RMSE and best R^2 by feature bundle, over the
    three operators. The former panel (d), a best-configuration table drawn
    inside the image, was dropped: it duplicated a LaTeX table of the thesis.
    """
    logger.info("Creating GNN-TAT comparison plot...")

    gnn_data = combined_df[combined_df['Family'] == 'GNN-TAT'].copy()
    if gnn_data.empty:
        logger.warning("No GNN-TAT data found")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize_for('model_comparison_v4_gnn_tat', 0.36),
                             gridspec_kw={'wspace': 0.42})

    models = ['GNN_TAT_GCN', 'GNN_TAT_GAT', 'GNN_TAT_SAGE']
    feature_sets = ['BASIC', 'KCE', 'PAFC']
    x = np.arange(len(models))
    width = 0.26

    # (a) RMSE by operator and feature bundle
    ax1 = axes[0]
    for i, feat in enumerate(feature_sets):
        vals = []
        for model in models:
            sub = gnn_data[(gnn_data['Model'] == model) & (gnn_data['Experiment'] == feat)]
            vals.append(sub['RMSE'].min() if not sub.empty else np.nan)
        logger.info(f"  (a) RMSE {feat}: " + ', '.join(f'{v:.2f}' for v in vals))
        ax1.bar(x + (i - 1) * width, vals, width, label=feat, color=COLORS.get(feat, 'gray'))
    ax1.set_ylabel('Best RMSE (mm)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['GCN', 'GAT', 'SAGE'])
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_title('(a)', loc='left', fontsize=11, fontweight='bold')

    # (b) R^2 by operator and feature bundle
    ax2 = axes[1]
    for i, feat in enumerate(feature_sets):
        vals = []
        for model in models:
            sub = gnn_data[(gnn_data['Model'] == model) & (gnn_data['Experiment'] == feat)]
            vals.append(sub['R^2'].max() if not sub.empty else np.nan)
        logger.info(f"  (b) R2 {feat}: " + ', '.join(f'{v:.3f}' for v in vals))
        ax2.bar(x + (i - 1) * width, vals, width, label=feat, color=COLORS.get(feat, 'gray'))
    ax2.set_ylabel('Best $R^2$')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['GCN', 'GAT', 'SAGE'])
    ax2.set_ylim(0, 0.7)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title('(b)', loc='left', fontsize=11, fontweight='bold')

    # (c) feature-bundle effect over the three operators
    ax3 = axes[2]
    mean_rmse, best_r2 = [], []
    for feat in feature_sets:
        sub = gnn_data[gnn_data['Experiment'] == feat]
        mean_rmse.append(sub['RMSE'].mean())
        best_r2.append(sub['R^2'].max())
    logger.info("  (c) mean RMSE: " + ', '.join(f'{v:.2f}' for v in mean_rmse))
    logger.info("  (c) best R2:   " + ', '.join(f'{v:.3f}' for v in best_r2))
    xf = np.arange(len(feature_sets))
    w2 = 0.36
    b1 = ax3.bar(xf - w2 / 2, mean_rmse, w2, color='#D55E00', label='Mean RMSE (left)')
    ax3t = ax3.twinx()
    b2 = ax3t.bar(xf + w2 / 2, best_r2, w2, color='#56B4E9', label='Best $R^2$ (right)')
    ax3.set_ylabel('Mean RMSE (mm)')
    ax3t.set_ylabel('Best $R^2$')
    ax3.set_ylim(0, 110)
    ax3t.set_ylim(0, 0.7)
    ax3.set_xticks(xf)
    ax3.set_xticklabels(feature_sets)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3t.grid(False)
    ax3.set_title('(c)', loc='left', fontsize=11, fontweight='bold')

    h_a, l_a = ax1.get_legend_handles_labels()
    fig.legend(h_a + [b1, b2], l_a + ['Mean RMSE (c, left axis)', 'Best $R^2$ (c, right axis)'],
               loc='upper center', bbox_to_anchor=(0.5, 0.07), ncol=5, frameon=False,
               columnspacing=1.2, handletextpad=0.4)

    fig.subplots_adjust(bottom=0.18)
    return fig


def main():
    """Main execution function."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help='Directory for the PNG files (default: %(default)s)')
    parser.add_argument('--only', nargs='*', default=None,
                        help='Generate only these figure names')
    args = parser.parse_args()
    output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("Script 12: Generate Benchmark Figures (All Families)")
    logger.info("=" * 60)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from ALL three families
    convlstm_df = load_convlstm_metrics()
    fno_df = load_fno_metrics()
    gnn_df = load_gnn_tat_metrics()
    training_logs = load_training_logs()

    # Combine all data
    dfs_to_combine = [df for df in [convlstm_df, fno_df, gnn_df] if not df.empty]

    if not dfs_to_combine:
        logger.error("No metrics data found!")
        return

    combined_df = pd.concat(dfs_to_combine, ignore_index=True)
    logger.info(f"Total combined records: {len(combined_df)}")

    builders = {
        'horizon_degradation': lambda: create_horizon_degradation_plot(combined_df),
        'feature_heatmap': lambda: create_feature_heatmap(combined_df),
        'radar_chart': lambda: create_radar_chart(combined_df),
        'parameter_efficiency': lambda: create_parameter_efficiency_plot(combined_df),
        'model_ranking': lambda: create_model_ranking_barplot(combined_df),
        'model_comparison_v4_gnn_tat': lambda: create_gnn_tat_comparison_plot(combined_df),
    }
    if training_logs:
        builders['training_dynamics'] = lambda: create_training_dynamics_plot(training_logs)

    # Generate figures with ALL families
    figures = {name: build() for name, build in builders.items()
               if args.only is None or name in args.only}

    # Save figures
    for name, fig in figures.items():
        output_path = output_dir / f"{name}.png"
        fig.savefig(output_path, dpi=700, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved: {output_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("Generated Figures (All Families: ConvLSTM, FNO, GNN-TAT):")
    for name in figures.keys():
        logger.info(f"  - {name}.png")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
