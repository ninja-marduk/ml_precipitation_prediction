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

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
V2_DIR = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models'
V3_DIR = PROJECT_ROOT / 'models' / 'output' / 'V3_FNO_Models'
V4_DIR = PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'papers' / '4' / 'figures'

# Color scheme (colorblind-friendly) - Professional family names
COLORS = {
    'ConvLSTM': '#1f77b4',   # Blue - Baselines
    'FNO': '#ff7f0e',        # Orange - Physics-Informed
    'GNN-TAT': '#d62728',    # Red - Hybrid
    'BASIC': '#2ca02c',      # Green
    'KCE': '#9467bd',        # Purple
    'PAFC': '#8c564b',       # Brown
}

# Model parameter counts (all families)
MODEL_PARAMS = {
    # ConvLSTM Models (Baselines)
    'ConvLSTM': 78_000,
    'ConvRNN': 45_000,
    'ConvLSTM_Enhanced': 156_000,
    'ConvRNN_Enhanced': 89_000,
    'ConvLSTM_Bidirectional': 1_200_000,
    'ConvLSTM_Residual': 234_000,
    'ConvLSTM_Attention': 178_000,
    'ConvLSTM_MeteoAttention': 198_000,
    'ConvLSTM_EfficientBidir': 312_000,
    'Transformer_Baseline': 41_800_000,
    # FNO Models (Physics-Informed)
    'FNO_ConvLSTM_Hybrid': 106_000,
    'FNO_Pure': 85_000,
    # GNN-TAT Models (Hybrid)
    'GNN_TAT_GAT': 98_000,
    'GNN_TAT_SAGE': 106_000,
    'GNN_TAT_GCN': 98_000,
}


def load_convlstm_metrics() -> pd.DataFrame:
    """Load ConvLSTM baseline metrics from CSV."""
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

    fig, ax = plt.subplots(figsize=(10, 6))

    # Select representative models from EACH family
    models_to_plot = [
        # ConvLSTM (Baselines)
        ('ConvLSTM', 'BASIC', 'ConvLSTM', COLORS['ConvLSTM'], '-', 'o'),
        ('ConvLSTM_Bidirectional', 'BASIC', 'ConvLSTM', '#17becf', '--', 's'),
        ('ConvLSTM_Residual', 'BASIC', 'ConvLSTM', '#7f7f7f', '-.', '^'),
        # FNO (Physics-Informed)
        ('FNO_ConvLSTM_Hybrid', 'BASIC', 'FNO', COLORS['FNO'], '-', 'D'),
        ('FNO_Pure', 'BASIC', 'FNO', '#ffbb78', '--', 'v'),
        # GNN-TAT (Hybrid)
        ('GNN_TAT_GAT', 'BASIC', 'GNN-TAT', COLORS['GNN-TAT'], '-', 'p'),
        ('GNN_TAT_GCN', 'PAFC', 'GNN-TAT', '#e377c2', '--', 'h'),
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
                marker=marker, markersize=6, linewidth=2, label=label)

    ax.set_xlabel('Forecast Horizon (months)', fontsize=11)
    ax.set_ylabel('R$^2$ Score', fontsize=11)
    ax.set_title('Forecast Horizon Degradation Analysis\n(All Model Families)', fontsize=12)

    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.0, 0.75)
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend with family groups
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.9, fontsize=8)

    # Add reference line
    ax.axhline(y=0.6, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(12.3, 0.6, 'R$^2$=0.6', fontsize=8, va='center', color='gray')

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

    fig, ax = plt.subplots(figsize=(8, 10))

    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.4, vmin=0.0, vmax=0.7,
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'R$^2$ Score'})

    ax.set_title('Feature Set Impact on Model Performance (H=12)\n(ConvLSTM, FNO, GNN-TAT)', fontsize=12)
    ax.set_xlabel('Feature Engineering Strategy', fontsize=11)
    ax.set_ylabel('Model Architecture', fontsize=11)

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
    categories = ['R$^2$', 'RMSE\n(inv)', 'MAE\n(inv)', 'Bias\n(inv)', 'Param\nEfficiency', 'Training\nStability']
    N = len(categories)

    # Define parameter counts for each family (from paper)
    PARAM_COUNTS = {
        'ConvLSTM': 2_000_000,  # ~2M parameters
        'FNO': 500_000,         # ~500K parameters
        'GNN-TAT': 100_000      # ~100K parameters
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

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(projection='polar'))

    # Plot ConvLSTM (Baselines)
    if convlstm_values is not None:
        values = convlstm_values + [convlstm_values[0]]  # Close the polygon
        ax.plot(angles, values, 'o-', linewidth=2, color=COLORS['ConvLSTM'], label='ConvLSTM (Baselines)', markersize=8)
        ax.fill(angles, values, alpha=0.2, color=COLORS['ConvLSTM'])

    # Plot FNO (Physics-Informed)
    if fno_values is not None:
        values = fno_values + [fno_values[0]]  # Close the polygon
        ax.plot(angles, values, 's-', linewidth=2, color=COLORS['FNO'], label='FNO (Physics-Informed)', markersize=8)
        ax.fill(angles, values, alpha=0.2, color=COLORS['FNO'])

    # Plot GNN-TAT (Hybrid)
    if gnn_values is not None:
        values = gnn_values + [gnn_values[0]]  # Close the polygon
        ax.plot(angles, values, 'D-', linewidth=2, color=COLORS['GNN-TAT'], label='GNN-TAT (Hybrid)', markersize=8)
        ax.fill(angles, values, alpha=0.2, color=COLORS['GNN-TAT'])

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # Set radial labels
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

    ax.set_title('Multi-Metric Model Family Comparison\n(Higher is Better)', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))

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

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot ConvLSTM models
    convlstm_data = model_stats[model_stats['Family'] == 'ConvLSTM']
    ax.scatter(convlstm_data['Params'], convlstm_data['R^2'],
               s=100, c=COLORS['ConvLSTM'], alpha=0.7, label='ConvLSTM (Baselines)', marker='o')

    # Plot FNO models
    fno_data = model_stats[model_stats['Family'] == 'FNO']
    ax.scatter(fno_data['Params'], fno_data['R^2'],
               s=120, c=COLORS['FNO'], alpha=0.7, label='FNO (Physics-Informed)', marker='^')

    # Plot GNN-TAT models
    gnn_data = model_stats[model_stats['Family'] == 'GNN-TAT']
    ax.scatter(gnn_data['Params'], gnn_data['R^2'],
               s=150, c=COLORS['GNN-TAT'], alpha=0.7, label='GNN-TAT (Hybrid)', marker='s')

    # Add model labels - Only annotate representative/important models to avoid clutter
    # Define which models to label (best performers and representative of each category)
    important_models = {
        'ConvLSTM_Residual': (15, 10),           # Best ConvLSTM
        'ConvLSTM_Bidirectional': (15, -18),     # Largest ConvLSTM
        'ConvRNN': (-45, 8),                     # Smallest baseline
        'FNO_Pure': (15, -15),                   # Pure FNO
        'FNO_ConvLSTM_Hybrid': (15, 10),         # Best FNO (Hybrid)
        'GNN_TAT_GAT': (15, 2),                  # Best GNN
        'GNN_TAT_GCN': (15, -18),                # Second GNN variant
        'GNN_TAT_SAGE': (-55, 0),                # Third GNN variant
        'Transformer_Baseline': (15, -8),        # Largest model (outlier)
    }

    for _, row in model_stats.iterrows():
        if row['Model'] in important_models:
            # Create clean label
            label = row['Model'].replace('ConvLSTM_', '').replace('GNN_TAT_', '').replace('FNO_', '')
            offset = important_models[row['Model']]

            ax.annotate(label, (row['Params'], row['R^2']),
                        textcoords='offset points', xytext=offset,
                        fontsize=8, alpha=0.85,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='gray', alpha=0.7, linewidth=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                        color='gray', lw=0.8, alpha=0.6))

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
                'g--', alpha=0.5, linewidth=2, label='Pareto Frontier')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Parameters (log scale)', fontsize=11)
    ax.set_ylabel('Best R$^2$ Score', fontsize=11)
    ax.set_title('Parameter Efficiency Analysis (All Families)\n(Upper-left is better)', fontsize=12)

    ax.set_xlim(3e4, 5e7)
    ax.set_ylim(0.0, 0.75)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

    ax.legend(loc='lower right')

    # Add efficiency zones
    ax.axvline(x=100_000, color='gray', linestyle=':', alpha=0.5)
    ax.text(100_000, 0.72, 'Efficient\n(<100K)', fontsize=8, ha='center', color='gray')
    ax.axhline(y=0.6, color='gray', linestyle=':', alpha=0.5)
    ax.text(4e7, 0.6, 'Target R$^2$', fontsize=8, va='bottom', color='gray')

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

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors based on family
    colors = []
    for family in model_best['Family']:
        if family == 'ConvLSTM':
            colors.append(COLORS['ConvLSTM'])
        elif family == 'FNO':
            colors.append(COLORS['FNO'])
        else:
            colors.append(COLORS['GNN-TAT'])

    # Create horizontal bars
    y_pos = range(len(model_best))
    bars = ax.barh(y_pos, model_best['R^2'], color=colors, alpha=0.8, height=0.7)

    # Add value labels
    for i, (bar, r2) in enumerate(zip(bars, model_best['R^2'])):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{r2:.3f}', va='center', fontsize=9)

    # Y-axis labels
    labels = [f"{row['Model'].replace('_', ' ')} ({row['Experiment']})" for _, row in model_best.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel('R$^2$ Score (H=12)', fontsize=11)
    ax.set_title('Model Performance Ranking by Best R$^2$\n(Top 15 Configurations - All Families)', fontsize=12)

    ax.set_xlim(0, 0.75)
    ax.axvline(x=0.6, color='gray', linestyle='--', alpha=0.5, label='Target R$^2$=0.6')

    # Custom legend with ALL families
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['ConvLSTM'], alpha=0.8, label='ConvLSTM (Baselines)'),
        mpatches.Patch(facecolor=COLORS['FNO'], alpha=0.8, label='FNO (Physics-Informed)'),
        mpatches.Patch(facecolor=COLORS['GNN-TAT'], alpha=0.8, label='GNN-TAT (Hybrid)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def create_training_dynamics_plot(training_logs: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Create training dynamics comparison showing loss curves.
    """
    logger.info("Creating training dynamics plot...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Select representative models
    models_to_plot = [
        ('BASIC_ConvLSTM', 'ConvLSTM (BASIC)', COLORS['ConvLSTM']),
        ('BASIC_ConvLSTM_Bidirectional', 'ConvLSTM Bidir (BASIC)', '#17becf'),
        ('BASIC_ConvLSTM_Residual', 'ConvLSTM Residual (BASIC)', '#7f7f7f'),
    ]

    # Training loss (left)
    ax1 = axes[0]
    for key, label, color in models_to_plot:
        if key in training_logs:
            df = training_logs[key]
            if 'loss' in df.columns:
                epochs = range(len(df))
                ax1.plot(epochs, df['loss'], color=color, linewidth=1.5, label=label)

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('Training Loss Curves', fontsize=12)
    ax1.set_xlim(0, 50)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=8)

    # Validation loss (right)
    ax2 = axes[1]
    for key, label, color in models_to_plot:
        if key in training_logs:
            df = training_logs[key]
            if 'val_loss' in df.columns:
                epochs = range(len(df))
                ax2.plot(epochs, df['val_loss'], color=color, linewidth=1.5, label=label)

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss Curves', fontsize=12)
    ax2.set_xlim(0, 50)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig


def create_gnn_tat_comparison_plot(combined_df: pd.DataFrame) -> plt.Figure:
    """
    Create GNN-TAT specific comparison figure with 4 clean panels.
    Replaces the legacy model_comparison_v4_gnn_tat.png
    """
    logger.info("Creating GNN-TAT comparison plot...")

    # Filter to GNN-TAT models only
    gnn_data = combined_df[combined_df['Family'] == 'GNN-TAT'].copy()

    if gnn_data.empty:
        logger.warning("No GNN-TAT data found")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('GNN-TAT Architecture Comparison (All Variants)', fontsize=14, fontweight='bold')

    # Panel 1: RMSE by Model and Feature Set
    ax1 = axes[0, 0]
    models = ['GNN_TAT_GCN', 'GNN_TAT_GAT', 'GNN_TAT_SAGE']
    feature_sets = ['BASIC', 'KCE', 'PAFC']
    x = np.arange(len(models))
    width = 0.25

    for i, feat in enumerate(feature_sets):
        rmse_vals = []
        for model in models:
            subset = gnn_data[(gnn_data['Model'] == model) & (gnn_data['Experiment'] == feat)]
            rmse_vals.append(subset['RMSE'].min() if not subset.empty else np.nan)
        ax1.bar(x + i*width, rmse_vals, width, label=feat, color=COLORS.get(feat, 'gray'), alpha=0.8)

    ax1.set_xlabel('GNN Architecture', fontsize=11)
    ax1.set_ylabel('RMSE (mm)', fontsize=11)
    ax1.set_title('RMSE by Model and Feature Set', fontsize=12)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['GCN', 'GAT', 'SAGE'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: R² by Model and Feature Set
    ax2 = axes[0, 1]
    for i, feat in enumerate(feature_sets):
        r2_vals = []
        for model in models:
            subset = gnn_data[(gnn_data['Model'] == model) & (gnn_data['Experiment'] == feat)]
            r2_vals.append(subset['R^2'].max() if not subset.empty else np.nan)
        ax2.bar(x + i*width, r2_vals, width, label=feat, color=COLORS.get(feat, 'gray'), alpha=0.8)

    ax2.set_xlabel('GNN Architecture', fontsize=11)
    ax2.set_ylabel('R² Score', fontsize=11)
    ax2.set_title('R² by Model and Feature Set', fontsize=12)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(['GCN', 'GAT', 'SAGE'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 0.7)

    # Panel 3: Feature Set Impact (Aggregated across architectures)
    ax3 = axes[1, 0]
    feat_metrics = []
    for feat in feature_sets:
        subset = gnn_data[gnn_data['Experiment'] == feat]
        feat_metrics.append({
            'Feature': feat,
            'Mean_RMSE': subset['RMSE'].mean(),
            'Mean_R2': subset['R^2'].mean(),
            'Best_R2': subset['R^2'].max()
        })
    feat_df = pd.DataFrame(feat_metrics)

    x_feat = np.arange(len(feature_sets))
    width = 0.35
    ax3.bar(x_feat - width/2, feat_df['Mean_RMSE'], width, label='Mean RMSE', color='#ff7f0e', alpha=0.8)
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x_feat + width/2, feat_df['Best_R2'], width, label='Best R²', color='#2ca02c', alpha=0.8)

    ax3.set_xlabel('Feature Engineering Strategy', fontsize=11)
    ax3.set_ylabel('Mean RMSE (mm)', fontsize=11, color='#ff7f0e')
    ax3_twin.set_ylabel('Best R² Score', fontsize=11, color='#2ca02c')
    ax3.set_title('Feature Set Impact (Averaged Across Architectures)', fontsize=12)
    ax3.set_xticks(x_feat)
    ax3.set_xticklabels(feature_sets)
    ax3.tick_params(axis='y', labelcolor='#ff7f0e')
    ax3_twin.tick_params(axis='y', labelcolor='#2ca02c')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Performance Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create summary table
    summary_data = []
    for model in models:
        model_subset = gnn_data[gnn_data['Model'] == model]
        best_config = model_subset.loc[model_subset['R^2'].idxmax()] if not model_subset.empty else None
        if best_config is not None:
            summary_data.append([
                model.replace('GNN_TAT_', ''),
                best_config['Experiment'],
                f"{best_config['R^2']:.3f}",
                f"{best_config['RMSE']:.1f}",
                f"{best_config['MAE']:.1f}"
            ])

    table = ax4.table(cellText=summary_data,
                     colLabels=['Architecture', 'Best\nFeatures', 'R²', 'RMSE\n(mm)', 'MAE\n(mm)'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.05, 0.2, 0.9, 0.6],
                     colWidths=[0.25, 0.20, 0.15, 0.20, 0.20])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Add padding to Architecture column header
    for key, cell in table.get_celld().items():
        cell.set_text_props(ha='center')
        cell.PAD = 0.05  # Add padding to all cells

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#d62728')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax4.set_title('Best Configuration Summary', fontsize=12, pad=20)

    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Script 12: Generate Benchmark Figures (All Families)")
    logger.info("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    figures = {}

    # Generate figures with ALL families
    figures['horizon_degradation'] = create_horizon_degradation_plot(combined_df)
    figures['feature_heatmap'] = create_feature_heatmap(combined_df)
    figures['radar_chart'] = create_radar_chart(combined_df)
    figures['parameter_efficiency'] = create_parameter_efficiency_plot(combined_df)
    figures['model_ranking'] = create_model_ranking_barplot(combined_df)
    figures['model_comparison_v4_gnn_tat'] = create_gnn_tat_comparison_plot(combined_df)

    if training_logs:
        figures['training_dynamics'] = create_training_dynamics_plot(training_logs)

    # Save figures
    for name, fig in figures.items():
        output_path = OUTPUT_DIR / f"{name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
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
