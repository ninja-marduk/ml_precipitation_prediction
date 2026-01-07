"""
Regenerate V2 benchmark figures with high quality settings matching V4 GNN-TAT.

This script regenerates all comparison figures from V2 Enhanced Models with:
- Consistent DPI (300 for publication quality)
- Consistent colormap (viridis-based)
- Consistent font sizes (12pt minimum)
- Consistent figure sizes
- White backgrounds
- Tight layouts

Usage:
    python scripts/benchmark/regenerate_figures_high_quality.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# QUALITY SETTINGS (matching V4 GNN-TAT figures)
# ============================================================================
FIGURE_DPI = 300  # Publication quality
FIGURE_FORMAT = 'png'
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
LINE_WIDTH = 2.0
MARKER_SIZE = 8

# Color palette (consistent with V4)
COLORS = {
    'BASIC': '#1f77b4',  # Blue
    'KCE': '#ff7f0e',    # Orange
    'PAFC': '#2ca02c',   # Green
    'V2': '#1f77b4',     # Blue
    'V3': '#d62728',     # Red
    'V4': '#2ca02c',     # Green
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
V2_OUTPUT_DIR = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models'
V2_COMP_DIR = V2_OUTPUT_DIR / 'comparisons'
PAPER_FIGURES_DIR = PROJECT_ROOT / 'docs' / 'papers' / '4' / 'figures'
PAPER_H12_DIR = PAPER_FIGURES_DIR / 'h12'

# ============================================================================
# MATPLOTLIB CONFIGURATION
# ============================================================================
def configure_matplotlib():
    """Set consistent matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')

    mpl.rcParams.update({
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LEGEND_SIZE,
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

    logger.info("Matplotlib configured for high-quality output")

# ============================================================================
# DATA LOADING
# ============================================================================
def load_v2_metrics():
    """Load V2 metrics from CSV files."""
    # Try the main metrics file first
    metrics_file = V2_OUTPUT_DIR / 'metrics_spatial_v2_refactored_h12.csv'

    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            # Rename columns to match expected format
            df = df.rename(columns={
                'Experiment': 'experiment',
                'Model': 'model',
                'RMSE': 'rmse',
                'MAE': 'mae',
                'R^2': 'r2',
                'H': 'horizon'
            })
            # Calculate bias if not present
            if 'bias' not in df.columns and 'Mean_True_mm' in df.columns and 'Mean_Pred_mm' in df.columns:
                df['bias'] = df['Mean_Pred_mm'] - df['Mean_True_mm']

            logger.info(f"Loaded {len(df)} V2 metric records from {metrics_file.name}")
            return df
        except Exception as e:
            logger.warning(f"Could not load {metrics_file}: {e}")

    # Fallback to searching
    metrics_files = list(V2_OUTPUT_DIR.glob('**/metrics*.csv'))
    metrics_files = [f for f in metrics_files if 'training' not in f.name.lower()]

    all_data = []
    for f in metrics_files:
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined)} V2 metric records")
        return combined

    logger.warning("No V2 metrics found")
    return pd.DataFrame()

def load_training_logs():
    """Load training logs for evolution plots."""
    log_files = list(V2_OUTPUT_DIR.glob('h12/**/training_metrics/*training_log*.csv'))

    all_logs = {}
    for f in log_files:
        try:
            df = pd.read_csv(f)
            model_name = f.stem.replace('_training_log_h12', '')
            # Get experiment
            for part in f.parts:
                if part in ['BASIC', 'KCE', 'PAFC']:
                    key = f"{part}_{model_name}"
                    all_logs[key] = df
                    break
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")

    logger.info(f"Loaded {len(all_logs)} training logs")
    return all_logs

# ============================================================================
# FIGURE GENERATION
# ============================================================================
def plot_metrics_evolution(training_logs, output_dir, horizon='h12'):
    """Generate metrics evolution plots."""
    logger.info("Generating metrics evolution plots...")

    metrics = ['rmse', 'mae', 'r2', 'totprecip']
    metric_labels = {
        'rmse': 'RMSE (mm)',
        'mae': 'MAE (mm)',
        'r2': 'R²',
        'totprecip': 'Total Precipitation (mm)'
    }

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for exp_model, df in training_logs.items():
            exp = exp_model.split('_')[0]
            color = COLORS.get(exp, '#333333')

            # Find metric column
            metric_col = None
            for col in df.columns:
                if metric.lower() in col.lower():
                    metric_col = col
                    break

            if metric_col and metric_col in df.columns:
                ax.plot(df.index, df[metric_col],
                       color=color, alpha=0.6, linewidth=1.5,
                       label=exp_model if len(training_logs) < 10 else None)

        ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
        ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=LABEL_SIZE)
        ax.set_title(f'{metric.upper()} Evolution ({horizon.upper()})', fontsize=TITLE_SIZE)

        if len(training_logs) < 10:
            ax.legend(loc='best', fontsize=LEGEND_SIZE-2)

        ax.grid(True, alpha=0.3)

        output_path = output_dir / f'metrics_evolution_{horizon}_{metric}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        logger.info(f"  Saved: {output_path.name}")

def plot_normalized_comparison(metrics_df, output_dir, horizon='h12'):
    """Generate normalized metrics comparison plot."""
    logger.info("Generating normalized metrics comparison...")

    if metrics_df.empty:
        logger.warning("No metrics data for normalized comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['rmse', 'mae', 'r2', 'bias']
    titles = ['RMSE (normalized)', 'MAE (normalized)',
              'R² (normalized)', 'Bias (normalized)']

    for ax, metric, title in zip(axes.flat, metrics, titles):
        if metric in metrics_df.columns and 'experiment' in metrics_df.columns:
            for exp in ['BASIC', 'KCE', 'PAFC']:
                exp_data = metrics_df[metrics_df['experiment'] == exp]
                if not exp_data.empty and metric in exp_data.columns:
                    values = exp_data[metric].dropna()
                    if len(values) > 0:
                        # Normalize to 0-1
                        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
                        ax.hist(norm_values, bins=20, alpha=0.5,
                               color=COLORS.get(exp), label=exp)

            ax.set_title(title, fontsize=TITLE_SIZE)
            ax.set_xlabel('Normalized Value', fontsize=LABEL_SIZE)
            ax.set_ylabel('Count', fontsize=LABEL_SIZE)
            ax.legend(fontsize=LEGEND_SIZE)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {metric}',
                   ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    output_path = output_dir / f'normalized_metrics_comparison_{horizon}.png'
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    logger.info(f"  Saved: {output_path.name}")

def plot_rmse_by_model(metrics_df, output_dir, horizon='h12'):
    """Generate RMSE by model bar plots."""
    logger.info("Generating RMSE by model plots...")

    if metrics_df.empty or 'model' not in metrics_df.columns:
        logger.warning("No model data for RMSE plots")
        return

    for exp in ['BASIC', 'KCE', 'PAFC']:
        exp_data = metrics_df[metrics_df.get('experiment', '') == exp]

        if exp_data.empty or 'rmse' not in exp_data.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by model
        model_rmse = exp_data.groupby('model')['rmse'].mean().sort_values()

        bars = ax.barh(range(len(model_rmse)), model_rmse.values,
                      color=COLORS.get(exp, '#1f77b4'), alpha=0.8)

        ax.set_yticks(range(len(model_rmse)))
        ax.set_yticklabels(model_rmse.index, fontsize=TICK_SIZE)
        ax.set_xlabel('RMSE (mm)', fontsize=LABEL_SIZE)
        ax.set_title(f'RMSE by Model - {exp} ({horizon.upper()})', fontsize=TITLE_SIZE)
        ax.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, (idx, val) in enumerate(model_rmse.items()):
            ax.text(val + 1, i, f'{val:.1f}', va='center', fontsize=TICK_SIZE-1)

        plt.tight_layout()

        output_path = output_dir / f'rmse_by_model_{horizon}_{exp.lower()}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        logger.info(f"  Saved: {output_path.name}")

def plot_r2_by_model(metrics_df, output_dir, horizon='h12'):
    """Generate R² by model bar plots."""
    logger.info("Generating R² by model plots...")

    if metrics_df.empty or 'model' not in metrics_df.columns:
        logger.warning("No model data for R² plots")
        return

    for exp in ['BASIC', 'KCE', 'PAFC']:
        exp_data = metrics_df[metrics_df.get('experiment', '') == exp]

        if exp_data.empty or 'r2' not in exp_data.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        model_r2 = exp_data.groupby('model')['r2'].mean().sort_values(ascending=False)

        bars = ax.barh(range(len(model_r2)), model_r2.values,
                      color=COLORS.get(exp, '#1f77b4'), alpha=0.8)

        ax.set_yticks(range(len(model_r2)))
        ax.set_yticklabels(model_r2.index, fontsize=TICK_SIZE)
        ax.set_xlabel('R²', fontsize=LABEL_SIZE)
        ax.set_title(f'R² by Model - {exp} ({horizon.upper()})', fontsize=TITLE_SIZE)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlim(0, 1)

        for i, (idx, val) in enumerate(model_r2.items()):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=TICK_SIZE-1)

        plt.tight_layout()

        output_path = output_dir / f'r2_by_model_{horizon}_{exp.lower()}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        logger.info(f"  Saved: {output_path.name}")

def plot_bias_by_model(metrics_df, output_dir, horizon='h12'):
    """Generate bias by model bar plots."""
    logger.info("Generating bias by model plots...")

    if metrics_df.empty or 'model' not in metrics_df.columns:
        logger.warning("No model data for bias plots")
        return

    for exp in ['BASIC', 'KCE', 'PAFC']:
        exp_data = metrics_df[metrics_df.get('experiment', '') == exp]

        if exp_data.empty or 'bias' not in exp_data.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        model_bias = exp_data.groupby('model')['bias'].mean().sort_values()

        colors = [COLORS.get(exp) if v >= 0 else '#d62728' for v in model_bias.values]

        bars = ax.barh(range(len(model_bias)), model_bias.values,
                      color=colors, alpha=0.8)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_yticks(range(len(model_bias)))
        ax.set_yticklabels(model_bias.index, fontsize=TICK_SIZE)
        ax.set_xlabel('Bias (mm)', fontsize=LABEL_SIZE)
        ax.set_title(f'Bias by Model - {exp} ({horizon.upper()})', fontsize=TITLE_SIZE)
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()

        output_path = output_dir / f'bias_by_model_{horizon}_{exp.lower()}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        logger.info(f"  Saved: {output_path.name}")

def plot_best_val_loss_matrix(training_logs, output_dir, horizon='h12'):
    """Generate validation loss heatmap matrix."""
    logger.info("Generating validation loss matrix...")

    if not training_logs:
        logger.warning("No training logs for loss matrix")
        return

    # Extract best validation loss per model
    data = {}
    for key, df in training_logs.items():
        if 'val_loss' in df.columns:
            best_loss = df['val_loss'].min()
            parts = key.split('_', 1)
            if len(parts) == 2:
                exp, model = parts
                if exp not in data:
                    data[exp] = {}
                data[exp][model] = best_loss

    if not data:
        logger.warning("No validation loss data found")
        return

    # Create matrix
    experiments = sorted(data.keys())
    models = sorted(set(m for exp in data.values() for m in exp.keys()))

    matrix = np.full((len(models), len(experiments)), np.nan)
    for j, exp in enumerate(experiments):
        for i, model in enumerate(models):
            if model in data.get(exp, {}):
                matrix[i, j] = data[exp][model]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='viridis_r', aspect='auto')

    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, fontsize=TICK_SIZE)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=TICK_SIZE-2)

    ax.set_xlabel('Experiment', fontsize=LABEL_SIZE)
    ax.set_ylabel('Model', fontsize=LABEL_SIZE)
    ax.set_title(f'Best Validation Loss ({horizon.upper()})', fontsize=TITLE_SIZE)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Validation Loss', fontsize=LABEL_SIZE)

    plt.tight_layout()

    output_path = output_dir / f'best_val_loss_matrix.png'
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    logger.info(f"  Saved: {output_path.name}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main execution."""
    logger.info("="*60)
    logger.info("Regenerating V2 Figures with High Quality Settings")
    logger.info("="*60)

    # Configure matplotlib
    configure_matplotlib()

    # Create output directories
    PAPER_H12_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    metrics_df = load_v2_metrics()
    training_logs = load_training_logs()

    # Generate all figures
    plot_metrics_evolution(training_logs, PAPER_H12_DIR, 'h12')
    plot_normalized_comparison(metrics_df, PAPER_FIGURES_DIR, 'h12')
    plot_rmse_by_model(metrics_df, PAPER_H12_DIR, 'h12')
    plot_r2_by_model(metrics_df, PAPER_H12_DIR, 'h12')
    plot_bias_by_model(metrics_df, PAPER_H12_DIR, 'h12')
    plot_best_val_loss_matrix(training_logs, PAPER_H12_DIR, 'h12')

    logger.info("="*60)
    logger.info("Figure regeneration complete!")
    logger.info(f"Output directory: {PAPER_FIGURES_DIR}")
    logger.info("="*60)

if __name__ == '__main__':
    main()
