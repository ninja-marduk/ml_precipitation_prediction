"""
Benchmark Analysis Script 11: Generate LaTeX Tables

Part of V2 vs V3 Comparative Analysis Pipeline
Creates publication-ready LaTeX tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'docs' / 'models' / 'comparative' / 'data'
TABLES_DIR = PROJECT_ROOT / 'docs' / 'models' / 'comparative' / 'tables'

# Ensure directories exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all necessary data files."""
    logger.info("Loading data files...")

    data = {
        'aggregate': pd.read_csv(DATA_DIR / 'aggregate_statistics_h12.csv'),
        'per_horizon': pd.read_csv(DATA_DIR / 'per_horizon_comparison_h12.csv'),
        'convergence': pd.read_csv(DATA_DIR / 'convergence_summary.csv'),
        'stats': pd.read_csv(DATA_DIR / 'statistical_tests_summary.csv')
    }

    logger.info("  All data files loaded successfully")
    return data


def create_table1_overall_summary(data: dict) -> str:
    """Create Table 1: Overall Performance Summary."""
    logger.info("Creating Table 1: Overall Performance Summary...")

    agg_df = data['aggregate']

    latex = r"""\begin{table}[htbp]
\centering
\caption{Overall Performance Comparison: V2 Enhanced Models vs V3 FNO Models (H=12)}
\label{tab:overall_summary}
\begin{tabular}{llrrrr}
\toprule
\textbf{Experiment} & \textbf{Family} & \textbf{RMSE (mm)} & \textbf{MAE (mm)} & \textbf{R\textsuperscript{2}} & \textbf{Bias (mm)} \\
\midrule
"""

    # Process each experiment
    for exp in ['BASIC', 'KCE', 'PAFC']:
        v2_row = agg_df[(agg_df['Experiment'] == exp) & (agg_df['Model_Family'] == 'V2_Enhanced')]
        v3_row = agg_df[(agg_df['Experiment'] == exp) & (agg_df['Model_Family'] == 'V3_FNO')]

        if not v2_row.empty:
            rmse_v2 = f"${v2_row['RMSE_mean'].values[0]:.2f} \\pm {v2_row['RMSE_std'].values[0]:.2f}$"
            mae_v2 = f"${v2_row['MAE_mean'].values[0]:.2f} \\pm {v2_row['MAE_std'].values[0]:.2f}$"
            r2_v2 = f"${v2_row['R^2_mean'].values[0]:.3f} \\pm {v2_row['R^2_std'].values[0]:.3f}$"
            bias_v2 = f"${v2_row['bias_mean'].values[0]:.2f}$"

            latex += f"{exp} & V2 Enhanced & {rmse_v2} & {mae_v2} & {r2_v2} & {bias_v2} \\\\\n"

        if not v3_row.empty:
            rmse_v3 = f"${v3_row['RMSE_mean'].values[0]:.2f} \\pm {v3_row['RMSE_std'].values[0]:.2f}$"
            mae_v3 = f"${v3_row['MAE_mean'].values[0]:.2f} \\pm {v3_row['MAE_std'].values[0]:.2f}$"
            r2_v3 = f"${v3_row['R^2_mean'].values[0]:.3f} \\pm {v3_row['R^2_std'].values[0]:.3f}$"
            bias_v3 = f"${v3_row['bias_mean'].values[0]:.2f}$"

            latex += f"      & V3 FNO      & {rmse_v3} & {mae_v3} & {r2_v3} & {bias_v3} \\\\\n"

        latex += "\\midrule\n"

    latex = latex.rstrip("\\midrule\n") + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Values shown as mean $\pm$ standard deviation across all models and horizons
\end{tablenotes}
\end{table}
"""

    return latex


def create_table2_best_per_horizon(data: dict) -> str:
    """Create Table 2: Best Models Per Horizon."""
    logger.info("Creating Table 2: Best Models Per Horizon...")

    per_h_df = data['per_horizon']

    latex = r"""\begin{table}[htbp]
\centering
\caption{Best Model Performance by Horizon: V2 vs V3 Comparison}
\label{tab:best_per_horizon}
\begin{tabular}{clrrrr}
\toprule
\textbf{Exp} & \textbf{H} & \textbf{RMSE V2} & \textbf{RMSE V3} & \textbf{R\textsuperscript{2} V2} & \textbf{R\textsuperscript{2} V3} \\
\midrule
"""

    for exp in ['BASIC', 'KCE', 'PAFC']:
        exp_df = per_h_df[per_h_df['Experiment'] == exp]

        for h in range(1, 13):
            h_row = exp_df[exp_df['H'] == h]

            if not h_row.empty:
                rmse_v2 = f"${h_row['RMSE_V2'].values[0]:.2f}$"
                rmse_v3 = f"${h_row['RMSE_V3'].values[0]:.2f}$"
                r2_v2 = f"${h_row['R2_V2'].values[0]:.3f}$"
                r2_v3 = f"${h_row['R2_V3'].values[0]:.3f}$"

                latex += f"{exp} & {h} & {rmse_v2} & {rmse_v3} & {r2_v2} & {r2_v3} \\\\\n"

        latex += "\\midrule\n"

    latex = latex.rstrip("\\midrule\n") + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def create_table3_training_efficiency(data: dict) -> str:
    """Create Table 3: Training Efficiency Summary."""
    logger.info("Creating Table 3: Training Efficiency Summary...")

    conv_df = data['convergence']

    latex = r"""\begin{table}[htbp]
\centering
\caption{Training Efficiency Comparison: V2 vs V3 Models}
\label{tab:training_efficiency}
\begin{tabular}{llrrr}
\toprule
\textbf{Exp} & \textbf{Version} & \textbf{Epochs to Best} & \textbf{Best Val Loss} & \textbf{Stability} \\
\midrule
"""

    for exp in ['BASIC', 'KCE', 'PAFC']:
        v2_row = conv_df[(conv_df['version'] == 'V2') & (conv_df['experiment'] == exp)]
        v3_row = conv_df[(conv_df['version'] == 'V3') & (conv_df['experiment'] == exp)]

        if not v2_row.empty:
            epochs = f"${v2_row['epochs_to_best_mean'].values[0]:.1f} \\pm {v2_row['epochs_to_best_std'].values[0]:.1f}$"
            val_loss = f"${v2_row['best_val_loss_mean'].values[0]:.3f}$"
            stability = f"${v2_row['training_stability_mean'].values[0]:.3f}$"

            latex += f"{exp} & V2 & {epochs} & {val_loss} & {stability} \\\\\n"

        if not v3_row.empty:
            epochs = f"${v3_row['epochs_to_best_mean'].values[0]:.1f} \\pm {v3_row['epochs_to_best_std'].values[0]:.1f}$"
            val_loss = f"${v3_row['best_val_loss_mean'].values[0]:.3f}$"
            stability = f"${v3_row['training_stability_mean'].values[0]:.3f}$"

            latex += f"      & V3 & {epochs} & {val_loss} & {stability} \\\\\n"

        latex += "\\midrule\n"

    latex = latex.rstrip("\\midrule\n") + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Stability measured as standard deviation of validation loss in final 10 epochs
\end{tablenotes}
\end{table}
"""

    return latex


def save_tables(tables: dict):
    """Save all LaTeX tables."""
    logger.info("Saving LaTeX tables...")

    for name, content in tables.items():
        table_path = TABLES_DIR / f"{name}.tex"
        with open(table_path, 'w') as f:
            f.write(content)
        logger.info(f"  Saved: {table_path}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Script 11: Generate LaTeX Tables")
    logger.info("="*60)

    # Load data
    data = load_data()

    # Generate tables
    tables = {
        'table1_overall_summary': create_table1_overall_summary(data),
        'table2_best_per_horizon': create_table2_best_per_horizon(data),
        'table3_training_efficiency': create_table3_training_efficiency(data)
    }

    # Save tables
    save_tables(tables)

    logger.info("="*60)
    logger.info("Completed successfully")
    logger.info(f"Generated {len(tables)} LaTeX tables")
    logger.info("="*60)


if __name__ == '__main__':
    main()
