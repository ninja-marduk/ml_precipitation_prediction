"""
Benchmark Analysis Script 11: Generate LaTeX Tables

Generates publication-quality LaTeX tables for Paper 4:
1. Master Model Comparison Table (all models: ConvLSTM, FNO, GNN-TAT)
2. Hyperparameter Summary Table (all families)
3. Statistical Significance Table
4. Per-Horizon Degradation Table

Following WeatherBench 2 and GraphCast visualization standards.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
V2_DIR = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models'
V3_DIR = PROJECT_ROOT / 'models' / 'output' / 'V3_FNO_Models'
V4_DIR = PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'papers' / '4' / 'tables'

# Model parameter counts (from architecture analysis)
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
        logger.info(f"Loaded ConvLSTM metrics: {len(df)} records")
        return df
    else:
        logger.warning(f"ConvLSTM metrics not found at {metrics_path}")
        return pd.DataFrame()


def load_fno_metrics() -> pd.DataFrame:
    """Load FNO metrics from CSV."""
    metrics_path = V3_DIR / 'metrics_spatial_v2_refactored_h12.csv'

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df['Family'] = 'FNO'
        logger.info(f"Loaded FNO metrics: {len(df)} records")
        return df
    else:
        logger.warning(f"FNO metrics not found at {metrics_path}")
        return pd.DataFrame()


def load_gnn_tat_metrics() -> pd.DataFrame:
    """Load GNN-TAT metrics from CSV."""
    metrics_path = V4_DIR / 'metrics_spatial_v4_gnn_tat_h12.csv'

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df['Family'] = 'GNN-TAT'
        logger.info(f"Loaded GNN-TAT metrics: {len(df)} records")
        return df
    else:
        logger.warning(f"GNN-TAT metrics not found at {metrics_path}")
        return pd.DataFrame()


def load_gnn_hyperparams() -> Dict:
    """Load GNN-TAT hyperparameters from experiment state."""
    state_path = V4_DIR / 'experiment_state_v4.json'

    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
        logger.info("Loaded GNN-TAT experiment state")
        return state
    else:
        logger.warning(f"GNN-TAT state not found at {state_path}")
        return {}


def load_fno_hyperparams() -> Dict:
    """Load FNO hyperparameters from JSON files."""
    hp_path = V3_DIR / 'h12' / 'BASIC' / 'training_metrics' / 'FNO_ConvLSTM_Hybrid_hyperparameters.json'

    if hp_path.exists():
        with open(hp_path, 'r') as f:
            hp = json.load(f)
        logger.info("Loaded FNO hyperparameters")
        return hp
    else:
        # Default FNO hyperparameters
        return {
            'modes': 12,
            'width': 32,
            'learning_rate': 0.001,
            'epochs': 200,
            'batch_size': 4,
            'patience': 20
        }


def format_params(params: int) -> str:
    """Format parameter count for display."""
    if params >= 1_000_000:
        return f"{params/1_000_000:.1f}M"
    elif params >= 1_000:
        return f"{params/1_000:.0f}K"
    else:
        return str(params)


def generate_master_comparison_table(convlstm_df: pd.DataFrame, fno_df: pd.DataFrame, gnn_df: pd.DataFrame) -> str:
    """
    Generate master comparison table with ALL model families.

    Format: Family | Model | Params | Features | H=1 R2 | H=6 R2 | H=12 R2 | RMSE | MAE
    """
    logger.info("Generating master comparison table...")

    # Combine all datasets
    dfs = [df for df in [convlstm_df, fno_df, gnn_df] if not df.empty]
    if not dfs:
        return ""
    combined = pd.concat(dfs, ignore_index=True)

    # Get best metrics per model/experiment combination at H=12
    h12_data = combined[combined['H'] == 12].copy()

    # Group and aggregate
    summary = h12_data.groupby(['Family', 'Model', 'Experiment']).agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'R^2': 'mean'
    }).reset_index()

    # Get H=1 and H=6 R2 for degradation analysis
    h1_r2 = combined[combined['H'] == 1].groupby(['Model', 'Experiment'])['R^2'].mean()
    h6_r2 = combined[combined['H'] == 6].groupby(['Model', 'Experiment'])['R^2'].mean()

    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Master Model Comparison: All Architectures at H=12 Forecast Horizon}",
        r"\label{tab:master-comparison}",
        r"\footnotesize",
        r"\begin{tabular}{llllrrrrr}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Model} & \textbf{Params} & \textbf{Features} & \textbf{H=1 R$^2$} & \textbf{H=6 R$^2$} & \textbf{H=12 R$^2$} & \textbf{RMSE} & \textbf{MAE} \\",
        r"\midrule",
    ]

    # ConvLSTM Section
    latex_lines.append(r"\multicolumn{9}{l}{\textit{ConvLSTM Family (Baselines)}} \\")
    convlstm_summary = summary[summary['Family'] == 'ConvLSTM'].sort_values('R^2', ascending=False).head(10)
    for _, row in convlstm_summary.iterrows():
        model = row['Model']
        exp = row['Experiment']
        params = format_params(MODEL_PARAMS.get(model, 0))
        key = (model, exp)
        r2_h1 = h1_r2.get(key, np.nan)
        r2_h6 = h6_r2.get(key, np.nan)
        model_display = model.replace('_', r'\_')
        latex_lines.append(
            f"ConvLSTM & {model_display} & {params} & {exp} & "
            f"{r2_h1:.3f} & {r2_h6:.3f} & {row['R^2']:.3f} & "
            f"{row['RMSE']:.1f} & {row['MAE']:.1f} \\\\"
        )

    latex_lines.append(r"\midrule")

    # FNO Section
    latex_lines.append(r"\multicolumn{9}{l}{\textit{Physics-Informed (FNO)}} \\")
    fno_summary = summary[summary['Family'] == 'FNO'].sort_values('R^2', ascending=False)
    for _, row in fno_summary.iterrows():
        model = row['Model']
        exp = row['Experiment']
        params = format_params(MODEL_PARAMS.get(model, 0))
        key = (model, exp)
        r2_h1 = h1_r2.get(key, np.nan)
        r2_h6 = h6_r2.get(key, np.nan)
        model_display = model.replace('_', r'\_')
        latex_lines.append(
            f"FNO & {model_display} & {params} & {exp} & "
            f"{r2_h1:.3f} & {r2_h6:.3f} & {row['R^2']:.3f} & "
            f"{row['RMSE']:.1f} & {row['MAE']:.1f} \\\\"
        )

    latex_lines.append(r"\midrule")

    # GNN-TAT Section
    latex_lines.append(r"\multicolumn{9}{l}{\textit{Hybrid GNN-TAT}} \\")
    gnn_summary = summary[summary['Family'] == 'GNN-TAT'].sort_values('R^2', ascending=False)
    for _, row in gnn_summary.iterrows():
        model = row['Model']
        exp = row['Experiment']
        params = format_params(MODEL_PARAMS.get(model, 0))
        key = (model, exp)
        r2_h1 = h1_r2.get(key, np.nan)
        r2_h6 = h6_r2.get(key, np.nan)
        model_display = model.replace('_', r'\_')
        latex_lines.append(
            f"GNN-TAT & {model_display} & {params} & {exp} & "
            f"{r2_h1:.3f} & {r2_h6:.3f} & {row['R^2']:.3f} & "
            f"{row['RMSE']:.1f} & {row['MAE']:.1f} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return '\n'.join(latex_lines)


def generate_hyperparameter_table(gnn_state: Dict, fno_hp: Dict) -> str:
    """
    Generate hyperparameter summary table for ALL model families.
    """
    logger.info("Generating hyperparameter table...")

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hyperparameter Configuration for All Model Families}",
        r"\label{tab:hyperparameters}",
        r"\footnotesize",
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Parameter} & \textbf{Value} \\",
        r"\midrule",
    ]

    # ConvLSTM baseline hyperparameters
    latex_lines.append(r"\multicolumn{3}{l}{\textit{ConvLSTM Baselines}} \\")
    latex_lines.append(f"Training & Epochs & 200 \\\\")
    latex_lines.append(f"Training & Batch Size & 4 \\\\")
    latex_lines.append(f"Training & Learning Rate & 0.001 \\\\")
    latex_lines.append(f"Training & Early Stop Patience & 20 \\\\")
    latex_lines.append(f"ConvLSTM & Filters & 32, 16 \\\\")
    latex_lines.append(r"ConvLSTM & Kernel Size & 3$\times$3 \\")

    latex_lines.append(r"\midrule")

    # FNO hyperparameters
    latex_lines.append(r"\multicolumn{3}{l}{\textit{Physics-Informed (FNO)}} \\")
    latex_lines.append(f"FNO & Fourier Modes & {fno_hp.get('modes', 12)} \\\\")
    latex_lines.append(f"FNO & Width & {fno_hp.get('width', 32)} \\\\")
    latex_lines.append(f"Training & Learning Rate & {fno_hp.get('learning_rate', 0.001)} \\\\")
    latex_lines.append(f"Training & Epochs & {fno_hp.get('epochs', 200)} \\\\")
    latex_lines.append(f"Training & Batch Size & {fno_hp.get('batch_size', 4)} \\\\")
    latex_lines.append(f"Training & Early Stop Patience & {fno_hp.get('patience', 20)} \\\\")

    latex_lines.append(r"\midrule")

    # GNN-TAT hyperparameters
    if gnn_state:
        config = gnn_state.get('config', {})
        gnn_config = gnn_state.get('gnn_config', {})

        latex_lines.append(r"\multicolumn{3}{l}{\textit{Hybrid GNN-TAT}} \\")
        latex_lines.append(f"Training & Epochs & {config.get('epochs', 150)} \\\\")
        latex_lines.append(f"Training & Batch Size & {config.get('batch_size', 2)} \\\\")
        latex_lines.append(f"Training & Learning Rate & {config.get('learning_rate', 0.001)} \\\\")
        latex_lines.append(f"Training & Weight Decay & {config.get('weight_decay', 1e-5)} \\\\")
        latex_lines.append(f"Training & Early Stop Patience & {config.get('patience', 50)} \\\\")
        latex_lines.append(f"GNN & Hidden Dimension & {gnn_config.get('hidden_dim', 64)} \\\\")
        latex_lines.append(f"GNN & Number of Layers & {gnn_config.get('num_gnn_layers', 2)} \\\\")
        latex_lines.append(f"GNN & Attention Heads (GAT) & {gnn_config.get('num_heads', 4)} \\\\")
        latex_lines.append(f"GNN & Dropout Rate & {gnn_config.get('dropout', 0.1)} \\\\")
        latex_lines.append(f"Temporal & Attention Heads & {gnn_config.get('num_temporal_heads', 4)} \\\\")
        latex_lines.append(f"LSTM & Hidden Dimension & {gnn_config.get('lstm_hidden_dim', 64)} \\\\")
        latex_lines.append(f"Graph & Edge Threshold & {gnn_config.get('edge_threshold', 0.3)} \\\\")
        latex_lines.append(f"Graph & Max Neighbors & {gnn_config.get('max_neighbors', 8)} \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return '\n'.join(latex_lines)


def generate_horizon_degradation_table(convlstm_df: pd.DataFrame, fno_df: pd.DataFrame, gnn_df: pd.DataFrame) -> str:
    """
    Generate per-horizon degradation analysis table for ALL families.
    Shows R2 at H=1,3,6,9,12 with degradation rate.
    """
    logger.info("Generating horizon degradation table...")

    dfs = [df for df in [convlstm_df, fno_df, gnn_df] if not df.empty]
    if not dfs:
        return ""
    combined = pd.concat(dfs, ignore_index=True)

    # Select best models from EACH family for degradation analysis
    best_models = [
        ('ConvLSTM', 'BASIC', 'ConvLSTM'),
        ('ConvLSTM_Bidirectional', 'BASIC', 'ConvLSTM'),
        ('FNO_ConvLSTM_Hybrid', 'BASIC', 'FNO'),
        ('FNO_Pure', 'BASIC', 'FNO'),
        ('GNN_TAT_GAT', 'BASIC', 'GNN-TAT'),
        ('GNN_TAT_GCN', 'PAFC', 'GNN-TAT'),
    ]

    horizons = [1, 3, 6, 9, 12]

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Forecast Horizon Degradation Analysis: R$^2$ Performance from H=1 to H=12}",
        r"\label{tab:horizon-degradation}",
        r"\footnotesize",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Model} & \textbf{Features} & \textbf{H=1} & \textbf{H=3} & \textbf{H=6} & \textbf{H=9} & \textbf{H=12} & \textbf{Degradation} \\",
        r"\midrule",
    ]

    for model, exp, family in best_models:
        model_data = combined[(combined['Model'] == model) &
                              (combined['Experiment'] == exp) &
                              (combined['Family'] == family)]

        if model_data.empty:
            continue

        r2_values = []
        for h in horizons:
            h_data = model_data[model_data['H'] == h]
            if not h_data.empty:
                r2_values.append(h_data['R^2'].mean())
            else:
                r2_values.append(np.nan)

        # Calculate degradation (H=1 to H=12)
        if not np.isnan(r2_values[0]) and not np.isnan(r2_values[-1]):
            degradation = ((r2_values[-1] - r2_values[0]) / abs(r2_values[0])) * 100
        else:
            degradation = np.nan

        model_display = model.replace('_', r'\_')
        r2_str = ' & '.join([f"{r2:.3f}" if not np.isnan(r2) else "---" for r2 in r2_values])
        deg_str = f"{degradation:+.1f}\\%" if not np.isnan(degradation) else "---"

        latex_lines.append(f"{family} & {model_display} & {exp} & {r2_str} & {deg_str} \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\multicolumn{9}{l}{\footnotesize Note: Degradation = (R$^2_{\mathrm{H=12}}$ - R$^2_{\mathrm{H=1}}$) / $|$R$^2_{\mathrm{H=1}}$$|$ $\times$ 100\%} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return '\n'.join(latex_lines)


def generate_statistical_tests_table(convlstm_df: pd.DataFrame, fno_df: pd.DataFrame, gnn_df: pd.DataFrame) -> str:
    """
    Generate statistical significance tests table comparing ALL families.
    """
    logger.info("Generating statistical tests table...")

    from scipy import stats

    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical Significance Tests: Pairwise Family Comparisons}",
        r"\label{tab:statistical-tests}",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Test} & \textbf{Statistic} & \textbf{p-value} & \textbf{Effect (d)} & \textbf{Significant?} \\",
        r"\midrule",
    ]

    # Get H=12 RMSE values for each family
    h12_convlstm = convlstm_df[convlstm_df['H'] == 12]['RMSE'].values if not convlstm_df.empty else np.array([])
    h12_fno = fno_df[fno_df['H'] == 12]['RMSE'].values if not fno_df.empty else np.array([])
    h12_gnn = gnn_df[gnn_df['H'] == 12]['RMSE'].values if not gnn_df.empty else np.array([])

    comparisons = [
        ('ConvLSTM vs GNN-TAT', h12_convlstm, h12_gnn),
        ('ConvLSTM vs FNO', h12_convlstm, h12_fno),
        ('GNN-TAT vs FNO', h12_gnn, h12_fno),
    ]

    for name, group1, group2 in comparisons:
        if len(group1) > 0 and len(group2) > 0:
            stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            d = cohens_d(pd.Series(group1), pd.Series(group2))
            sig = "Yes" if p < 0.05 else "No"
            latex_lines.append(
                f"{name} & Mann-Whitney U & {stat:.1f} & {p:.4f} & {abs(d):.2f} & {sig} \\\\"
            )

    latex_lines.extend([
        r"\bottomrule",
        r"\multicolumn{6}{l}{\footnotesize Effect size: $|d|<0.2$ negligible, $<0.5$ small, $<0.8$ medium, $\geq0.8$ large} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return '\n'.join(latex_lines)


def generate_feature_comparison_table(convlstm_df: pd.DataFrame, fno_df: pd.DataFrame, gnn_df: pd.DataFrame) -> str:
    """
    Generate feature set comparison table for ALL families.
    Shows best R2 per feature set for each model family.
    """
    logger.info("Generating feature comparison table...")

    dfs = [df for df in [convlstm_df, fno_df, gnn_df] if not df.empty]
    if not dfs:
        return ""
    combined = pd.concat(dfs, ignore_index=True)
    h12_data = combined[combined['H'] == 12]

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Feature Set Comparison: Best R$^2$ at H=12 by Feature Engineering Strategy}",
        r"\label{tab:feature-comparison}",
        r"\footnotesize",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"\textbf{Features} & \textbf{Best Model} & \textbf{Family} & \textbf{R$^2$} & \textbf{RMSE} & \textbf{MAE} \\",
        r"\midrule",
    ]

    for exp in ['BASIC', 'KCE', 'PAFC']:
        exp_data = h12_data[h12_data['Experiment'] == exp]
        if exp_data.empty:
            continue

        # Find best model for this feature set
        best_idx = exp_data['R^2'].idxmax()
        best = exp_data.loc[best_idx]

        model_display = best['Model'].replace('_', r'\_')

        latex_lines.append(
            f"{exp} & {model_display} & {best['Family']} & "
            f"{best['R^2']:.3f} & {best['RMSE']:.1f} & {best['MAE']:.1f} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\multicolumn{6}{l}{\footnotesize BASIC: Temporal + Topographic; KCE: + Elevation classes; PAFC: + Precipitation lags} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return '\n'.join(latex_lines)


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Script 11: Generate LaTeX Tables (All Families: ConvLSTM, FNO, GNN-TAT)")
    logger.info("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for ALL families
    convlstm_df = load_convlstm_metrics()
    fno_df = load_fno_metrics()
    gnn_df = load_gnn_tat_metrics()
    gnn_state = load_gnn_hyperparams()
    fno_hp = load_fno_hyperparams()

    # Generate tables
    tables = {}

    if not convlstm_df.empty or not fno_df.empty or not gnn_df.empty:
        tables['master_comparison'] = generate_master_comparison_table(convlstm_df, fno_df, gnn_df)
        tables['horizon_degradation'] = generate_horizon_degradation_table(convlstm_df, fno_df, gnn_df)
        tables['feature_comparison'] = generate_feature_comparison_table(convlstm_df, fno_df, gnn_df)
        tables['statistical_tests'] = generate_statistical_tests_table(convlstm_df, fno_df, gnn_df)

    tables['hyperparameters'] = generate_hyperparameter_table(gnn_state, fno_hp)

    # Save tables
    for name, content in tables.items():
        if content:
            output_path = OUTPUT_DIR / f"table_{name}.tex"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved: {output_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("Generated LaTeX Tables:")
    for name in tables.keys():
        if tables[name]:
            logger.info(f"  - table_{name}.tex")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
