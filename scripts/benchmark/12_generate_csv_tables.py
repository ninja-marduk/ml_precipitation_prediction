"""
Benchmark Analysis Script 12: Generate CSV Tables

Part of V2 vs V3 Comparative Analysis Pipeline
Creates comprehensive CSV summary tables
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
        'unified': pd.read_csv(DATA_DIR / 'unified_metrics_h12.csv'),
        'aggregate': pd.read_csv(DATA_DIR / 'aggregate_statistics_h12.csv'),
        'per_horizon': pd.read_csv(DATA_DIR / 'per_horizon_comparison_h12.csv'),
        'delta': pd.read_csv(DATA_DIR / 'delta_metrics_h12.csv')
    }

    logger.info("  All data files loaded successfully")
    return data


def create_summary_statistics(data: dict) -> pd.DataFrame:
    """Create comprehensive summary statistics table."""
    logger.info("Creating summary statistics table...")

    agg_df = data['aggregate']
    delta_df = data['delta']

    # Merge aggregate with delta
    summary = []

    for exp in ['BASIC', 'KCE', 'PAFC']:
        v2_row = agg_df[(agg_df['Experiment'] == exp) & (agg_df['Model_Family'] == 'V2_Enhanced')]
        v3_row = agg_df[(agg_df['Experiment'] == exp) & (agg_df['Model_Family'] == 'V3_FNO')]
        delta_row = delta_df[delta_df['Experiment'] == exp]

        if not v2_row.empty and not v3_row.empty:
            summary.append({
                'Experiment': exp,
                'V2_RMSE_mean': v2_row['RMSE_mean'].values[0],
                'V2_RMSE_std': v2_row['RMSE_std'].values[0],
                'V3_RMSE_mean': v3_row['RMSE_mean'].values[0],
                'V3_RMSE_std': v3_row['RMSE_std'].values[0],
                'Delta_RMSE': v3_row['RMSE_mean'].values[0] - v2_row['RMSE_mean'].values[0],
                'V2_MAE_mean': v2_row['MAE_mean'].values[0],
                'V2_MAE_std': v2_row['MAE_std'].values[0],
                'V3_MAE_mean': v3_row['MAE_mean'].values[0],
                'V3_MAE_std': v3_row['MAE_std'].values[0],
                'Delta_MAE': v3_row['MAE_mean'].values[0] - v2_row['MAE_mean'].values[0],
                'V2_R2_mean': v2_row['R^2_mean'].values[0],
                'V2_R2_std': v2_row['R^2_std'].values[0],
                'V3_R2_mean': v3_row['R^2_mean'].values[0],
                'V3_R2_std': v3_row['R^2_std'].values[0],
                'Delta_R2': v3_row['R^2_mean'].values[0] - v2_row['R^2_mean'].values[0],
                'V2_Bias': v2_row['bias_mean'].values[0],
                'V3_Bias': v3_row['bias_mean'].values[0]
            })

    summary_df = pd.DataFrame(summary)

    logger.info(f"  Summary statistics: {len(summary_df)} experiments")

    return summary_df


def create_best_models_ranking(data: dict) -> pd.DataFrame:
    """Create ranking of best models across all metrics."""
    logger.info("Creating best models ranking...")

    unified_df = data['unified']

    # Calculate overall ranking score (lower RMSE, higher R2)
    rankings = []

    for exp in ['BASIC', 'KCE', 'PAFC']:
        exp_df = unified_df[unified_df['Experiment'] == exp]

        for version in ['V2', 'V3']:
            ver_df = exp_df[exp_df['Version'] == version]

            if not ver_df.empty:
                rankings.append({
                    'Experiment': exp,
                    'Version': version,
                    'RMSE_mean': ver_df['RMSE'].mean(),
                    'RMSE_median': ver_df['RMSE'].median(),
                    'MAE_mean': ver_df['MAE'].mean(),
                    'R2_mean': ver_df['R^2'].mean(),
                    'R2_median': ver_df['R^2'].median(),
                    'n_models': len(ver_df)
                })

    ranking_df = pd.DataFrame(rankings)

    # Sort by RMSE (ascending)
    ranking_df = ranking_df.sort_values('RMSE_mean')

    logger.info(f"  Best models ranking: {len(ranking_df)} entries")

    return ranking_df


def create_horizon_degradation_rates(data: dict) -> pd.DataFrame:
    """Calculate degradation rates across horizons."""
    logger.info("Creating horizon degradation rates...")

    per_h_df = data['per_horizon']

    degradation = []

    for exp in ['BASIC', 'KCE', 'PAFC']:
        exp_df = per_h_df[per_h_df['Experiment'] == exp]

        if len(exp_df) < 2:
            continue

        # Calculate degradation slope for V2
        h_values = exp_df['H'].values
        rmse_v2 = exp_df['RMSE_V2'].values
        rmse_v3 = exp_df['RMSE_V3'].values
        r2_v2 = exp_df['R2_V2'].values
        r2_v3 = exp_df['R2_V3'].values

        # Linear fit
        slope_rmse_v2 = np.polyfit(h_values, rmse_v2, 1)[0]
        slope_rmse_v3 = np.polyfit(h_values, rmse_v3, 1)[0]
        slope_r2_v2 = np.polyfit(h_values, r2_v2, 1)[0]
        slope_r2_v3 = np.polyfit(h_values, r2_v3, 1)[0]

        degradation.append({
            'Experiment': exp,
            'RMSE_V2_slope': slope_rmse_v2,
            'RMSE_V3_slope': slope_rmse_v3,
            'R2_V2_slope': slope_r2_v2,
            'R2_V3_slope': slope_r2_v3,
            'RMSE_V2_H1': exp_df[exp_df['H'] == 1]['RMSE_V2'].values[0] if len(exp_df[exp_df['H'] == 1]) > 0 else np.nan,
            'RMSE_V2_H12': exp_df[exp_df['H'] == 12]['RMSE_V2'].values[0] if len(exp_df[exp_df['H'] == 12]) > 0 else np.nan,
            'R2_V2_H1': exp_df[exp_df['H'] == 1]['R2_V2'].values[0] if len(exp_df[exp_df['H'] == 1]) > 0 else np.nan,
            'R2_V2_H12': exp_df[exp_df['H'] == 12]['R2_V2'].values[0] if len(exp_df[exp_df['H'] == 12]) > 0 else np.nan
        })

    degradation_df = pd.DataFrame(degradation)

    logger.info(f"  Horizon degradation: {len(degradation_df)} experiments")

    return degradation_df


def create_feature_set_impact(data: dict) -> pd.DataFrame:
    """Analyze impact of feature sets (BASIC vs KCE vs PAFC)."""
    logger.info("Creating feature set impact analysis...")

    agg_df = data['aggregate']

    # Compare within each version
    impact = []

    for version in ['V2_Enhanced', 'V3_FNO']:
        ver_df = agg_df[agg_df['Model_Family'] == version]

        basic = ver_df[ver_df['Experiment'] == 'BASIC']
        kce = ver_df[ver_df['Experiment'] == 'KCE']
        pafc = ver_df[ver_df['Experiment'] == 'PAFC']

        if not basic.empty:
            basic_rmse = basic['RMSE_mean'].values[0]
            basic_r2 = basic['R^2_mean'].values[0]

            impact.append({
                'Version': version,
                'Feature_Set': 'BASIC',
                'RMSE': basic_rmse,
                'R2': basic_r2,
                'RMSE_vs_BASIC': 0.0,
                'R2_vs_BASIC': 0.0
            })

            if not kce.empty:
                impact.append({
                    'Version': version,
                    'Feature_Set': 'KCE',
                    'RMSE': kce['RMSE_mean'].values[0],
                    'R2': kce['R^2_mean'].values[0],
                    'RMSE_vs_BASIC': kce['RMSE_mean'].values[0] - basic_rmse,
                    'R2_vs_BASIC': kce['R^2_mean'].values[0] - basic_r2
                })

            if not pafc.empty:
                impact.append({
                    'Version': version,
                    'Feature_Set': 'PAFC',
                    'RMSE': pafc['RMSE_mean'].values[0],
                    'R2': pafc['R^2_mean'].values[0],
                    'RMSE_vs_BASIC': pafc['RMSE_mean'].values[0] - basic_rmse,
                    'R2_vs_BASIC': pafc['R^2_mean'].values[0] - basic_r2
                })

    impact_df = pd.DataFrame(impact)

    logger.info(f"  Feature set impact: {len(impact_df)} entries")

    return impact_df


def save_tables(tables: dict):
    """Save all CSV tables."""
    logger.info("Saving CSV tables...")

    for name, df in tables.items():
        table_path = TABLES_DIR / f"{name}.csv"
        df.to_csv(table_path, index=False)
        logger.info(f"  Saved: {table_path}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Script 12: Generate CSV Tables")
    logger.info("="*60)

    # Load data
    data = load_data()

    # Generate tables
    tables = {
        'summary_statistics': create_summary_statistics(data),
        'best_models_ranking': create_best_models_ranking(data),
        'horizon_degradation_rates': create_horizon_degradation_rates(data),
        'feature_set_impact': create_feature_set_impact(data)
    }

    # Save tables
    save_tables(tables)

    logger.info("="*60)
    logger.info("Completed successfully")
    logger.info(f"Generated {len(tables)} CSV tables")
    logger.info("="*60)

    # Print summary
    print("\n" + "="*60)
    print("CSV TABLES GENERATED")
    print("="*60)
    for name in tables.keys():
        print(f"  - {name}.csv")
    print("="*60)


if __name__ == '__main__':
    main()
