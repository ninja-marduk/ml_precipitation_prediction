"""
Benchmark Analysis Script 2: Consolidate Metrics & Calculate Deltas

Part of V2 vs V3 Comparative Analysis Pipeline
Merges V2 and V3 metrics, computes performance deltas, generates aggregate statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'docs' / 'models' / 'comparative' / 'data'
V2_METRICS_PATH = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / 'metrics_spatial_v2_refactored_h12.csv'
V3_METRICS_PATH = PROJECT_ROOT / 'models' / 'output' / 'V3_FNO_Models' / 'metrics_spatial_v2_refactored_h12.csv'

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load V2 and V3 metrics data.

    Returns:
        Tuple of (v2_df, v3_df)
    """
    logger.info("Loading metrics data...")

    v2_df = pd.read_csv(V2_METRICS_PATH)
    v3_df = pd.read_csv(V3_METRICS_PATH)

    logger.info(f"  V2 metrics: {len(v2_df)} rows, {len(v2_df.columns)} columns")
    logger.info(f"  V3 metrics: {len(v3_df)} rows, {len(v3_df.columns)} columns")

    return v2_df, v3_df


def create_unified_metrics(v2_df: pd.DataFrame, v3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unified metrics table with V2 and V3 data.

    Args:
        v2_df: V2 metrics DataFrame
        v3_df: V3 metrics DataFrame

    Returns:
        Unified DataFrame
    """
    logger.info("Creating unified metrics table...")

    # Add version identifier
    v2_df = v2_df.copy()
    v3_df = v3_df.copy()
    v2_df['Version'] = 'V2'
    v3_df['Version'] = 'V3'

    # Concatenate
    unified_df = pd.concat([v2_df, v3_df], ignore_index=True)

    logger.info(f"  Unified table: {len(unified_df)} rows")

    return unified_df


def calculate_per_horizon_comparison(v2_df: pd.DataFrame, v3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-horizon comparison between V2 and V3.

    Args:
        v2_df: V2 metrics DataFrame
        v3_df: V3 metrics DataFrame

    Returns:
        Per-horizon comparison DataFrame
    """
    logger.info("Calculating per-horizon comparison...")

    # Group by Experiment and H, calculate means
    v2_grouped = v2_df.groupby(['Experiment', 'H']).agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'R^2': 'mean',
        'Mean_True_mm': 'mean',
        'Mean_Pred_mm': 'mean'
    }).reset_index()
    v2_grouped.columns = ['Experiment', 'H', 'RMSE_V2', 'MAE_V2', 'R2_V2', 'Mean_True_V2', 'Mean_Pred_V2']

    v3_grouped = v3_df.groupby(['Experiment', 'H']).agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'R^2': 'mean',
        'Mean_True_mm': 'mean',
        'Mean_Pred_mm': 'mean'
    }).reset_index()
    v3_grouped.columns = ['Experiment', 'H', 'RMSE_V3', 'MAE_V3', 'R2_V3', 'Mean_True_V3', 'Mean_Pred_V3']

    # Merge
    comparison_df = pd.merge(v2_grouped, v3_grouped, on=['Experiment', 'H'], how='outer')

    # Calculate deltas
    comparison_df['delta_RMSE'] = comparison_df['RMSE_V3'] - comparison_df['RMSE_V2']
    comparison_df['delta_MAE'] = comparison_df['MAE_V3'] - comparison_df['MAE_V2']
    comparison_df['delta_R2'] = comparison_df['R2_V3'] - comparison_df['R2_V2']
    comparison_df['delta_bias'] = (comparison_df['Mean_Pred_V3'] - comparison_df['Mean_True_V3']) - \
                                   (comparison_df['Mean_Pred_V2'] - comparison_df['Mean_True_V2'])

    # Calculate percent changes
    comparison_df['percent_change_RMSE'] = (comparison_df['delta_RMSE'] / comparison_df['RMSE_V2']) * 100
    comparison_df['percent_change_MAE'] = (comparison_df['delta_MAE'] / comparison_df['MAE_V2']) * 100
    comparison_df['percent_change_R2'] = (comparison_df['delta_R2'] / comparison_df['R2_V2'].abs()) * 100

    logger.info(f"  Per-horizon comparison: {len(comparison_df)} rows")

    return comparison_df


def calculate_aggregate_statistics(v2_df: pd.DataFrame, v3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregate statistics per experiment and model family.

    Args:
        v2_df: V2 metrics DataFrame
        v3_df: V3 metrics DataFrame

    Returns:
        Aggregate statistics DataFrame
    """
    logger.info("Calculating aggregate statistics...")

    v2_df = v2_df.copy()
    v3_df = v3_df.copy()
    v2_df['Model_Family'] = 'V2_Enhanced'
    v3_df['Model_Family'] = 'V3_FNO'

    combined = pd.concat([v2_df, v3_df], ignore_index=True)

    agg_stats = combined.groupby(['Experiment', 'Model_Family']).agg({
        'RMSE': ['mean', 'std', 'min', 'max', 'median'],
        'MAE': ['mean', 'std', 'min', 'max', 'median'],
        'R^2': ['mean', 'std', 'min', 'max', 'median'],
        'Mean_Pred_mm': ['mean'],
        'Mean_True_mm': ['mean']
    }).reset_index()

    # Flatten column names
    agg_stats.columns = ['_'.join(col).strip('_') for col in agg_stats.columns.values]

    # Calculate bias
    agg_stats['bias_mean'] = agg_stats['Mean_Pred_mm_mean'] - agg_stats['Mean_True_mm_mean']

    logger.info(f"  Aggregate statistics: {len(agg_stats)} rows")

    return agg_stats


def calculate_delta_metrics(v2_df: pd.DataFrame, v3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate detailed delta metrics.

    Args:
        v2_df: V2 metrics DataFrame
        v3_df: V3 metrics DataFrame

    Returns:
        Delta metrics DataFrame
    """
    logger.info("Calculating delta metrics...")

    # Calculate overall deltas by experiment
    deltas = []

    for exp in ['BASIC', 'KCE', 'PAFC']:
        v2_exp = v2_df[v2_df['Experiment'] == exp]
        v3_exp = v3_df[v3_df['Experiment'] == exp]

        if len(v2_exp) == 0 or len(v3_exp) == 0:
            logger.warning(f"  No data for experiment: {exp}")
            continue

        delta_row = {
            'Experiment': exp,
            'V2_RMSE_mean': v2_exp['RMSE'].mean(),
            'V2_RMSE_std': v2_exp['RMSE'].std(),
            'V3_RMSE_mean': v3_exp['RMSE'].mean(),
            'V3_RMSE_std': v3_exp['RMSE'].std(),
            'delta_RMSE_mean': v3_exp['RMSE'].mean() - v2_exp['RMSE'].mean(),
            'V2_MAE_mean': v2_exp['MAE'].mean(),
            'V2_MAE_std': v2_exp['MAE'].std(),
            'V3_MAE_mean': v3_exp['MAE'].mean(),
            'V3_MAE_std': v3_exp['MAE'].std(),
            'delta_MAE_mean': v3_exp['MAE'].mean() - v2_exp['MAE'].mean(),
            'V2_R2_mean': v2_exp['R^2'].mean(),
            'V2_R2_std': v2_exp['R^2'].std(),
            'V3_R2_mean': v3_exp['R^2'].mean(),
            'V3_R2_std': v3_exp['R^2'].std(),
            'delta_R2_mean': v3_exp['R^2'].mean() - v2_exp['R^2'].mean(),
            'V2_n_samples': len(v2_exp),
            'V3_n_samples': len(v3_exp)
        }

        deltas.append(delta_row)

    delta_df = pd.DataFrame(deltas)

    logger.info(f"  Delta metrics: {len(delta_df)} experiments")

    return delta_df


def save_results(unified_df: pd.DataFrame, per_horizon_df: pd.DataFrame,
                 aggregate_df: pd.DataFrame, delta_df: pd.DataFrame):
    """Save all processed results."""
    logger.info("Saving results...")

    unified_path = DATA_DIR / 'unified_metrics_h12.csv'
    per_horizon_path = DATA_DIR / 'per_horizon_comparison_h12.csv'
    aggregate_path = DATA_DIR / 'aggregate_statistics_h12.csv'
    delta_path = DATA_DIR / 'delta_metrics_h12.csv'

    unified_df.to_csv(unified_path, index=False)
    per_horizon_df.to_csv(per_horizon_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    logger.info(f"  Unified metrics saved to {unified_path}")
    logger.info(f"  Per-horizon comparison saved to {per_horizon_path}")
    logger.info(f"  Aggregate statistics saved to {aggregate_path}")
    logger.info(f"  Delta metrics saved to {delta_path}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Script 2: Consolidate Metrics & Calculate Deltas")
    logger.info("="*60)

    # Load data
    v2_df, v3_df = load_metrics_data()

    # Create unified metrics
    unified_df = create_unified_metrics(v2_df, v3_df)

    # Calculate per-horizon comparison
    per_horizon_df = calculate_per_horizon_comparison(v2_df, v3_df)

    # Calculate aggregate statistics
    aggregate_df = calculate_aggregate_statistics(v2_df, v3_df)

    # Calculate delta metrics
    delta_df = calculate_delta_metrics(v2_df, v3_df)

    # Save results
    save_results(unified_df, per_horizon_df, aggregate_df, delta_df)

    logger.info("="*60)
    logger.info("Completed successfully")
    logger.info("="*60)

    # Print summary
    print("\n" + "="*60)
    print("CONSOLIDATION SUMMARY")
    print("="*60)
    print(f"Unified metrics: {len(unified_df)} rows")
    print(f"Per-horizon comparison: {len(per_horizon_df)} rows")
    print(f"Aggregate statistics: {len(aggregate_df)} rows")
    print(f"Delta metrics: {len(delta_df)} experiments")
    print("\nDelta Summary (RMSE):")
    print(delta_df[['Experiment', 'V2_RMSE_mean', 'V3_RMSE_mean', 'delta_RMSE_mean']])
    print("="*60)


if __name__ == '__main__':
    main()
