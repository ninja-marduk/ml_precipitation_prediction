"""
Benchmark Analysis Script 3: Training Dynamics Analysis

Part of V2 vs V3 Comparative Analysis Pipeline
Analyzes training efficiency, convergence speed, and stability
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'docs' / 'models' / 'comparative' / 'data'
V2_TRAINING_DIR = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models' / 'h12'
V3_TRAINING_DIR = PROJECT_ROOT / 'models' / 'output' / 'V3_FNO_Models' / 'h12'

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)


def find_training_logs(base_dir: Path, version: str) -> List[Path]:
    """
    Find all training log CSV files.

    Args:
        base_dir: Base directory to search
        version: 'V2' or 'V3'

    Returns:
        List of Path objects to training logs
    """
    logger.info(f"Searching for {version} training logs in {base_dir}")

    training_logs = []

    # Search in BASIC, KCE, PAFC subdirectories
    for exp in ['BASIC', 'KCE', 'PAFC']:
        exp_dir = base_dir / exp / 'training_metrics'
        if exp_dir.exists():
            logs = list(exp_dir.glob('*_training_log_h12.csv'))
            training_logs.extend(logs)
            logger.info(f"  Found {len(logs)} logs in {exp}")

    logger.info(f"  Total {version} logs found: {len(training_logs)}")

    return training_logs


def analyze_training_log(log_path: Path) -> Dict:
    """
    Analyze a single training log file.

    Args:
        log_path: Path to training log CSV

    Returns:
        Dictionary with convergence metrics
    """
    try:
        df = pd.read_csv(log_path)

        # Extract model name and experiment from path
        parts = log_path.parts
        experiment = parts[-3]  # BASIC, KCE, or PAFC
        model_name = log_path.stem.replace('_training_log_h12', '')

        # Calculate convergence metrics
        if 'val_loss' in df.columns:
            best_epoch = df['val_loss'].idxmin()
            best_val_loss = df['val_loss'].min()
            final_val_loss = df['val_loss'].iloc[-1]

            # Training stability (std of last 10 epochs)
            if len(df) >= 10:
                stability = df['val_loss'].iloc[-10:].std()
            else:
                stability = df['val_loss'].std()

            # Overfitting detection
            if 'loss' in df.columns:
                train_val_gap = df['loss'].iloc[-1] - final_val_loss
            else:
                train_val_gap = np.nan

            # Training improvement
            initial_val_loss = df['val_loss'].iloc[0]
            improvement = initial_val_loss - best_val_loss
            improvement_pct = (improvement / initial_val_loss) * 100

        else:
            logger.warning(f"  No val_loss column in {log_path.name}")
            best_epoch = np.nan
            best_val_loss = np.nan
            final_val_loss = np.nan
            stability = np.nan
            train_val_gap = np.nan
            improvement_pct = np.nan

        metrics = {
            'experiment': experiment,
            'model': model_name,
            'total_epochs': len(df),
            'best_epoch': best_epoch,
            'epochs_to_best': best_epoch + 1 if not np.isnan(best_epoch) else np.nan,
            'best_val_loss': best_val_loss,
            'final_val_loss': final_val_loss,
            'training_stability': stability,
            'train_val_gap': train_val_gap,
            'improvement_percent': improvement_pct
        }

        return metrics

    except Exception as e:
        logger.error(f"  Error analyzing {log_path.name}: {e}")
        return None


def process_training_logs(log_paths: List[Path], version: str) -> pd.DataFrame:
    """
    Process all training logs for a version.

    Args:
        log_paths: List of training log paths
        version: 'V2' or 'V3'

    Returns:
        DataFrame with convergence metrics
    """
    logger.info(f"Processing {version} training logs...")

    all_metrics = []

    for log_path in log_paths:
        metrics = analyze_training_log(log_path)
        if metrics:
            metrics['version'] = version
            all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    logger.info(f"  Processed {len(df)} {version} training logs")

    return df


def create_convergence_summary(v2_df: pd.DataFrame, v3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary comparison of convergence metrics.

    Args:
        v2_df: V2 convergence DataFrame
        v3_df: V3 convergence DataFrame

    Returns:
        Summary DataFrame
    """
    logger.info("Creating convergence summary...")

    combined = pd.concat([v2_df, v3_df], ignore_index=True)

    summary = combined.groupby(['version', 'experiment']).agg({
        'epochs_to_best': ['mean', 'std', 'min', 'max'],
        'best_val_loss': ['mean', 'std', 'min', 'max'],
        'training_stability': ['mean', 'std'],
        'train_val_gap': ['mean', 'std'],
        'improvement_percent': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    logger.info(f"  Convergence summary: {len(summary)} rows")

    return summary


def save_results(v2_df: pd.DataFrame, v3_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Save training dynamics results."""
    logger.info("Saving results...")

    # Combine V2 and V3 for epochs to convergence
    epochs_df = pd.concat([v2_df, v3_df], ignore_index=True)
    epochs_df = epochs_df[['version', 'experiment', 'model', 'epochs_to_best', 'best_val_loss']]

    epochs_path = DATA_DIR / 'epochs_to_convergence.csv'
    summary_path = DATA_DIR / 'convergence_summary.csv'

    epochs_df.to_csv(epochs_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"  Epochs to convergence saved to {epochs_path}")
    logger.info(f"  Convergence summary saved to {summary_path}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Script 3: Training Dynamics Analysis")
    logger.info("="*60)

    # Find training logs
    v2_logs = find_training_logs(V2_TRAINING_DIR, 'V2')
    v3_logs = find_training_logs(V3_TRAINING_DIR, 'V3')

    if not v2_logs:
        logger.warning("No V2 training logs found!")
    if not v3_logs:
        logger.warning("No V3 training logs found!")

    # Process training logs
    v2_df = process_training_logs(v2_logs, 'V2')
    v3_df = process_training_logs(v3_logs, 'V3')

    # Create convergence summary
    summary_df = create_convergence_summary(v2_df, v3_df)

    # Save results
    save_results(v2_df, v3_df, summary_df)

    logger.info("="*60)
    logger.info("Completed successfully")
    logger.info("="*60)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING DYNAMICS SUMMARY")
    print("="*60)
    print("\nAverage Epochs to Best Validation Loss:")
    print(summary_df[['version', 'experiment', 'epochs_to_best_mean', 'best_val_loss_mean']])
    print("="*60)


if __name__ == '__main__':
    main()
