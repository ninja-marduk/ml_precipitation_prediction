"""
Benchmark Analysis Script 4: Statistical Significance Testing

Part of V2 vs V3 Comparative Analysis Pipeline
Performs comprehensive statistical hypothesis testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats
from typing import Dict, Tuple

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
DATA_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load per-horizon comparison data."""
    logger.info("Loading per-horizon comparison data...")

    data_path = DATA_DIR / 'per_horizon_comparison_h12.csv'

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please run script 02_consolidate_metrics.py first")
        raise FileNotFoundError(f"Required file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"  Loaded {len(df)} rows")

    return df


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        x1: First sample
        x2: Second sample

    Returns:
        Cohen's d value
    """
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(x1) - np.mean(x2)) / pooled_std


def perform_paired_t_test(df: pd.DataFrame, experiment: str, metric: str) -> Dict:
    """
    Perform paired t-test for V2 vs V3.

    Args:
        df: Comparison DataFrame
        experiment: Experiment name (BASIC, KCE, PAFC)
        metric: Metric name (RMSE, MAE, R2)

    Returns:
        Dictionary with test results
    """
    exp_df = df[df['Experiment'] == experiment].copy()

    v2_col = f'{metric}_V2'
    v3_col = f'{metric}_V3'

    if v2_col not in exp_df.columns or v3_col not in exp_df.columns:
        logger.warning(f"  Columns not found for {metric} in {experiment}")
        return None

    # Remove NaN values
    valid_df = exp_df[[v2_col, v3_col]].dropna()

    if len(valid_df) < 3:
        logger.warning(f"  Insufficient data for {experiment} {metric} (n={len(valid_df)})")
        return None

    v2_values = valid_df[v2_col].values
    v3_values = valid_df[v3_col].values

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(v2_values, v3_values)

    # Cohen's d
    effect_size = cohens_d(v2_values, v3_values)

    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, w_pvalue = stats.wilcoxon(v2_values, v3_values)
    except Exception as e:
        logger.warning(f"  Wilcoxon test failed: {e}")
        w_stat, w_pvalue = np.nan, np.nan

    # 95% Confidence interval for difference
    diff = v2_values - v3_values
    mean_diff = np.mean(diff)
    sem_diff = stats.sem(diff)
    ci_95 = stats.t.interval(0.95, len(diff)-1, mean_diff, sem_diff)

    results = {
        'experiment': experiment,
        'metric': metric,
        'n_samples': len(valid_df),
        'v2_mean': np.mean(v2_values),
        'v2_std': np.std(v2_values, ddof=1),
        'v3_mean': np.mean(v3_values),
        'v3_std': np.std(v3_values, ddof=1),
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': effect_size,
        'wilcoxon_stat': w_stat,
        'wilcoxon_p': w_pvalue,
        'ci_95_lower': ci_95[0],
        'ci_95_upper': ci_95[1],
        'significant_alpha_0.05': p_value < 0.05,
        'significant_alpha_0.01': p_value < 0.01,
        'significant_alpha_0.001': p_value < 0.001
    }

    return results


def perform_all_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform statistical tests for all experiments and metrics.

    Args:
        df: Comparison DataFrame

    Returns:
        DataFrame with all test results
    """
    logger.info("Performing statistical tests...")

    experiments = ['BASIC', 'KCE', 'PAFC']
    metrics = ['RMSE', 'MAE', 'R2']

    all_results = []

    for exp in experiments:
        for metric in metrics:
            logger.info(f"  Testing {exp} - {metric}")
            result = perform_paired_t_test(df, exp, metric)
            if result:
                all_results.append(result)

    results_df = pd.DataFrame(all_results)

    logger.info(f"  Completed {len(results_df)} statistical tests")

    return results_df


def create_significance_matrix(results_df: pd.DataFrame) -> str:
    """
    Create LaTeX significance matrix.

    Args:
        results_df: Statistical test results

    Returns:
        LaTeX table string
    """
    logger.info("Creating LaTeX significance matrix...")

    latex = r"""\begin{table}[htbp]
\centering
\caption{Statistical Significance Tests: V2 vs V3 Models}
\label{tab:significance_tests}
\begin{tabular}{llrrrr}
\toprule
\textbf{Experiment} & \textbf{Metric} & \textbf{$t$-statistic} & \textbf{$p$-value} & \textbf{Cohen's $d$} & \textbf{Sig.} \\
\midrule
"""

    for _, row in results_df.iterrows():
        # Determine significance markers
        if row['significant_alpha_0.001']:
            sig_marker = '***'
        elif row['significant_alpha_0.01']:
            sig_marker = '**'
        elif row['significant_alpha_0.05']:
            sig_marker = '*'
        else:
            sig_marker = 'ns'

        # Format p-value
        if row['p_value'] < 0.001:
            p_str = '$< 0.001$'
        else:
            p_str = f"${row['p_value']:.3f}$"

        latex += f"{row['experiment']} & {row['metric']} & ${row['t_statistic']:.2f}$ & {p_str} & ${row['cohens_d']:.2f}$ & {sig_marker} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$, ns = not significant
\item Cohen's $d$: small (0.2), medium (0.5), large (0.8)
\end{tablenotes}
\end{table}
"""

    return latex


def save_results(results_df: pd.DataFrame):
    """Save statistical test results."""
    logger.info("Saving results...")

    # Save CSV
    csv_path = DATA_DIR / 'statistical_tests_summary.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"  Statistical tests saved to {csv_path}")

    # Save LaTeX table
    latex_table = create_significance_matrix(results_df)
    latex_path = TABLES_DIR / 'significance_matrix.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    logger.info(f"  LaTeX table saved to {latex_path}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Script 4: Statistical Significance Testing")
    logger.info("="*60)

    # Load data
    df = load_data()

    # Perform all tests
    results_df = perform_all_tests(df)

    # Save results
    save_results(results_df)

    logger.info("="*60)
    logger.info("Completed successfully")
    logger.info("="*60)

    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL TESTS SUMMARY")
    print("="*60)
    print("\nSignificance Summary:")
    print(results_df[['experiment', 'metric', 'p_value', 'cohens_d', 'significant_alpha_0.05']])
    print("\nEffect Sizes (Cohen's d):")
    print("  < 0.2: negligible")
    print("  0.2-0.5: small")
    print("  0.5-0.8: medium")
    print("  > 0.8: large")
    print("="*60)


if __name__ == '__main__':
    main()
