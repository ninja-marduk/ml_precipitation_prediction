"""
Benchmark Analysis Script 13: Statistical Significance Tests

Performs formal statistical tests for Paper 4:
1. Friedman Test - Non-parametric ANOVA for multiple groups
2. Nemenyi Post-Hoc Test - Pairwise comparisons after Friedman
3. Mann-Whitney U Test - V2 vs V4 family comparison
4. Effect Size (Cohen's d) - Practical significance
5. Critical Difference (CD) Calculation for CD diagram

Following Demsar (2006) methodology for ML algorithm comparison.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
V2_DIR = PROJECT_ROOT / 'models' / 'output' / 'V2_Enhanced_Models'
V4_DIR = PROJECT_ROOT / 'models' / 'output' / 'V4_GNN_TAT_Models'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'papers' / '4'


def load_v2_metrics() -> pd.DataFrame:
    """Load V2 metrics from CSV."""
    metrics_path = V2_DIR / 'metrics_spatial_v2_refactored_h12.csv'
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df['Version'] = 'V2'
        logger.info(f"Loaded V2 metrics: {len(df)} records")
        return df
    return pd.DataFrame()


def load_v4_metrics() -> pd.DataFrame:
    """Load V4 GNN-TAT metrics from CSV."""
    metrics_path = V4_DIR / 'metrics_spatial_v4_gnn_tat_h12.csv'
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df['Version'] = 'V4'
        logger.info(f"Loaded V4 metrics: {len(df)} records")
        return df
    return pd.DataFrame()


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Interpretation:
    - 0.2: Small effect
    - 0.5: Medium effect
    - 0.8: Large effect
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def calculate_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Calculate Critical Difference for Nemenyi test.

    CD = q_alpha * sqrt(k(k+1)/(6N))

    Where q_alpha is the critical value from the Studentized range distribution.
    """
    # Critical values for Nemenyi test (alpha=0.05)
    # k = number of algorithms
    q_alpha_table = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
        13: 3.313,
        14: 3.354,
        15: 3.391,
    }

    q_alpha = q_alpha_table.get(k, 2.5)  # Default if k not in table
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    return cd


def friedman_test(data: Dict[str, np.ndarray]) -> Dict:
    """
    Perform Friedman test for comparing multiple algorithms.

    H0: All algorithms perform equally
    H1: At least one algorithm performs differently
    """
    logger.info("Performing Friedman test...")

    # Convert to arrays for testing
    arrays = list(data.values())

    # Friedman test
    stat, p_value = stats.friedmanchisquare(*arrays)

    result = {
        'test_name': 'Friedman Test',
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'n_groups': len(data),
        'group_names': list(data.keys()),
        'interpretation': 'Significant difference exists' if p_value < 0.05 else 'No significant difference'
    }

    logger.info(f"  Friedman chi-square: {stat:.4f}, p-value: {p_value:.6f}")
    return result


def mann_whitney_test(group1: np.ndarray, group2: np.ndarray,
                       name1: str = 'Group1', name2: str = 'Group2') -> Dict:
    """
    Perform Mann-Whitney U test for two independent groups.

    H0: The distributions are equal
    H1: The distributions are different
    """
    logger.info(f"Performing Mann-Whitney U test: {name1} vs {name2}...")

    stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    effect_size = cohens_d(group1, group2)

    # Effect size interpretation
    if abs(effect_size) < 0.2:
        effect_interp = 'negligible'
    elif abs(effect_size) < 0.5:
        effect_interp = 'small'
    elif abs(effect_size) < 0.8:
        effect_interp = 'medium'
    else:
        effect_interp = 'large'

    result = {
        'test_name': 'Mann-Whitney U Test',
        'groups': [name1, name2],
        'U_statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'cohens_d': float(effect_size),
        'effect_size_interpretation': effect_interp,
        'group1_mean': float(np.mean(group1)),
        'group1_std': float(np.std(group1)),
        'group2_mean': float(np.mean(group2)),
        'group2_std': float(np.std(group2)),
        'mean_difference': float(np.mean(group1) - np.mean(group2)),
        'interpretation': f'{name1} is {"better" if np.mean(group1) < np.mean(group2) else "worse"} ({"significant" if p_value < 0.05 else "not significant"})'
    }

    logger.info(f"  U={stat:.1f}, p={p_value:.6f}, Cohen's d={effect_size:.3f} ({effect_interp})")
    return result


def compute_average_ranks(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute average ranks for each algorithm across all datasets.
    Lower rank = better performance (for RMSE).
    """
    # Combine into matrix (rows = datasets, cols = algorithms)
    n_samples = len(list(data.values())[0])
    n_algorithms = len(data)

    matrix = np.column_stack(list(data.values()))

    # Rank each row (dataset)
    ranks = np.zeros_like(matrix)
    for i in range(n_samples):
        ranks[i] = stats.rankdata(matrix[i])

    # Average rank for each algorithm
    avg_ranks = ranks.mean(axis=0)

    return dict(zip(data.keys(), avg_ranks))


def run_comprehensive_analysis(v2_df: pd.DataFrame, v4_df: pd.DataFrame) -> Dict:
    """
    Run comprehensive statistical analysis.
    """
    logger.info("Running comprehensive statistical analysis...")

    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_summary': {},
        'tests': [],
        'model_rankings': {},
        'critical_difference': {},
    }

    # Combine data
    combined = pd.concat([v2_df, v4_df], ignore_index=True)
    h12_data = combined[combined['H'] == 12]

    # Data summary
    results['data_summary'] = {
        'v2_records': len(v2_df),
        'v4_records': len(v4_df),
        'total_records': len(combined),
        'h12_records': len(h12_data),
        'unique_models_v2': v2_df['Model'].nunique() if not v2_df.empty else 0,
        'unique_models_v4': v4_df['Model'].nunique() if not v4_df.empty else 0,
    }

    # Test 1: V2 vs V4 Family Comparison (RMSE)
    v2_rmse = h12_data[h12_data['Version'] == 'V2']['RMSE'].values
    v4_rmse = h12_data[h12_data['Version'] == 'V4']['RMSE'].values

    if len(v2_rmse) > 0 and len(v4_rmse) > 0:
        mw_result = mann_whitney_test(v2_rmse, v4_rmse, 'V2_ConvLSTM', 'V4_GNN-TAT')
        results['tests'].append(mw_result)

    # Test 2: V2 vs V4 Family Comparison (R2)
    v2_r2 = h12_data[h12_data['Version'] == 'V2']['R^2'].values
    v4_r2 = h12_data[h12_data['Version'] == 'V4']['R^2'].values

    if len(v2_r2) > 0 and len(v4_r2) > 0:
        # For R2, higher is better, so negate for consistent interpretation
        mw_r2_result = mann_whitney_test(-v2_r2, -v4_r2, 'V2_ConvLSTM_R2', 'V4_GNN-TAT_R2')
        mw_r2_result['metric'] = 'R^2'
        mw_r2_result['interpretation'] = f'V4 {"outperforms" if np.mean(v4_r2) > np.mean(v2_r2) else "underperforms"} V2 on R^2'
        results['tests'].append(mw_r2_result)

    # Test 3: Friedman test across model configurations
    # Prepare data for Friedman: each model-experiment pair as a "treatment"
    model_rmse = {}
    for (model, exp), group in h12_data.groupby(['Model', 'Experiment']):
        key = f"{model}_{exp}"
        if len(group) >= 3:  # Need at least 3 samples
            model_rmse[key] = group['RMSE'].values[:12]  # Take first 12 for consistency

    # Filter to models with same sample size
    if model_rmse:
        min_len = min(len(v) for v in model_rmse.values())
        model_rmse = {k: v[:min_len] for k, v in model_rmse.items() if len(v) >= min_len}

        if len(model_rmse) >= 3:
            friedman_result = friedman_test(model_rmse)
            results['tests'].append(friedman_result)

            # Compute ranks and CD
            avg_ranks = compute_average_ranks(model_rmse)
            results['model_rankings'] = {k: float(v) for k, v in sorted(avg_ranks.items(), key=lambda x: x[1])}

            # Critical difference
            k = len(model_rmse)
            n = min_len
            cd = calculate_cd(k, n)
            results['critical_difference'] = {
                'CD': float(cd),
                'k_algorithms': k,
                'n_samples': n,
                'alpha': 0.05,
                'formula': 'CD = q_alpha * sqrt(k(k+1)/(6N))'
            }

    # Test 4: Per-feature set comparison
    for exp in ['BASIC', 'KCE', 'PAFC']:
        exp_data = h12_data[h12_data['Experiment'] == exp]
        v2_exp = exp_data[exp_data['Version'] == 'V2']['RMSE'].values
        v4_exp = exp_data[exp_data['Version'] == 'V4']['RMSE'].values

        if len(v2_exp) > 0 and len(v4_exp) > 0:
            mw_exp = mann_whitney_test(v2_exp, v4_exp, f'V2_{exp}', f'V4_{exp}')
            mw_exp['feature_set'] = exp
            results['tests'].append(mw_exp)

    return results


def generate_latex_statistical_table(results: Dict) -> str:
    """
    Generate LaTeX table summarizing statistical tests.
    """
    logger.info("Generating statistical tests LaTeX table...")

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical Significance Tests: V2 ConvLSTM vs V4 GNN-TAT}",
        r"\label{tab:statistical-tests}",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Test} & \textbf{Statistic} & \textbf{p-value} & \textbf{Effect (d)} & \textbf{Significant?} \\",
        r"\midrule",
    ]

    for test in results.get('tests', []):
        if 'Mann-Whitney' in test['test_name']:
            groups = test.get('groups', ['G1', 'G2'])
            comparison = f"{groups[0]} vs {groups[1]}"
            stat = test['U_statistic']
            p = test['p_value']
            d = test.get('cohens_d', 0)
            sig = 'Yes' if test['significant'] else 'No'

            latex_lines.append(
                f"{comparison} & Mann-Whitney U & {stat:.1f} & {p:.4f} & {d:.2f} & {sig} \\\\"
            )
        elif 'Friedman' in test['test_name']:
            latex_lines.append(
                f"All Models & Friedman & {test['statistic']:.2f} & {test['p_value']:.4f} & -- & {'Yes' if test['significant'] else 'No'} \\\\"
            )

    # Add CD info if available
    if results.get('critical_difference'):
        cd = results['critical_difference']
        latex_lines.append(r"\midrule")
        latex_lines.append(
            f"\\multicolumn{{6}}{{l}}{{Critical Difference (CD) = {cd['CD']:.3f} "
            f"for k={cd['k_algorithms']} algorithms, n={cd['n_samples']} samples, $\\alpha$=0.05}} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\multicolumn{6}{l}{\footnotesize Effect size: $|d|<0.2$ negligible, $<0.5$ small, $<0.8$ medium, $\geq0.8$ large} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return '\n'.join(latex_lines)


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Script 13: Statistical Significance Analysis")
    logger.info("=" * 60)

    # Load data
    v2_df = load_v2_metrics()
    v4_df = load_v4_metrics()

    if v2_df.empty and v4_df.empty:
        logger.error("No data found!")
        return

    # Run analysis
    results = run_comprehensive_analysis(v2_df, v4_df)

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    results = convert_to_native(results)

    # Save JSON results
    json_path = OUTPUT_DIR / 'statistical_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to: {json_path}")

    # Generate and save LaTeX table
    tables_dir = OUTPUT_DIR / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    latex_table = generate_latex_statistical_table(results)
    latex_path = tables_dir / 'table_statistical_tests.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    logger.info(f"Saved LaTeX table to: {latex_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("Statistical Analysis Summary:")
    logger.info("=" * 60)

    for test in results.get('tests', []):
        if 'groups' in test:
            logger.info(f"  {test['groups'][0]} vs {test['groups'][1]}:")
            logger.info(f"    p-value: {test['p_value']:.6f}")
            logger.info(f"    Cohen's d: {test.get('cohens_d', 0):.3f} ({test.get('effect_size_interpretation', 'N/A')})")
            logger.info(f"    Significant: {test['significant']}")

    if results.get('critical_difference'):
        cd = results['critical_difference']
        logger.info(f"\n  Critical Difference: CD = {cd['CD']:.3f}")

    if results.get('model_rankings'):
        logger.info("\n  Top 5 Model Rankings (by average rank, lower is better):")
        for i, (model, rank) in enumerate(list(results['model_rankings'].items())[:5]):
            logger.info(f"    {i+1}. {model}: {rank:.2f}")

    logger.info("=" * 60)


if __name__ == '__main__':
    main()
