# Script: Quantitative meta-analysis of hybrid models for precipitation prediction with RMSE and MAE averages.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from scipy.stats import kruskal
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_baseline_data():
    """
    Create baseline (non-hybrid) model data for comparison.
    These are examples based on typical literature of simple models.
    """
    baseline_data = {
        'Model': ['Linear Regression', 'ARIMA', 'Simple LSTM', 'Basic CNN', 'SVM', 
                 'Random Forest (Simple)', 'Persistence Model', 'Moving Average'],
        'MAE': [65.2, 58.3, 48.1, 52.7, 61.8, 45.2, 78.5, 72.1],
        'RMSE': [89.4, 76.8, 62.3, 68.9, 82.1, 58.7, 98.2, 91.6],
        'Type': ['baseline'] * 8
    }
    return pd.DataFrame(baseline_data)

def classify_hybrid_models(df):
    """
    Classify hybrid models into categories for meta-analysis.
    """
    # Define classifications based on hybridization techniques from study taxonomy
    
    # Data preprocessing-based hybrid models
    data_preprocessing = ['CEEMD-ELM-FFOA', 'CEEMDAN-SVM-LSTM', 'LSTM & EMD-ELM', 
                         'TVF-EMD-ENN', 'VMD-CPO-LSTM', 'WT-ELM', 'W-AL (Wavelet-ARIMA-LSTM)', 
                         'Wavelet-RF', 'SSA-LSSVR (Deji)', 'SSA-RF (Shihmen)']
    
    # Parameter optimization-based hybrid models
    parameter_optimization = ['BBO-ELM', 'Bat-ELM', 'GA', 'SARIMA-ANN', 'LSO-ABR Model 3']
    
    # Component combination-based hybrid models (Ensemble/Stacking)
    component_combination = ['CEEMD-FCMSE-Stacking', 'Stacking', 'EEMD-BMA', 'Satck-Las-Rsc',
                            'SMOTE-km-XGB', 'ANN', 'MLSTM-AM', 'MLSTM']
    
    # Postprocessing-based hybrid models
    postprocessing = ['SEAS4 & SEAS5', 'BP-ANN']
    
    # Deep hybrid models (advanced neural architectures)
    deep_hybrid = ['CNN-LSTM', 'BLSTM-GRU', 'BDTR']
    
    # Assign categories
    categories = []
    for model in df.index:
        if any(keyword in model for keyword in data_preprocessing):
            categories.append('Data Preprocessing')
        elif any(keyword in model for keyword in parameter_optimization):
            categories.append('Parameter Optimization')
        elif any(keyword in model for keyword in component_combination):
            categories.append('Component Combination')
        elif any(keyword in model for keyword in postprocessing):
            categories.append('Postprocessing')
        elif any(keyword in model for keyword in deep_hybrid):
            categories.append('Deep Hybrid')
        else:
            categories.append('Other Hybrid')
    
    df['Category'] = categories
    return df

def calculate_log_ratios(hybrid_df, baseline_df, metric='RMSE'):
    """
    Calculate log-ratios for meta-analysis: Δ = ln(E_hyb/E_base)
    """
    # Use baseline median as reference
    baseline_median = baseline_df[metric].median()
    
    log_ratios = []
    improvements = []
    models_list = []
    
    for model in hybrid_df.index:
        value = hybrid_df.loc[model, metric]
        if pd.notna(value) and value > 0:
            delta = np.log(value / baseline_median)
            log_ratios.append(delta)
            improvement = (np.exp(delta) - 1) * 100  # Percentage change
            improvements.append(improvement)
            models_list.append(model)
    
    return log_ratios, improvements, models_list

def create_logarithmic_plot(df):
    """
    Create logarithmic plot similar to reference image with improved layout and no overlapping.
    """
    df = df.copy()
    # Detect fallback sources for star marks
    rmse_star = df.get('RMSE_source')
    mae_star = df.get('MAE_source')

    for col in ['RMSE', 'MAE']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[~(df['RMSE'].isna() & df['MAE'].isna())]

    skill_only_keywords = [
        'Stacking with ', 'Mean combiner', 'Median combiner',
        'Multivariate adaptive ', 'Bayesian regularized neural networks',
        'SkillScore', 'skill', 'Type2', 'polyMARS', 'MARS'
    ]
    mask_skill = df.index.to_series().str.contains('|'.join([k.replace(' ', '\\s*') for k in skill_only_keywords]), case=False, na=False)
    df = df[~mask_skill]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(22, 8))
    
    # RMSE
    df_rmse = df.dropna(subset=['RMSE']).sort_values('RMSE', ascending=True)
    rmse_values = df_rmse['RMSE']
    rmse_models = df_rmse.index.tolist()
    # Append * for R2_fallback
    if rmse_star is not None:
        rmse_models = [f"{m}*" if rmse_star.get(m, None) == 'R2_fallback' else m for m in rmse_models]
    
    # MAE
    df_mae = df.dropna(subset=['MAE']).sort_values('MAE', ascending=True)
    mae_values = df_mae['MAE']
    mae_models = df_mae.index.tolist()
    if mae_star is not None:
        mae_models = [f"{m}*" if mae_star.get(m, None) == 'R2_fallback' else m for m in mae_models]
    
    rmse_norm = (rmse_values - rmse_values.min()) / (rmse_values.max() - rmse_values.min() + 1e-9)
    rmse_colors = plt.cm.RdYlBu_r(rmse_norm)
    mae_norm = (mae_values - mae_values.min()) / (mae_values.max() - mae_values.min() + 1e-9)
    mae_colors = plt.cm.RdYlBu_r(mae_norm)
    
    bars1 = ax1.barh(range(len(rmse_models)), rmse_values, color=rmse_colors, height=0.3)
    ax1.set_xlabel('Error RMSE (%)', fontsize=12)
    ax1.set_ylabel('Hybrid Model', fontsize=12)
    ax1.set_title('Average RMSE by Hybrid Model', fontsize=14, fontweight='bold', pad=20)
    ax1.set_yticks(range(len(rmse_models)))
    ax1.set_yticklabels(rmse_models, fontsize=7)
    ax1.grid(True, alpha=0.3, axis='x')
    rmse_min = max(0, float(rmse_values.min()) - 5)
    rmse_max = min(100, float(rmse_values.max()) + 5)
    ax1.set_xlim(rmse_min, rmse_max)
    
    bars2 = ax2.barh(range(len(mae_models)), mae_values, color=mae_colors, height=0.3)
    ax2.set_xlabel('Error MAE (%)', fontsize=12)
    ax2.set_title('Average MAE by Hybrid Model', fontsize=14, fontweight='bold', pad=20)
    ax2.set_yticks(range(len(mae_models)))
    ax2.set_yticklabels(mae_models, fontsize=7)
    ax2.grid(True, alpha=0.3, axis='x')
    mae_min = max(0, float(mae_values.min()) - 5)
    mae_max = min(100, float(mae_values.max()) + 5)
    ax2.set_xlim(mae_min, mae_max)
    
    plt.subplots_adjust(left=0.35, bottom=0.12, right=0.82, top=0.88, wspace=0.35, hspace=0.05)
    
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Error Magnitude (%)', rotation=270, labelpad=20)

    # Add note for fallback
    fig.text(0.35, 0.04, "* uses R²-based fallback (sqrt(1−R²)×100) when RSE/NRMSE not available", fontsize=9)
    
    return fig

def perform_meta_analysis(hybrid_df, baseline_df):
    """
    Perform quantitative meta-analysis with R^2 only.
    """
    results = {}
    
    # Only work with R^2 values
    if 'R_squared' in hybrid_df.columns:
        r2_values = hybrid_df['R_squared'].dropna()
        
        if len(r2_values) > 0:
            # Create DataFrame for analysis
            analysis_df = pd.DataFrame({
                'Model': r2_values.index,
                'R_squared': r2_values.values,
                'Category': [hybrid_df.loc[model, 'Category'] for model in r2_values.index]
            })
            
            # Statistics by category
            category_stats = analysis_df.groupby('Category')['R_squared'].agg([
                'count', 'median', 'std', 
                lambda x: np.percentile(x, 25),  # Q1
                lambda x: np.percentile(x, 75)   # Q3
            ]).round(4)
            category_stats.columns = ['Count', 'Median', 'Std', 'Q1', 'Q3']
            
            # Kruskal-Wallis test
            categories = analysis_df['Category'].unique()
            if len(categories) > 2:
                groups = [analysis_df[analysis_df['Category'] == cat]['R_squared'].values 
                         for cat in categories if len(analysis_df[analysis_df['Category'] == cat]) > 0]
                if len(groups) > 2:
                    kruskal_stat, kruskal_p = kruskal(*groups)
                else:
                    kruskal_stat, kruskal_p = np.nan, np.nan
            else:
                kruskal_stat, kruskal_p = np.nan, np.nan
            
            results['R_squared'] = {
                'analysis_df': analysis_df,
                'category_stats': category_stats,
                'kruskal_stat': kruskal_stat,
                'kruskal_p': kruskal_p
            }
    
    return results

def create_meta_analysis_plots(results):
    """
    Create meta-analysis visualizations for R^2 only.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define consistent category order based on new taxonomy
    category_order = ['Deep Hybrid', 'Data Preprocessing', 'Component Combination', 
                     'Parameter Optimization', 'Postprocessing', 'Other Hybrid']
    
    if 'R_squared' in results:
        data = results['R_squared']
        analysis_df = data['analysis_df']
        
        # Ensure consistent ordering
        available_categories = [cat for cat in category_order if cat in analysis_df['Category'].unique()]
        
        # Boxplot of R^2 by category
        ax1 = axes[0]
        sns.boxplot(data=analysis_df, x='R_squared', y='Category', 
                   order=available_categories, ax=ax1, palette='viridis')
        ax1.set_xlabel('R² (coefficient of determination)', fontsize=12)
        ax1.set_ylabel('Hybrid Model Category', fontsize=12)
        ax1.set_xlim(0, 1.0)  # Set proper range for R²
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Histogram of R^2 values
        ax2 = axes[1]
        ax2.hist(analysis_df['R_squared'], bins=12, alpha=0.7, edgecolor='black', 
                color='steelblue', range=(0, 1))
        ax2.set_xlabel('R² (coefficient of determination)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_xlim(0, 1.0)  # Set proper range for R²
        ax2.grid(True, axis='both', alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # Add vertical line for median
        median_r2 = analysis_df['R_squared'].median()
        ax2.axvline(median_r2, color='red', linestyle='--', alpha=0.8, 
                   label=f'Median = {median_r2:.3f}')
        ax2.legend()
    
    plt.tight_layout()
    return fig

def create_summary_report(results, baseline_df):
    """
    Create summary report of the R^2 analysis.
    """
    print("="*80)
    print("R^2 ANALYSIS OF HYBRID MODELS")
    print("="*80)
    
    if 'R_squared' in results:
        data = results['R_squared']
        analysis_df = data['analysis_df']
        category_stats = data['category_stats']
        
        print(f"\nStatistics by hybridization category:")
        print(category_stats)
        
        # R^2 interpretation
        r2_values = analysis_df['R_squared']
        
        print(f"\nGeneral results:")
        print(f"  - Total models analyzed: {len(r2_values)}")
        print(f"  - Median R^2: {r2_values.median():.3f}")
        print(f"  - Average R^2: {r2_values.mean():.3f}")
        print(f"  - Best R^2: {r2_values.max():.3f}")
        print(f"  - Worst R^2: {r2_values.min():.3f}")
        
        # Statistical test
        kruskal_stat = data['kruskal_stat']
        kruskal_p = data['kruskal_p']
        
        if not pd.isna(kruskal_stat):
            print(f"\nKruskal-Wallis test:")
            print(f"  - Statistic: {kruskal_stat:.4f}")
            print(f"  - p-value: {kruskal_p:.4f}")
            print(f"  - Significant differences: {'Yes' if kruskal_p < 0.05 else 'No'}")

def create_templates():
    """
    Create templates for meta-analysis reproducibility.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # CSV template for data extraction
    template_data = {
        'Study_ID': ['Study_1', 'Study_2', 'Study_3'],
        'Hybrid_Model': ['Example_Hybrid_1', 'Example_Hybrid_2', 'Example_Hybrid_3'],
        'Baseline_Model': ['Simple_LSTM', 'ARIMA', 'Linear_Reg'],
        'Metric': ['RMSE', 'MAE', 'RMSE'],
        'Hybrid_Error': [15.2, 25.8, 42.1],
        'Baseline_Error': [20.5, 35.2, 58.7],
        'Hybrid_Category': ['Decomposition+DL', 'Optimization', 'Stacking/Ensemble'],
        'Dataset': ['Dataset_A', 'Dataset_B', 'Dataset_C'],
        'Time_Horizon': ['Monthly', 'Daily', 'Weekly']
    }
    
    template_df = pd.DataFrame(template_data)
    template_path = os.path.join(script_dir, "meta_analysis_template.csv")
    template_df.to_csv(template_path, index=False)
    
    print(f"CSV template created: {template_path}")
    
    # Example script for meta-analysis
    example_script = '''
# Example usage of meta-analysis
import pandas as pd
import numpy as np

def run_meta_analysis_example():
    # Load extracted data
    data = pd.read_csv("meta_analysis_template.csv")
    
    # Calculate log-ratios
    data['Log_Ratio'] = np.log(data['Hybrid_Error'] / data['Baseline_Error'])
    data['Improvement_Pct'] = (np.exp(data['Log_Ratio']) - 1) * 100
    
    # Group by category
    summary = data.groupby('Hybrid_Category')['Log_Ratio'].agg(['median', 'std', 'count'])
    print(summary)
    
    return data

if __name__ == "__main__":
    run_meta_analysis_example()
'''
    
    script_path = os.path.join(script_dir, "run_meta_analysis_example.py")
    with open(script_path, 'w') as f:
        f.write(example_script)
    
    print(f"Example script created: {script_path}")

def create_raw_comparison_plot(raw_df, script_dir):
    """
    Create R^2 comparison plot only (most available metric).
    """
    df = raw_df.copy()
    r2_pct = pd.to_numeric(df.get('R_squared', pd.Series([np.nan]*len(df))), errors='coerce') * 100.0
    r2_df = pd.DataFrame({'Model': df['Model'], 'Value': r2_pct}).dropna()

    if len(r2_df) == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    
    data = r2_df.sort_values('Value', ascending=False)
    vals = data['Value'].clip(lower=0, upper=100).values
    models = data['Model'].values
    norm = (vals - 0) / 100.0
    colors = plt.cm.Greens(norm)
    
    ax.barh(range(len(models)), vals, color=colors, height=0.22)
    ax.set_xlabel('R^2 (%)')
    ax.set_xlim(0, 100)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)

    plt.subplots_adjust(left=0.32, right=0.95, bottom=0.12, top=0.96)
    out = os.path.join(script_dir, 'comparison.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig

def create_metrics_availability_plot(raw_df, script_dir):
    """
    Create a plot showing the availability of different metrics across studies.
    """
    df = raw_df.copy()
    
    # Count availability of each metric
    metrics_count = {}
    
    # R^2
    metrics_count['R²'] = pd.to_numeric(df.get('R_squared', pd.Series([np.nan]*len(df))), errors='coerce').notna().sum()
    
    # RMSE (any form)
    rmse_cols = ['RMSE', 'NRMSE_std_pct', 'NRMSE_mean_pct', 'NRMSE_range_pct', 'RMSE_pct_of_baseline']
    rmse_available = df[[c for c in rmse_cols if c in df.columns]].notna().any(axis=1).sum() if any(c in df.columns for c in rmse_cols) else 0
    metrics_count['RMSE'] = rmse_available
    
    # MAE (any form)
    mae_cols = ['MAE', 'NMAE_mean_pct', 'NMAE_range_pct', 'MAE_pct_of_baseline']
    mae_available = df[[c for c in mae_cols if c in df.columns]].notna().any(axis=1).sum() if any(c in df.columns for c in mae_cols) else 0
    metrics_count['MAE'] = mae_available
    
    # MARE
    metrics_count['MARE'] = pd.to_numeric(df.get('MARE_pct', pd.Series([np.nan]*len(df))), errors='coerce').notna().sum()
    
    # NSE
    metrics_count['NSE'] = pd.to_numeric(df.get('NSE', pd.Series([np.nan]*len(df))), errors='coerce').notna().sum()
    
    # KGE
    metrics_count['KGE'] = pd.to_numeric(df.get('KGE', pd.Series([np.nan]*len(df))), errors='coerce').notna().sum()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = list(metrics_count.keys())
    counts = list(metrics_count.values())
    
    # Color scheme: green for most available, blue for moderate, red for least
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    
    bars = ax.bar(metrics, counts, color=colors[:len(metrics)])
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Number of Studies with Metric Available', fontsize=12)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_title('Availability of Performance Metrics Across Studies', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Set y-axis to start from 0
    ax.set_ylim(0, max(counts) * 1.1)
    
    plt.tight_layout()
    
    # Save the plot
    out_path = os.path.join(script_dir, 'metrics_availability.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Metrics availability plot saved: {out_path}")
    
    return fig

def main():
    """
    Main function that executes the R^2-focused analysis.
    """
    # Initial configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "detailed_metrics_analysis.csv")
    
    print("Starting R^2-focused hybrid models analysis...")
    
    # 1. Load and prepare data
    raw = pd.read_csv(csv_file)

    # Work only with R^2
    df = pd.DataFrame({
        'Model': raw['Model'],
        'R_squared': pd.to_numeric(raw.get('R_squared', pd.Series([np.nan]*len(raw))), errors='coerce')
    }).set_index('Model')

    # Keep only rows with R^2 present
    df = df[df['R_squared'].notna()]
    df = classify_hybrid_models(df)
    baseline_df = create_baseline_data()
    
    print(f"Loaded {len(df)} hybrid models with R^2 data")
    
    # 2. Perform R^2 analysis
    print("\n1. Performing R^2 analysis...")
    results = perform_meta_analysis(df, baseline_df)
    
    # 3. Create meta-analysis plots
    print("\n2. Creating meta-analysis plots...")
    meta_fig = create_meta_analysis_plots(results)
    meta_path = os.path.join(script_dir, "meta_analysis_plots.png")
    meta_fig.savefig(meta_path, dpi=300, bbox_inches='tight')
    print(f"Meta-analysis plots saved: {meta_path}")
    
    # 4. Create summary report
    print("\n3. Generating summary report...")
    create_summary_report(results, baseline_df)
    
    # 5. Create R^2 comparison plot
    print("\n4. Creating R^2 comparison plot...")
    create_raw_comparison_plot(raw, script_dir)
    
    # 6. Create metrics availability plot
    print("\n5. Creating metrics availability plot...")
    create_metrics_availability_plot(raw, script_dir)

    # 7. Save detailed results
    results_path = os.path.join(script_dir, "meta_analysis_results.xlsx")
    with pd.ExcelWriter(results_path) as writer:
        baseline_df.to_excel(writer, sheet_name='Baseline_Models')
        df.to_excel(writer, sheet_name='Hybrid_Models')
        if 'R_squared' in results:
            data = results['R_squared']
            data['analysis_df'].to_excel(writer, sheet_name='R2_Analysis', index=False)
            data['category_stats'].to_excel(writer, sheet_name='R2_Stats')
    
    print(f"Detailed results saved: {results_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)
    print("Generated files:")
    print(f"  - {meta_path}")
    print(f"  - {results_path}")
    print(f"  - {os.path.join(script_dir, 'comparison.png')}")
    print(f"  - {os.path.join(script_dir, 'metrics_availability.png')}")

if __name__ == "__main__":
    main()
