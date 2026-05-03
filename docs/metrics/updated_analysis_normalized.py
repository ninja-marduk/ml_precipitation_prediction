#!/usr/bin/env python3
"""
Updated analysis script using normalized precipitation prediction metrics.
All metrics are now consistently scaled (normalized percentages when R² available).
"""

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
    Using normalized error percentages for consistency.
    """
    baseline_data = {
        'Model': ['Linear Regression', 'ARIMA', 'Simple LSTM', 'Basic CNN', 'SVM', 
                 'Random Forest (Simple)', 'Persistence Model', 'Moving Average'],
        'MAE': [85.2, 78.3, 65.1, 72.7, 81.8, 62.2, 95.5, 89.1],  # Normalized percentages
        'RMSE': [89.4, 76.8, 68.3, 75.9, 85.1, 65.7, 98.2, 91.6],  # Normalized percentages
        'Type': ['baseline'] * 8
    }
    return pd.DataFrame(baseline_data)

def classify_hybrid_models(df):
    """
    Classify hybrid models into categories for meta-analysis.
    """
    # Define classifications based on hybridization techniques
    decomposition_dl = ['CEEMD-ELM-FFOA', 'CEEMD-FCMSE-Stacking', 'CEEMDAN-SVM-LSTM', 
                       'LSTM & EMD-ELM', 'TVF-EMD-ENN', 'VMD-CPO-LSTM', 'WT-ELM']
    
    optimization = ['BBO-ELM', 'Bat-ELM', 'GA', 'SARIMA-ANN', 'SSA-LSSVR (Deji)', 'SSA-RF (Shihmen)']
    
    stacking_ensemble = ['CEEMD-FCMSE-Stacking', 'Stacking', 'EEMD-BMA', 'Satck-Las-Rsc']
    
    wavelet_based = ['W-AL (Wavelet-ARIMA-LSTM)', 'Wavelet-RF', 'WT-ELM']
    
    deep_hybrid = ['CNN-LSTM', 'BLSTM-GRU', 'LSTM & EMD-ELM']
    
    # Assign categories
    categories = []
    for model in df.index:
        if any(keyword in model for keyword in decomposition_dl):
            categories.append('Decomposition+DL')
        elif any(keyword in model for keyword in optimization):
            categories.append('Optimization')
        elif any(keyword in model for keyword in stacking_ensemble):
            categories.append('Stacking/Ensemble')
        elif any(keyword in model for keyword in wavelet_based):
            categories.append('Wavelet-based')
        elif any(keyword in model for keyword in deep_hybrid):
            categories.append('Deep Hybrid')
        else:
            categories.append('Other Hybrid')
    
    df['Category'] = categories
    return df

def calculate_log_ratios(hybrid_df, baseline_df, metric='RMSE'):
    """
    Calculate log-ratios for meta-analysis: Δ = ln(E_hyb/E_base)
    Now using normalized percentages for consistent comparison.
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
    Create logarithmic plot with normalized data - all values now in consistent scale.
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(22, 8))
    
    # Prepare RMSE data
    df_rmse = df.dropna(subset=['RMSE']).sort_values('RMSE', ascending=True)
    rmse_values = df_rmse['RMSE']
    rmse_models = df_rmse.index
    
    # Prepare MAE data  
    df_mae = df.dropna(subset=['MAE']).sort_values('MAE', ascending=True)
    mae_values = df_mae['MAE']
    mae_models = df_mae.index
    
    # Create color map based on logarithmic values
    rmse_log_norm = np.log10(rmse_values)
    rmse_colors = plt.cm.RdYlBu_r((rmse_log_norm - rmse_log_norm.min()) / 
                                  (rmse_log_norm.max() - rmse_log_norm.min()))
    
    mae_log_norm = np.log10(mae_values)
    mae_colors = plt.cm.RdYlBu_r((mae_log_norm - mae_log_norm.min()) / 
                                 (mae_log_norm.max() - mae_log_norm.min()))
    
    # RMSE plot with logarithmic scale
    bars1 = ax1.barh(range(len(rmse_models)), rmse_values, color=rmse_colors, height=0.3)
    ax1.set_xscale('log')
    ax1.set_xlabel('Log₁₀(Normalized RMSE %)', fontsize=12)
    ax1.set_ylabel('Hybrid Model', fontsize=12)
    ax1.set_title('Normalized RMSE by Hybrid Model', fontsize=14, fontweight='bold', pad=20)
    ax1.set_yticks(range(len(rmse_models)))
    ax1.set_yticklabels(rmse_models, fontsize=7)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(10**-1, 10**2)
    
    # MAE plot with logarithmic scale
    bars2 = ax2.barh(range(len(mae_models)), mae_values, color=mae_colors, height=0.3)
    ax2.set_xscale('log')
    ax2.set_xlabel('Log₁₀(Normalized MAE %)', fontsize=12)
    ax2.set_title('Normalized MAE by Hybrid Model', fontsize=14, fontweight='bold', pad=20)
    ax2.set_yticks(range(len(mae_models)))
    ax2.set_yticklabels(mae_models, fontsize=7)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(10**-1, 10**2)
    
    # Adjust layout to prevent overlapping and make room for colorbar
    plt.subplots_adjust(left=0.35, bottom=0.12, right=0.82, top=0.88, wspace=0.35, hspace=0.05)
    
    # Add colorbar positioned to the right of both plots
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalized Error Magnitude (%)', rotation=270, labelpad=20)
    
    return fig

def perform_meta_analysis(hybrid_df, baseline_df):
    """
    Perform quantitative meta-analysis with normalized log-ratios and statistical tests.
    """
    results = {}
    
    for metric in ['RMSE', 'MAE']:
        if metric in hybrid_df.columns:
            # Calculate log-ratios
            log_ratios, improvements, models_list = calculate_log_ratios(hybrid_df, baseline_df, metric)
            
            if len(log_ratios) > 0:
                # Create DataFrame for analysis
                analysis_df = pd.DataFrame({
                    'Model': models_list,
                    'Log_Ratio': log_ratios,
                    'Improvement_Pct': improvements,
                    'Category': [hybrid_df.loc[model, 'Category'] for model in models_list]
                })
                
                # Statistics by category
                category_stats = analysis_df.groupby('Category')['Log_Ratio'].agg([
                    'count', 'median', 'std', 
                    lambda x: np.percentile(x, 25),  # Q1
                    lambda x: np.percentile(x, 75)   # Q3
                ]).round(4)
                category_stats.columns = ['Count', 'Median', 'Std', 'Q1', 'Q3']
                
                # Kruskal-Wallis test
                categories = analysis_df['Category'].unique()
                if len(categories) > 2:
                    groups = [analysis_df[analysis_df['Category'] == cat]['Log_Ratio'].values 
                             for cat in categories if len(analysis_df[analysis_df['Category'] == cat]) > 0]
                    if len(groups) > 2:
                        kruskal_stat, kruskal_p = kruskal(*groups)
                    else:
                        kruskal_stat, kruskal_p = np.nan, np.nan
                else:
                    kruskal_stat, kruskal_p = np.nan, np.nan
                
                results[metric] = {
                    'analysis_df': analysis_df,
                    'category_stats': category_stats,
                    'kruskal_stat': kruskal_stat,
                    'kruskal_p': kruskal_p
                }
    
    return results

def create_meta_analysis_plots(results):
    """
    Create meta-analysis visualizations with consistent category ordering.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define consistent category order based on effectiveness
    category_order = ['Deep Hybrid', 'Decomposition+DL', 'Stacking/Ensemble', 
                     'Wavelet-based', 'Other Hybrid', 'Optimization']
    
    for i, (metric, data) in enumerate(results.items()):
        # Boxplot of log-ratios by category
        ax1 = axes[i, 0]
        analysis_df = data['analysis_df']
        
        # Ensure consistent ordering across metrics
        available_categories = [cat for cat in category_order if cat in analysis_df['Category'].unique()]
        
        sns.boxplot(data=analysis_df, x='Log_Ratio', y='Category', 
                   order=available_categories, ax=ax1)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No improvement')
        ax1.set_xlabel(f'Log-Ratio Δ (Normalized {metric} %)')
        ax1.set_title(f'Effect Size Distribution by Category - {metric}')
        ax1.legend()
        
        # Histogram of percentage improvements
        ax2 = axes[i, 1]
        ax2.hist(analysis_df['Improvement_Pct'], bins=15, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No improvement')
        ax2.set_xlabel(f'Improvement Percentage (Normalized {metric} %)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Distribution of Improvements - {metric}')
        ax2.legend()
    
    plt.tight_layout()
    return fig

def create_summary_report(results, baseline_df):
    """
    Create summary report of the normalized meta-analysis.
    """
    print("="*80)
    print("QUANTITATIVE META-ANALYSIS OF HYBRID MODELS (NORMALIZED DATA)")
    print("="*80)
    
    baseline_stats = baseline_df[['RMSE', 'MAE']].describe()
    print("\n1. BASELINE MODELS STATISTICS (Normalized %):")
    print(baseline_stats.round(3))
    
    for metric, data in results.items():
        print(f"\n2. LOG-RATIO ANALYSIS FOR NORMALIZED {metric} (%):")
        print("-" * 60)
        
        analysis_df = data['analysis_df']
        category_stats = data['category_stats']
        
        print(f"\nStatistics by hybridization category:")
        print(category_stats)
        
        # Improvement interpretation
        improvements = analysis_df['Improvement_Pct']
        better_models = (improvements < 0).sum()
        worse_models = (improvements > 0).sum()
        
        print(f"\nGeneral results:")
        print(f"  - Models improving baseline: {better_models}/{len(improvements)} ({100*better_models/len(improvements):.1f}%)")
        print(f"  - Models worsening baseline: {worse_models}/{len(improvements)} ({100*worse_models/len(improvements):.1f}%)")
        print(f"  - Median improvement: {improvements.median():.1f}%")
        print(f"  - Average improvement: {improvements.mean():.1f}%")
        
        # Statistical test
        kruskal_stat = data['kruskal_stat']
        kruskal_p = data['kruskal_p']
        
        if not pd.isna(kruskal_stat):
            print(f"\nKruskal-Wallis test:")
            print(f"  - Statistic: {kruskal_stat:.4f}")
            print(f"  - p-value: {kruskal_p:.4f}")
            print(f"  - Significant differences: {'Yes' if kruskal_p < 0.05 else 'No'}")

def main():
    """
    Main function that executes the complete normalized analysis.
    """
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "hybrid_model_final_normalized.csv")
    
    print("="*80)
    print("STARTING NORMALIZED HYBRID MODELS ANALYSIS")
    print("="*80)
    print("[INFO] Using normalized data where R2 was available")
    print("[INFO] All metrics now consistently scaled")

    # 1. Load and prepare normalized data
    df = pd.read_csv(csv_file, index_col=0)
    df = classify_hybrid_models(df)
    baseline_df = create_baseline_data()

    print(f"\n[OK] Loaded {len(df)} normalized hybrid models and {len(baseline_df)} baseline models")

    # 2. Create normalized logarithmic plot
    print("\n1. Creating normalized logarithmic plot...")
    log_fig = create_logarithmic_plot(df)
    log_path = os.path.join(script_dir, "normalized_logarithmic_comparison.png")
    log_fig.savefig(log_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Normalized logarithmic plot saved: {log_path}")

    # 3. Perform meta-analysis with normalized data
    print("\n2. Performing normalized meta-analysis...")
    results = perform_meta_analysis(df, baseline_df)

    # 4. Create meta-analysis plots
    print("\n3. Creating normalized meta-analysis plots...")
    meta_fig = create_meta_analysis_plots(results)
    meta_path = os.path.join(script_dir, "normalized_meta_analysis_plots.png")
    meta_fig.savefig(meta_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Normalized meta-analysis plots saved: {meta_path}")

    # 5. Create summary report
    print("\n4. Generating normalized summary report...")
    create_summary_report(results, baseline_df)

    # 6. Save normalized results
    results_path = os.path.join(script_dir, "normalized_meta_analysis_results.xlsx")
    with pd.ExcelWriter(results_path) as writer:
        baseline_df.to_excel(writer, sheet_name='Baseline_Models')
        df.to_excel(writer, sheet_name='Normalized_Hybrid_Models')
        for metric, data in results.items():
            data['analysis_df'].to_excel(writer, sheet_name=f'{metric}_Analysis', index=False)
            data['category_stats'].to_excel(writer, sheet_name=f'{metric}_Stats')

    print(f"\n[SAVED] Normalized results saved: {results_path}")

    # Show plots
    plt.show()

    print("\n" + "="*80)
    print("[OK] NORMALIZED ANALYSIS COMPLETED")
    print("="*80)
    print("[FILES] Generated files:")
    print(f"  - {log_path}")
    print(f"  - {meta_path}")
    print(f"  - {results_path}")
    print("\n[INFO] All metrics are now consistently normalized!")

if __name__ == "__main__":
    main()
