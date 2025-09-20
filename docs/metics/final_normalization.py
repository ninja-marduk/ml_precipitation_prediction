#!/usr/bin/env python3
"""
Final script to properly normalize precipitation prediction metrics.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os


def create_proper_csv():
    """Create a properly formatted CSV from the original data."""
    
    with open('models_avg_metrics.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_data = []
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            ref = parts[0].strip()
            metric = parts[1].strip()
            result = parts[2].replace(',', '.').strip()
            unity = parts[3].strip()
            model = parts[4].strip()
            processed_data.append({
                'Reference': ref,
                'Metric': metric,
                'Result': result,
                'Unit': unity,
                'Model': model
            })
    df = pd.DataFrame(processed_data)
    df['Result_Numeric'] = pd.to_numeric(df['Result'], errors='coerce')
    df.to_csv('processed_metrics.csv', index=False)
    return df


def apply_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Apply curated overrides/fixes based on study-level verification."""
    overrides = {
        'W-AL (Wavelet-ARIMA-LSTM)': {
            'RMSE': 31.307,
            'MAE': 22.853,
            'R^2': 0.712,
            'RMSE_pct_of_baseline': 86.71,
            'MAE_pct_of_baseline': 93.20,
        },
        'Stacking': {
            'RMSE (%)': 23.34,
            'MAE (%)': 19.40,
        },
        'ANN': {
            'RMSE (%)': 24.63,
            'MAE (%)': 21.02,
        },
        'BDTR': {
            'RMSE': 0.117037,
            'MAE': 0.064627,
            'R^2': 0.7921 ** 2,
            'RSE': 0.2079,
        },
        'AdaBoostRegressor Model': {
            'RMSE': 56.40,
            'MAE': 42.49,
        },
        # Monthly climate prediction using deep CNN and LSTM (predicting set)
        'CNN-LSTM': {
            'RMSE': 8.1762,
            'MAE': 6.7051,
            'R^2': 0.9924,
            'RMSE_pct_of_baseline': 48.56,
            'MAE_pct_of_baseline': 56.52,
        },
        # Wavelet-outlier robust extreme learning machine (WORELM)
        'WORELM 14': {
            'MARE (%)': 4.479,
        },
        # MLSTM-AM: average RAE ≈ 0.35 → 35%
        'MLSTM-AM': {
            'MARE (%)': 35.0,
        }
    }
    for model_name, metrics_map in overrides.items():
        for metric_name, value in metrics_map.items():
            mask = (df['Model'].str.strip() == model_name) & (df['Metric'].str.strip() == metric_name)
            if mask.any():
                df.loc[mask, 'Result_Numeric'] = float(value)
                df.loc[mask, 'Result'] = str(value)
                if metric_name == 'R^2':
                    df.loc[mask, 'Unit'] = ''
                elif metric_name.endswith('(%)') or metric_name.endswith('_pct_of_baseline'):
                    if df.loc[mask, 'Unit'].eq('').all():
                        df.loc[mask, 'Unit'] = '%'
            else:
                df = pd.concat([
                    df,
                    pd.DataFrame([{
                        'Reference': 'Override from verified study tables',
                        'Metric': metric_name,
                        'Result': str(value),
                        'Unit': '' if metric_name == 'R^2' else '%'
                        if (metric_name.endswith('(%)') or metric_name.endswith('_pct_of_baseline')) else 'mm',
                        'Model': model_name,
                        'Result_Numeric': float(value)
                    }])
                ], ignore_index=True)
    return df


def normalize_metrics(df):
    """Normalize metrics prioritizing NRMSE_std from RSE and adding preferred columns."""
    print("="*80)
    print("PRECIPITATION PREDICTION METRICS NORMALIZATION")
    print("="*80)
    models_data = []
    unique_models = df['Model'].unique()
    print(f"\nProcessing {len(unique_models)} unique models...")

    def is_mean(metric: str) -> bool:
        metric_l = (metric or '').strip().lower()
        return metric_l in {'mean', 'avg', 'average', 'mean precipitation', 'mean rainfall', 'μ'}

    def is_min(metric: str) -> bool:
        metric_l = (metric or '').strip().lower()
        return metric_l in {'min', 'minimum'}

    def is_max(metric: str) -> bool:
        metric_l = (metric or '').strip().lower()
        return metric_l in {'max', 'maximum'}

    for model_name in unique_models:
        if pd.isna(model_name) or not model_name.strip():
            continue
        model_data = df[df['Model'] == model_name].copy()
        metrics = {}
        reference = ""
        for _, row in model_data.iterrows():
            if pd.notna(row['Result_Numeric']):
                metrics[row['Metric']] = {'value': row['Result_Numeric'], 'unit': row['Unit']}
                if not reference:
                    reference = row['Reference']
        r2_val = None
        rse_val = None
        rmse_val = None
        mae_val = None
        rmse_unit = ""
        mae_unit = ""
        mean_val = None
        min_val = None
        max_val = None
        for metric in list(metrics.keys()):
            if metric in ['R^2', 'R²', 'R2']:
                r2_val = metrics[metric]['value']
            if metric == 'RSE':
                rse_val = metrics[metric]['value']
            if is_mean(metric):
                mean_val = metrics[metric]['value']
            if is_min(metric):
                min_val = metrics[metric]['value']
            if is_max(metric):
                max_val = metrics[metric]['value']
        if 'RMSE' in metrics:
            rmse_val = metrics['RMSE']['value']
            rmse_unit = metrics['RMSE']['unit']
        if 'MAE' in metrics:
            mae_val = metrics['MAE']['value']
            mae_unit = metrics['MAE']['unit']
        nrmse_std_pct = None
        if rse_val is not None and rse_val >= 0:
            nrmse_std_pct = float(np.sqrt(rse_val) * 100)
        norm_rmse = None
        norm_mae = None
        if r2_val is not None and 0 <= r2_val <= 1:
            normalized_error_pct = float(np.sqrt(1 - r2_val) * 100)
            if rmse_val is not None:
                norm_rmse = normalized_error_pct
            if mae_val is not None:
                norm_mae = normalized_error_pct
        nrmse_mean_pct = None
        nmae_mean_pct = None
        nrmse_range_pct = None
        nmae_range_pct = None
        mare_pct = None
        if 'RMSE (%)' in metrics:
            nrmse_mean_pct = metrics['RMSE (%)']['value']
        if 'MAE (%)' in metrics:
            nmae_mean_pct = metrics['MAE (%)']['value']
        if 'MARE (%)' in metrics:
            mare_pct = metrics['MARE (%)']['value']
        if mean_val is not None and mean_val > 0:
            if rmse_val is not None and nrmse_mean_pct is None:
                nrmse_mean_pct = (rmse_val / mean_val) * 100
            if mae_val is not None and nmae_mean_pct is None:
                nmae_mean_pct = (mae_val / mean_val) * 100
        if min_val is not None and max_val is not None and (max_val - min_val) > 0:
            denom = (max_val - min_val)
            if rmse_val is not None:
                nrmse_range_pct = (rmse_val / denom) * 100
            if mae_val is not None:
                nmae_range_pct = (mae_val / denom) * 100
        rmse_pct_of_baseline = metrics.get('RMSE_pct_of_baseline', {}).get('value') if 'RMSE_pct_of_baseline' in metrics else None
        mae_pct_of_baseline = metrics.get('MAE_pct_of_baseline', {}).get('value') if 'MAE_pct_of_baseline' in metrics else None
        rmse_pref = None
        rmse_source = None
        mae_pref = None
        mae_source = None
        if nrmse_std_pct is not None:
            rmse_pref = nrmse_std_pct; rmse_source = 'NRMSE_std'
        elif rmse_pct_of_baseline is not None:
            rmse_pref = rmse_pct_of_baseline; rmse_source = 'RMSE_pct_of_baseline'
        elif nrmse_mean_pct is not None:
            rmse_pref = nrmse_mean_pct; rmse_source = 'NRMSE_mean'
        elif nrmse_range_pct is not None:
            rmse_pref = nrmse_range_pct; rmse_source = 'NRMSE_range'
        if mae_pct_of_baseline is not None:
            mae_pref = mae_pct_of_baseline; mae_source = 'MAE_pct_of_baseline'
        elif nmae_mean_pct is not None:
            mae_pref = nmae_mean_pct; mae_source = 'NMAE_mean'
        elif nmae_range_pct is not None:
            mae_pref = nmae_range_pct; mae_source = 'NMAE_range'
        models_data.append({
            'Model': model_name.strip(),
            'Original_RMSE': rmse_val,
            'Original_MAE': mae_val,
            'R_squared': r2_val,
            'RSE': rse_val,
            'Normalized_RMSE_pct': norm_rmse,
            'Normalized_MAE_pct': norm_mae,
            'NRMSE_std_pct': nrmse_std_pct,
            'NRMSE_mean_pct': nrmse_mean_pct,
            'NMAE_mean_pct': nmae_mean_pct,
            'NRMSE_range_pct': nrmse_range_pct,
            'NMAE_range_pct': nmae_range_pct,
            'MARE_pct': mare_pct,
            'RMSE_pct_of_baseline': rmse_pct_of_baseline,
            'MAE_pct_of_baseline': mae_pct_of_baseline,
            'Error_RMSE_pct_preferred': rmse_pref,
            'Error_MAE_pct_preferred': mae_pref,
            'RMSE_source': rmse_source if rmse_source else '',
            'MAE_source': mae_source if mae_source else '',
            'RMSE_Unit': rmse_unit,
            'MAE_Unit': mae_unit,
            'Reference': reference[:100] + "..." if isinstance(reference, str) and len(reference) > 100 else reference,
            'All_Metrics': ', '.join(metrics.keys())
        })

    analysis_df = pd.DataFrame(models_data)

    # Baseline-relative calculations (unchanged)
    analysis_df['Improvement_RMSE_pct'] = np.nan
    analysis_df['Improvement_MAE_pct'] = np.nan
    for ref, grp in analysis_df.groupby('Reference'):
        base = grp[grp['Model'].str.contains('AdaBoostRegressor Model', case=False, na=False)]
        if len(base) == 0:
            continue
        base_row = base.iloc[0]
        base_rmse = base_row['Original_RMSE']
        base_mae = base_row['Original_MAE']
        idx_ref = analysis_df['Reference'].eq(ref)
        if pd.notna(base_rmse):
            rmse_vals = analysis_df.loc[idx_ref, 'Original_RMSE']
            pct = (rmse_vals / base_rmse) * 100.0
            analysis_df.loc[idx_ref & rmse_vals.notna(), 'RMSE_pct_of_baseline'] = pct[rmse_vals.notna()]
            analysis_df.loc[idx_ref & rmse_vals.notna(), 'Improvement_RMSE_pct'] = (base_rmse - rmse_vals[rmse_vals.notna()]) / base_rmse * 100.0
        if pd.notna(base_mae):
            mae_vals = analysis_df.loc[idx_ref, 'Original_MAE']
            pct = (mae_vals / base_mae) * 100.0
            analysis_df.loc[idx_ref & mae_vals.notna(), 'MAE_pct_of_baseline'] = pct[mae_vals.notna()]
            analysis_df.loc[idx_ref & mae_vals.notna(), 'Improvement_MAE_pct'] = (base_mae - mae_vals[mae_vals.notna()]) / base_mae * 100.0

    # Preferred fill + fallback by R2
    empty_rmse_pref = analysis_df['Error_RMSE_pct_preferred'].isna() | (analysis_df['Error_RMSE_pct_preferred'] == '')
    can_use_base_rmse = analysis_df['RMSE_pct_of_baseline'].notna()
    analysis_df.loc[empty_rmse_pref & can_use_base_rmse, 'Error_RMSE_pct_preferred'] = analysis_df.loc[empty_rmse_pref & can_use_base_rmse, 'RMSE_pct_of_baseline']
    analysis_df.loc[empty_rmse_pref & can_use_base_rmse, 'RMSE_source'] = 'RMSE_pct_of_baseline'

    empty_mae_pref = analysis_df['Error_MAE_pct_preferred'].isna() | (analysis_df['Error_MAE_pct_preferred'] == '')
    can_use_base_mae = analysis_df['MAE_pct_of_baseline'].notna()
    analysis_df.loc[empty_mae_pref & can_use_base_mae, 'Error_MAE_pct_preferred'] = analysis_df.loc[empty_mae_pref & can_use_base_mae, 'MAE_pct_of_baseline']
    analysis_df.loc[empty_mae_pref & can_use_base_mae, 'MAE_source'] = 'MAE_pct_of_baseline'

    still_empty_rmse = analysis_df['Error_RMSE_pct_preferred'].isna() | (analysis_df['Error_RMSE_pct_preferred'] == '')
    analysis_df.loc[still_empty_rmse & analysis_df['Normalized_RMSE_pct'].notna(), 'Error_RMSE_pct_preferred'] = analysis_df['Normalized_RMSE_pct']
    analysis_df.loc[still_empty_rmse & analysis_df['Normalized_RMSE_pct'].notna(), 'RMSE_source'] = 'R2_fallback'

    still_empty_mae = analysis_df['Error_MAE_pct_preferred'].isna() | (analysis_df['Error_MAE_pct_preferred'] == '')
    analysis_df.loc[still_empty_mae & analysis_df['Normalized_MAE_pct'].notna(), 'Error_MAE_pct_preferred'] = analysis_df['Normalized_MAE_pct']
    analysis_df.loc[still_empty_mae & analysis_df['Normalized_MAE_pct'].notna(), 'MAE_source'] = 'R2_fallback'

    analysis_df.to_csv('detailed_metrics_analysis.csv', index=False)

    final_df = analysis_df[['Model','Error_RMSE_pct_preferred','Error_MAE_pct_preferred','RMSE_source','MAE_source']].rename(columns={'Error_RMSE_pct_preferred':'RMSE','Error_MAE_pct_preferred':'MAE'})
    final_df.to_csv('hybrid_model_final_normalized.csv', index=False)

    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   Total models processed: {len(analysis_df)}")
    print(f"   Models with R² available: {analysis_df['R_squared'].notna().sum()}")
    print(f"   RSE available: {analysis_df['RSE'].notna().sum()}")
    print(f"   Preferred RMSE available: {final_df['RMSE'].notna().sum()}")
    print(f"   Preferred MAE available: {final_df['MAE'].notna().sum()}")

    print("\n" + "="*80)
    print("✅ NORMALIZATION COMPLETED!")
    print("="*80)

    return final_df, [], analysis_df


def main():
    print("Starting comprehensive metrics normalization...")
    df = create_proper_csv()
    print(f"✅ Processed {len(df)} records from {len(df['Model'].unique())} models")
    df = apply_overrides(df)
    final_df, _, analysis_df = normalize_metrics(df)
    print(f"\n🎯 NEXT STEPS:\n   1. Review 'detailed_metrics_analysis.csv'\n   2. Use 'hybrid_model_final_normalized.csv' for visualizations")
    return final_df

if __name__ == "__main__":
    main()
