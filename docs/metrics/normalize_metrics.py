#!/usr/bin/env python3
"""
Script to normalize precipitation prediction metrics and identify inconsistencies.
Converts all RMSE and MAE to normalized percentages (0-100%) when R² is available.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_numeric_value(value):
    """Clean and convert string values to numeric."""
    if pd.isna(value) or value == '':
        return np.nan
    
    # Replace comma with dot for decimal separator
    if isinstance(value, str):
        value = value.replace(',', '.')
        # Remove percentage signs and other characters
        value = value.replace('%', '').strip()
    
    try:
        return float(value)
    except:
        return np.nan

def calculate_normalized_error(rmse_or_mae, r_squared):
    """
    Calculate normalized error percentage from RMSE/MAE and R².
    Formula: Normalized Error (%) = sqrt(1 - R²) * 100
    """
    if pd.isna(r_squared) or pd.isna(rmse_or_mae):
        return np.nan
    
    if r_squared < 0 or r_squared > 1:
        return np.nan
    
    # Normalized error as percentage
    normalized_error = np.sqrt(1 - r_squared) * 100
    return normalized_error

def analyze_and_normalize_data():
    """Main function to analyze and normalize the metrics data."""
    
    print("="*80)
    print("ANALYSIS AND NORMALIZATION OF PRECIPITATION PREDICTION METRICS")
    print("="*80)
    
    # Load the cleaned CSV file
    df = pd.read_csv('models_avg_metrics_clean.csv')
    
    # Clean numeric values
    df['Result_Clean'] = df['Result'].apply(clean_numeric_value)
    
    # Get unique models
    models = df['Model'].unique()
    
    print(f"\nTotal models found: {len(models)}")
    print(f"Total records: {len(df)}")
    
    # Analyze data by model
    normalized_data = []
    issues_to_verify = []
    
    for model in models:
        model_data = df[df['Model'] == model].copy()
        
        # Get available metrics for this model
        metrics = model_data['Metric'].values
        results = model_data['Result_Clean'].values
        units = model_data['Unity'].values
        refs = model_data['Ref'].values
        
        # Create metric dictionary
        metric_dict = {}
        for i, metric in enumerate(metrics):
            if not pd.isna(results[i]):
                metric_dict[metric] = {
                    'value': results[i],
                    'unit': units[i] if not pd.isna(units[i]) else '',
                    'ref': refs[i]
                }
        
        # Check if we have R² and RMSE/MAE
        r2_value = None
        rmse_value = None
        mae_value = None
        
        # Look for R² (various formats)
        for key in metric_dict.keys():
            if isinstance(key, str) and ('R^2' in key or 'R²' in key or key == 'R2'):
                r2_value = metric_dict[key]['value']
                break
        
        # Look for RMSE
        if 'RMSE' in metric_dict:
            rmse_value = metric_dict['RMSE']['value']
            rmse_unit = metric_dict['RMSE']['unit']
        
        # Look for MAE
        if 'MAE' in metric_dict:
            mae_value = metric_dict['MAE']['value']
            mae_unit = metric_dict['MAE']['unit']
        
        # Normalize if possible
        normalized_rmse = None
        normalized_mae = None
        
        if r2_value is not None:
            if rmse_value is not None:
                normalized_rmse = calculate_normalized_error(rmse_value, r2_value)
            if mae_value is not None:
                normalized_mae = calculate_normalized_error(mae_value, r2_value)
        
        # Determine if we need to verify this model
        needs_verification = False
        verification_reason = []
        
        # Check for inconsistencies
        if rmse_value is not None and r2_value is None:
            needs_verification = True
            verification_reason.append("Missing R² for normalization")
        
        if mae_value is not None and r2_value is None:
            needs_verification = True
            verification_reason.append("Missing R² for normalization")
        
        # Check for suspicious values
        if rmse_value is not None:
            if rmse_value > 1000:  # Very large RMSE values
                needs_verification = True
                verification_reason.append(f"Unusually large RMSE: {rmse_value}")
            elif rmse_value < 0.001:  # Very small RMSE values
                needs_verification = True
                verification_reason.append(f"Unusually small RMSE: {rmse_value}")
        
        if r2_value is not None and (r2_value < 0 or r2_value > 1):
            needs_verification = True
            verification_reason.append(f"Invalid R² value: {r2_value}")
        
        # Store results
        result_entry = {
            'Model': model,
            'Original_RMSE': rmse_value,
            'Original_MAE': mae_value,
            'R_squared': r2_value,
            'Normalized_RMSE_pct': normalized_rmse,
            'Normalized_MAE_pct': normalized_mae,
            'RMSE_Unit': rmse_unit if rmse_value is not None else '',
            'MAE_Unit': mae_unit if mae_value is not None else '',
            'Reference': refs[0] if len(refs) > 0 else '',
            'Needs_Verification': needs_verification,
            'Verification_Reason': '; '.join(verification_reason)
        }
        
        normalized_data.append(result_entry)
        
        if needs_verification:
            issues_to_verify.append(result_entry)
    
    # Create normalized dataframe
    normalized_df = pd.DataFrame(normalized_data)
    
    # Save normalized data
    normalized_df.to_csv('normalized_metrics.csv', index=False)
    
    # Create the final hybrid model file with normalized values
    final_data = []
    for _, row in normalized_df.iterrows():
        if not pd.isna(row['Normalized_RMSE_pct']) or not pd.isna(row['Normalized_MAE_pct']):
            final_data.append({
                'Model': row['Model'],
                'MAE': row['Normalized_MAE_pct'],
                'RMSE': row['Normalized_RMSE_pct']
            })
        elif not pd.isna(row['Original_RMSE']) or not pd.isna(row['Original_MAE']):
            # Keep original values if normalization not possible
            final_data.append({
                'Model': row['Model'],
                'MAE': row['Original_MAE'],
                'RMSE': row['Original_RMSE']
            })
    
    final_df = pd.DataFrame(final_data)
    final_df.to_csv('hybrid_model_normalized.csv', index=False)
    
    # Print analysis results
    print(f"\n1. NORMALIZATION RESULTS:")
    print(f"   - Models with R² available: {normalized_df['R_squared'].notna().sum()}")
    print(f"   - Models normalized for RMSE: {normalized_df['Normalized_RMSE_pct'].notna().sum()}")
    print(f"   - Models normalized for MAE: {normalized_df['Normalized_MAE_pct'].notna().sum()}")
    
    print(f"\n2. MODELS REQUIRING VERIFICATION:")
    print(f"   - Total models needing verification: {len(issues_to_verify)}")
    
    if issues_to_verify:
        print("\n   Detailed issues:")
        for i, issue in enumerate(issues_to_verify, 1):
            print(f"   {i:2d}. {issue['Model']}")
            print(f"       Reason: {issue['Verification_Reason']}")
            print(f"       Reference: {issue['Reference'][:80]}...")
            print()
    
    # Print unit analysis
    print(f"\n3. UNIT ANALYSIS:")
    rmse_units = normalized_df[normalized_df['RMSE_Unit'] != '']['RMSE_Unit'].value_counts()
    mae_units = normalized_df[normalized_df['MAE_Unit'] != '']['MAE_Unit'].value_counts()
    
    print("   RMSE Units found:")
    for unit, count in rmse_units.items():
        print(f"     - {unit}: {count} models")
    
    print("   MAE Units found:")
    for unit, count in mae_units.items():
        print(f"     - {unit}: {count} models")
    
    # Print summary statistics
    print(f"\n4. SUMMARY STATISTICS:")
    print("   Original RMSE values:")
    print(f"     - Min: {normalized_df['Original_RMSE'].min():.4f}")
    print(f"     - Max: {normalized_df['Original_RMSE'].max():.4f}")
    print(f"     - Mean: {normalized_df['Original_RMSE'].mean():.4f}")
    
    print("   Normalized RMSE (%):")
    print(f"     - Min: {normalized_df['Normalized_RMSE_pct'].min():.2f}%")
    print(f"     - Max: {normalized_df['Normalized_RMSE_pct'].max():.2f}%")
    print(f"     - Mean: {normalized_df['Normalized_RMSE_pct'].mean():.2f}%")
    
    print("\n5. FILES CREATED:")
    print("   - normalized_metrics.csv: Complete analysis with all details")
    print("   - hybrid_model_normalized.csv: Final normalized data for plotting")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    analyze_and_normalize_data()
