#!/usr/bin/env python3
"""
Script to fix CSV format issues with decimal commas and create proper normalized metrics.
"""

import pandas as pd
import numpy as np
import re

def fix_csv_format():
    """Fix the CSV format by handling decimal commas properly."""
    
    with open('models_avg_metrics.csv', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines
    lines = content.strip().split('\n')
    
    fixed_lines = []
    for line in lines:
        # Split by tabs first
        parts = line.split('\t')
        
        if len(parts) >= 5:  # Should have 5 columns: Ref, Metric, Result, Unity, Model
            ref = parts[0]
            metric = parts[1] 
            result = parts[2]
            unity = parts[3] if len(parts) > 3 else ''
            model = parts[4] if len(parts) > 4 else ''
            
            # Fix decimal comma in result
            result = result.replace(',', '.')
            
            # Create proper CSV line
            fixed_line = f'"{ref}","{metric}","{result}","{unity}","{model}"'
            fixed_lines.append(fixed_line)
    
    # Write fixed CSV
    with open('models_avg_metrics_fixed.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print(f"Fixed CSV created with {len(fixed_lines)} lines")
    return 'models_avg_metrics_fixed.csv'

def analyze_normalized_data():
    """Analyze and normalize the fixed data."""
    
    print("="*80)
    print("COMPREHENSIVE METRICS ANALYSIS AND NORMALIZATION")
    print("="*80)
    
    # Fix CSV format first
    csv_file = fix_csv_format()
    
    # Load the fixed CSV
    df = pd.read_csv(csv_file)
    
    print(f"\nLoaded data: {len(df)} records")
    print(f"Unique models: {len(df['Model'].unique())}")
    
    # Clean and convert Result column to numeric
    def clean_numeric(val):
        if pd.isna(val) or val == '':
            return np.nan
        try:
            # Remove percentage signs and convert
            val_str = str(val).replace('%', '').strip()
            return float(val_str)
        except:
            return np.nan
    
    df['Result_Numeric'] = df['Result'].apply(clean_numeric)
    
    # Group by model to analyze
    models_analysis = []
    verification_needed = []
    
    for model_name in df['Model'].unique():
        if pd.isna(model_name) or model_name.strip() == '':
            continue
            
        model_data = df[df['Model'] == model_name].copy()
        
        # Extract metrics for this model
        metrics = {}
        for _, row in model_data.iterrows():
            metric_name = row['Metric']
            if pd.notna(metric_name) and pd.notna(row['Result_Numeric']):
                metrics[metric_name] = {
                    'value': row['Result_Numeric'],
                    'unit': row['Unity'] if pd.notna(row['Unity']) else '',
                    'ref': row['Ref']
                }
        
        # Look for key metrics
        r2_value = None
        rmse_value = None
        mae_value = None
        rmse_unit = ''
        mae_unit = ''
        
        # Find R²
        for key in metrics.keys():
            if 'R^2' in str(key) or 'R²' in str(key) or str(key).strip() == 'R2':
                r2_value = metrics[key]['value']
                break
        
        # Find RMSE
        if 'RMSE' in metrics:
            rmse_value = metrics['RMSE']['value']
            rmse_unit = metrics['RMSE']['unit']
        
        # Find MAE
        if 'MAE' in metrics:
            mae_value = metrics['MAE']['value']
            mae_unit = metrics['MAE']['unit']
        
        # Calculate normalized errors if R² is available
        normalized_rmse = None
        normalized_mae = None
        
        if r2_value is not None and 0 <= r2_value <= 1:
            if rmse_value is not None:
                # Convert to normalized percentage: sqrt(1 - R²) * 100
                normalized_rmse = np.sqrt(1 - r2_value) * 100
            if mae_value is not None:
                normalized_mae = np.sqrt(1 - r2_value) * 100
        
        # Determine if verification is needed
        needs_verification = False
        reasons = []
        
        if (rmse_value is not None or mae_value is not None) and r2_value is None:
            needs_verification = True
            reasons.append("Missing R² for normalization")
        
        if r2_value is not None and (r2_value < 0 or r2_value > 1):
            needs_verification = True
            reasons.append(f"Invalid R² value: {r2_value}")
        
        # Check for suspicious RMSE/MAE values
        if rmse_value is not None:
            if rmse_value > 500:
                needs_verification = True
                reasons.append(f"Unusually large RMSE: {rmse_value} {rmse_unit}")
            elif rmse_value < 0.001 and rmse_unit == 'mm':
                needs_verification = True
                reasons.append(f"Unusually small RMSE: {rmse_value} {rmse_unit}")
        
        # Store analysis
        analysis = {
            'Model': model_name.strip(),
            'Original_RMSE': rmse_value,
            'Original_MAE': mae_value,
            'R_squared': r2_value,
            'Normalized_RMSE_pct': normalized_rmse,
            'Normalized_MAE_pct': normalized_mae,
            'RMSE_Unit': rmse_unit,
            'MAE_Unit': mae_unit,
            'Reference': model_data.iloc[0]['Ref'] if len(model_data) > 0 else '',
            'Available_Metrics': list(metrics.keys()),
            'Needs_Verification': needs_verification,
            'Verification_Reasons': '; '.join(reasons)
        }
        
        models_analysis.append(analysis)
        
        if needs_verification:
            verification_needed.append(analysis)
    
    # Create results DataFrame
    results_df = pd.DataFrame(models_analysis)
    results_df.to_csv('comprehensive_metrics_analysis.csv', index=False)
    
    # Create final normalized dataset for plotting
    final_data = []
    for _, row in results_df.iterrows():
        entry = {'Model': row['Model']}
        
        # Use normalized values if available, otherwise original
        if pd.notna(row['Normalized_RMSE_pct']):
            entry['RMSE'] = row['Normalized_RMSE_pct']
        elif pd.notna(row['Original_RMSE']):
            entry['RMSE'] = row['Original_RMSE']
        else:
            entry['RMSE'] = np.nan
            
        if pd.notna(row['Normalized_MAE_pct']):
            entry['MAE'] = row['Normalized_MAE_pct']
        elif pd.notna(row['Original_MAE']):
            entry['MAE'] = row['Original_MAE']
        else:
            entry['MAE'] = np.nan
        
        final_data.append(entry)
    
    final_df = pd.DataFrame(final_data)
    # Remove rows where both RMSE and MAE are NaN
    final_df = final_df.dropna(subset=['RMSE', 'MAE'], how='all')
    final_df.to_csv('hybrid_model_normalized_final.csv', index=False)
    
    # Print comprehensive analysis
    print(f"\n1. MODELS ANALYZED: {len(models_analysis)}")
    
    print(f"\n2. NORMALIZATION SUCCESS:")
    normalized_rmse = results_df['Normalized_RMSE_pct'].notna().sum()
    normalized_mae = results_df['Normalized_MAE_pct'].notna().sum()
    print(f"   - Models with normalized RMSE: {normalized_rmse}")
    print(f"   - Models with normalized MAE: {normalized_mae}")
    print(f"   - Models with R² available: {results_df['R_squared'].notna().sum()}")
    
    print(f"\n3. VERIFICATION REQUIRED: {len(verification_needed)} models")
    if verification_needed:
        print("\n   Models needing verification:")
        for i, model in enumerate(verification_needed, 1):
            print(f"   {i:2d}. {model['Model']}")
            print(f"       Issues: {model['Verification_Reasons']}")
            print(f"       Paper: {model['Reference'][:60]}...")
            print()
    
    print(f"\n4. UNIT DISTRIBUTION:")
    rmse_units = results_df[results_df['RMSE_Unit'] != '']['RMSE_Unit'].value_counts()
    mae_units = results_df[results_df['MAE_Unit'] != '']['MAE_Unit'].value_counts()
    
    print("   RMSE Units:")
    for unit, count in rmse_units.items():
        print(f"     - {unit}: {count} models")
    
    print("   MAE Units:")
    for unit, count in mae_units.items():
        print(f"     - {unit}: {count} models")
    
    print(f"\n5. FILES CREATED:")
    print("   - comprehensive_metrics_analysis.csv: Complete detailed analysis")
    print("   - hybrid_model_normalized_final.csv: Final dataset for plotting")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED - READY FOR PLOTTING")
    print("="*80)
    
    return final_df, verification_needed

if __name__ == "__main__":
    analyze_normalized_data()
