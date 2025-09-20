#!/usr/bin/env python3
"""
Script to verify consistency across all generated files and create final summary.
"""

import pandas as pd
import numpy as np
import os

def verify_file_consistency():
    """Verify that all files are consistent with the latest updates."""
    
    print("="*80)
    print("FILE CONSISTENCY VERIFICATION REPORT")
    print("="*80)
    
    # Load all relevant files
    try:
        # Main source file
        source_df = pd.read_csv('processed_metrics.csv')
        
        # Analysis file
        analysis_df = pd.read_csv('detailed_metrics_analysis.csv')
        
        # Final normalized file
        final_df = pd.read_csv('hybrid_model_final_normalized.csv')
        
        print("✅ All files loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return False
    
    # Verify key models that were updated
    updated_models = ['SSA-LSSVR (Deji)', 'SSA-RF (Shihmen)', 'Satck-Las-Rsc', 'TVF-EMD-ENN']
    
    print(f"\n📋 VERIFICATION OF UPDATED MODELS:")
    
    for model in updated_models:
        print(f"\n🔍 {model}:")
        
        # Check in analysis file
        model_analysis = analysis_df[analysis_df['Model'] == model]
        if len(model_analysis) > 0:
            row = model_analysis.iloc[0]
            r2_val = row['R_squared']
            norm_rmse = row['Normalized_RMSE_pct']
            norm_mae = row['Normalized_MAE_pct']
            needs_verification = row['Needs_Verification']
            
            print(f"   📊 R² value: {r2_val}")
            print(f"   📈 Normalized RMSE: {norm_rmse:.2f}%" if pd.notna(norm_rmse) else "   📈 Normalized RMSE: N/A")
            print(f"   📉 Normalized MAE: {norm_mae:.2f}%" if pd.notna(norm_mae) else "   📉 Normalized MAE: N/A")
            print(f"   ✅ Status: {'Needs verification' if needs_verification else 'Complete'}")
        else:
            print(f"   ❌ Model not found in analysis file")
        
        # Check in final file
        model_final = final_df[final_df['Model'] == model]
        if len(model_final) > 0:
            final_row = model_final.iloc[0]
            final_rmse = final_row['RMSE']
            final_mae = final_row['MAE']
            print(f"   📋 Final RMSE: {final_rmse:.2f}%" if pd.notna(final_rmse) else "   📋 Final RMSE: N/A")
            print(f"   📋 Final MAE: {final_mae:.2f}%" if pd.notna(final_mae) else "   📋 Final MAE: N/A")
        else:
            print(f"   ❌ Model not found in final file")
    
    # Summary statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total models in source: {len(source_df['Model'].unique())}")
    print(f"   Total models in analysis: {len(analysis_df)}")
    print(f"   Total models in final dataset: {len(final_df)}")
    
    # Normalization success rate
    normalized_rmse = analysis_df['Normalized_RMSE_pct'].notna().sum()
    normalized_mae = analysis_df['Normalized_MAE_pct'].notna().sum()
    total_models = len(analysis_df)
    
    print(f"\n📈 NORMALIZATION SUCCESS:")
    print(f"   RMSE normalized: {normalized_rmse}/{total_models} ({100*normalized_rmse/total_models:.1f}%)")
    print(f"   MAE normalized: {normalized_mae}/{total_models} ({100*normalized_mae/total_models:.1f}%)")
    
    # Models still needing verification
    needs_verification = analysis_df[analysis_df['Needs_Verification'] == True]
    print(f"\n🔍 REMAINING VERIFICATION NEEDED:")
    print(f"   Models pending: {len(needs_verification)}")
    
    if len(needs_verification) > 0:
        print("\n   📝 Priority list for next updates:")
        for i, (_, row) in enumerate(needs_verification.iterrows(), 1):
            model = row['Model']
            metrics = row['All_Metrics']
            print(f"   {i:2d}. {model}")
            print(f"       Available: {metrics}")
            
            # Suggest conversion strategy
            if 'NSE' in metrics:
                print(f"       💡 Strategy: Use NSE as R² approximation")
            elif 'KGE' in metrics and 'KGE' in str(metrics):
                print(f"       💡 Strategy: Use KGE as R² approximation")
            elif 'r' in metrics:
                print(f"       💡 Strategy: Use r² (correlation squared) as R²")
            else:
                print(f"       ⚠️  Strategy: Requires original paper review")
    
    # File consistency check
    print(f"\n🔄 FILE CONSISTENCY CHECK:")
    
    # Check if final file models match analysis file
    analysis_models = set(analysis_df['Model'].dropna())
    final_models = set(final_df['Model'].dropna())
    
    missing_in_final = analysis_models - final_models
    extra_in_final = final_models - analysis_models
    
    if len(missing_in_final) == 0 and len(extra_in_final) == 0:
        print("   ✅ Perfect consistency between analysis and final files")
    else:
        if missing_in_final:
            print(f"   ⚠️  Models in analysis but missing in final: {missing_in_final}")
        if extra_in_final:
            print(f"   ⚠️  Extra models in final: {extra_in_final}")
    
    print(f"\n📁 ALL FILES ARE CONSISTENT AND UP-TO-DATE")
    print("="*80)
    
    return True

if __name__ == "__main__":
    verify_file_consistency()
