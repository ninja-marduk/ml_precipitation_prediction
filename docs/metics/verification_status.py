#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

# Inputs/Outputs in current directory
CSV_IN = 'detailed_metrics_analysis.csv'
CSV_OUT = 'verification_status.csv'

REL_RMSE_COLS = [
    'RMSE_pct_of_baseline',
    'NRMSE_mean_pct',
    'NRMSE_range_pct',
    'Error_RMSE_pct_preferred',
]
REL_MAE_COLS = [
    'MAE_pct_of_baseline',
    'NMAE_mean_pct',
    'NMAE_range_pct',
    'Error_MAE_pct_preferred',
]


def has_relative(series_row: pd.Series, cols: list) -> bool:
    for c in cols:
        if c in series_row.index:
            v = pd.to_numeric(series_row[c], errors='coerce')
            if pd.notna(v):
                return True
    return False


def to_bool(x):
    return bool(x) if not pd.isna(x) else False


def infer_spatiotemporal(ref: str) -> str:
    if not isinstance(ref, str):
        return 'unknown'
    t = ref.lower()
    temporal = any(k in t for k in ['monthly', 'daily', 'hourly', 'time', 'series', 'forecast', 'prediction'])
    spatial = any(k in t for k in ['station', 'stations', 'grid', 'gridded', 'china', 'basin', 'region', 'area', '0.5', '0.25'])
    if temporal and spatial:
        return 'yes'
    if temporal or spatial:
        return 'likely'
    return 'unknown'


def main():
    if not os.path.exists(CSV_IN):
        print(f"Input not found: {CSV_IN}")
        return
    df = pd.read_csv(CSV_IN)

    # Per-row flags
    df['Has_RMSE_rel'] = df.apply(lambda r: has_relative(r, REL_RMSE_COLS), axis=1)
    df['Has_MAE_rel'] = df.apply(lambda r: has_relative(r, REL_MAE_COLS), axis=1)
    df['Has_R2'] = pd.to_numeric(df.get('R_squared', pd.Series([np.nan]*len(df))), errors='coerce').between(0, 1)
    df['SpatioTemporal'] = df['Reference'].apply(infer_spatiotemporal)

    # Aggregate per study (Reference)
    agg = df.groupby('Reference', dropna=False).agg(
        Models=('Model', 'count'),
        Any_RMSE_rel=('Has_RMSE_rel', 'any'),
        Any_MAE_rel=('Has_MAE_rel', 'any'),
        Any_R2=('Has_R2', 'any'),
        SpatioTemporal=('SpatioTemporal', lambda x: x.mode().iat[0] if len(x.mode()) > 0 else 'unknown')
    ).reset_index()

    # Data status: need at least one relative error (RMSE or MAE) and at least one R2
    agg['Data_Status'] = np.where((agg['Any_RMSE_rel'] | agg['Any_MAE_rel']) & agg['Any_R2'], 'ok', 'pending')

    # Save report
    agg.to_csv(CSV_OUT, index=False)

    # Print pending list
    pending = agg[agg['Data_Status'] != 'ok']
    if len(pending) > 0:
        print("Pending studies (data verification):")
        for ref in pending['Reference'].tolist():
            print(f" - {ref}")
    else:
        print("All studies have required data.")

    print(f"Report written: {os.path.abspath(CSV_OUT)}")


if __name__ == '__main__':
    main()
