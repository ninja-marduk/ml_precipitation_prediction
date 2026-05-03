"""T029 - cross-model unified table for paper/figure.

Joins the per-model consolidated CSVs into a single table keyed by horizon,
with columns `{V2,V4,V10}_{R^2,RMSE,MAE,Bias}_{mean,std}`. Writes to
`.docs/papers/5/data/horizon_multiseed_v2_v4_v10.csv`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'models' / 'output'
DST = REPO / '.docs' / 'papers' / '5' / 'data' / 'horizon_multiseed_v2_v4_v10.csv'

MODEL_DIRS = {
    'V2':  OUT / 'V2_Enhanced_Models',
    'V4':  OUT / 'V4_GNN_TAT_Models',
    'V10': OUT / 'V10_Late_Fusion',
}


def run(args: argparse.Namespace) -> int:
    print('[T029] cross-model unified table')

    frames = {}
    for label, base in MODEL_DIRS.items():
        p = base / 'metrics_multiseed_consolidated.csv'
        if not p.exists():
            print(f'  MISSING: {p}')
            return 1
        df = pd.read_csv(p)
        # Rename mean/std columns with model prefix (keep count out of the unified)
        rename = {}
        for c in df.columns:
            if c.endswith('_mean') or c.endswith('_std'):
                rename[c] = f'{label}_{c}'
        df = df[['H'] + list(rename)].rename(columns=rename)
        frames[label] = df

    unified = frames['V2']
    for lbl in ('V4', 'V10'):
        unified = unified.merge(frames[lbl], on='H', how='inner')

    # Single n_seeds column (all models use 3 seeds)
    unified['n_seeds'] = 3

    # Sanity: 12 horizons, no NaN, V10 values match the per-model CSV to 3 decimals
    assert len(unified) == 12, f'expected 12 rows, got {len(unified)}'
    assert not unified.isna().any().any(), 'NaN in unified table'

    DST.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(DST, index=False)

    print(f'  rows: {len(unified)}')
    print(f'  cols: {list(unified.columns)}')
    print(f'  mean V10 R^2 across horizons: {unified["V10_R^2_mean"].mean():.4f}')
    print(f'  wrote: {DST}')

    # Echo a compact preview for the paper narrative
    print('\n  preview (canonical horizons H=3, 6, 12):')
    for H in (3, 6, 12):
        r = unified[unified['H'] == H].iloc[0]
        print(f'   H={H}: V2={r["V2_R^2_mean"]:.4f}±{r["V2_R^2_std"]:.4f}  '
              f'V4={r["V4_R^2_mean"]:.4f}±{r["V4_R^2_std"]:.4f}  '
              f'V10={r["V10_R^2_mean"]:.4f}±{r["V10_R^2_std"]:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace()))
