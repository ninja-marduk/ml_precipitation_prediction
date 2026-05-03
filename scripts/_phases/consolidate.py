"""T025-T028 - per-model consolidation of per-seed metrics.

Aggregates the per-seed CSVs (V2, V4, V10 × seeds {42,123,456}) into one
per-model consolidated CSV with mean/std/count per horizon. Schema:
  [H, R^2_mean, R^2_std, R^2_count, RMSE_mean, ..., Bias_count]

Writes to each model's root output directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'models' / 'output'
SEEDS = (42, 123, 456)

MODEL_DIRS = {
    'v2':  OUT / 'V2_Enhanced_Models',
    'v4':  OUT / 'V4_GNN_TAT_Models',
    'v10': OUT / 'V10_Late_Fusion',
}


def _per_seed_csv(model: str, seed: int) -> Path:
    if model == 'v10':
        return MODEL_DIRS['v10'] / f'SEED{seed}' / 'v10_metrics.csv'
    return MODEL_DIRS[model] / f'SEED{seed}' / 'metrics_per_horizon.csv'


def _consolidate_one(model: str) -> Path:
    frames = []
    for seed in SEEDS:
        p = _per_seed_csv(model, seed)
        if not p.exists():
            raise FileNotFoundError(f'{model} seed {seed}: {p}')
        df = pd.read_csv(p)
        df['seed'] = seed
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    metric_cols = ['R^2', 'RMSE', 'MAE', 'Bias']
    agg = all_df.groupby('H')[metric_cols].agg(['mean', 'std', 'count'])
    agg.columns = [f'{m}_{s}' for m, s in agg.columns]
    agg = agg.reset_index()
    out_path = MODEL_DIRS[model] / 'metrics_multiseed_consolidated.csv'
    agg.to_csv(out_path, index=False)
    return out_path


def run(args: argparse.Namespace) -> int:
    print('[T025] per-model consolidation (mean/std/count across 3 seeds)')
    results = {}
    for model in ('v2', 'v4', 'v10'):
        if args.model and args.model != model:
            continue
        try:
            path = _consolidate_one(model)
        except Exception as e:
            print(f'  {model}: FAIL {e}')
            return 1
        df = pd.read_csv(path)
        counts_ok = all((df[c] == 3).all() for c in df.columns if c.endswith('_count'))
        mean_r2 = df['R^2_mean'].mean()
        mean_rmse = df['RMSE_mean'].mean()
        print(f'  {model:<4s}: {path.relative_to(REPO)} '
              f'(count=3 OK: {counts_ok}, mean R^2 across horizons={mean_r2:.4f}, '
              f'mean RMSE={mean_rmse:.2f})')
        results[model] = {'path': path, 'counts_ok': counts_ok, 'mean_r2': mean_r2, 'mean_rmse': mean_rmse}

    if not all(r['counts_ok'] for r in results.values()):
        print('[T025] HALT: at least one model has count != 3 on some horizon')
        return 2

    print('[T025] consolidation complete; every horizon has count=3 per model.')
    return 0


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace(model=None)))
