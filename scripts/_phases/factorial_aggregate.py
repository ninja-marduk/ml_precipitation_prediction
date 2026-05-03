"""Feature 003 T022 - aggregate factorial CSV across seeds.

Reads `metrics_factorial_consolidated.csv` (216 rows) and produces two
paper-ready artefacts:

1. `.docs/papers/5/data/factorial_feat_variant.csv` — 6 rows (feat x variant) at H=12
   with mean/std/count across seeds. Schema:
   [feat, variant, R^2_mean, R^2_std, R^2_count, RMSE_mean, ..., Bias_count]

2. `.docs/papers/5/data/factorial_feat_variant_byhorizon.csv` — 72 rows
   (feat x variant x 12 horizons) with mean/std across seeds. Used for the
   factorial horizon-degradation figure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / 'models' / 'output' / 'V4_GNN_TAT_Models' / 'metrics_factorial_consolidated.csv'
DST_DIR = REPO / '.docs' / 'papers' / '5' / 'data'

METRIC_COLS = ['R^2', 'RMSE', 'MAE', 'Bias']


def _agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols)[METRIC_COLS].agg(['mean', 'std', 'count'])
    agg.columns = [f'{m}_{s}' for m, s in agg.columns]
    return agg.reset_index()


def run(args: argparse.Namespace) -> int:
    print('[T022] Aggregate factorial CSV across seeds')
    if not SRC.exists():
        print(f'  MISSING: {SRC}')
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(SRC)

    # 1) H=12 only, grouped by (feat, variant) -> 6 rows
    h12 = long[long['H'] == 12]
    h12_agg = _agg(h12, ['feat', 'variant'])
    h12_dst = DST_DIR / 'factorial_feat_variant.csv'
    h12_agg.to_csv(h12_dst, index=False)
    print(f'  wrote: {h12_dst.relative_to(REPO)}  ({len(h12_agg)} rows)')

    # Sanity
    assert len(h12_agg) == 6, f'expected 6 rows, got {len(h12_agg)}'
    for c in h12_agg.columns:
        if c.endswith('_count'):
            assert (h12_agg[c] == 3).all(), f'count column {c} != 3'

    # 2) All horizons, grouped by (feat, variant, H) -> 72 rows
    bh_agg = _agg(long, ['feat', 'variant', 'H'])
    bh_dst = DST_DIR / 'factorial_feat_variant_byhorizon.csv'
    bh_agg.to_csv(bh_dst, index=False)
    print(f'  wrote: {bh_dst.relative_to(REPO)}  ({len(bh_agg)} rows)')
    assert len(bh_agg) == 72, f'expected 72 rows, got {len(bh_agg)}'

    # Preview the H=12 aggregated table (6 rows, paper-ready)
    print(f'\n  H=12 aggregated (feat x variant, mean +/- s.d. over 3 seeds):')
    print(f'  {"feat":<6}  {"variant":<5}  {"R^2_mean":>9}  {"R^2_std":>9}  '
          f'{"RMSE_mean":>10}  {"RMSE_std":>9}')
    print(f'  {"-"*6}  {"-"*5}  {"-"*9}  {"-"*9}  {"-"*10}  {"-"*9}')
    for _, r in h12_agg.iterrows():
        print(f'  {r["feat"]:<6}  {r["variant"]:<5}  '
              f'{r["R^2_mean"]:>+9.4f}  {r["R^2_std"]:>9.4f}  '
              f'{r["RMSE_mean"]:>10.2f}  {r["RMSE_std"]:>9.2f}')

    # Identify best cell for narrative
    best_idx = h12_agg['R^2_mean'].idxmax()
    best = h12_agg.loc[best_idx]
    print(f'\n  BEST cell at H=12: feat={best["feat"]} variant={best["variant"]}  '
          f'R^2 = {best["R^2_mean"]:.4f} +/- {best["R^2_std"]:.4f}  '
          f'RMSE = {best["RMSE_mean"]:.2f} +/- {best["RMSE_std"]:.2f}')

    print('\n[T022] aggregate complete')
    return 0


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace()))
