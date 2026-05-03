"""Feature 003 T021 - consolidate the 18 per-cell factorial CSVs into one long table.

Reads each cell's `metrics_per_horizon.csv` (12 rows, columns [H, R^2, RMSE, MAE, Bias]),
tags it with (seed, feat, variant), and concatenates. Writes a single 216-row
CSV at `V4_GNN_TAT_Models/metrics_factorial_consolidated.csv` with schema
[seed, feat, variant, H, R^2, RMSE, MAE, Bias].
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'models' / 'output' / 'V4_GNN_TAT_Models'

SEEDS = (42, 123, 456)
FEATS = ('BASIC', 'PAFC')
VARIANTS = ('GAT', 'GCN', 'SAGE')
HORIZON = 12


def _cell_dir(seed: int, feat: str, variant: str) -> Path:
    if seed == 42:
        base = OUT / 'map_exports'
    else:
        base = OUT / f'SEED{seed}' / 'map_exports'
    return base / f'H{HORIZON}' / feat / f'GNN_TAT_{variant}'


def run(args: argparse.Namespace) -> int:
    print('[T021] Consolidate 18 factorial cells into one long table')
    frames = []
    missing = []
    for seed in SEEDS:
        for feat in FEATS:
            for variant in VARIANTS:
                p = _cell_dir(seed, feat, variant) / 'metrics_per_horizon.csv'
                if not p.exists():
                    missing.append(p)
                    continue
                df = pd.read_csv(p)
                df.insert(0, 'variant', variant)
                df.insert(0, 'feat', feat)
                df.insert(0, 'seed', seed)
                frames.append(df)

    if missing:
        print('[T021] HALT: missing per-cell CSVs')
        for p in missing:
            print(f'  - {p.relative_to(REPO)}')
        return 1

    long = pd.concat(frames, ignore_index=True)

    expected_rows = len(SEEDS) * len(FEATS) * len(VARIANTS) * HORIZON
    if len(long) != expected_rows:
        print(f'[T021] WARN: expected {expected_rows} rows, got {len(long)}')

    out_csv = OUT / 'metrics_factorial_consolidated.csv'
    long.to_csv(out_csv, index=False)
    print(f'  wrote: {out_csv.relative_to(REPO)}  ({len(long)} rows)')

    # H=12 summary preview (18 rows)
    h12 = long[long['H'] == 12].copy()
    h12 = h12.sort_values(['feat', 'variant', 'seed'])
    print(f'\n  H=12 per-cell R^2 preview (should be 18 rows):')
    print(f'  {"seed":>4}  {"feat":<6}  {"variant":<5}  {"R^2":>7}  {"RMSE":>7}')
    print(f'  {"-"*4}  {"-"*6}  {"-"*5}  {"-"*7}  {"-"*7}')
    for _, r in h12.iterrows():
        print(f'  {int(r["seed"]):>4}  {r["feat"]:<6}  {r["variant"]:<5}  '
              f'{r["R^2"]:>+7.4f}  {r["RMSE"]:>7.2f}')

    print(f'\n[T021] Consolidation complete: {len(long)} rows across '
          f'{len(SEEDS)} seeds x {len(FEATS)} feats x {len(VARIANTS)} variants x '
          f'{HORIZON} horizons')
    return 0


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace()))
