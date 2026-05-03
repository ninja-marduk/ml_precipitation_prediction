"""Feature 003 T020 - per-horizon metrics for each of the 18 factorial cells.

Factorial: seeds {42, 123, 456} x feats {BASIC, PAFC} x variants {GAT, GCN, SAGE}.
Writes `metrics_per_horizon.csv` inside each cell's map_exports subdir with schema
matching the existing per-seed format: [H, R^2, RMSE, MAE, Bias].

seed 42 cells live at the legacy root:
  V4_GNN_TAT_Models/map_exports/H12/{feat}/GNN_TAT_{variant}/
seeds 123/456 under:
  V4_GNN_TAT_Models/SEED{seed}/map_exports/H12/{feat}/GNN_TAT_{variant}/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    yt = y_true.flatten()
    yp = y_pred.flatten()
    return {
        'R^2': float(r2_score(yt, yp)),
        'RMSE': float(np.sqrt(mean_squared_error(yt, yp))),
        'MAE': float(mean_absolute_error(yt, yp)),
        'Bias': float(np.mean(yp - yt)),
    }


def _per_horizon(pred: np.ndarray, tgt: np.ndarray) -> pd.DataFrame:
    rows = []
    for h in range(pred.shape[1]):
        m = _metrics(tgt[:, h], pred[:, h])
        rows.append({'H': h + 1, **m})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    print('[T020] Factorial per-cell per-horizon metrics (18 cells x 12 horizons)')
    n_written = 0
    halts = []

    for seed in SEEDS:
        for feat in FEATS:
            for variant in VARIANTS:
                cell = _cell_dir(seed, feat, variant)
                pred_p = cell / 'predictions.npy'
                tgt_p = cell / 'targets.npy'
                if not pred_p.exists() or not tgt_p.exists():
                    halts.append(f'seed={seed} feat={feat} variant={variant}: MISSING {pred_p.name}')
                    continue
                pred = np.load(pred_p).astype(np.float32)
                tgt = np.load(tgt_p).astype(np.float32)
                if pred.shape != tgt.shape:
                    halts.append(f'seed={seed} feat={feat} variant={variant}: shape mismatch '
                                 f'{pred.shape} vs {tgt.shape}')
                    continue
                df = _per_horizon(pred, tgt)
                out_csv = cell / 'metrics_per_horizon.csv'
                df.to_csv(out_csv, index=False)
                r2_h1 = df['R^2'].iloc[0]
                r2_h12 = df['R^2'].iloc[-1]
                print(f'  seed={seed} {feat:<6} {variant:<5}: '
                      f'H=1 R^2={r2_h1:+.4f}  H=12 R^2={r2_h12:+.4f}  '
                      f'-> {out_csv.relative_to(REPO)}')
                n_written += 1

    if halts:
        print('\n[T020] HALT: missing cells')
        for h in halts:
            print(f'  - {h}')
        return 1

    print(f'\n[T020] {n_written} per-cell CSVs written (expected 18)')
    return 0 if n_written == len(SEEDS) * len(FEATS) * len(VARIANTS) else 2


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace()))
