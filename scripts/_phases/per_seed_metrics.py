"""T020/T021 - per-horizon metrics for every (model, seed) combination.

Writes `metrics_per_horizon.csv` for V2 and V4 at each SEED{n}/ directory with
schema matching V10: [H, R^2, RMSE, MAE, Bias]. V10 already has this file
(v10_metrics.csv) produced by phase_v10_fusion.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'models' / 'output'
SEEDS = (42, 123, 456)
V2_VARIANT = 'ConvLSTM_Bidirectional'


def _metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y_t = y_true.flatten()
    y_p = y_pred.flatten()
    return {
        'R^2': float(r2_score(y_t, y_p)),
        'RMSE': float(np.sqrt(mean_squared_error(y_t, y_p))),
        'MAE': float(mean_absolute_error(y_t, y_p)),
        'Bias': float(np.mean(y_p - y_t)),
    }


def _per_horizon(pred: np.ndarray, tgt: np.ndarray):
    import pandas as pd
    rows = []
    for h in range(pred.shape[1]):
        m = _metrics(tgt[:, h], pred[:, h])
        rows.append({'H': h + 1, **m})
    return pd.DataFrame(rows)


def _paths(model: str, seed: int) -> tuple[Path, Path, Path]:
    """Return (pred_path, tgt_path, out_csv) for a (model, seed) combination."""
    if model == 'v2':
        base = OUT / 'V2_Enhanced_Models' / f'SEED{seed}' / 'map_exports' / 'H12' / 'BASIC' / V2_VARIANT
        return base / 'predictions.npy', base / 'targets.npy', \
               OUT / 'V2_Enhanced_Models' / f'SEED{seed}' / 'metrics_per_horizon.csv'
    if model == 'v4':
        if seed == 42:
            base = OUT / 'V4_GNN_TAT_Models' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT'
        else:
            base = OUT / 'V4_GNN_TAT_Models' / f'SEED{seed}' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT'
        out_dir = OUT / 'V4_GNN_TAT_Models' / f'SEED{seed}'
        return base / 'predictions.npy', base / 'targets.npy', out_dir / 'metrics_per_horizon.csv'
    if model == 'v10':
        base = OUT / 'V10_Late_Fusion' / f'SEED{seed}'
        return base / 'predictions.npy', base / 'targets.npy', base / 'v10_metrics.csv'
    raise ValueError(f'unknown model {model}')


def run(args: argparse.Namespace) -> int:
    print('[T021] per-seed per-horizon metrics for V2, V4, V10 × 3 seeds')
    import pandas as pd  # noqa

    n_written = 0
    for model in ('v2', 'v4', 'v10'):
        if args.model and args.model != model:
            continue
        print(f'\n  === {model.upper()} ===')
        for seed in SEEDS:
            pred_p, tgt_p, out_csv = _paths(model, seed)
            if not pred_p.exists() or not tgt_p.exists():
                print(f'    seed {seed}: MISSING {pred_p.name if not pred_p.exists() else tgt_p.name}')
                return 1
            pred = np.load(pred_p).astype(np.float32)
            tgt = np.load(tgt_p).astype(np.float32)
            df = _per_horizon(pred, tgt)
            # V10 SEED{n}/v10_metrics.csv was already written by phase_v10_fusion;
            # overwrite only to ensure schema consistency (same columns).
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
            r2_h1 = df['R^2'].iloc[0]
            r2_h12 = df['R^2'].iloc[-1]
            print(f'    seed {seed}: wrote {out_csv.name}  '
                  f'(H=1 R^2={r2_h1:+.4f}, H=12 R^2={r2_h12:+.4f})')
            n_written += 1

    print(f'\n[T021] {n_written} per-seed CSVs written/verified')
    return 0


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace(model=None)))
