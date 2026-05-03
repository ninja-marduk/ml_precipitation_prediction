"""T012 - V10 Ridge fusion re-run for all 3 seeds (path C: Bidirectional V2).

Refreshes `V10_Late_Fusion/SEED{42,123,456}/` using ConvLSTM_Bidirectional V2
inputs. Previous V10 SEED123/SEED456 results (produced with bare ConvLSTM via
the multi-seed notebook) are archived as `v10_summary.legacy.json` before
overwrite.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'models' / 'output'

V2_VARIANT = 'ConvLSTM_Bidirectional'

SEED_CONFIG = {
    42: {
        'v2': OUT / 'V2_Enhanced_Models' / 'SEED42' / 'map_exports' / 'H12' / 'BASIC' / V2_VARIANT / 'predictions.npy',
        'v4': OUT / 'V4_GNN_TAT_Models' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT' / 'predictions.npy',
        'v4_tgt': OUT / 'V4_GNN_TAT_Models' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT' / 'targets.npy',
        'out': OUT / 'V10_Late_Fusion' / 'SEED42',
    },
    123: {
        'v2': OUT / 'V2_Enhanced_Models' / 'SEED123' / 'map_exports' / 'H12' / 'BASIC' / V2_VARIANT / 'predictions.npy',
        'v4': OUT / 'V4_GNN_TAT_Models' / 'SEED123' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT' / 'predictions.npy',
        'v4_tgt': OUT / 'V4_GNN_TAT_Models' / 'SEED123' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT' / 'targets.npy',
        'out': OUT / 'V10_Late_Fusion' / 'SEED123',
    },
    456: {
        'v2': OUT / 'V2_Enhanced_Models' / 'SEED456' / 'map_exports' / 'H12' / 'BASIC' / V2_VARIANT / 'predictions.npy',
        'v4': OUT / 'V4_GNN_TAT_Models' / 'SEED456' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT' / 'predictions.npy',
        'v4_tgt': OUT / 'V4_GNN_TAT_Models' / 'SEED456' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT' / 'targets.npy',
        'out': OUT / 'V10_Late_Fusion' / 'SEED456',
    },
}

RIDGE_ALPHA = 1.0
N_FOLDS = 5


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return {
        'R2': float(r2_score(y_true_flat, y_pred_flat)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
        'MAE': float(mean_absolute_error(y_true_flat, y_pred_flat)),
        'Bias': float(np.mean(y_pred_flat - y_true_flat)),
    }


def _ridge_fusion_oof(v2_pred: np.ndarray, v4_pred: np.ndarray, targets: np.ndarray,
                     alpha: float, n_folds: int, seed: int):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    X_meta = np.column_stack([v2_pred.flatten(), v4_pred.flatten()])
    y_flat = targets.flatten()
    np.random.seed(seed)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y_flat)
    for fold, (tr, va) in enumerate(kf.split(X_meta)):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_meta[tr], y_flat[tr])
        oof[va] = ridge.predict(X_meta[va])
        print(f'      Fold {fold + 1}: R^2 = {r2_score(y_flat[va], oof[va]):.4f}')
    ridge_final = Ridge(alpha=alpha)
    ridge_final.fit(X_meta, y_flat)
    return oof.reshape(targets.shape), ridge_final


def _fuse_one_seed(seed: int) -> int:
    cfg = SEED_CONFIG[seed]
    print(f'\n  === seed {seed} ===')
    for p in (cfg['v2'], cfg['v4'], cfg['v4_tgt']):
        if not p.exists():
            print(f'    MISSING: {p}')
            return 1

    v2_pred = np.load(cfg['v2']).astype(np.float32)
    v4_pred = np.load(cfg['v4']).astype(np.float32)
    targets = np.load(cfg['v4_tgt']).astype(np.float32)
    print(f'    V2={v2_pred.shape}, V4={v4_pred.shape}, targets={targets.shape}')

    if not (v2_pred.shape == v4_pred.shape == targets.shape):
        print('    HALT: shape mismatch')
        return 2

    # Archive existing summary (only if first refresh)
    cfg['out'].mkdir(parents=True, exist_ok=True)
    legacy_summary = cfg['out'] / 'v10_summary.json'
    legacy_backup = cfg['out'] / 'v10_summary.legacy.json'
    if legacy_summary.exists() and not legacy_backup.exists():
        shutil.copy2(legacy_summary, legacy_backup)
        print(f'    archived legacy summary: {legacy_backup.name}')

    v2_m = _compute_metrics(targets, v2_pred)
    v4_m = _compute_metrics(targets, v4_pred)
    print(f'    V2 baseline: R^2={v2_m["R2"]:.4f}  RMSE={v2_m["RMSE"]:.2f}')
    print(f'    V4 baseline: R^2={v4_m["R2"]:.4f}  RMSE={v4_m["RMSE"]:.2f}')

    # Method 1: simple average
    avg_pred = (v2_pred + v4_pred) / 2.0
    simple_m = _compute_metrics(targets, avg_pred)
    print(f'    simple avg: R^2={simple_m["R2"]:.4f}  RMSE={simple_m["RMSE"]:.2f}')

    # Method 2: weighted (grid search)
    best_w_v2, best_r2 = 0.5, -np.inf
    for w_v2 in np.arange(0.1, 0.9, 0.05):
        pred = w_v2 * v2_pred + (1 - w_v2) * v4_pred
        r2 = _compute_metrics(targets, pred)['R2']
        if r2 > best_r2:
            best_r2, best_w_v2 = r2, float(w_v2)
    best_weighted = best_w_v2 * v2_pred + (1 - best_w_v2) * v4_pred
    weighted_m = _compute_metrics(targets, best_weighted)
    weighted_m['w_v2'] = best_w_v2
    weighted_m['w_v4'] = 1 - best_w_v2
    print(f'    weighted (w_v2={best_w_v2:.2f}): R^2={weighted_m["R2"]:.4f}  RMSE={weighted_m["RMSE"]:.2f}')

    # Method 3: Ridge OOF
    print('    Ridge OOF 5-fold:')
    ridge_pred, ridge_model = _ridge_fusion_oof(v2_pred, v4_pred, targets,
                                                 RIDGE_ALPHA, N_FOLDS, seed)
    ridge_m = _compute_metrics(targets, ridge_pred)
    ridge_m['w_v2'] = float(ridge_model.coef_[0])
    ridge_m['w_v4'] = float(ridge_model.coef_[1])
    ridge_m['bias'] = float(ridge_model.intercept_)
    print(f'    Ridge: R^2={ridge_m["R2"]:.4f}  RMSE={ridge_m["RMSE"]:.2f}  '
          f'w_v2={ridge_m["w_v2"]:.4f} w_v4={ridge_m["w_v4"]:.4f} bias={ridge_m["bias"]:.3f}')

    # Write outputs
    np.save(cfg['out'] / 'predictions.npy', ridge_pred.astype(np.float32))
    np.save(cfg['out'] / 'targets.npy', targets)

    # Per-horizon CSV
    import pandas as pd
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    rows = []
    for h in range(targets.shape[1]):
        y_t = targets[:, h].flatten()
        y_p = ridge_pred[:, h].flatten()
        rows.append({
            'H': h + 1,
            'R^2': float(r2_score(y_t, y_p)),
            'RMSE': float(np.sqrt(mean_squared_error(y_t, y_p))),
            'MAE': float(mean_absolute_error(y_t, y_p)),
            'Bias': float(np.mean(y_p - y_t)),
        })
    pd.DataFrame(rows).to_csv(cfg['out'] / 'v10_metrics.csv', index=False)

    # Summary JSON
    best_baseline_r2 = max(v2_m['R2'], v4_m['R2'])
    improvement = (ridge_m['R2'] - best_baseline_r2) / best_baseline_r2 * 100
    summary = {
        'seed': seed,
        'version': 'V10_Late_Fusion',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'ridge_alpha': RIDGE_ALPHA,
            'n_folds': N_FOLDS,
            'seed': seed,
            'feature_set': 'BASIC',
            'v2_variant': V2_VARIANT,
        },
        'baselines': {'v2_convlstm_bidirectional': v2_m, 'v4_gnn_tat': v4_m},
        'results': {
            'simple_average': simple_m,
            'weighted_average': weighted_m,
            'ridge_oof': ridge_m,
        },
        'learned_weights': {
            'w_v2': ridge_m['w_v2'],
            'w_v4': ridge_m['w_v4'],
            'bias': ridge_m['bias'],
        },
        'improvement_over_baseline': f'+{improvement:.2f}%',
        'regenerated_by': 'scripts/regenerate_multiseed_horizon.py --phase v10-fusion (path C)',
        'regenerated_note': f'V2 variant switched to {V2_VARIANT} to match paper naming',
    }
    (cfg['out'] / 'v10_summary.json').write_text(
        json.dumps(summary, indent=2, default=str), encoding='utf-8',
    )
    print(f'    wrote: {cfg["out"] / "predictions.npy"}')
    print(f'    wrote: {cfg["out"] / "v10_metrics.csv"}')
    print(f'    wrote: {cfg["out"] / "v10_summary.json"}')
    return 0


def run(args: argparse.Namespace) -> int:
    print('[T012] V10 Ridge fusion refresh for seeds {42, 123, 456} '
          f'with V2={V2_VARIANT}')
    for seed in (42, 123, 456):
        rc = _fuse_one_seed(seed)
        if rc != 0:
            print(f'[T012] HALT at seed {seed} (rc={rc})')
            return rc
    print('\n[T012] V10 fusion refresh complete for all 3 seeds.')
    return 0


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace()))
