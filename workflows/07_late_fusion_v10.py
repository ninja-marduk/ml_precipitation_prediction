"""
Pipeline Stage 07: V10 Late Fusion Ridge Stacking

Combines V2 ConvLSTM and V4 GNN-TAT predictions using out-of-fold
Ridge regression (decision-level ensemble).

Literature:
- Multi-view Stacking (Frontiers in Water 2024)
- GNN Ensemble Post-Processing (arXiv 2407.11050)
- TransLSTMUNet (J. Hydrology 2024)

Usage:
    python workflows/07_late_fusion_v10.py
    python workflows/07_late_fusion_v10.py --config workflows/config.yaml
    python workflows/07_late_fusion_v10.py --alpha 1.0 --n-folds 5
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import json
import warnings
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class V10Config:
    """V10 Late Fusion configuration."""
    v2_predictions_path: str = ''
    v4_predictions_path: str = ''
    targets_path: str = ''
    output_dir: str = ''
    ridge_alpha: float = 1.0
    n_folds: int = 5
    seed: int = 42
    feature_set: str = 'BASIC'
    grid_height: int = 61
    grid_width: int = 65
    n_horizons: int = 12

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'V10Config':
        """Load configuration from YAML file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        models = cfg.get('models', {})
        v2 = models.get('v2', {})
        v4 = models.get('v4', {})
        v10 = models.get('v10', {})
        region = cfg.get('region', {})
        env = cfg.get('environment', {})

        return cls(
            v2_predictions_path=v2.get('predictions_path', ''),
            v4_predictions_path=v4.get('predictions_path', ''),
            targets_path=v2.get('targets_path', ''),
            output_dir=v10.get('output_dir', 'models/output/V10_Late_Fusion'),
            ridge_alpha=v10.get('ridge_alpha', 1.0),
            n_folds=v10.get('n_folds', 5),
            seed=env.get('random_seed', 42),
            feature_set=models.get('feature_set', 'BASIC'),
            grid_height=region.get('grid_height', 61),
            grid_width=region.get('grid_width', 65),
            n_horizons=models.get('n_horizons', 12),
        )

    @classmethod
    def default(cls) -> 'V10Config':
        """Default configuration matching the original notebook."""
        return cls(
            v2_predictions_path='models/output/V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM/predictions.npy',
            v4_predictions_path='models/output/V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT/predictions.npy',
            targets_path='models/output/V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM/targets.npy',
            output_dir='models/output/V10_Late_Fusion',
        )


def load_predictions(config: V10Config, base_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load V2 and V4 predictions and targets."""
    logger.info('Loading base model predictions...')

    v2_pred = np.load(base_path / config.v2_predictions_path)
    logger.info(f'  V2 ConvLSTM: {v2_pred.shape}')

    v4_pred = np.load(base_path / config.v4_predictions_path)
    logger.info(f'  V4 GNN-TAT:  {v4_pred.shape}')

    targets = np.load(base_path / config.targets_path)
    logger.info(f'  Targets:     {targets.shape}')

    assert v2_pred.shape == v4_pred.shape == targets.shape, \
        f'Shape mismatch: V2={v2_pred.shape}, V4={v4_pred.shape}, targets={targets.shape}'

    return v2_pred, v4_pred, targets


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R2, RMSE, MAE, Bias."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return {
        'R2': float(r2_score(y_true_flat, y_pred_flat)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
        'MAE': float(mean_absolute_error(y_true_flat, y_pred_flat)),
        'Bias': float(np.mean(y_pred_flat - y_true_flat)),
    }


def simple_average_fusion(v2_pred: np.ndarray, v4_pred: np.ndarray) -> np.ndarray:
    """Simple averaging of predictions."""
    return (v2_pred + v4_pred) / 2


def weighted_average_fusion(v2_pred: np.ndarray, v4_pred: np.ndarray,
                            w_v2: float = 0.5, w_v4: float = 0.5) -> np.ndarray:
    """Weighted averaging of predictions."""
    return w_v2 * v2_pred + w_v4 * v4_pred


def ridge_fusion_oof(v2_pred: np.ndarray, v4_pred: np.ndarray,
                     targets: np.ndarray, config: V10Config) -> Tuple[np.ndarray, Ridge]:
    """Out-of-fold Ridge Regression fusion.

    Avoids information leakage through cross-validation.
    Returns (out-of-fold predictions, final Ridge model).
    """
    v2_flat = v2_pred.flatten()
    v4_flat = v4_pred.flatten()
    y_flat = targets.flatten()

    X_meta = np.column_stack([v2_flat, v4_flat])

    np.random.seed(config.seed)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    oof_predictions = np.zeros_like(y_flat)

    logger.info(f'Ridge OOF ({config.n_folds}-fold cross-validation):')

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_meta)):
        ridge = Ridge(alpha=config.ridge_alpha)
        ridge.fit(X_meta[train_idx], y_flat[train_idx])
        oof_predictions[val_idx] = ridge.predict(X_meta[val_idx])

        fold_r2 = r2_score(y_flat[val_idx], oof_predictions[val_idx])
        logger.info(f'  Fold {fold+1}: R2 = {fold_r2:.4f}')

    ridge_final = Ridge(alpha=config.ridge_alpha)
    ridge_final.fit(X_meta, y_flat)

    oof_predictions = oof_predictions.reshape(targets.shape)
    return oof_predictions, ridge_final


def compute_per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute metrics for each forecast horizon."""
    n_horizons = y_true.shape[1]
    rows = []
    for h in range(n_horizons):
        y_true_h = y_true[:, h].flatten()
        y_pred_h = y_pred[:, h].flatten()
        rows.append({
            'Horizon': h + 1,
            'R2': r2_score(y_true_h, y_pred_h),
            'RMSE': np.sqrt(mean_squared_error(y_true_h, y_pred_h)),
            'MAE': mean_absolute_error(y_true_h, y_pred_h),
            'Bias': np.mean(y_pred_h - y_true_h),
        })
    return pd.DataFrame(rows)


def run_experiments(v2_pred, v4_pred, targets, config):
    """Run all fusion experiments and return results."""
    results = {}

    # Baselines
    v2_metrics = compute_metrics(targets, v2_pred)
    v4_metrics = compute_metrics(targets, v4_pred)
    logger.info(f"V2 ConvLSTM baseline: R2={v2_metrics['R2']:.4f}, RMSE={v2_metrics['RMSE']:.2f} mm")
    logger.info(f"V4 GNN-TAT baseline:  R2={v4_metrics['R2']:.4f}, RMSE={v4_metrics['RMSE']:.2f} mm")

    # Method 1: Simple Average
    avg_pred = simple_average_fusion(v2_pred, v4_pred)
    results['Simple_Average'] = compute_metrics(targets, avg_pred)
    logger.info(f"Simple Average: R2={results['Simple_Average']['R2']:.4f}")

    # Method 2: Optimized Weighted Average
    best_w_v2, best_r2 = 0.5, 0
    for w_v2 in np.arange(0.1, 0.9, 0.05):
        w_v4 = 1 - w_v2
        weighted_pred = weighted_average_fusion(v2_pred, v4_pred, w_v2, w_v4)
        r2 = compute_metrics(targets, weighted_pred)['R2']
        if r2 > best_r2:
            best_r2 = r2
            best_w_v2 = w_v2

    best_weighted_pred = weighted_average_fusion(v2_pred, v4_pred, best_w_v2, 1 - best_w_v2)
    results['Weighted_Average'] = compute_metrics(targets, best_weighted_pred)
    results['Weighted_Average']['w_v2'] = float(best_w_v2)
    results['Weighted_Average']['w_v4'] = float(1 - best_w_v2)
    logger.info(f"Weighted Average: R2={results['Weighted_Average']['R2']:.4f} (w_V2={best_w_v2:.2f})")

    # Method 3: Ridge OOF
    ridge_pred, ridge_model = ridge_fusion_oof(v2_pred, v4_pred, targets, config)
    results['Ridge_OOF'] = compute_metrics(targets, ridge_pred)
    results['Ridge_OOF']['w_v2'] = float(ridge_model.coef_[0])
    results['Ridge_OOF']['w_v4'] = float(ridge_model.coef_[1])
    results['Ridge_OOF']['bias'] = float(ridge_model.intercept_)
    logger.info(f"Ridge OOF: R2={results['Ridge_OOF']['R2']:.4f}, "
                f"weights=[{ridge_model.coef_[0]:.4f}, {ridge_model.coef_[1]:.4f}]")

    return results, v2_metrics, v4_metrics, ridge_pred, ridge_model


def save_results(config, results, v2_metrics, v4_metrics,
                 ridge_pred, ridge_model, targets, base_path):
    """Save predictions, metrics, and summary."""
    output_dir = base_path / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Predictions and targets
    np.save(output_dir / 'predictions.npy', ridge_pred)
    np.save(output_dir / 'targets.npy', targets)

    # Per-horizon metrics
    horizon_df = compute_per_horizon_metrics(targets, ridge_pred)
    horizon_df.to_csv(output_dir / 'v10_metrics.csv', index=False)

    # Summary JSON
    best_baseline = max(v2_metrics['R2'], v4_metrics['R2'])
    improvement = (results['Ridge_OOF']['R2'] - best_baseline) / best_baseline * 100

    summary = {
        'version': 'V10_Late_Fusion',
        'timestamp': datetime.now().isoformat(),
        'config': asdict(config),
        'baselines': {
            'v2_convlstm': v2_metrics,
            'v4_gnn_tat': v4_metrics,
        },
        'results': results,
        'learned_weights': {
            'w_v2': float(ridge_model.coef_[0]),
            'w_v4': float(ridge_model.coef_[1]),
            'bias': float(ridge_model.intercept_),
        },
        'improvement_over_baseline': f'+{improvement:.2f}%',
    }

    with open(output_dir / 'v10_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f'Results saved to {output_dir}')
    logger.info(f'  predictions.npy, targets.npy, v10_metrics.csv, v10_summary.json')

    # Print final summary
    print('\n' + '=' * 60)
    print('  V10 LATE FUSION - RESULTS')
    print('=' * 60)
    print(f"  V2 ConvLSTM (baseline): R2={v2_metrics['R2']:.4f}, RMSE={v2_metrics['RMSE']:.2f} mm")
    print(f"  V4 GNN-TAT (baseline):  R2={v4_metrics['R2']:.4f}, RMSE={v4_metrics['RMSE']:.2f} mm")
    print(f"  V10 Ridge OOF:          R2={results['Ridge_OOF']['R2']:.4f}, RMSE={results['Ridge_OOF']['RMSE']:.2f} mm")
    print(f"  Improvement: +{improvement:.2f}%")
    print(f"  Weights: V2={ridge_model.coef_[0]:.4f}, V4={ridge_model.coef_[1]:.4f}, Bias={ridge_model.intercept_:.4f}")
    print('=' * 60)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='V10 Late Fusion Ridge Stacking')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Ridge alpha (regularization)')
    parser.add_argument('--n-folds', type=int, default=None,
                        help='Number of cross-validation folds')
    parser.add_argument('--feature-set', type=str, default=None,
                        help='Feature set: BASIC, KCE, or PAFC')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--intracell-dem', action='store_true',
                        help='Use intra-cell DEM predictions (Paper 5)')
    parser.add_argument('--bundle', type=str, default='BASIC_D10',
                        choices=['BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS'],
                        help='Feature bundle for --intracell-dem (default: BASIC_D10)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if args.config:
        config = V10Config.from_yaml(Path(args.config))
    else:
        default_config = PROJECT_ROOT / 'workflows' / 'config.yaml'
        if default_config.exists():
            config = V10Config.from_yaml(default_config)
        else:
            config = V10Config.default()

    # Apply CLI overrides
    if args.alpha is not None:
        config.ridge_alpha = args.alpha
    if args.n_folds is not None:
        config.n_folds = args.n_folds
    if args.feature_set is not None:
        config.feature_set = args.feature_set
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    # Intracell DEM mode: override paths to Paper 5 directories
    if args.intracell_dem:
        bundle = args.bundle
        config.feature_set = bundle
        config.v2_predictions_path = f'models/output/V2_Enhanced_Models_intracell_dem/map_exports/H12/{bundle}/ConvLSTM/predictions.npy'
        config.v4_predictions_path = f'models/output/V4_GNN_TAT_Models_intracell_dem/map_exports/H12/{bundle}/GNN_TAT_GAT/predictions.npy'
        config.targets_path = f'models/output/V2_Enhanced_Models_intracell_dem/map_exports/H12/{bundle}/ConvLSTM/targets.npy'
        config.output_dir = f'models/output/V10_Late_Fusion_intracell_dem/{bundle}'
        logger.info(f'Intracell DEM mode: bundle={bundle}')

    logger.info(f'V10 Late Fusion - config: alpha={config.ridge_alpha}, folds={config.n_folds}, '
                f'feature_set={config.feature_set}')

    # Load predictions
    v2_pred, v4_pred, targets = load_predictions(config, PROJECT_ROOT)

    # Run experiments
    results, v2_metrics, v4_metrics, ridge_pred, ridge_model = \
        run_experiments(v2_pred, v4_pred, targets, config)

    # Save results
    save_results(config, results, v2_metrics, v4_metrics,
                 ridge_pred, ridge_model, targets, PROJECT_ROOT)


if __name__ == '__main__':
    main()
