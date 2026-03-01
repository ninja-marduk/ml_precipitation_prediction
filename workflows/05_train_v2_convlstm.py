"""
Pipeline Stage 05: Train V2 Enhanced ConvLSTM

Trains the V2 ConvLSTM model for spatiotemporal precipitation prediction.
Replicates the full training pipeline from the notebooks into a standalone,
region-agnostic CLI script for use in Barcelona and other deployments.

Architecture:
- ConvLSTM2D(32) -> ConvLSTM2D(16) -> spatial_head (Conv2D)
- CombinedLoss: multi-horizon MSE + temporal consistency
- Dual StandardScaler (BASE features + DEM features separately)

Source: models/intracell_dem/train_v2_convlstm_intracell_dem.ipynb
        models/base_models_conv_sthymountain_v2.ipynb

Usage:
    python workflows/05_train_v2_convlstm.py --dry-run
    python workflows/05_train_v2_convlstm.py --config workflows/config.yaml
    python workflows/05_train_v2_convlstm.py --intracell-dem --bundle BASIC_D10
    python workflows/05_train_v2_convlstm.py --light-mode --epochs 5

Note: Requires TensorFlow >= 2.6 and significant GPU memory.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import gc
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

FEATURES_BASIC = [
    'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
    'elevation', 'slope', 'aspect',
]

FEATURES_BASIC_D10 = FEATURES_BASIC + [
    'dem_p10', 'dem_p20', 'dem_p30', 'dem_p40', 'dem_p50',
    'dem_p60', 'dem_p70', 'dem_p80', 'dem_p90', 'dem_p100',
]

FEATURES_BASIC_PCA6 = FEATURES_BASIC + [
    'dem_pca_1', 'dem_pca_2', 'dem_pca_3',
    'dem_pca_4', 'dem_pca_5', 'dem_pca_6',
]

FEATURES_BASIC_D10_STATS = FEATURES_BASIC + [
    'dem_p10', 'dem_p20', 'dem_p30', 'dem_p40', 'dem_p50',
    'dem_p60', 'dem_p70', 'dem_p80', 'dem_p90', 'dem_p100',
    'dem_mean', 'dem_std', 'dem_skewness', 'dem_kurtosis', 'dem_range',
]

# Feature scaling groups
BASE_CONTINUOUS_FEATURES = [
    'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
    'elevation', 'slope', 'aspect',
]

DEM_CONTINUOUS_FEATURES = [
    'dem_p10', 'dem_p20', 'dem_p30', 'dem_p40', 'dem_p50',
    'dem_p60', 'dem_p70', 'dem_p80', 'dem_p90', 'dem_p100',
    'dem_mean', 'dem_std', 'dem_skewness', 'dem_kurtosis', 'dem_range',
    'dem_pca_1', 'dem_pca_2', 'dem_pca_3', 'dem_pca_4', 'dem_pca_5', 'dem_pca_6',
]

# Feature bundles for Paper 4 (original, without --intracell-dem)
FEATURE_SETS_PAPER4 = {
    'BASIC': FEATURES_BASIC,
}

# Feature bundles for Paper 5 (with --intracell-dem)
FEATURE_SETS_INTRACELL = {
    'BASIC_D10': FEATURES_BASIC_D10,
    'BASIC_PCA6': FEATURES_BASIC_PCA6,
}

ALL_BUNDLES = {
    'BASIC': FEATURES_BASIC,
    'BASIC_D10': FEATURES_BASIC_D10,
    'BASIC_PCA6': FEATURES_BASIC_PCA6,
    'BASIC_D10_STATS': FEATURES_BASIC_D10_STATS,
}


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class V2Config:
    """V2 ConvLSTM training configuration."""
    data_file: str = ''
    output_dir: str = 'models/output/V2_Enhanced_Models'
    input_window: int = 60
    epochs: int = 150
    batch_size: int = 2
    learning_rate: float = 1e-3
    patience: int = 50
    train_val_split: float = 0.8
    loss_weighting: str = 'uniform'
    consistency_weight: float = 0.1
    prediction_batch_size: int = 1
    enabled_horizons: List[int] = field(default_factory=lambda: [12])
    feature_sets: Dict[str, List[str]] = field(default_factory=dict)
    seed: int = 42
    light_mode: bool = False
    light_grid_size: int = 5

    @classmethod
    def from_yaml(cls, config_path: Path, intracell_dem: bool = False,
                  bundle: str = None) -> 'V2Config':
        """Load configuration from YAML file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        models = cfg.get('models', {})
        v2 = models.get('v2', {})
        env = cfg.get('environment', {})

        obj = cls(
            data_file=str(cfg.get('data', {}).get('dataset_nc', '')),
            output_dir=v2.get('output_dir', 'models/output/V2_Enhanced_Models'),
            input_window=v2.get('input_window', 60),
            epochs=v2.get('epochs', 150),
            batch_size=v2.get('batch_size', 2),
            learning_rate=v2.get('learning_rate', 1e-3),
            patience=v2.get('early_stopping_patience', 50),
            train_val_split=v2.get('train_val_split', 0.8),
            loss_weighting=v2.get('loss_weighting', 'uniform'),
            consistency_weight=v2.get('consistency_weight', 0.1),
            prediction_batch_size=v2.get('prediction_batch_size', 1),
            enabled_horizons=v2.get('enabled_horizons', [12]),
            seed=env.get('random_seed', 42),
        )

        if intracell_dem:
            ic = models.get('intracell_dem', {})
            obj.data_file = ic.get('dataset_nc',
                                   'data/output/complete_dataset_extended_dem_features.nc')
            obj.output_dir = ic.get('v2_output_dir',
                                    'models/output/V2_Enhanced_Models_intracell_dem')
            if bundle:
                obj.feature_sets = {bundle: ALL_BUNDLES[bundle]}
            else:
                obj.feature_sets = dict(FEATURE_SETS_INTRACELL)
        else:
            fs = models.get('feature_set', 'BASIC')
            obj.feature_sets = {fs: ALL_BUNDLES.get(fs, FEATURES_BASIC)}

        return obj

    @classmethod
    def default(cls, intracell_dem: bool = False, bundle: str = None) -> 'V2Config':
        """Default configuration matching the notebooks."""
        obj = cls(
            data_file='data/output/complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc',
        )
        if intracell_dem:
            obj.data_file = 'data/output/complete_dataset_extended_dem_features.nc'
            obj.output_dir = 'models/output/V2_Enhanced_Models_intracell_dem'
            if bundle:
                obj.feature_sets = {bundle: ALL_BUNDLES[bundle]}
            else:
                obj.feature_sets = dict(FEATURE_SETS_INTRACELL)
        else:
            obj.feature_sets = dict(FEATURE_SETS_PAPER4)
        return obj


# ============================================================================
# TENSORFLOW SETUP
# ============================================================================

def setup_tensorflow(seed=42, allow_missing=False):
    """Initialize TensorFlow with GPU memory growth and reproducibility.

    If allow_missing=True (dry-run), returns None when TF is not installed.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    try:
        import tensorflow as tf
    except ImportError:
        if allow_missing:
            logger.warning('TensorFlow not installed (dry-run mode)')
            return None
        logger.error('TensorFlow not installed. Install with: pip install tensorflow>=2.6.0')
        sys.exit(1)

    tf.random.set_seed(seed)
    np.random.seed(seed)

    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f'GPU available: {len(gpus)} device(s)')
        else:
            logger.warning('No GPU detected. Training will be slow.')
    except RuntimeError as e:
        logger.warning(f'GPU configuration: {e}')

    logger.info(f'TensorFlow {tf.__version__}')
    return tf


# ============================================================================
# CUSTOM LAYERS AND LOSSES
# ============================================================================

def define_custom_objects(tf):
    """Define custom Keras layers and losses. Returns dict of classes."""
    from tensorflow.keras.layers import Layer, Dense, Conv2D, Lambda
    from tensorflow.keras.losses import Loss

    class MultiHorizonLoss(Loss):
        """Weighted loss for multi-horizon forecasting."""
        def __init__(self, horizon_weights=None, name='multi_horizon_loss', **kwargs):
            super().__init__(name=name, **kwargs)
            if horizon_weights is None:
                horizon_weights = [0.4, 0.35, 0.25]
            self.horizon_weights = tf.constant(horizon_weights, dtype=tf.float32)

        def call(self, y_true, y_pred):
            y_pred = tf.maximum(y_pred, 0.0)
            mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[2, 3, 4])
            weights = tf.reshape(self.horizon_weights, [1, -1])
            return tf.reduce_mean(tf.reduce_sum(mse * weights, axis=1))

        def get_config(self):
            config = super().get_config()
            config.update({'horizon_weights': self.horizon_weights.numpy().tolist()})
            return config

    class CombinedLoss(Loss):
        """Multi-horizon MSE + temporal consistency."""
        def __init__(self, horizon_weights=None, consistency_weight=0.1,
                     name='combined_loss', **kwargs):
            super().__init__(name=name, **kwargs)
            if horizon_weights is None:
                horizon_weights = [0.4, 0.35, 0.25]
            self.horizon_weights = tf.constant(horizon_weights, dtype=tf.float32)
            self.consistency_weight = consistency_weight

        def call(self, y_true, y_pred):
            y_pred = tf.maximum(y_pred, 0.0)
            mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[2, 3, 4])
            weights = tf.reshape(self.horizon_weights, [1, -1])
            mh_loss = tf.reduce_mean(tf.reduce_sum(mse * weights, axis=1))
            temporal_diffs = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
            tc_loss = tf.reduce_mean(temporal_diffs)
            return mh_loss + self.consistency_weight * tc_loss

        def get_config(self):
            config = super().get_config()
            config.update({
                'horizon_weights': self.horizon_weights.numpy().tolist(),
                'consistency_weight': self.consistency_weight,
            })
            return config

    class SpatialReshapeLayer(Layer):
        """Reshape: (batch, time, H, W, C) -> (batch, time, H*W*C)."""
        def call(self, inputs):
            batch_size, time_steps, height, width, channels = tf.unstack(tf.shape(inputs))
            return tf.reshape(inputs, [batch_size, time_steps, height * width * channels])

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[1],
                    input_shape[2] * input_shape[3] * input_shape[4])

    class SpatialRestoreLayer(Layer):
        """Restore spatial dimensions after attention."""
        def __init__(self, height, width, channels, **kwargs):
            super().__init__(**kwargs)
            self.height = height
            self.width = width
            self.channels = channels

        def call(self, inputs):
            batch_size, time_steps, _ = tf.unstack(tf.shape(inputs))
            return tf.reshape(inputs, [batch_size, time_steps,
                                       self.height, self.width, self.channels])

        def get_config(self):
            config = super().get_config()
            config.update({'height': self.height, 'width': self.width,
                           'channels': self.channels})
            return config

    return {
        'MultiHorizonLoss': MultiHorizonLoss,
        'CombinedLoss': CombinedLoss,
        'SpatialReshapeLayer': SpatialReshapeLayer,
        'SpatialRestoreLayer': SpatialRestoreLayer,
    }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_conv_lstm(tf, n_feats, lat, lon, horizon):
    """Build ConvLSTM model (primary V2 architecture).

    Architecture: ConvLSTM2D(32) -> ConvLSTM2D(16) -> spatial_head
    """
    from tensorflow.keras.layers import Input, ConvLSTM2D, Conv2D, Lambda
    from tensorflow.keras.models import Model

    inp = Input(shape=(None, lat, lon, n_feats))
    x = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(inp)
    x = ConvLSTM2D(16, (3, 3), padding='same', return_sequences=False)(x)

    # Spatial head: Conv2D(horizon) -> transpose -> expand_dims
    x = Conv2D(horizon, (1, 1), padding='same', activation='linear',
               name='head_conv1x1')(x)
    x = Lambda(lambda t: tf.transpose(t, [0, 3, 1, 2]),
               name='head_transpose')(x)
    x = Lambda(lambda t: tf.expand_dims(t, -1),
               name='head_expand_dim')(x)

    model = Model(inp, x, name='ConvLSTM')

    expected = (None, horizon, lat, lon, 1)
    if model.output_shape != expected:
        raise ValueError(f'Output shape {model.output_shape} != expected {expected}')

    return model


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def load_and_validate_data(config: V2Config, base_path: Path):
    """Load dataset and validate features."""
    import xarray as xr

    data_file = base_path / config.data_file
    logger.info(f'Loading dataset: {data_file}')
    if not data_file.exists():
        raise FileNotFoundError(f'Dataset not found: {data_file}')

    ds = xr.open_dataset(data_file)

    if config.light_mode:
        g = config.light_grid_size
        lat_c = len(ds.latitude) // 2
        lon_c = len(ds.longitude) // 2
        ds = ds.isel(
            latitude=slice(lat_c - g // 2, lat_c - g // 2 + g),
            longitude=slice(lon_c - g // 2, lon_c - g // 2 + g),
        )
        logger.info(f'Light mode: {g}x{g} subset')

    n_lat, n_lon = len(ds.latitude), len(ds.longitude)
    logger.info(f'Dataset: time={len(ds.time)}, lat={n_lat}, lon={n_lon}')

    available = set(list(ds.data_vars) + list(ds.coords))
    for exp_name, feats in config.feature_sets.items():
        missing = [f for f in feats if f not in available]
        if missing:
            raise ValueError(f'{exp_name} missing features: {missing}')
        logger.info(f'  {exp_name}: all {len(feats)} features present')

    return ds, n_lat, n_lon


def fill_nan_with_median(X, features):
    """Fill NaN in features with per-feature spatial median.

    603 cells (15.2%) have NaN in DEM features because the 90m DEM
    doesn't cover those cells. Median fill minimizes bias (~200mm
    avoided vs fill=0).
    """
    nan_count = 0
    for i, feat in enumerate(features):
        if np.isnan(X[..., i]).any():
            valid_vals = X[0, :, :, i][~np.isnan(X[0, :, :, i])]
            fill_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0
            X[..., i] = np.nan_to_num(X[..., i], nan=fill_val)
            nan_count += 1
            logger.info(f'    Filled {feat} NaN with median={fill_val:.1f}')
    if nan_count > 0:
        logger.info(f'    Filled NaN in {nan_count} features')
    return X


def windowed_arrays(X, y, input_window, horizon, start_indices=None):
    """Create windowed arrays for sequence-to-sequence learning."""
    seq_X, seq_y = [], []
    T = len(X)
    if T < (input_window + horizon):
        raise ValueError(f'Not enough timesteps ({T}) for windows')
    iterator = start_indices if start_indices is not None else range(
        T - input_window - horizon + 1)
    for start in iterator:
        if start < 0:
            continue
        end_w = start + input_window
        end_y = end_w + horizon
        if end_y > T:
            continue
        Xw = X[start:end_w]
        yw = y[end_w:end_y]
        if np.isnan(Xw).any() or np.isnan(yw).any():
            continue
        seq_X.append(Xw)
        seq_y.append(yw)
    if not seq_X:
        raise ValueError('No valid windows found')
    return np.asarray(seq_X, dtype=np.float32), np.asarray(seq_y, dtype=np.float32)


def compute_split_indices(total_steps, input_window, horizon, split_ratio):
    """Return train/validation start indices ensuring no temporal leakage."""
    cutoff = max(0, int(total_steps * split_ratio))
    max_start = total_steps - input_window - horizon + 1
    train_end = max(0, cutoff - input_window - horizon + 1)
    train_indices = list(range(0, train_end))
    val_start = min(max_start, max(cutoff, 0))
    val_indices = list(range(val_start, max_start))
    if not train_indices or not val_indices:
        raise ValueError('Split produced empty train/val sets')
    return train_indices, val_indices


def scale_feature_blocks(X_tr, X_va, features):
    """Scale continuous features with DUAL scalers (BASE + DEM separately)."""
    from sklearn.preprocessing import StandardScaler

    X_tr_scaled = X_tr.copy()
    X_va_scaled = X_va.copy()

    def apply_scaler(idxs):
        if not idxs:
            return
        scaler = StandardScaler()
        tr_block = X_tr[..., idxs].reshape(-1, len(idxs))
        va_block = X_va[..., idxs].reshape(-1, len(idxs))
        scaler.fit(tr_block)
        X_tr_scaled[..., idxs] = scaler.transform(tr_block).reshape(
            X_tr[..., idxs].shape)
        X_va_scaled[..., idxs] = scaler.transform(va_block).reshape(
            X_va[..., idxs].shape)

    cont_indices = [i for i, f in enumerate(features) if f in BASE_CONTINUOUS_FEATURES]
    apply_scaler(cont_indices)

    dem_indices = [i for i, f in enumerate(features) if f in DEM_CONTINUOUS_FEATURES]
    apply_scaler(dem_indices)

    return X_tr_scaled, X_va_scaled


def preprocess_data(ds, config: V2Config, n_lat, n_lon, horizon):
    """Preprocess data per experiment with no leakage."""
    from sklearn.preprocessing import StandardScaler

    data_splits = {}

    for exp_name, features in config.feature_sets.items():
        logger.info(f'Preprocessing {exp_name}...')
        X = np.stack([ds[feat].values for feat in features], axis=-1)
        y = ds['total_precipitation'].values[..., np.newaxis]

        X = fill_nan_with_median(X, features)

        total_steps = X.shape[0]
        train_idx, val_idx = compute_split_indices(
            total_steps, config.input_window, horizon, config.train_val_split)

        X_tr, y_tr = windowed_arrays(X, y, config.input_window, horizon, train_idx)
        X_va, y_va = windowed_arrays(X, y, config.input_window, horizon, val_idx)

        X_tr_s, X_va_s = scale_feature_blocks(X_tr, X_va, features)

        y_scaler = StandardScaler()
        y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).reshape(y_tr.shape)
        y_va_s = y_scaler.transform(y_va.reshape(-1, 1)).reshape(y_va.shape)

        data_splits[exp_name] = (
            X_tr_s.astype(np.float32), y_tr_s.astype(np.float32),
            X_va_s.astype(np.float32), y_va_s.astype(np.float32),
            y_scaler,
        )
        logger.info(f'  {exp_name}: {X_tr.shape[0]} train, {X_va.shape[0]} val windows')
        logger.info(f'  X shape: {X_tr_s.shape}, y shape: {y_tr_s.shape}')

    return data_splits


# ============================================================================
# TRAINING INFRASTRUCTURE
# ============================================================================

def compute_horizon_weights(H, strategy='uniform'):
    """Compute per-horizon loss weights."""
    if H <= 0:
        return []
    if strategy == 'linear_decay':
        w = np.linspace(1.0, 0.2, H, dtype=np.float32)
    else:
        w = np.ones(H, dtype=np.float32)
    return (w / w.sum()).tolist()


def batch_predict(model, X, batch_size=1):
    """Predict in mini-batches to avoid OOM."""
    n_samples = X.shape[0]
    num_batches = int(math.ceil(n_samples / float(batch_size)))
    predictions = []
    for b in range(num_batches):
        start = b * batch_size
        end = min(n_samples, start + batch_size)
        batch_pred = model.predict(X[start:end], verbose=0, batch_size=batch_size)
        predictions.append(np.asarray(batch_pred))
    return np.concatenate(predictions, axis=0)


def train_single_model(tf, custom_objects, model, X_tr, y_tr, X_va, y_va,
                       config: V2Config, model_name, exp_name, out_dir, horizon):
    """Train a single model with callbacks and save results."""
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback
    )
    from tensorflow.keras.optimizers import Adam

    metrics_dir = out_dir / f'h{horizon}' / exp_name / 'training_metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_path = metrics_dir / f'{model_name}_best_h{horizon}.h5'

    # Save hyperparameters
    hyperparams = {
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'patience': config.patience,
        'model_params': model.count_params(),
        'version': 'V2_Workflow',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(metrics_dir / f'{model_name}_hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=4)

    # Loss
    CombinedLoss = custom_objects['CombinedLoss']
    horizon_weights = compute_horizon_weights(horizon, config.loss_weighting)
    loss = CombinedLoss(horizon_weights=horizon_weights,
                        consistency_weight=config.consistency_weight)

    model.compile(optimizer=Adam(config.learning_rate), loss=loss, metrics=['mae'])
    logger.info(f'Model compiled: {model_name} ({model.count_params():,} params)')

    # Training monitor callback
    class TrainingMonitor(Callback):
        def __init__(self):
            super().__init__()
            self.start_time = None

        def on_train_begin(self, logs=None):
            self.start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.time() - self.start_time
            loss_val = logs.get('loss', float('nan'))
            vloss = logs.get('val_loss', float('nan'))
            try:
                lr = float(self.model.optimizer.learning_rate)
            except Exception:
                lr = 0.0
            logger.info(f'{exp_name}-{model_name} | Epoch {epoch+1} | '
                        f'Loss: {loss_val:.4f} | Val: {vloss:.4f} | '
                        f'LR: {lr:.2e} | {elapsed:.0f}s')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.patience,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_path), save_best_only=True,
                        monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=config.patience // 2, min_lr=1e-6, verbose=1),
        CSVLogger(str(metrics_dir / f'{model_name}_training_log_h{horizon}.csv'),
                  separator=',', append=False),
        TrainingMonitor(),
    ]

    logger.info(f'Training {model_name} on {exp_name} '
                f'(batch_size={config.batch_size})...')

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=config.epochs,
        batch_size=max(1, config.batch_size),
        callbacks=callbacks,
        verbose=0,
    )

    # Save history
    val_losses = history.history['val_loss']
    best_idx = int(np.argmin(val_losses))
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in val_losses],
        'mae': [float(x) for x in history.history.get('mae', [])],
        'val_mae': [float(x) for x in history.history.get('val_mae', [])],
        'best_epoch': best_idx,
        'best_val_loss': float(val_losses[best_idx]),
    }
    with open(metrics_dir / f'{model_name}_history.json', 'w') as f:
        json.dump(history_dict, f, indent=4)

    logger.info(f'Best val_loss: {val_losses[best_idx]:.4f} at epoch {best_idx+1}')

    # Save learning curve plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, len(history_dict['loss']) + 1)
        ax.plot(epochs_range, history_dict['loss'], 'b-', label='Train')
        ax.plot(epochs_range, history_dict['val_loss'], 'r-', label='Val')
        ax.plot(best_idx + 1, val_losses[best_idx], 'r*', markersize=12,
                label=f'Best: {val_losses[best_idx]:.4f}')
        ax.set_title(f'{model_name} - {exp_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig(metrics_dir / f'{model_name}_learning_curve.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except ImportError:
        pass

    return model, history_dict


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def run_training(tf, config: V2Config, base_path: Path):
    """Run the full V2 ConvLSTM training pipeline."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    custom_objects = define_custom_objects(tf)
    ds, n_lat, n_lon = load_and_validate_data(config, base_path)
    out_dir = base_path / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for horizon in config.enabled_horizons:
        logger.info(f'\n{"="*60}')
        logger.info(f'Training for horizon H={horizon}')
        logger.info(f'{"="*60}')

        data_splits = preprocess_data(ds, config, n_lat, n_lon, horizon)

        for exp_name in config.feature_sets:
            if exp_name not in data_splits:
                logger.warning(f'Skipping {exp_name}: data not available')
                continue

            X_tr, y_tr, X_va, y_va, scaler = data_splits[exp_name]
            n_features = len(config.feature_sets[exp_name])
            model_name = 'ConvLSTM'

            logger.info(f'\n--- {exp_name}: {n_features} features ---')

            tf.keras.backend.clear_session()

            try:
                model = build_conv_lstm(tf, n_features, n_lat, n_lon, horizon)
                model, history = train_single_model(
                    tf, custom_objects, model,
                    X_tr, y_tr, X_va, y_va,
                    config, model_name, exp_name, out_dir, horizon,
                )

                if not history or not history.get('val_loss'):
                    logger.error(f'Training failed for {model_name}')
                    continue

                # Generate predictions
                y_hat_sc = batch_predict(model, X_va,
                                         batch_size=config.prediction_batch_size)
                if y_hat_sc.shape != y_va.shape:
                    y_hat_sc = y_hat_sc.reshape(y_va.shape)
                y_hat = scaler.inverse_transform(
                    y_hat_sc.reshape(-1, 1)).reshape(y_va.shape)
                y_true = scaler.inverse_transform(
                    y_va.reshape(-1, 1)).reshape(y_va.shape)

                # Per-horizon metrics
                for h in range(horizon):
                    y_true_h = y_true[:, h, ..., 0]
                    y_pred_h = y_hat[:, h, ..., 0]
                    rmse = np.sqrt(mean_squared_error(
                        y_true_h.ravel(), y_pred_h.ravel()))
                    mae = mean_absolute_error(
                        y_true_h.ravel(), y_pred_h.ravel())
                    r2 = r2_score(y_true_h.ravel(), y_pred_h.ravel())
                    results.append({
                        'TotalHorizon': horizon,
                        'Experiment': exp_name,
                        'Model': model_name,
                        'H': h + 1,
                        'RMSE': rmse, 'MAE': mae, 'R^2': r2,
                        'Mean_True_mm': float(np.mean(y_true_h)),
                        'Mean_Pred_mm': float(np.mean(y_pred_h)),
                    })
                    logger.info(f'  H={h+1}: RMSE={rmse:.2f}, '
                                f'MAE={mae:.2f}, R2={r2:.4f}')

                # Compute forecast_dates for ACC (benchmark script 14)
                total_steps = len(ds.time)
                _, val_idx = compute_split_indices(
                    total_steps, config.input_window, horizon,
                    config.train_val_split)
                times = pd.to_datetime(ds.time.values)
                forecast_dates = []
                for vi in val_idx:
                    sample_dates = []
                    for h_offset in range(horizon):
                        t_idx = vi + config.input_window + h_offset
                        if t_idx < total_steps:
                            sample_dates.append(
                                times[t_idx].strftime('%Y-%m'))
                    if len(sample_dates) == horizon:
                        forecast_dates.append(sample_dates)

                # Export predictions for V10 Late Fusion
                export_dir = (out_dir / 'map_exports' / f'H{horizon}'
                              / exp_name / model_name)
                export_dir.mkdir(parents=True, exist_ok=True)
                np.save(export_dir / 'predictions.npy',
                        y_hat.astype(np.float32))
                np.save(export_dir / 'targets.npy',
                        y_true.astype(np.float32))
                metadata = {
                    'model': model_name,
                    'experiment': exp_name,
                    'horizon': horizon,
                    'shape': list(y_hat.shape),
                    'rmse_mean': float(np.sqrt(np.mean((y_hat - y_true)**2))),
                    'r2_mean': float(r2_score(y_true.ravel(), y_hat.ravel())),
                    'forecast_dates': forecast_dates,
                    'generated_at': datetime.now().isoformat(),
                }
                with open(export_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f'  Predictions saved: {export_dir}')
                logger.info(f'  forecast_dates: {len(forecast_dates)} samples '
                            f'x {horizon} horizons')

            except Exception as e:
                logger.error(f'ERROR training {model_name} for {exp_name}: {e}')
                import traceback
                traceback.print_exc()

            tf.keras.backend.clear_session()
            gc.collect()

    # Save results CSV
    if results:
        res_df = pd.DataFrame(results)
        out_csv = out_dir / 'metrics_spatial_v2_h12.csv'
        res_df.to_csv(out_csv, index=False)
        logger.info(f'Results saved to {out_csv}')

    # Verification summary
    print('\n' + '=' * 60)
    print('  V2 ConvLSTM TRAINING COMPLETE')
    print('=' * 60)
    for horizon in config.enabled_horizons:
        for exp_name in config.feature_sets:
            pred_dir = (out_dir / 'map_exports' / f'H{horizon}'
                        / exp_name / 'ConvLSTM')
            pred_file = pred_dir / 'predictions.npy'
            targ_file = pred_dir / 'targets.npy'
            if pred_file.exists() and targ_file.exists():
                from sklearn.metrics import r2_score
                pred = np.load(pred_file)
                targ = np.load(targ_file)
                r2 = r2_score(targ.ravel(), pred.ravel())
                rmse = np.sqrt(np.mean((pred - targ)**2))
                print(f'  {exp_name} (H{horizon}): R2={r2:.4f}, '
                      f'RMSE={rmse:.2f}mm, shape={pred.shape}')
            else:
                print(f'  {exp_name} (H{horizon}): MISSING')
    print('=' * 60)

    ds.close()
    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train V2 Enhanced ConvLSTM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflows/05_train_v2_convlstm.py --dry-run
  python workflows/05_train_v2_convlstm.py --intracell-dem --bundle BASIC_D10
  python workflows/05_train_v2_convlstm.py --light-mode --epochs 5
  python workflows/05_train_v2_convlstm.py --config workflows/config.yaml
        """,
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml')
    parser.add_argument('--feature-set', type=str, default=None,
                        help='Feature set override')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check setup without training')
    parser.add_argument('--intracell-dem', action='store_true',
                        help='Use intra-cell DEM features (Paper 5)')
    parser.add_argument('--bundle', type=str, default=None,
                        choices=['BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS'],
                        help='Feature bundle for --intracell-dem')
    parser.add_argument('--light-mode', action='store_true',
                        help='Use small grid subset for testing')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if args.config:
        config = V2Config.from_yaml(Path(args.config),
                                    intracell_dem=args.intracell_dem,
                                    bundle=args.bundle)
    else:
        default_config = PROJECT_ROOT / 'workflows' / 'config.yaml'
        if default_config.exists():
            config = V2Config.from_yaml(default_config,
                                        intracell_dem=args.intracell_dem,
                                        bundle=args.bundle)
        else:
            config = V2Config.default(intracell_dem=args.intracell_dem,
                                      bundle=args.bundle)

    # CLI overrides
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.feature_set is not None:
        if args.feature_set in ALL_BUNDLES:
            config.feature_sets = {args.feature_set: ALL_BUNDLES[args.feature_set]}
        else:
            logger.error(f'Unknown feature set: {args.feature_set}')
            sys.exit(1)
    if args.light_mode:
        config.light_mode = True

    # Setup TF
    tf = setup_tensorflow(config.seed, allow_missing=args.dry_run)

    logger.info('=' * 60)
    logger.info('  V2 ConvLSTM Training Pipeline')
    logger.info('=' * 60)
    logger.info(f'  Dataset: {config.data_file}')
    logger.info(f'  Output: {config.output_dir}')
    logger.info(f'  Bundles: {list(config.feature_sets.keys())}')
    logger.info(f'  Epochs: {config.epochs}, Batch: {config.batch_size}')
    logger.info(f'  Horizons: {config.enabled_horizons}')
    if config.light_mode:
        logger.info(f'  Light mode: {config.light_grid_size}x{config.light_grid_size}')
    logger.info('=' * 60)

    if args.dry_run:
        # Verify dataset exists and features are present
        import xarray as xr
        data_file = PROJECT_ROOT / config.data_file
        if data_file.exists():
            ds = xr.open_dataset(data_file)
            n_lat, n_lon = len(ds.latitude), len(ds.longitude)
            available = set(list(ds.data_vars) + list(ds.coords))
            for exp_name, feats in config.feature_sets.items():
                missing = [f for f in feats if f not in available]
                status = 'OK' if not missing else f'MISSING: {missing}'
                logger.info(f'  {exp_name}: {len(feats)} features - {status}')
            logger.info(f'  Grid: {n_lat}x{n_lon}')
            ds.close()
        else:
            logger.warning(f'  Dataset not found: {data_file}')
        logger.info(f'  TensorFlow: {"available" if tf is not None else "NOT installed"}')
        logger.info('Dry run complete.')
        return

    if tf is None:
        logger.error('TensorFlow is required for training. '
                     'Install with: pip install tensorflow>=2.6.0')
        sys.exit(1)

    run_training(tf, config, PROJECT_ROOT)


if __name__ == '__main__':
    main()
