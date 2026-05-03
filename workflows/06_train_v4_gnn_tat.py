"""
Pipeline Stage 06: Train V4 GNN-TAT (Graph Neural Network with Temporal Attention)

Trains the V4 GNN-TAT model for spatiotemporal precipitation prediction.
Replicates the full training pipeline from the notebooks into a standalone,
region-agnostic CLI script for use in Barcelona and other deployments.

Architecture:
- SpatialGraphBuilder: Haversine distance + elevation similarity + Pearson correlation
- SpatialGNNEncoder (GAT/SAGE/GCN): residual + GELU + LayerNorm
- TemporalAttention: Multi-head with residual connection
- LSTM decoder -> output projection
- Chunked GNN processing to avoid OOM on large grids

Source: models/intracell_dem/train_v4_gnn_tat_intracell_dem.ipynb
        models/base_models_gnn_tat_v4.ipynb

Usage:
    python workflows/06_train_v4_gnn_tat.py --dry-run
    python workflows/06_train_v4_gnn_tat.py --config workflows/config.yaml
    python workflows/06_train_v4_gnn_tat.py --intracell-dem --bundle BASIC_D10
    python workflows/06_train_v4_gnn_tat.py --light-mode --epochs 5

Note: Requires PyTorch >= 1.9 and PyTorch Geometric.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import gc
import json
import os
import time
import warnings
from collections import defaultdict
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _versions import log_environment, log_script_version

# ============================================================================
# FEATURE DEFINITIONS (same as V2 for consistency)
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

FEATURE_SETS_PAPER4 = {'BASIC': FEATURES_BASIC}
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

# Default GNN config (matches notebook CONFIG['gnn_config'])
DEFAULT_GNN_CONFIG = {
    'hidden_dim': 64,
    'num_gnn_layers': 2,
    'gnn_type': 'GAT',
    'num_heads': 4,
    'dropout': 0.1,
    'temporal_hidden_dim': 64,
    'num_temporal_heads': 4,
    'temporal_dropout': 0.1,
    'lstm_hidden_dim': 64,
    'num_lstm_layers': 2,
    'edge_threshold': 0.3,
    'max_neighbors': 8,
    'use_distance_edges': True,
    'use_elevation_edges': True,
    'use_correlation_edges': True,
    'distance_scale_km': 10.0,
    'elevation_scale': 0.2,
    'elevation_weight': 0.3,
    'correlation_weight': 0.5,
    'min_edge_weight': 0.01,
}


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class V4Config:
    """V4 GNN-TAT training configuration."""
    data_file: str = ''
    output_dir: str = 'models/output/V4_GNN_TAT_Models'
    input_window: int = 60
    epochs: int = 150
    batch_size: int = 2
    learning_rate: float = 1e-3
    patience: int = 50
    train_val_split: float = 0.8
    weight_decay: float = 1e-5
    enabled_horizons: List[int] = field(default_factory=lambda: [12])
    feature_sets: Dict[str, List[str]] = field(default_factory=dict)
    gnn_config: Dict = field(default_factory=lambda: dict(DEFAULT_GNN_CONFIG))
    seed: int = 42
    light_mode: bool = False
    light_grid_size: int = 5

    @classmethod
    def from_yaml(cls, config_path: Path, intracell_dem: bool = False,
                  bundle: str = None) -> 'V4Config':
        """Load configuration from YAML file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        models = cfg.get('models', {})
        v4 = models.get('v4', {})
        env = cfg.get('environment', {})

        gnn_cfg = dict(DEFAULT_GNN_CONFIG)
        if 'gnn_config' in v4:
            gnn_cfg.update(v4['gnn_config'])

        obj = cls(
            data_file=str(cfg.get('data', {}).get('dataset_nc', '')),
            output_dir=v4.get('output_dir', 'models/output/V4_GNN_TAT_Models'),
            input_window=v4.get('input_window', 60),
            epochs=v4.get('epochs', 150),
            batch_size=v4.get('batch_size', 2),
            learning_rate=v4.get('learning_rate', 1e-3),
            patience=v4.get('early_stopping_patience', 50),
            train_val_split=v4.get('train_val_split', 0.8),
            weight_decay=v4.get('weight_decay', 1e-5),
            enabled_horizons=v4.get('enabled_horizons', [12]),
            gnn_config=gnn_cfg,
            seed=env.get('random_seed', 42),
        )

        if intracell_dem:
            ic = models.get('intracell_dem', {})
            obj.data_file = ic.get('dataset_nc',
                                   'data/output/complete_dataset_extended_dem_features.nc')
            obj.output_dir = ic.get('v4_output_dir',
                                    'models/output/intracell_dem/GNN_TAT_GAT')
            if bundle:
                obj.feature_sets = {bundle: ALL_BUNDLES[bundle]}
            else:
                obj.feature_sets = dict(FEATURE_SETS_INTRACELL)
        else:
            fs = models.get('feature_set', 'BASIC')
            obj.feature_sets = {fs: ALL_BUNDLES.get(fs, FEATURES_BASIC)}

        return obj

    @classmethod
    def default(cls, intracell_dem: bool = False, bundle: str = None) -> 'V4Config':
        """Default configuration matching the notebooks."""
        obj = cls(
            data_file='data/output/complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc',
        )
        if intracell_dem:
            obj.data_file = 'data/output/complete_dataset_extended_dem_features.nc'
            obj.output_dir = 'models/output/intracell_dem/GNN_TAT_GAT'
            if bundle:
                obj.feature_sets = {bundle: ALL_BUNDLES[bundle]}
            else:
                obj.feature_sets = dict(FEATURE_SETS_INTRACELL)
        else:
            obj.feature_sets = dict(FEATURE_SETS_PAPER4)
        return obj


# ============================================================================
# PYTORCH SETUP
# ============================================================================

def setup_pytorch(seed=42, allow_missing=False):
    """Initialize PyTorch with GPU detection and reproducibility.

    If allow_missing=True (dry-run), returns None when torch is not installed.
    """
    try:
        import torch
    except ImportError:
        if allow_missing:
            logger.warning('PyTorch not installed (dry-run mode)')
            return None
        logger.error('PyTorch not installed. Install with: pip install torch>=1.9.0')
        sys.exit(1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logger.info(f'PyTorch {torch.__version__} - '
                    f'CUDA: {torch.cuda.get_device_name(0)}')
    else:
        logger.warning(f'PyTorch {torch.__version__} - No CUDA GPU detected.')

    try:
        import torch_geometric
        logger.info(f'PyTorch Geometric {torch_geometric.__version__}')
    except ImportError:
        if not allow_missing:
            logger.error('PyTorch Geometric not installed. '
                         'Install with: pip install torch-geometric')
            sys.exit(1)
        logger.warning('PyTorch Geometric not installed (dry-run mode)')

    return torch


# ============================================================================
# SPATIAL GRAPH CONSTRUCTION
# ============================================================================

class SpatialGraphBuilder:
    """Builds spatial graph from precipitation grid data.

    Edge types: Haversine distance, elevation similarity, temporal correlation.
    """

    def __init__(self, lat_coords, lon_coords, elevation, config):
        self.lat_coords = lat_coords
        self.lon_coords = lon_coords
        self.elevation = elevation
        self.config = config
        self.n_lat = len(lat_coords)
        self.n_lon = len(lon_coords)
        self.n_nodes = self.n_lat * self.n_lon
        self.node_positions = np.array(
            [[la, lo] for la in lat_coords for lo in lon_coords])
        self.node_elevations = elevation.flatten()
        logger.info(f'SpatialGraphBuilder: {self.n_nodes} nodes '
                    f'({self.n_lat}x{self.n_lon})')

    def compute_distance_matrix(self):
        """Haversine distance between all node pairs (km)."""
        pos_rad = np.radians(self.node_positions)
        lat1, lon1 = pos_rad[:, 0:1], pos_rad[:, 1:2]
        dlat = lat1.T - lat1
        dlon = lon1.T - lon1
        a = (np.sin(dlat / 2)**2 +
             np.cos(lat1) * np.cos(lat1.T) * np.sin(dlon / 2)**2)
        return 6371.0 * 2 * np.arcsin(np.sqrt(a))

    def compute_elevation_similarity(self):
        """Exponential decay similarity based on elevation difference."""
        elev = self.node_elevations.reshape(-1, 1)
        elev_diff = np.abs(elev - elev.T)
        elev_range = elev.max() - elev.min() + 1e-6
        scale = self.config.get('elevation_scale', 0.2)
        return np.exp(-elev_diff / (elev_range * scale))

    def compute_correlation_matrix(self, precip_series):
        """Pearson temporal correlation between nodes."""
        T = precip_series.shape[0]
        flat = precip_series.reshape(T, -1)
        flat = np.nan_to_num(flat, nan=0.0)
        centered = flat - flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True) + 1e-8
        norm = centered / std
        return np.clip((norm.T @ norm) / T, -1, 1)

    def build_adjacency_matrix(self, precip_series=None):
        """Build adjacency and extract edge_index/edge_weight."""
        cfg = self.config
        adj = np.zeros((self.n_nodes, self.n_nodes))

        if cfg.get('use_distance_edges', True):
            dist = self.compute_distance_matrix()
            dist[dist == 0] = 1e-6
            sim = 1.0 / (1.0 + dist / cfg.get('distance_scale_km', 10.0))
            for i in range(self.n_nodes):
                neighbors = np.argsort(dist[i])[:cfg['max_neighbors'] + 1]
                neighbors = neighbors[neighbors != i][:cfg['max_neighbors']]
                adj[i, neighbors] += sim[i, neighbors]
            logger.info(f'  Distance edges added (k={cfg["max_neighbors"]})')

        if cfg.get('use_elevation_edges', True):
            adj += (self.compute_elevation_similarity()
                    * cfg.get('elevation_weight', 0.3))
            logger.info('  Elevation edges added')

        if cfg.get('use_correlation_edges', True) and precip_series is not None:
            corr = self.compute_correlation_matrix(precip_series)
            adj += (np.maximum(corr - cfg['edge_threshold'], 0)
                    * cfg.get('correlation_weight', 0.5))
            logger.info('  Correlation edges added')

        np.fill_diagonal(adj, 0)
        adj_max = adj.max()
        if adj_max > 0:
            adj = adj / adj_max
        adj = (adj + adj.T) / 2

        # Vectorized edge extraction
        min_w = cfg.get('min_edge_weight', 0.01)
        rows, cols = np.where(adj > min_w)
        edge_index = np.stack([rows, cols], axis=0)
        edge_weight = adj[rows, cols]
        logger.info(f'  Edges: {len(edge_weight)}, '
                    f'avg degree: {len(edge_weight)/self.n_nodes:.1f}')

        # Limit edges if too many
        max_edges = 500_000
        if len(edge_weight) > max_edges:
            top_idx = np.argsort(edge_weight)[-max_edges:]
            edge_index = edge_index[:, top_idx]
            edge_weight = edge_weight[top_idx]
            logger.info(f'  Reduced to {len(edge_weight)} edges')

        return edge_index, edge_weight


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def load_and_validate_data(config: V4Config, base_path: Path):
    """Load dataset and validate features."""
    import xarray as xr

    data_file = base_path / config.data_file
    logger.info(f'Loading dataset: {data_file}')
    if not data_file.exists():
        raise FileNotFoundError(f'Dataset not found: {data_file}')

    engines = ['h5netcdf', 'netcdf4', 'scipy']
    ds = None
    for engine in engines:
        try:
            ds = xr.open_dataset(data_file, engine=engine)
            logger.info(f'Loaded with engine: {engine}')
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError('Could not open dataset with any engine')

    if config.light_mode:
        g = config.light_grid_size
        lc = len(ds.latitude) // 2
        lnc = len(ds.longitude) // 2
        ds = ds.isel(
            latitude=slice(lc - g // 2, lc - g // 2 + g),
            longitude=slice(lnc - g // 2, lnc - g // 2 + g),
        )
        logger.info(f'Light mode: {g}x{g} subset')

    n_lat, n_lon = len(ds.latitude), len(ds.longitude)
    lat_coords = ds.latitude.values
    lon_coords = ds.longitude.values

    available = set(list(ds.data_vars) + list(ds.coords))
    for exp_name, feats in config.feature_sets.items():
        missing = [f for f in feats if f not in available]
        if missing:
            raise ValueError(f'{exp_name} missing: {missing}')
        logger.info(f'  {exp_name}: all {len(feats)} features present')

    logger.info(f'Grid: {n_lat}x{n_lon} = {n_lat * n_lon} nodes, '
                f'{len(ds.time)} timesteps')
    return ds, n_lat, n_lon, lat_coords, lon_coords


def fill_nan_with_median(X, feature_list):
    """Fill NaN in features with per-feature spatial median."""
    nan_count = 0
    for i, feat in enumerate(feature_list):
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
    iterator = (start_indices if start_indices is not None
                else range(T - input_window - horizon + 1))
    for start in iterator:
        if start < 0:
            continue
        end_w = start + input_window
        end_y = end_w + horizon
        if end_y > T:
            continue
        Xw, yw = X[start:end_w], y[end_w:end_y]
        if np.isnan(Xw).any() or np.isnan(yw).any():
            continue
        seq_X.append(Xw)
        seq_y.append(yw)
    if not seq_X:
        raise ValueError('No valid windows')
    return np.asarray(seq_X, np.float32), np.asarray(seq_y, np.float32)


def compute_split_indices(total_steps, input_window, horizon, split_ratio):
    """Return train/val start indices with no temporal leakage."""
    cutoff = max(0, int(total_steps * split_ratio))
    max_start = total_steps - input_window - horizon + 1
    train_end = max(0, cutoff - input_window - horizon + 1)
    train_indices = list(range(0, train_end))
    val_start = min(max_start, max(cutoff, 0))
    val_indices = list(range(val_start, max_start))
    if not train_indices or not val_indices:
        raise ValueError('Split produced empty train/val sets')
    return train_indices, val_indices


def preprocess_data(ds, config: V4Config, n_lat, n_lon, horizon):
    """Preprocess data per experiment. V4 uses SINGLE scaler for all features."""
    from sklearn.preprocessing import StandardScaler

    results = {}
    for exp_name, feature_list in config.feature_sets.items():
        logger.info(f'Preprocessing {exp_name}...')
        arrays = []
        for feat in feature_list:
            if feat in ds.data_vars:
                arr = ds[feat].values
            elif feat in ds.coords:
                arr = ds[feat].values
            else:
                logger.warning(f'  Feature {feat} not found, skipping')
                continue

            if arr.ndim == 2:
                arr = np.broadcast_to(arr, (len(ds.time), n_lat, n_lon))
            elif arr.ndim == 1:
                arr = np.broadcast_to(
                    arr[:, np.newaxis, np.newaxis], (len(ds.time), n_lat, n_lon))
            arrays.append(arr)

        X = np.stack(arrays, axis=-1).astype(np.float32)
        y = ds['total_precipitation'].values.astype(np.float32)

        X = fill_nan_with_median(X, feature_list)
        y = np.nan_to_num(y, nan=0.0)

        train_idx, val_idx = compute_split_indices(
            len(ds.time), config.input_window, horizon, config.train_val_split)

        X_tr, y_tr = windowed_arrays(X, y, config.input_window, horizon, train_idx)
        X_va, y_va = windowed_arrays(X, y, config.input_window, horizon, val_idx)

        # V4: single scaler for all features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        scaler_X.fit(X_tr.reshape(-1, X_tr.shape[-1]))
        scaler_y.fit(y_tr.reshape(-1, 1))

        X_tr_sc = scaler_X.transform(
            X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
        X_va_sc = scaler_X.transform(
            X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)
        y_tr_sc = scaler_y.transform(y_tr.reshape(-1, 1)).reshape(y_tr.shape)
        y_va_sc = scaler_y.transform(y_va.reshape(-1, 1)).reshape(y_va.shape)

        # V4 expects y with trailing dim
        y_tr_sc = np.expand_dims(y_tr_sc, -1)
        y_va_sc = np.expand_dims(y_va_sc, -1)

        results[exp_name] = (X_tr_sc, y_tr_sc, X_va_sc, y_va_sc, scaler_y)
        logger.info(f'  {exp_name}: X_tr={X_tr_sc.shape}, y_tr={y_tr_sc.shape}')

    return results


# ============================================================================
# GNN-TAT MODEL ARCHITECTURE
# ============================================================================

def define_model_classes(torch):
    """Define PyTorch model classes. Returns dict of classes."""
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv

    class TemporalAttention(nn.Module):
        """Multi-Head Temporal Attention with residual connection."""
        def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            self.hidden_dim = hidden_dim
            assert hidden_dim % num_heads == 0

            self.q_proj = nn.Linear(input_dim, hidden_dim)
            self.k_proj = nn.Linear(input_dim, hidden_dim)
            self.v_proj = nn.Linear(input_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.residual_proj = (nn.Linear(input_dim, hidden_dim)
                                  if input_dim != hidden_dim else nn.Identity())
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.scale = self.head_dim ** -0.5

        def forward(self, x):
            B, S, _ = x.shape
            residual = self.residual_proj(x)
            Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            attn = F.softmax(
                torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
            attn = self.dropout(attn)
            out = (torch.matmul(attn, V).transpose(1, 2)
                   .contiguous().view(B, S, self.hidden_dim))
            return self.layer_norm(residual + self.dropout(self.out_proj(out)))

    class SpatialGNNEncoder(nn.Module):
        """GNN Encoder supporting GAT, SAGE, GCN."""
        def __init__(self, input_dim, hidden_dim, num_layers,
                     gnn_type='GAT', num_heads=4, dropout=0.1):
            super().__init__()
            self.gnn_type = gnn_type
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.gnn_layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                if gnn_type == 'GAT':
                    layer = GATConv(hidden_dim, hidden_dim // num_heads,
                                    heads=num_heads, dropout=dropout, concat=True)
                elif gnn_type == 'SAGE':
                    layer = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
                else:
                    layer = GCNConv(hidden_dim, hidden_dim,
                                    add_self_loops=True, normalize=True)
                self.gnn_layers.append(layer)
                self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index, edge_weight=None):
            x = self.input_proj(x)
            for gnn, norm in zip(self.gnn_layers, self.norms):
                residual = x
                if self.gnn_type == 'GCN':
                    x = gnn(x, edge_index, edge_weight)
                else:
                    x = gnn(x, edge_index)
                x = F.gelu(x)
                x = self.dropout(x)
                x = norm(x + residual)
            return x

    class GNN_TAT(nn.Module):
        """Graph Neural Network with Temporal Attention (memory-optimized)."""
        def __init__(self, n_features, n_nodes, n_lat, n_lon,
                     horizon, gnn_config, gnn_chunk_size=15):
            super().__init__()
            self.n_features = n_features
            self.n_nodes = n_nodes
            self.n_lat = n_lat
            self.n_lon = n_lon
            self.horizon = horizon
            self.gnn_chunk_size = gnn_chunk_size

            hidden_dim = gnn_config['hidden_dim']
            self.hidden_dim = hidden_dim

            self.gnn_encoder = SpatialGNNEncoder(
                n_features, hidden_dim,
                gnn_config['num_gnn_layers'],
                gnn_config['gnn_type'],
                gnn_config['num_heads'],
                gnn_config['dropout'],
            )
            self.temporal_attention = TemporalAttention(
                hidden_dim, hidden_dim,
                gnn_config['num_temporal_heads'],
                gnn_config['temporal_dropout'],
            )
            self.lstm = nn.LSTM(
                hidden_dim, hidden_dim,
                gnn_config['num_lstm_layers'],
                batch_first=True,
                dropout=(gnn_config['dropout']
                         if gnn_config['num_lstm_layers'] > 1 else 0),
            )
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(gnn_config['dropout']),
                nn.Linear(hidden_dim, horizon),
            )

            total_params = sum(p.numel() for p in self.parameters())
            logger.info(f'GNN-TAT: {n_features} features, {n_nodes} nodes, '
                        f'{total_params:,} params, chunk={gnn_chunk_size}')

        def _batch_edge_index(self, edge_index, num_graphs, device):
            num_edges = edge_index.shape[1]
            offsets = (torch.arange(num_graphs, device=device)
                       .view(-1, 1, 1) * self.n_nodes)
            batch_ei = (edge_index.unsqueeze(0).expand(num_graphs, -1, -1)
                        + offsets.expand(-1, 2, num_edges))
            return batch_ei.permute(1, 0, 2).reshape(2, -1)

        def _process_gnn_chunk(self, x_chunk, edge_index, edge_weight=None):
            chunk_size = x_chunk.shape[0]
            batch_ei = self._batch_edge_index(
                edge_index, chunk_size, x_chunk.device)
            batch_ew = (edge_weight.repeat(chunk_size)
                        if edge_weight is not None else None)
            x_nodes = x_chunk.view(-1, self.n_features)
            return self.gnn_encoder(
                x_nodes, batch_ei, batch_ew
            ).view(chunk_size, self.n_nodes, self.hidden_dim)

        def forward(self, x, edge_index, edge_weight=None):
            B, S = x.shape[:2]
            x = x.view(B, S, self.n_nodes, self.n_features)
            x_flat = x.view(B * S, self.n_nodes, self.n_features)
            total = B * S

            # Chunked GNN processing
            outputs = []
            for i in range(0, total, self.gnn_chunk_size):
                chunk = x_flat[i:min(i + self.gnn_chunk_size, total)]
                outputs.append(self._process_gnn_chunk(
                    chunk, edge_index, edge_weight))
            gnn_out = torch.cat(outputs, dim=0).view(
                B, S, self.n_nodes, self.hidden_dim)
            del outputs

            # Temporal processing per node
            temporal_in = gnn_out.permute(0, 2, 1, 3).reshape(
                B * self.n_nodes, S, self.hidden_dim)
            temporal_out = self.temporal_attention(temporal_in)
            lstm_out, _ = self.lstm(temporal_out)
            out = self.output_proj(lstm_out[:, -1, :])

            # Reshape to grid
            out = out.view(B, self.n_nodes, self.horizon).permute(0, 2, 1)
            return out.view(B, self.horizon, self.n_lat, self.n_lon, 1)

    return {
        'TemporalAttention': TemporalAttention,
        'SpatialGNNEncoder': SpatialGNNEncoder,
        'GNN_TAT': GNN_TAT,
    }


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_pytorch_model(torch, model, X_tr, y_tr, X_va, y_va,
                        edge_index, edge_weight, config: V4Config,
                        model_name, exp_name, out_dir, horizon, device):
    """Train PyTorch GNN-TAT model with early stopping."""
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class PrecipitationDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    metrics_dir = out_dir / f'h{horizon}' / exp_name / 'training_metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_path = metrics_dir / f'{model_name}_best_h{horizon}.pt'

    train_loader = DataLoader(PrecipitationDataset(X_tr, y_tr),
                              batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(PrecipitationDataset(X_va, y_va),
                            batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=config.patience // 2)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    start_epoch = 0

    # --- Resume from checkpoint if available ---
    resume_path = model_path.parent / f'{model_name}_resume_h{horizon}.pt'
    if resume_path.exists():
        ckpt = torch.load(resume_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        best_epoch = ckpt['best_epoch']
        patience_counter = ckpt['patience_counter']
        history = ckpt.get('history', history)
        logger.info(f'  RESUMED from epoch {start_epoch} '
                    f'(best_val={best_val_loss:.4f} @ epoch {best_epoch+1})')

    # --- Log GPU memory before training ---
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(f'  GPU memory: {free_mem/1e9:.1f}GB free / '
                    f'{total_mem/1e9:.1f}GB total')

    logger.info(f'Training {model_name} on {exp_name}...')

    for epoch in range(start_epoch, config.epochs):
        # Train
        model.train()
        train_losses = []
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            output = model(bX, edge_index, edge_weight)
            loss = criterion(output, by)

            # NaN detection
            if torch.isnan(loss):
                logger.warning(f'  NaN loss at epoch {epoch+1}, skipping batch')
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        if not train_losses:
            logger.error(f'  All batches produced NaN at epoch {epoch+1}, stopping')
            break

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bX, by in val_loader:
                bX, by = bX.to(device), by.to(device)
                output = model(bX, edge_index, edge_weight)
                val_losses.append(criterion(output, by).item())

        epoch_train = np.mean(train_losses)
        epoch_val = np.mean(val_losses)
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(epoch_train)
        history['val_loss'].append(epoch_val)
        history['lr'].append(lr)
        scheduler.step(epoch_val)

        if epoch_val < best_val_loss:
            best_val_loss = epoch_val
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f'  Epoch {epoch+1:3d}/{config.epochs}: '
                        f'train={epoch_train:.4f}, val={epoch_val:.4f}, '
                        f'lr={lr:.2e}')

        # --- Periodic resume checkpoint (every 25 epochs) ---
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'history': history,
            }, resume_path)
            logger.info(f'  Checkpoint saved at epoch {epoch+1}')

        if patience_counter >= config.patience:
            logger.info(f'  Early stopping at epoch {epoch+1}')
            break

    # Load best checkpoint
    if model_path.exists():
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Clean up resume checkpoint after successful completion
    if resume_path.exists():
        resume_path.unlink()
        logger.info('  Resume checkpoint cleaned up')

    # Save history
    pd.DataFrame(history).to_csv(
        metrics_dir / f'{model_name}_training_log_h{horizon}.csv', index=False)
    summary = {
        'model_name': model_name,
        'experiment': exp_name,
        'horizon': horizon,
        'best_epoch': best_epoch + 1,
        'best_val_loss': float(best_val_loss),
        'total_epochs': len(history['train_loss']),
        'parameters': sum(p.numel() for p in model.parameters()),
    }
    with open(metrics_dir / f'{model_name}_history.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save learning curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs_range, history['train_loss'], 'b-', label='Train')
        ax.plot(epochs_range, history['val_loss'], 'r-', label='Val')
        ax.plot(best_epoch + 1, best_val_loss, 'r*', markersize=12,
                label=f'Best: {best_val_loss:.4f}')
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

    logger.info(f'  Best val_loss: {best_val_loss:.4f} at epoch {best_epoch+1}')
    return model, summary


def evaluate_model(torch, model, X_va, y_va, edge_index, edge_weight,
                   scaler, device, horizon):
    """Evaluate model and return metrics + inverse-transformed predictions."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    model.eval()
    preds = []
    eval_bs = min(4, len(X_va))  # eval batch size, capped at validation set size
    with torch.no_grad():
        for i in range(0, len(X_va), eval_bs):
            bX = torch.tensor(X_va[i:i+eval_bs], dtype=torch.float32).to(device)
            preds.append(
                model(bX, edge_index, edge_weight).cpu().numpy())
    y_hat_sc = np.concatenate(preds, axis=0)
    y_hat = scaler.inverse_transform(
        y_hat_sc.reshape(-1, 1)).reshape(y_hat_sc.shape)
    y_true = scaler.inverse_transform(
        y_va.reshape(-1, 1)).reshape(y_va.shape)

    results = []
    for h in range(horizon):
        t = y_true[:, h].flatten()
        p = y_hat[:, h].flatten()
        results.append({
            'H': h + 1,
            'RMSE': np.sqrt(mean_squared_error(t, p)),
            'MAE': mean_absolute_error(t, p),
            'R2': r2_score(t, p),
        })
    return results, y_hat, y_true


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def run_training(torch, config: V4Config, base_path: Path):
    """Run the full V4 GNN-TAT training pipeline."""
    model_classes = define_model_classes(torch)
    GNN_TAT = model_classes['GNN_TAT']

    ds, n_lat, n_lon, lat_coords, lon_coords = load_and_validate_data(
        config, base_path)
    out_dir = base_path / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select GPU: use CUDA_VISIBLE_DEVICES if set, otherwise GPU:0
    if torch.cuda.is_available():
        gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
        device = torch.device(f'cuda:0')  # always :0 after CUDA_VISIBLE_DEVICES filtering
        gpu_name = torch.cuda.get_device_name(0)
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        logger.info(f'Device: {gpu_name} ({free_mem/1e9:.1f}GB free / {total_mem/1e9:.1f}GB total)')
    else:
        device = torch.device('cpu')
        logger.warning('No GPU detected - training will be very slow')

    # Build spatial graph
    logger.info('Building spatial graph...')
    elevation = ds['elevation'].values
    if elevation.ndim == 3:
        elevation = elevation[0]
    precip_series = ds['total_precipitation'].values

    graph_builder = SpatialGraphBuilder(
        lat_coords, lon_coords, elevation, config.gnn_config)
    edge_index_np, edge_weight_np = graph_builder.build_adjacency_matrix(
        precip_series)

    edge_index_tensor = torch.tensor(
        edge_index_np, dtype=torch.long).to(device)
    edge_weight_tensor = torch.tensor(
        edge_weight_np, dtype=torch.float32).to(device)

    # Determine GNN chunk size based on grid
    n_nodes = n_lat * n_lon
    if n_nodes <= 50:
        gnn_chunk_size = 60
    elif n_nodes <= 500:
        gnn_chunk_size = 8
    else:
        gnn_chunk_size = 2
    logger.info(f'GNN chunk size: {gnn_chunk_size} (n_nodes={n_nodes})')

    all_results = []

    for horizon in config.enabled_horizons:
        logger.info(f'\n{"="*60}')
        logger.info(f'Training for H={horizon}')
        logger.info(f'{"="*60}')

        data_splits = preprocess_data(ds, config, n_lat, n_lon, horizon)

        for exp_name in config.feature_sets:
            if exp_name not in data_splits:
                continue

            X_tr, y_tr, X_va, y_va, scaler = data_splits[exp_name]
            n_features = X_tr.shape[-1]
            model_name = 'GNN_TAT_GAT'

            logger.info(f'\n--- {exp_name}: {n_features} features ---')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            try:
                model = GNN_TAT(
                    n_features=n_features,
                    n_nodes=graph_builder.n_nodes,
                    n_lat=n_lat,
                    n_lon=n_lon,
                    horizon=horizon,
                    gnn_config=config.gnn_config,
                    gnn_chunk_size=gnn_chunk_size,
                ).to(device)

                model, summary = train_pytorch_model(
                    torch, model, X_tr, y_tr, X_va, y_va,
                    edge_index_tensor, edge_weight_tensor,
                    config, model_name, exp_name, out_dir, horizon, device,
                )

                # Evaluate
                results, y_hat, y_true = evaluate_model(
                    torch, model, X_va, y_va,
                    edge_index_tensor, edge_weight_tensor,
                    scaler, device, horizon,
                )

                for r in results:
                    r['Experiment'] = exp_name
                    r['Model'] = model_name
                all_results.extend(results)

                avg_r2 = np.mean([r['R2'] for r in results])
                avg_rmse = np.mean([r['RMSE'] for r in results])
                logger.info(f'  Overall: R2={avg_r2:.4f}, RMSE={avg_rmse:.2f}mm')

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
                    'r2_mean': float(avg_r2),
                    'rmse_mean': float(avg_rmse),
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'generated_at': datetime.now().isoformat(),
                }
                with open(export_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f'  Saved: {export_dir}')

            except Exception as e:
                logger.error(f'ERROR in {exp_name} H{horizon} '
                             f'({type(e).__name__}): {e}')
                import traceback
                logger.error(traceback.format_exc())
            finally:
                for var in ('model', 'X_tr', 'y_tr', 'X_va', 'y_va'):
                    if var in dir():
                        try:
                            del locals()[var]
                        except KeyError:
                            pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    # Save results CSV
    if all_results:
        res_df = pd.DataFrame(all_results)
        out_csv = out_dir / 'metrics_spatial_v4_h12.csv'
        res_df.to_csv(out_csv, index=False)
        logger.info(f'Results saved: {out_csv}')

    # Verification summary
    print('\n' + '=' * 60)
    print('  V4 GNN-TAT TRAINING COMPLETE')
    print('=' * 60)
    for horizon in config.enabled_horizons:
        for exp_name in config.feature_sets:
            pred_dir = (out_dir / 'map_exports' / f'H{horizon}'
                        / exp_name / 'GNN_TAT_GAT')
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
    return all_results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train V4 GNN-TAT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflows/06_train_v4_gnn_tat.py --dry-run
  python workflows/06_train_v4_gnn_tat.py --intracell-dem --bundle BASIC_D10
  python workflows/06_train_v4_gnn_tat.py --light-mode --epochs 5
  python workflows/06_train_v4_gnn_tat.py --gnn-type SAGE
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
    parser.add_argument('--gnn-type', type=str, default=None,
                        choices=['GAT', 'SAGE', 'GCN'],
                        help='GNN type override')
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
        config = V4Config.from_yaml(Path(args.config),
                                    intracell_dem=args.intracell_dem,
                                    bundle=args.bundle)
    else:
        default_config = PROJECT_ROOT / 'workflows' / 'config.yaml'
        if default_config.exists():
            config = V4Config.from_yaml(default_config,
                                        intracell_dem=args.intracell_dem,
                                        bundle=args.bundle)
        else:
            config = V4Config.default(intracell_dem=args.intracell_dem,
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
    if args.gnn_type is not None:
        config.gnn_config['gnn_type'] = args.gnn_type
    if args.light_mode:
        config.light_mode = True

    log_environment(logger, ['numpy', 'pandas', 'xarray', 'netCDF4', 'h5netcdf',
                              'torch', 'torch_geometric', 'sklearn', 'matplotlib'])
    log_script_version(logger, __file__)

    # Setup PyTorch
    torch = setup_pytorch(config.seed, allow_missing=args.dry_run)

    logger.info('=' * 60)
    logger.info('  V4 GNN-TAT Training Pipeline')
    logger.info('=' * 60)
    logger.info(f'  Dataset: {config.data_file}')
    logger.info(f'  Output: {config.output_dir}')
    logger.info(f'  Bundles: {list(config.feature_sets.keys())}')
    logger.info(f'  GNN type: {config.gnn_config["gnn_type"]}')
    logger.info(f'  Epochs: {config.epochs}, Batch: {config.batch_size}')
    logger.info(f'  Horizons: {config.enabled_horizons}')
    if config.light_mode:
        logger.info(f'  Light mode: {config.light_grid_size}x{config.light_grid_size}')
    logger.info('=' * 60)

    if args.dry_run:
        import xarray as xr
        data_file = PROJECT_ROOT / config.data_file
        if data_file.exists():
            ds = None
            for engine in ('h5netcdf', 'netcdf4', 'scipy'):
                try:
                    ds = xr.open_dataset(data_file, engine=engine)
                    logger.info(f'  Loaded with engine: {engine}')
                    break
                except Exception as e:
                    logger.warning(f'  Engine {engine} failed: {type(e).__name__}: {e}')
            if ds is None:
                raise RuntimeError(f'Could not open {data_file} with any engine')
            n_lat, n_lon = len(ds.latitude), len(ds.longitude)
            available = set(list(ds.data_vars) + list(ds.coords))
            for exp_name, feats in config.feature_sets.items():
                missing = [f for f in feats if f not in available]
                status = 'OK' if not missing else f'MISSING: {missing}'
                logger.info(f'  {exp_name}: {len(feats)} features - {status}')
            logger.info(f'  Grid: {n_lat}x{n_lon} = {n_lat*n_lon} nodes')
            ds.close()
        else:
            logger.warning(f'  Dataset not found: {data_file}')
        logger.info(f'  PyTorch: {"available" if torch is not None else "NOT installed"}')
        logger.info('Dry run complete.')
        return

    if torch is None:
        logger.error('PyTorch is required for training. Install with: pip install torch')
        sys.exit(1)

    run_training(torch, config, PROJECT_ROOT)


if __name__ == '__main__':
    main()
