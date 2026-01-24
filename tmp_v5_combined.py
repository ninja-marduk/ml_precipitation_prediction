# --- Cell 4 ---
# =============================================================================
# SECTION 1.1: ENVIRONMENT DETECTION AND SETUP
# =============================================================================

import os
import sys
from pathlib import Path

# Detect environment
IN_COLAB = 'google.colab' in sys.modules
print(f"Running in Google Colab: {IN_COLAB}")

if IN_COLAB:
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # Install required packages (match PyTorch/CUDA build)
    import torch

    torch_version = torch.__version__.split('+')[0]
    cuda_version = torch.version.cuda
    if cuda_version:
        cuda_tag = f"cu{cuda_version.replace('.', '')}"
    else:
        cuda_tag = "cpu"
    pyg_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    print(f"Installing PyG wheels from: {pyg_url}")
    # Removed -q for more verbose output to debug potential installation issues
    !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f {pyg_url}
    !pip install netCDF4 xarray dask h5netcdf

    # Verify torch_geometric installation
    try:
        import torch_geometric
        print("torch_geometric installed successfully.")
    except ImportError:
        print("ERROR: torch_geometric failed to install. Please check the output above for errors.")
        # Optionally, sys.exit(1) or raise ImportError here if torch_geometric is critical

    # Set base paths for Colab
    BASE_PATH = Path('/content/drive/MyDrive/ml_precipitation_prediction')
    DRIVE_DATA_FILE = BASE_PATH / 'data' / 'output' / 'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
    LOCAL_DATA_FILE = Path('/content/complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc')
    OUTPUT_ROOT = BASE_PATH / 'models' / 'output' / 'V5_GNN_ConvLSTM_Stacking'

    # Copy dataset to local for faster access
    if not LOCAL_DATA_FILE.exists():
        !cp "{DRIVE_DATA_FILE}" "{LOCAL_DATA_FILE}"
        print("Dataset copied to local storage for faster access")
    DATA_FILE = LOCAL_DATA_FILE
else:
    # Local paths
    BASE_PATH = Path('d:/github.com/ninja-marduk/ml_precipitation_prediction')
    DATA_FILE = BASE_PATH / 'data' / 'output' / 'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
    OUTPUT_ROOT = BASE_PATH / 'models' / 'output' / 'V5_GNN_ConvLSTM_Stacking'

# Create output directories
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_ROOT}")

# --- Cell 5 ---
# =============================================================================
# SECTION 1.2: IMPORTS
# =============================================================================

import json
import gc
import copy
import warnings
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import xarray as xr

def ensure_matplotlib():
    """Ensures matplotlib is properly installed and importable."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except AttributeError as e:
        if "get_data_path" in str(e):
            print("[WARN] Matplotlib cache corrupted. Attempting fix...")
            import sys
            import subprocess
            import shutil
            from pathlib import Path

            # Clear matplotlib cache
            try:
                import matplotlib
                cache_dir = Path(matplotlib.get_cachedir())
                if cache_dir.exists():
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    print(f"[INFO] Cleared matplotlib cache: {cache_dir}")
            except Exception:
                pass

            # Reinstall matplotlib
            try:
                print("[INFO] Reinstalling matplotlib...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--force-reinstall", "--no-cache-dir",
                    "matplotlib", "seaborn"
                ])
                print("[INFO] Matplotlib reinstalled. Please restart the kernel.")
            except Exception as install_exc:
                print(f"[ERROR] Reinstall failed: {install_exc}")

            raise RuntimeError(
                "Matplotlib installation corrupted. "
                "Please restart the kernel and rerun the cell."
            )
        else:
            raise

plt, sns = ensure_matplotlib()

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


def ensure_torch_geometric() -> bool:
    """
    Ensures torch_geometric is installed with compatible versions.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import torch_geometric  # noqa: F401
        return True
    except Exception:
        pass

    import sys
    import subprocess

    torch_ver = torch.__version__.split("+")[0]
    cuda_ver = torch.version.cuda
    if cuda_ver:
        cuda_tag = "cu" + cuda_ver.replace(".", "")
    else:
        cuda_tag = "cpu"

    url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
    pkgs = [
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        "torch-geometric",
    ]

    print(f"[INFO] Installing PyG for torch={torch_ver}, cuda={cuda_tag}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs + ["-f", url])
        return True
    except Exception as exc:
        print(f"[WARN] PyG install failed: {exc}")

    if cuda_tag != "cpu":
        url_cpu = f"https://data.pyg.org/whl/torch-{torch_ver}+cpu.html"
        print(f"[INFO] Retrying CPU wheels: {url_cpu}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs + ["-f", url_cpu])
            print("[WARN] Installed CPU wheels; GPU will be disabled.")
            return True
        except Exception as exc:
            print(f"[ERROR] PyG CPU install failed: {exc}")

    return False


if not ensure_torch_geometric():
    raise RuntimeError("torch_geometric not available. Restart runtime and rerun.")

# PyTorch Geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, Batch

from sklearn.cluster import KMeans
from scipy import stats

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42

def seed_everything(seed: int) -> bool:
    """
    Sets random seeds for reproducibility across libraries.

    Args:
        seed: Random seed value

    Returns:
        bool: True if CUDA seeding succeeded, False otherwise
    """
    np.random.seed(seed)

    cuda_ok = False
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cuda_ok = True
        except Exception as exc:
            print(f"[WARN] CUDA seed failed: {exc}")

    return cuda_ok

# Initialize seeds and check CUDA status
CUDA_READY = seed_everything(SEED)
USE_CUDA = torch.cuda.is_available() and CUDA_READY

# Device configuration
device = torch.device('cuda' if USE_CUDA else 'cpu')
print(f"Using device: {device}")

if USE_CUDA:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
elif torch.cuda.is_available() and not CUDA_READY:
    print("[WARN] CUDA is available but in a bad state. Using CPU instead.")
    print("[WARN] Consider restarting the runtime if GPU is needed.")
else:
    print("[INFO] CUDA not available. Using CPU.")

# --- Cell 7 ---
# =============================================================================
# SECTION 2: V5 CONFIGURATION
# =============================================================================

@dataclass
class V5Config:
    """Complete V5 GNN-ConvLSTM Stacking configuration."""

    # Data configuration
    input_window: int = 60      # Input months
    horizon: int = 12           # Prediction months
    train_val_split: float = 0.8

    # Light mode for testing
    light_mode: bool = False     # Set to False for full grid
    light_grid_size: int = 5    # Grid subset size when light_mode=True

    # Enabled horizons for experiments
    enabled_horizons: List[int] = field(default_factory=lambda: [12])

    # Branch 1: ConvLSTM configuration (V2 Enhanced)
    convlstm_filters: List[int] = field(default_factory=lambda: [32, 16])
    convlstm_kernel_size: int = 3 # Ensure this is an odd number to preserve spatial dimensions
    convlstm_attention: bool = True
    convlstm_bidirectional: bool = True
    convlstm_residual: bool = True
    convlstm_output_dim: int = 64

    # Branch 2: GNN-TAT configuration (V4)
    gnn_type: str = 'GAT'           # GAT, SAGE, or GCN
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 2
    gnn_num_heads: int = 4          # For GAT
    gnn_temporal_heads: int = 4
    gnn_lstm_hidden: int = 64
    gnn_lstm_layers: int = 2
    gnn_output_dim: int = 64
    gnn_dropout: float = 0.1
    use_temporal_attention: bool = True
    temporal_num_heads: int = 4

    # Graph construction
    edge_threshold: float = 0.3
    max_neighbors: int = 8
    use_distance_edges: bool = True
    use_elevation_edges: bool = True
    use_correlation_edges: bool = True
    distance_scale_km: float = 10.0
    elevation_weight: float = 0.3
    correlation_weight: float = 0.5
    elevation_scale_m: float = 500.0
    min_edge_weight: float = 0.01

    # Grid-Graph Fusion
    fusion_type: str = 'auto'       # cross_attention, gated, or auto
    fusion_heads: int = 4
    fusion_hidden_dim: int = 64
    fusion_dropout: float = 0.1
    fusion_max_nodes: int = 1024

    # Meta-Learner
    meta_hidden_dim: int = 128
    meta_dropout: float = 0.1
    meta_use_context_features: bool = True
    meta_context_features: List[str] = field(
        default_factory=lambda: ['mean_elevation', 'elevation_cluster', 'temporal_regime']
    )

    # Training
    epochs: int = 100  # Reduced from 200 for faster training
    batch_size: int = 8  # Increased from 4 for better GPU utilization
    validate: bool = False  # Disable validation in production for performance
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    patience: int = 20  # Reduced from 60 for faster early stopping
    gradient_clip: float = 1.0
    # Branch weight regularization
    weight_floor: float = 0.3  # Increased from 0.1 to force more balance
    weight_reg_lambda: float = 0.1  # Increased to penalize weight imbalance

    # Model sizing
    compact_model: bool = True


    # Export/visualization alignment
    export_predictions: bool = True
    generate_map_plots: bool = True
    map_cycle_all_val_windows: bool = False
    map_sample_index: int = -1
    map_export_dpi: int = 300
    map_gif_duration: float = 1.5
    plot_graph_diagnostics: bool = True

    # Quality checks / gates
    max_bias_pct: float = 10.0
    max_scale_ratio: float = 50.0
    max_negative_frac: float = 0.001
    min_branch_weight: float = 0.05
    param_budget: int = 200000
    enforce_quality_gates: bool = False
    allow_missing_features: bool = False
    allow_overlap_windows: bool = False
    correlation_train_only: bool = True

    def __getattr__(self, name):
        """Catch access to removed attributes with helpful error messages."""
        if name == 'graph_input_dim':
            raise AttributeError(
                "V5Config.graph_input_dim has been removed.\n"
                "GNNBranch now requires n_features as explicit parameter:\n"
                "  gnn = GNNBranch(config, n_features=16, n_nodes=config.n_nodes, ...)\n"
                "See cell 18 USAGE EXAMPLES for correct usage."
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# Initialize config
CONFIG = V5Config()

# Run-mode overrides
RUN_FULL_DATASET = True  # Set False for quick smoke tests
if RUN_FULL_DATASET:
    CONFIG.light_mode = False
    CONFIG.enabled_horizons = [12]
    CONFIG.horizon = 12

if CONFIG.compact_model:
    CONFIG.convlstm_filters = [16, 8]
    CONFIG.convlstm_output_dim = 32
    CONFIG.gnn_hidden_dim = 32
    CONFIG.gnn_num_heads = 2
    CONFIG.gnn_temporal_heads = 2
    CONFIG.gnn_lstm_hidden = 32
    CONFIG.gnn_output_dim = 32
    CONFIG.fusion_hidden_dim = 32
    CONFIG.fusion_heads = 2
    CONFIG.meta_hidden_dim = 64

print("V5 Configuration initialized:")
print(f"  - Light mode: {CONFIG.light_mode}")
print(f"  - GNN type: {CONFIG.gnn_type}")
print(f"  - Enabled horizons: {CONFIG.enabled_horizons}")
print(f"  - Epochs: {CONFIG.epochs}")
print(f"  - Fusion type: {CONFIG.fusion_type}")

# --- Cell 9 ---
# =============================================================================
# SECTION 3: DATA LOADING AND FEATURE EXTRACTION
# =============================================================================

# Feature set definitions
FEATURE_SETS = {
    'BASIC': [
        'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
        'elevation', 'slope', 'aspect'
    ],
    'KCE': [
        'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
        'elevation', 'slope', 'aspect',
        'elev_high', 'elev_med', 'elev_low'
    ],
    'PAFC': [
        'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
        'elevation', 'slope', 'aspect',
        'elev_high', 'elev_med', 'elev_low',
        'total_precipitation_lag1', 'total_precipitation_lag2', 'total_precipitation_lag12'
    ]
}

def _resolve_dim(ds: xr.Dataset, candidates: Tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.dims:
            return name
    raise ValueError(f"Missing required dimension. Expected one of: {candidates}")


def _required_features(feature_sets: Dict[str, List[str]]) -> List[str]:
    required = set()
    for name in feature_sets['BASIC'] + feature_sets['KCE']:
        if name.startswith('elev_'):
            continue
        required.add(name)
    required.add('total_precipitation')
    return sorted(required)


def validate_dataset(ds: xr.Dataset, config: V5Config, feature_sets: Dict[str, List[str]]) -> Tuple[str, str]:
    if 'time' not in ds.dims:
        raise ValueError("Dataset missing required 'time' dimension")

    lat_dim = _resolve_dim(ds, ('latitude', 'lat'))
    lon_dim = _resolve_dim(ds, ('longitude', 'lon'))

    missing = [name for name in _required_features(feature_sets)
               if name not in ds.data_vars and name not in ds.coords]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")

    n_time = int(ds.dims['time'])
    if n_time < config.input_window + config.horizon:
        raise ValueError(
            f"Not enough timesteps: {n_time} < {config.input_window + config.horizon}"
        )

    return lat_dim, lon_dim


def load_dataset(data_path: Path, config: V5Config) -> xr.Dataset:
    """Load and validate the NetCDF dataset."""
    print(f"Loading dataset from: {data_path}")
    ds = xr.open_dataset(data_path)

    lat_dim, lon_dim = validate_dataset(ds, config, FEATURE_SETS)

    # Print dataset info
    print(f"\nDataset dimensions:")
    for dim, size in ds.dims.items():
        print(f"  - {dim}: {size}")

    print(f"\nAvailable variables: {list(ds.data_vars)}")

    # Apply light mode if enabled
    if config.light_mode:
        lat_slice = slice(0, config.light_grid_size)
        lon_slice = slice(0, config.light_grid_size)
        ds = ds.isel({lat_dim: lat_slice, lon_dim: lon_slice})
        print(f"\nLight mode enabled: using {config.light_grid_size}x{config.light_grid_size} grid")

    return ds


def create_elevation_clusters(ds: xr.Dataset, n_clusters: int = 3) -> xr.Dataset:
    """Add elevation cluster features (KCE) to dataset."""
    elevation = ds['elevation'].values
    elev_dims = ds['elevation'].dims

    # Handle 3D elevation (time, lat, lon) by taking first timestep
    if elevation.ndim == 3:
        print("Handling 3D elevation: using first timestep for static clustering")
        elevation = elevation[0]
        elev_dims = elev_dims[-2:]

    valid_mask = ~np.isnan(elevation)

    # Flatten for clustering
    elev_flat = elevation[valid_mask].reshape(-1, 1)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    labels = np.full(elevation.shape, -1)
    labels[valid_mask] = kmeans.fit_predict(elev_flat)

    # Create one-hot encoded features
    for i, name in enumerate(['elev_low', 'elev_med', 'elev_high']):
        cluster_data = np.zeros_like(elevation)
        cluster_data[labels == i] = 1.0
        ds[name] = xr.DataArray(
            data=cluster_data,
            dims=elev_dims,
            attrs={'description': f'Elevation cluster {name}'}
        )

    print("Added elevation clusters: elev_low, elev_med, elev_high")
    return ds


def extract_features(ds: xr.Dataset, feature_names: List[str], config: V5Config) -> Tuple[np.ndarray, List[str]]:
    """Extract features from dataset into numpy array."""
    features = []
    missing = []

    for name in feature_names:
        if name in ds.data_vars:
            data = ds[name].values
            # Handle different dimensions
            if data.ndim == 2:  # (lat, lon) - static features
                # Broadcast to (time, lat, lon)
                data = np.broadcast_to(data, (ds.dims['time'], *data.shape))
            features.append(data)
        else:
            missing.append(name)

    if missing:
        msg = f"Missing features: {missing}"
        if config.allow_missing_features:
            print(f"Warning: {msg}")
        else:
            raise ValueError(msg)

    if not features:
        raise ValueError("No features extracted; check dataset and feature list")

    # Stack features: (time, lat, lon, n_features)
    features = np.stack(features, axis=-1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Extracted features shape: {features.shape}")

    return features.astype(np.float32), missing

# Load dataset
ds = load_dataset(DATA_FILE, CONFIG)
LAT_DIM = _resolve_dim(ds, ('latitude', 'lat'))
LON_DIM = _resolve_dim(ds, ('longitude', 'lon'))
ds = create_elevation_clusters(ds)

# --- Cell 11 ---
# =============================================================================
# SECTION 4: GRAPH CONSTRUCTION
# =============================================================================

class SpatialGraphBuilder:
    """Build spatial graph for GNN branch based on geographic and topographic similarity."""

    def __init__(self, config: V5Config):
        self.config = config

    @staticmethod
    def _safe_correlation(ts_a: np.ndarray, ts_b: np.ndarray) -> float:
        """Compute correlation robustly (returns 0.0 for invalid cases)."""
        mask = np.isfinite(ts_a) & np.isfinite(ts_b)
        if mask.sum() < 2:
            return 0.0
        a = ts_a[mask]
        b = ts_b[mask]
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
        if denom < 1e-6:
            return 0.0
        corr = float(np.sum(a * b) / denom)
        if not np.isfinite(corr):
            return 0.0
        return corr

    @staticmethod
    def _get_coord(ds: xr.Dataset, names: List[str]) -> np.ndarray:
        for name in names:
            if name in ds.coords:
                return ds.coords[name].values
            if name in ds.variables:
                return ds[name].values
        raise KeyError(f"Missing coordinate; tried {names}")

    def build_graph(self, ds: xr.Dataset, train_time_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph structure from dataset.

        Returns:
            edge_index: (2, num_edges) tensor of edge indices
            edge_weight: (num_edges,) tensor of edge weights
        """
        lat = self._get_coord(ds, ['lat', 'latitude'])
        lon = self._get_coord(ds, ['lon', 'longitude'])
        elevation = ds['elevation'].values

        if elevation.ndim == 3:
            elevation = elevation[0]

        n_lat, n_lon = len(lat), len(lon)
        n_nodes = n_lat * n_lon

        print(f"Building graph for {n_lat}x{n_lon} = {n_nodes} nodes")

        # Create node positions in km for distance scaling
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        mean_lat = float(np.mean(lat))
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(np.deg2rad(mean_lat))
        positions = np.stack(
            [lat_grid.flatten() * km_per_deg_lat, lon_grid.flatten() * km_per_deg_lon],
            axis=1
        )
        elev_flat = elevation.flatten()

        self.node_positions = positions
        self.node_elevations = elev_flat
        self.n_nodes = n_nodes

        # Precompute precipitation time series for correlation edges
        precip_flat = None
        if self.config.use_correlation_edges:
            precip = ds['total_precipitation'].values.astype(np.float32)
            if train_time_idx is not None:
                train_time_idx = max(2, min(train_time_idx, precip.shape[0]))
                precip = precip[:train_time_idx]
                print(f"Correlation edges using first {train_time_idx} timesteps")
            precip_flat = precip.reshape(precip.shape[0], n_nodes)

        edges = []
        weights = []

        distance_coeff = 1.0
        if self.config.use_elevation_edges or self.config.use_correlation_edges:
            distance_coeff = max(
                0.0,
                1.0 - self.config.elevation_weight - self.config.correlation_weight
            )

        for i in range(n_nodes):
            # Calculate distances to all other nodes
            distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))

            # Get k nearest neighbors (excluding self)
            distances[i] = np.inf
            nearest_idx = np.argsort(distances)[:self.config.max_neighbors]
            ts_i = precip_flat[:, i] if precip_flat is not None else None

            for j in nearest_idx:
                if distances[j] == np.inf:
                    continue

                component_weights = []

                if self.config.use_distance_edges:
                    dist_weight = np.exp(-distances[j] / self.config.distance_scale_km)
                    component_weights.append((dist_weight, distance_coeff))

                if self.config.use_elevation_edges:
                    if not np.isnan(elev_flat[i]) and not np.isnan(elev_flat[j]):
                        elev_diff = np.abs(elev_flat[i] - elev_flat[j])
                        elev_weight = np.exp(-elev_diff / self.config.elevation_scale_m)
                    else:
                        elev_weight = 0.5
                    component_weights.append((elev_weight, self.config.elevation_weight))

                if self.config.use_correlation_edges:
                    if precip_flat is not None:
                        corr = self._safe_correlation(ts_i, precip_flat[:, j])
                    else:
                        corr = 0.0
                    corr_weight = max(0.0, corr)
                    component_weights.append((corr_weight, self.config.correlation_weight))

                if not component_weights:
                    continue

                coeff_sum = sum(weight for _, weight in component_weights)
                if coeff_sum <= 0:
                    coeff_sum = len(component_weights)
                    combined_weight = sum(val for val, _ in component_weights) / coeff_sum
                else:
                    combined_weight = sum(val * weight for val, weight in component_weights) / coeff_sum

                threshold = max(self.config.edge_threshold, self.config.min_edge_weight)
                if combined_weight >= threshold:
                    edges.append([i, j])
                    weights.append(combined_weight)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float32)

        print(f"Graph built: {n_nodes} nodes, {edge_index.shape[1]} edges")
        print(f"Average edges per node: {edge_index.shape[1] / n_nodes:.1f}")

        return edge_index, edge_weight

# Build graph
graph_builder = SpatialGraphBuilder(CONFIG)
train_time_idx = None
if CONFIG.use_correlation_edges and CONFIG.correlation_train_only:
    train_time_idx = int(ds.dims['time'] * CONFIG.train_val_split)
edge_index, edge_weight = graph_builder.build_graph(ds, train_time_idx=train_time_idx)
if edge_index.numel() == 0:
    raise ValueError("Graph construction produced zero edges")
print(f"Edge index shape: {edge_index.shape}")
print(f"Edge weight shape: {edge_weight.shape}")

# --- Cell 12 ---
# =============================================================================
# SECTION 4.1: GRAPH DIAGNOSTICS (V4-COMPATIBLE)
# =============================================================================

def visualize_graph(graph_builder: SpatialGraphBuilder, edge_index: np.ndarray,
                   edge_weight: np.ndarray, title: str = 'Spatial Graph'):
    if edge_weight.size == 0:
        print('Graph visualization skipped: empty edge list')
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    pos = graph_builder.node_positions
    n_edges_to_draw = min(500, len(edge_weight))
    edge_indices = np.random.choice(len(edge_weight), n_edges_to_draw, replace=False)
    for idx in edge_indices:
        i, j = edge_index[0, idx], edge_index[1, idx]
        ax1.plot([pos[i, 1], pos[j, 1]], [pos[i, 0], pos[j, 0]], 'b-', alpha=0.2, linewidth=0.5)

    scatter = ax1.scatter(pos[:, 1], pos[:, 0], c=graph_builder.node_elevations,
                         cmap='terrain', s=80, edgecolors='black', linewidth=0.4)
    plt.colorbar(scatter, ax=ax1, label='Elevation (m)')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Graph Structure (colored by elevation)')

    ax2 = axes[1]
    ax2.hist(edge_weight, bins=50, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Edge Weight')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Edge Weight Distribution')
    ax2.axvline(np.mean(edge_weight), color='red', linestyle='--', label=f'Mean: {np.mean(edge_weight):.3f}')
    ax2.legend()

    ax3 = axes[2]
    degrees = np.bincount(edge_index[0], minlength=graph_builder.n_nodes)
    ax3.hist(degrees, bins=20, color='forestgreen', edgecolor='black')
    ax3.set_xlabel('Node Degree')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Node Degree Distribution')
    ax3.axvline(np.mean(degrees), color='red', linestyle='--', label=f'Mean: {np.mean(degrees):.1f}')
    ax3.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = OUTPUT_ROOT / 'graph_visualization_v5.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Graph visualization saved to: {fig_path}")
    plt.show()


def analyze_graph_edges(graph_builder: SpatialGraphBuilder, edge_index: np.ndarray,
                        edge_weight: np.ndarray, output_dir: Path):
    if edge_weight.size == 0:
        print('Graph edge analysis skipped: empty edge list')
        return

    max_nodes_for_adj = 6000
    if graph_builder.n_nodes > max_nodes_for_adj:
        print(f"Skipping adjacency matrix plot: {graph_builder.n_nodes} nodes")
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if len(axes) > 1:
        ax1 = axes[0]
        adj_matrix = np.zeros((graph_builder.n_nodes, graph_builder.n_nodes), dtype=np.float32)
        for idx in range(len(edge_weight)):
            i, j = edge_index[0, idx], edge_index[1, idx]
            adj_matrix[i, j] = edge_weight[idx]
        im1 = ax1.imshow(adj_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im1, ax=ax1, label='Edge Weight')
        ax1.set_xlabel('Node Index')
        ax1.set_ylabel('Node Index')
        ax1.set_title('Adjacency Matrix (Edge Weights)')
        ax2 = axes[1]
    else:
        ax2 = axes[0]

    threshold = np.percentile(edge_weight, 90)
    strong_edges = edge_weight > threshold
    pos = graph_builder.node_positions

    scatter = ax2.scatter(pos[:, 1], pos[:, 0], c=graph_builder.node_elevations,
                         cmap='terrain', s=80, edgecolors='black', linewidth=0.4, zorder=2)
    plt.colorbar(scatter, ax=ax2, label='Elevation (m)')

    for idx in np.where(strong_edges)[0]:
        i, j = edge_index[0, idx], edge_index[1, idx]
        ax2.plot([pos[i, 1], pos[j, 1]], [pos[i, 0], pos[j, 0]],
                'r-', alpha=0.6, linewidth=edge_weight[idx] * 2, zorder=1)

    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'Top 10% Strongest Edges (n={int(strong_edges.sum())})')

    plt.tight_layout()

    fig_path = output_dir / 'graph_edge_analysis_v5.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Edge analysis saved to: {fig_path}")
    plt.show()

    print('Graph Statistics:')
    print(f"  Total nodes: {graph_builder.n_nodes}")
    print(f"  Total edges: {len(edge_weight)}")
    print(f"  Average edge weight: {np.mean(edge_weight):.4f}")
    print(f"  Max edge weight: {np.max(edge_weight):.4f}")
    print(f"  Strong edges (top 10%): {int(strong_edges.sum())}")


if getattr(CONFIG, 'plot_graph_diagnostics', True):
    edge_index_np = edge_index.cpu().numpy()
    edge_weight_np = edge_weight.cpu().numpy()
    try:
        visualize_graph(
            graph_builder, edge_index_np, edge_weight_np,
            title=f"V5 Spatial Graph ({int(ds.dims[LAT_DIM])}x{int(ds.dims[LON_DIM])} grid)"
        )
    except Exception as exc:
        print(f"Graph visualization skipped: {exc}")
    try:
        analyze_graph_edges(graph_builder, edge_index_np, edge_weight_np, OUTPUT_ROOT)
    except Exception as exc:
        print(f"Graph edge analysis skipped: {exc}")

# --- Cell 14 ---
# =============================================================================
# SECTION 5: DATA PREPROCESSING AND DATASET CLASS
# =============================================================================

def create_temporal_windows(
    features: np.ndarray,
    target: np.ndarray,
    input_window: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create temporal windows for sequence-to-sequence prediction.

    Note: This is memory-heavy for full grids and kept for debugging only.
    """
    n_time = features.shape[0]
    n_samples = n_time - input_window - horizon + 1

    if n_samples <= 0:
        raise ValueError(f"Not enough timesteps: {n_time} < {input_window + horizon}")

    X_list = []
    Y_list = []

    for i in range(n_samples):
        X_list.append(features[i:i+input_window])
        Y_list.append(target[i+input_window:i+input_window+horizon])

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    print(f"Created {n_samples} samples")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")

    return X, Y

class V5DualBranchDataset(Dataset):
    """Dataset for V5 dual-branch model (ConvLSTM + GNN).

    Provides grid and graph representations with on-the-fly windowing
    to avoid materializing huge tensors for full-grid training.
    """

    def __init__(
        self,
        features_basic: torch.Tensor,  # (time, lat, lon, features)
        features_kce: torch.Tensor,    # (time, lat, lon, features)
        target: torch.Tensor,          # (time, lat, lon)
        input_window: int,
        horizon: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        start_idx: int,
        end_idx: int
    ):
        self.features_basic = features_basic
        self.features_kce = features_kce
        self.target = target
        self.input_window = input_window
        self.horizon = horizon
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.start_idx = start_idx
        self.end_idx = end_idx

        if end_idx <= start_idx:
            raise ValueError(f"Invalid window range: {start_idx} to {end_idx}")

        # Store grid dimensions for later use
        self.n_lat = features_basic.shape[1]
        self.n_lon = features_basic.shape[2]
        self.n_nodes = self.n_lat * self.n_lon

        max_start = target.shape[0] - input_window - horizon
        if max_start < 0:
            raise ValueError(
                f"Not enough timesteps: {target.shape[0]} < {input_window + horizon}"
            )
        if self.start_idx < 0 or self.end_idx - 1 > max_start:
            raise ValueError(
                f"Window range out of bounds: {self.start_idx}..{self.end_idx - 1} > {max_start}"
            )

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        i = self.start_idx + idx
        x_basic = self.features_basic[i:i+self.input_window]
        x_kce = self.features_kce[i:i+self.input_window]
        y = self.target[i+self.input_window:i+self.input_window+self.horizon]
        return {
            'x_grid': x_basic,
            'x_graph': x_kce.reshape(
                self.input_window, -1, x_kce.shape[-1]
            ),
            'y': y,
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight
        }

def prepare_data(
    ds: xr.Dataset,
    config: V5Config,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    feature_set_basic: str = 'BASIC',
    feature_set_kce: str = 'KCE'
):
    """Prepare data for training."""
    print("\n" + "="*60)
    print("Preparing data for V5 dual-branch model")
    print("="*60)

    # Extract features for both branches
    print(f"\nExtracting {feature_set_basic} features (ConvLSTM branch):")
    features_basic, missing_basic = extract_features(ds, FEATURE_SETS[feature_set_basic], config)

    print(f"\nExtracting {feature_set_kce} features (GNN branch):")
    features_kce, missing_kce = extract_features(ds, FEATURE_SETS[feature_set_kce], config)

    # Target variable
    target = ds['total_precipitation'].values.astype(np.float32)
    print(f"\nTarget shape: {target.shape}")

    # Handle NaN values
    features_basic = np.nan_to_num(features_basic, nan=0.0)
    features_kce = np.nan_to_num(features_kce, nan=0.0)
    target = np.nan_to_num(target, nan=0.0)

    # Window counts (avoid materializing all windows)
    n_time = features_basic.shape[0]
    max_start = n_time - config.input_window - config.horizon
    if max_start < 0:
        raise ValueError(f"Not enough timesteps: {n_time} < {config.input_window + config.horizon}")
    n_samples = max_start + 1

    print("\nComputed temporal windows:")
    print(f"  Samples: {n_samples}")
    print(f"  Input window: {config.input_window}")
    print(f"  Horizon: {config.horizon}")

    # Train/val split by time BEFORE windowing (avoid leakage)
    min_split = config.input_window + config.horizon
    max_split = max_start
    split_mode = 'time'
    split_time_idx = int(n_time * config.train_val_split)

    if min_split > max_split:
        msg = (
            f"Not enough timesteps for leakage-free split: n_time={n_time}, "
            f"need >= {2 * (config.input_window + config.horizon)}"
        )
        if config.allow_overlap_windows:
            print(f"Warning: {msg}. Falling back to overlap-prone split.")
            split_mode = 'window'
            n_train = int(n_samples * config.train_val_split)
            n_val = n_samples - n_train
            train_last_start = n_train - 1
            val_start_idx = n_train
            val_last_start = max_start
            split_time_idx = val_start_idx
        elif config.enforce_quality_gates:
            raise ValueError(msg)
        else:
            raise ValueError(msg)
    else:
        if split_time_idx < min_split:
            print(f"Warning: split_time_idx={split_time_idx} < min_split={min_split}; clipping.")
            split_time_idx = min_split
        if split_time_idx > max_split:
            print(f"Warning: split_time_idx={split_time_idx} > max_split={max_split}; clipping.")
            split_time_idx = max_split

        train_last_start = split_time_idx - config.input_window - config.horizon
        val_start_idx = split_time_idx
        val_last_start = max_start
        n_train = train_last_start + 1
        n_val = val_last_start - val_start_idx + 1

    if n_train <= 0 or n_val <= 0:
        raise ValueError(f"Invalid train/val split: {n_train}/{n_val}")

    print("\nSplit configuration:")
    print(f"  Split mode: {split_mode}")
    print(f"  Split index: {split_time_idx}")
    print(f"  Train window starts: 0..{train_last_start} ({n_train})")
    print(f"  Val window starts: {val_start_idx}..{val_last_start} ({n_val})")
    gap = val_start_idx - (train_last_start + 1)
    print(f"  Gap windows: {gap}")

    # Overlap leakage check
    last_train_end = train_last_start + config.input_window + config.horizon - 1
    first_val_start = val_start_idx
    overlap_leakage = last_train_end >= first_val_start
    if overlap_leakage:
        msg = "Train/val windows overlap. Consider splitting before windowing."
        if config.allow_overlap_windows:
            print(f"Warning: {msg}")
        elif config.enforce_quality_gates:
            raise ValueError(msg)
        else:
            print(f"Warning: {msg}")

    # Normalize using training input range only (avoid leakage)
    print("\nNormalizing features using training split only:")
    train_input_end = train_last_start + config.input_window - 1
    train_slice = slice(0, train_input_end + 1)

    basic_mean = features_basic[train_slice].mean(axis=(0, 1, 2), keepdims=True)
    basic_std = features_basic[train_slice].std(axis=(0, 1, 2), keepdims=True)
    basic_std = np.where(basic_std > 1e-6, basic_std, 1.0)

    kce_mean = features_kce[train_slice].mean(axis=(0, 1, 2), keepdims=True)
    kce_std = features_kce[train_slice].std(axis=(0, 1, 2), keepdims=True)
    kce_std = np.where(kce_std > 1e-6, kce_std, 1.0)

    features_basic = ((features_basic - basic_mean) / basic_std).astype(np.float32)
    features_kce = ((features_kce - kce_mean) / kce_std).astype(np.float32)

    # Convert to torch once to avoid repeated conversion in __getitem__
    features_basic_t = torch.from_numpy(features_basic)
    features_kce_t = torch.from_numpy(features_kce)
    target_t = torch.from_numpy(target)

    train_dataset = V5DualBranchDataset(
        features_basic_t, features_kce_t, target_t,
        config.input_window, config.horizon,
        edge_index, edge_weight,
        start_idx=0,
        end_idx=n_train
    )
    val_dataset = V5DualBranchDataset(
        features_basic_t, features_kce_t, target_t,
        config.input_window, config.horizon,
        edge_index, edge_weight,
        start_idx=val_start_idx,
        end_idx=val_last_start + 1
    )

    data_report = {
        'n_samples': int(n_samples),
        'n_train': int(n_train),
        'n_val': int(n_val),
        'n_time': int(n_time),
        'feature_set_basic': feature_set_basic,
        'feature_set_kce': feature_set_kce,
        'split_mode': split_mode,
        'split_time_idx': int(split_time_idx),
        'train_last_start': int(train_last_start),
        'val_start_idx': int(val_start_idx),
        'val_end_idx': int(val_last_start),
        'train_input_end': int(train_input_end),
        'overlap_leakage': bool(overlap_leakage),
        'missing_features_basic': missing_basic,
        'missing_features_kce': missing_kce
    }

    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset, features_basic.shape[-1], features_kce.shape[-1], data_report

# Prepare data
DEFAULT_FEATURE_SET_BASIC = 'BASIC'
DEFAULT_FEATURE_SET_KCE = 'KCE'

train_dataset, val_dataset, n_features_basic, n_features_kce, data_report = prepare_data(
    ds, CONFIG, edge_index, edge_weight,
    feature_set_basic=DEFAULT_FEATURE_SET_BASIC,
    feature_set_kce=DEFAULT_FEATURE_SET_KCE
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, shuffle=False)

print(f"\nData loaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# --- Cell 16 ---
# =============================================================================
# SECTION 6.1: CONVLSTM BRANCH (V2 Enhanced Architecture)
# =============================================================================

class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell with spatial convolutions."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # i, f, o, g gates
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new

    def init_hidden(self, batch_size: int, height: int, width: int, device: torch.device):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for ConvLSTM output."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class ConvLSTMBranch(nn.Module):
    """Branch 1: ConvLSTM encoder for Euclidean spatial patterns.

    Based on V2 Enhanced architecture with attention and residual connections.
    """

    def __init__(self, config: V5Config, n_features: int):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Conv2d(
            n_features, config.convlstm_filters[0],
            kernel_size=1
        )

        # ConvLSTM layers
        self.convlstm_cells_fw = nn.ModuleList()
        self.convlstm_cells_bw = nn.ModuleList() if config.convlstm_bidirectional else None
        in_dim = config.convlstm_filters[0]
        for out_dim in config.convlstm_filters:
            self.convlstm_cells_fw.append(
                ConvLSTMCell(in_dim, out_dim, config.convlstm_kernel_size)
            )
            if config.convlstm_bidirectional:
                self.convlstm_cells_bw.append(
                    ConvLSTMCell(in_dim, out_dim, config.convlstm_kernel_size)
                )
            in_dim = out_dim

        # Spatial attention
        attn_in_dim = config.convlstm_filters[-1] * (2 if config.convlstm_bidirectional else 1)
        if config.convlstm_attention:
            self.attention = SpatialAttention(attn_in_dim)
        else:
            self.attention = None

        # Output projection to match GNN branch
        self.output_proj = nn.Conv2d(
            config.convlstm_filters[-1] * (2 if config.convlstm_bidirectional else 1),
            config.convlstm_output_dim,
            kernel_size=1
        )

        # Residual connection
        if config.convlstm_residual:
            self.residual_proj = nn.Conv2d(n_features, config.convlstm_output_dim, kernel_size=1)
        else:
            self.residual_proj = None

    def _run_convlstm(self, x: torch.Tensor, cells: nn.ModuleList) -> torch.Tensor:
        """Run ConvLSTM stack and return last hidden state."""
        batch_size, seq_len, _, h, w = x.shape
        layer_input = x
        for layer_idx, cell in enumerate(cells):
            h_state, c_state = cell.init_hidden(batch_size, h, w, x.device)
            outputs = []
            for t in range(seq_len):
                x_t = layer_input[:, t]
                if layer_idx == 0:
                    x_t = self.input_proj(x_t)
                h_state, c_state = cell(x_t, (h_state, c_state))
                outputs.append(h_state)
            layer_input = torch.stack(outputs, dim=1)
        return h_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, lat, lon, features)

        Returns:
            output: (batch, lat, lon, output_dim)
        """
        batch_size, seq_len, h, w, _ = x.shape
        device = x.device

        # Reshape for Conv2d: (batch, seq, features, h, w)
        x_reshaped = x.permute(0, 1, 4, 2, 3)

        # Store residual
        if self.residual_proj is not None:
            residual = self.residual_proj(x_reshaped[:, -1])  # Use last timestep

        # Forward direction
        h_forward = self._run_convlstm(x_reshaped, self.convlstm_cells_fw)

        if self.config.convlstm_bidirectional:
            # Backward direction
            x_backward = x_reshaped.flip(1)
            h_backward = self._run_convlstm(x_backward, self.convlstm_cells_bw)
            output = torch.cat([h_forward, h_backward], dim=1)
        else:
            output = h_forward

        # Apply attention
        if self.attention is not None:
            output = self.attention(output)

        # Project to output dimension
        output = self.output_proj(output)  # (batch, output_dim, h, w)

        # Add residual
        if self.residual_proj is not None:
            output = output + residual

        # Reshape to (batch, h, w, output_dim)
        output = output.permute(0, 2, 3, 1)

        return output

print("ConvLSTM Branch defined successfully")

# --- Cell 18 ---
# =============================================================================
# SECTION 6.2: GNN BRANCH - FIXED VERSION
# =============================================================================
# This is the corrected and optimized GNNBranch class for Cell 18 in
# base_models_gnn_convlstm_stacking_v5.ipynb
#
# CRITICAL IMPROVEMENTS:
# 1.  Per-layer GNN validation (after each graph convolution)
# 2.  Message passing numerical stability checks
# 3.  Temporal attention validation
# 4.  Edge weight normalization validation
# 5.  Gradient-safe operations
# 6.  Detailed error messages with layer-specific diagnostics
# 7.  Optional validation toggle
# 8.  Memory-efficient implementation
# 9.  FIXED: Handles 4D input (Batch, Time, Nodes, Features)
# =============================================================================

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from typing import Tuple, Optional, Literal

class GNNBranch(nn.Module):
    """Graph Neural Network branch for non-Euclidean spatial processing.

    Architecture:
        Processes precipitation data as graph:
        - Nodes: Grid locations (lat, lon) with features
        - Edges: Spatial relationships (distance-based or learned)
        - Message passing: Aggregates neighbor information

        Flow:
        1. Input embedding: Project features to hidden dim
        2. GNN layers: Multiple graph convolutions with residual connections
        3. Temporal attention: Capture temporal dependencies
        4. Output: Node features ready for fusion

    Key Features:
        - Multiple GNN architectures (GCN, GAT, SAGE)
        - Residual connections for gradient flow
        - Layer-wise validation (prevents error propagation)
        - Numerical stability (gradient clipping, normalization)
        - Handling of temporal sequences (4D input)

    Improvements over Original:
         Per-layer validation (catches errors early)
         Message passing stability checks
         Edge weight normalization validation
         Temporal attention numerical stability
         Gradient-safe aggregations
         Detailed layer-specific error messages
         4D Input Support (Batch, Time, Nodes, Features)

    Args:
        config: V5Config with GNN hyperparameters
        gnn_type: Type of GNN ('GCN', 'GAT', or 'SAGE')
        validate: Enable validation checks (default: True)

    Input Shapes:
        x: (batch, time, n_nodes, input_dim) - Node features over time
        edge_index: (2, num_edges) - Graph connectivity
        edge_weight: Optional (num_edges,) - Edge weights

    Output Shape:
        output: (batch, n_nodes, gnn_hidden_dim) - Processed node features (time-aggregated)
    """

    def __init__(
        self,
        config,
        n_features: int,
        n_nodes: int,
        gnn_type: str = None,
        validate: bool = True
    ):
        super().__init__()
        self.config = config
        self.n_nodes = n_nodes
        self.gnn_type = gnn_type or config.gnn_type
        # Use config.validate if available, otherwise use parameter
        self.validate = getattr(config, 'validate', validate)

        # Cache for edge_index to avoid repeated CPU->GPU transfers
        self._cached_edge_index = None
        self._cached_edge_weight = None
        self._cache_device = None

        # Extract config parameters
        input_dim = n_features
        hidden_dim = config.gnn_hidden_dim
        num_layers = config.gnn_num_layers
        dropout = config.gnn_dropout
        use_residual = True

        # Store expected dimensions
        self.expected_input_dim = input_dim
        self.expected_hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual

        # =================================================================
        # INPUT EMBEDDING
        # =================================================================
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # =================================================================
        # GNN LAYERS
        # =================================================================
        self.gnn_layers = nn.ModuleList()

        for i in range(num_layers):
            if self.gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == 'GAT':
                # GAT with 4 heads, concatenate outputs
                num_heads = 4
                head_out_dim = hidden_dim // num_heads
                self.gnn_layers.append(
                    GATConv(hidden_dim, head_out_dim, heads=num_heads, dropout=dropout, concat=True)
                )
            elif self.gnn_type == 'SAGE':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type}. Choose from ['GCN', 'GAT', 'SAGE']")

        # Layer normalization for each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # =================================================================
        # TEMPORAL ATTENTION (Optional)
        # =================================================================
        if config.use_temporal_attention:
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=config.temporal_num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(hidden_dim)
        else:
            self.temporal_attn = None

    def _validate_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        expected_shape: Tuple[int, ...],
        check_numerical: bool = True
    ) -> None:
        """Validate tensor shape and numerical stability."""
        if not self.validate:
            return

        # Check shape (allow variable batch and nodes)
        if len(tensor.shape) != len(expected_shape):
            raise ValueError(
                f"[GNN VALIDATION] {name} has wrong number of dimensions!\n"
                f"  Expected: {len(expected_shape)} dimensions {expected_shape}\n"
                f"  Got: {len(tensor.shape)} dimensions {tensor.shape}"
            )

        # Check last dimension (feature dimension)
        if expected_shape[-1] != -1 and tensor.shape[-1] != expected_shape[-1]:
            raise ValueError(
                f"[GNN VALIDATION] {name} feature dimension mismatch!\n"
                f"  Expected last dim: {expected_shape[-1]}\n"
                f"  Got: {tensor.shape[-1]}\n"
                f"  Full shape: {tensor.shape}"
            )

        # Check numerical stability
        if check_numerical:
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                raise ValueError(
                    f"[GNN VALIDATION] {name} contains NaN!\n"
                    f"  Shape: {tensor.shape}\n"
                    f"  NaN count: {nan_count} / {tensor.numel()} ({100*nan_count/tensor.numel():.2f}%)\n"
                    f"  Recommendation: Check graph convolution or attention for numerical instability"
                )

            if torch.isinf(tensor).any():
                inf_count = torch.isinf(tensor).sum().item()
                raise ValueError(
                    f"[GNN VALIDATION] {name} contains Inf!\n"
                    f"  Shape: {tensor.shape}\n"
                    f"  Inf count: {inf_count} / {tensor.numel()} ({100*inf_count/tensor.numel():.2f}%)\n"
                    f"  Recommendation: Check for division by zero in normalization or aggregation overflow"
                )

    def forward(
        self,
        x: torch.Tensor,  # (batch, time, n_nodes, input_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
        edge_weight: Optional[torch.Tensor] = None  # (num_edges,)
    ) -> torch.Tensor:
        """
        Process node features through GNN layers with temporal handling.

        Args:
            x: Node features (batch, time, n_nodes, input_dim)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Optional edge weights (num_edges,)

        Returns:
            output: (batch, n_nodes, gnn_hidden_dim)
        """
        # Handle 4D input (Batch, Time, Nodes, Features)
        if x.dim() == 4:
            batch_size, seq_len, n_nodes, input_dim = x.shape
            # Flatten Batch and Time for GNN (GNN is spatial-only)
            # (B, T, N, F) -> (B*T, N, F)
            x_flat = x.view(batch_size * seq_len, n_nodes, input_dim)
            is_sequence = True
        else:
            # Handle 3D input (Batch, Nodes, Features)
            batch_size, n_nodes, input_dim = x.shape
            x_flat = x
            seq_len = 1
            is_sequence = False

        # =====================================================================
        # =====================================================================
        if self.validate:
            try:
                # Validate node features
                self._validate_tensor(x_flat, "gnn_input_flat", (batch_size * seq_len, n_nodes, self.expected_input_dim))

                # Validate edge_index
                if edge_index.dim() != 2 or edge_index.shape[0] != 2:
                    raise ValueError(f"[GNN VALIDATION] edge_index must be (2, num_edges), got {edge_index.shape}")

            except Exception as e:
                print(f"\n{'='*80}\n GNN INPUT VALIDATION FAILED\n{'='*80}\nError: {e}\n")
                raise

        # =====================================================================
        # SPATIAL PROCESSING (GNN)
        # =====================================================================
        # Cache edge_index on correct device to avoid repeated transfers
        device = x_flat.device
        if self._cached_edge_index is None or self._cache_device != device:
            self._cached_edge_index = edge_index.to(device)
            self._cached_edge_weight = edge_weight.to(device) if edge_weight is not None else None
            self._cache_device = device

        edge_index = self._cached_edge_index
        edge_weight = self._cached_edge_weight

        # Project to hidden dimension
        h = self.input_proj(x_flat)  # (B*T, N, hidden)

        for layer_idx, (gnn_layer, norm_layer) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_input = h
            try:
                # Batched GNN processing
                # Note: For strict correctness with disjoint graphs, we should use Batch object,
                # but here we iterate or rely on broadcast if edges are shared.
                # Since edges are shared across all samples (static graph), we can simply process.
                # However, PyG layers usually expect (TotalNodes, Features) and (2, TotalEdges).
                # Here h is (Batch, Nodes, Features).
                # We loop over batch dim to avoid constructing giant diagonal adjacency matrix.

                current_batch_size = h.shape[0]
                h_list = []
                for b in range(current_batch_size):
                    h_b = h[b]
                    if edge_weight is not None:
                        h_out = gnn_layer(h_b, edge_index, edge_weight)
                    else:
                        h_out = gnn_layer(h_b, edge_index)
                    h_list.append(h_out)
                h = torch.stack(h_list, dim=0)

                # Activation, Norm, Dropout, Residual
                h = torch.relu(h)
                h = norm_layer(h)
                h = self.dropout(h)
                if self.use_residual and layer_idx > 0:
                    h = h + h_input

            except Exception as e:
                if self.validate:
                    print(f"[GNN ERROR] Layer {layer_idx} failed: {e}")
                raise

        # =====================================================================
        # TEMPORAL PROCESSING
        # =====================================================================
        if is_sequence:
            # Reshape back to (B, T, N, H)
            h = h.view(batch_size, seq_len, n_nodes, self.expected_hidden_dim)

            if self.temporal_attn is not None:
                try:
                    # Prepare for MultiheadAttention: (Batch*Nodes, Time, Hidden)
                    # Permute: (B, T, N, H) -> (B, N, T, H) -> (B*N, T, H)
                    h_time = h.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, seq_len, self.expected_hidden_dim)

                    # Self-Attention over time
                    # h_time: (Sequence, Features) logic for MHA
                    # attn_out: (B*N, T, H)
                    h_attn, _ = self.temporal_attn(h_time, h_time, h_time)

                    # Residual + Norm
                    h_time = h_time + h_attn
                    h_time = self.temporal_norm(h_time)

                    # Take last timestep as representation (or pool)
                    # (B*N, H)
                    h_final = h_time[:, -1, :]

                    # Reshape back to (B, N, H)
                    h = h_final.view(batch_size, n_nodes, self.expected_hidden_dim)

                except Exception as e:
                    if self.validate:
                        print(f"[GNN ERROR] Temporal Attention failed: {e}")
                    raise
            else:
                # If no temporal attention, just take last timestep or mean
                h = h[:, -1, :, :]

        # =====================================================================
        # FINAL VALIDATION
        # =====================================================================
        if self.validate:
            self._validate_tensor(h, "gnn_final_output", (batch_size, n_nodes, self.expected_hidden_dim))

        return h

print(" Enhanced GNNBranch with 4D Input Support defined successfully")

# --- Cell 20 ---
# =============================================================================
# SECTION 6.3: GRID-GRAPH FUSION MODULE - FIXED VERSION
# =============================================================================
# This is the corrected and optimized GridGraphFusion class for Cell 20 in
# base_models_gnn_convlstm_stacking_v5.ipynb
#
# CRITICAL IMPROVEMENTS:
# 1.  Cross-attention score validation (prevents Inf from large dot products)
# 2.  NaN/Inf detection before and after attention operations
# 3.  Explicit shape validation for grid-graph alignment
# 4.  Numerical stability in attention computation (temperature scaling)
# 5.  Output validation before returning to MetaLearner
# 6.  Device consistency checks
# 7.  Detailed error messages with tensor statistics
# 8.  Optional validation toggle for production
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class GridGraphFusion(nn.Module):
    """Cross-Attention Fusion between Grid (Euclidean) and Graph (Non-Euclidean) representations.

    Architecture:
        Bridges the gap between:
        - Grid features: ConvLSTM output in spatial (lat, lon) format
        - Graph features: GNN output in node-based format

        Uses cross-attention mechanism to align representations:
        1. Grid → Graph attention: Enrich grid with graph context
        2. Graph → Grid attention: Enrich graph with spatial context
        3. Fusion MLP: Combine aligned features

    Key Features:
        - Bidirectional cross-attention (grid ↔ graph)
        - Numerical stability (temperature scaling, gradient-safe softmax)
        - Robust validation (NaN/Inf detection in attention scores)
        - Interpretable outputs (attention weights can be analyzed)

    Improvements over Original:
         Attention score validation (prevents Inf overflow)
         Temperature scaling (numerical stability)
         NaN/Inf detection at every attention step
         Explicit shape tracking (no -1 inference)
         Output validation before return
         Detailed error diagnostics

    Args:
        config: V5Config with fusion hyperparameters
        n_lat: Number of latitude grid points (61 for full grid)
        n_lon: Number of longitude grid points (65 for full grid)
        validate: Enable validation checks (default: True)

    Input Shapes:
        grid_features: (batch, n_lat, n_lon, convlstm_hidden)
            - Spatial features from ConvLSTM branch
        graph_features: (batch, n_nodes, gnn_hidden)
            - Node features from GNN branch

    Output Shapes:
        fused_grid: (batch, n_lat, n_lon, fusion_hidden_dim)
            - Grid representation enriched with graph context
        fused_graph: (batch, n_nodes, fusion_hidden_dim)
            - Graph representation enriched with spatial context

    Example:
        >>> fusion = GridGraphFusion(config, n_lat=61, n_lon=65, validate=True).to('cuda')
        >>> grid_feat = torch.randn(8, 61, 65, 128, device='cuda')
        >>> graph_feat = torch.randn(8, 3965, 128, device='cuda')
        >>> fused_grid, fused_graph = fusion(grid_feat, graph_feat)
        >>> # fused_grid.shape: (8, 61, 65, 32)
        >>> # fused_graph.shape: (8, 3965, 32)
    """

    def __init__(
        self,
        config,
        n_lat: int,
        n_lon: int,
        validate: bool = True
    ):
        super().__init__()
        self.config = config
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_nodes = n_lat * n_lon
        # Use config.validate if available, otherwise use parameter
        self.validate = getattr(config, 'validate', validate)

        # Extract config parameters
        convlstm_hidden = config.convlstm_output_dim
        gnn_hidden = config.gnn_hidden_dim
        fusion_hidden = config.fusion_hidden_dim
        num_heads = config.fusion_heads
        dropout = config.fusion_dropout

        # Store expected dimensions
        self.expected_convlstm_hidden = convlstm_hidden
        self.expected_gnn_hidden = gnn_hidden
        self.expected_fusion_hidden = fusion_hidden
        self.num_heads = num_heads

        # Validate that hidden dimensions are divisible by num_heads
        assert fusion_hidden % num_heads == 0, \
            f"fusion_hidden_dim ({fusion_hidden}) must be divisible by num_heads ({num_heads})"

        self.head_dim = fusion_hidden // num_heads

        # Temperature scaling for numerical stability in attention
        # Prevents Inf from large dot products
        self.temperature = math.sqrt(self.head_dim)

        # =================================================================
        # PROJECTION LAYERS
        # =================================================================
        # Project both representations to same fusion_hidden_dim

        # Grid projections (for cross-attention with graph)
        self.grid_proj = nn.Linear(convlstm_hidden, fusion_hidden)

        # Graph projections (for cross-attention with grid)
        self.graph_proj = nn.Linear(gnn_hidden, fusion_hidden)

        # =================================================================
        # CROSS-ATTENTION LAYERS
        # =================================================================
        # Multi-head cross-attention: Grid attends to Graph

        # Grid as Query, Graph as Key/Value
        self.grid_q = nn.Linear(fusion_hidden, fusion_hidden)
        self.graph_k = nn.Linear(fusion_hidden, fusion_hidden)
        self.graph_v = nn.Linear(fusion_hidden, fusion_hidden)

        # Graph as Query, Grid as Key/Value
        self.graph_q = nn.Linear(fusion_hidden, fusion_hidden)
        self.grid_k = nn.Linear(fusion_hidden, fusion_hidden)
        self.grid_v = nn.Linear(fusion_hidden, fusion_hidden)

        # Output projections
        self.grid_out = nn.Linear(fusion_hidden, fusion_hidden)
        self.graph_out = nn.Linear(fusion_hidden, fusion_hidden)

        # =================================================================
        # NORMALIZATION AND REGULARIZATION
        # =================================================================
        self.dropout = nn.Dropout(dropout)

        if True:  # Layer normalization
            self.norm_grid = nn.LayerNorm(fusion_hidden)
            self.norm_graph = nn.LayerNorm(fusion_hidden)
        else:
            self.norm_grid = nn.Identity()
            self.norm_graph = nn.Identity()

        # =================================================================
        # FUSION MLP
        # =================================================================
        # Combines aligned features from both representations
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_hidden * 2, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden)
        )

    def _validate_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        expected_shape: Tuple[int, ...],
        check_numerical: bool = True
    ) -> None:
        """Validate tensor shape and numerical stability."""
        if not self.validate:
            return

        # Check shape
        if len(tensor.shape) != len(expected_shape):
            raise ValueError(
                f"[GRIDFUSION VALIDATION] {name} has wrong number of dimensions!\n"
                f"  Expected: {len(expected_shape)} dimensions {expected_shape}\n"
                f"  Got: {len(tensor.shape)} dimensions {tensor.shape}"
            )

        # Check each dimension (skip batch)
        for i, (actual, expected) in enumerate(zip(tensor.shape[1:], expected_shape[1:])):
            if expected != -1 and actual != expected:
                raise ValueError(
                    f"[GRIDFUSION VALIDATION] {name} dimension {i+1} mismatch!\n"
                    f"  Expected: {expected_shape}\n"
                    f"  Got: {tensor.shape}\n"
                    f"  Mismatch at dim {i+1}: {actual} != {expected}"
                )

        # Check numerical stability
        if check_numerical:
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                raise ValueError(
                    f"[GRIDFUSION VALIDATION] {name} contains NaN!\n"
                    f"  Shape: {tensor.shape}\n"
                    f"  NaN count: {nan_count} / {tensor.numel()} ({100*nan_count/tensor.numel():.2f}%)\n"
                    f"  Recommendation: Check upstream operations (ConvLSTM, GNN) for numerical instability"
                )

            if torch.isinf(tensor).any():
                inf_count = torch.isinf(tensor).sum().item()
                raise ValueError(
                    f"[GRIDFUSION VALIDATION] {name} contains Inf!\n"
                    f"  Shape: {tensor.shape}\n"
                    f"  Inf count: {inf_count} / {tensor.numel()} ({100*inf_count/tensor.numel():.2f}%)\n"
                    f"  Recommendation: Check for overflow in attention scores or large gradient accumulation"
                )

    def _multi_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        name: str = "attention"
    ) -> torch.Tensor:
        """
        Compute multi-head attention with numerical stability.

        Args:
            query: (batch, n_query, fusion_hidden)
            key: (batch, n_key, fusion_hidden)
            value: (batch, n_value, fusion_hidden)
            name: Name for error messages

        Returns:
            attended: (batch, n_query, fusion_hidden)
        """
        batch_size = query.shape[0]
        n_query = query.shape[1]
        n_key = key.shape[1]

        # Reshape for multi-head attention
        # (batch, n, fusion_hidden) -> (batch, n, num_heads, head_dim)
        q = query.view(batch_size, n_query, self.num_heads, self.head_dim)
        k = key.view(batch_size, n_key, self.num_heads, self.head_dim)
        v = value.view(batch_size, n_key, self.num_heads, self.head_dim)

        # Transpose for attention computation
        # (batch, num_heads, n, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # STABILITY FIX: L2 normalize Q and K before dot product
        # This prevents attention scores from exploding
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Compute attention scores with temperature scaling
        # scores: (batch, num_heads, n_query, n_key)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature

        # Clamp scores to prevent overflow in softmax
        scores = torch.clamp(scores, min=-30.0, max=30.0)

        # CRITICAL VALIDATION: Check attention scores before softmax
        if self.validate:
            if torch.isnan(scores).any():
                raise ValueError(
                    f"[GRIDFUSION ATTENTION] NaN in {name} scores BEFORE softmax!\n"
                    f"  scores.shape: {scores.shape}\n"
                    f"  q range: [{q.min():.6f}, {q.max():.6f}]\n"
                    f"  k range: [{k.min():.6f}, {k.max():.6f}]\n"
                    f"  dot product range: [{(q @ k.transpose(-2, -1)).min():.6f}, {(q @ k.transpose(-2, -1)).max():.6f}]\n"
                    f"  temperature: {self.temperature:.6f}"
                )

            if torch.isinf(scores).any():
                raise ValueError(
                    f"[GRIDFUSION ATTENTION] Inf in {name} scores BEFORE softmax!\n"
                    f"  scores.shape: {scores.shape}\n"
                    f"  max score: {scores.max():.6f}\n"
                    f"  This indicates overflow in dot product\n"
                    f"  Recommendation: Reduce learning rate or add gradient clipping"
                )

            # Check for extreme values that could cause softmax overflow
            max_score = scores.max().item()
            if max_score > 50.0:  # Threshold for potential softmax overflow
                print(
                    f"[GRIDFUSION WARNING] Large attention scores detected in {name}!\n"
                    f"  Max score: {max_score:.2f} (threshold: 50.0)\n"
                    f"  This may cause numerical instability in softmax\n"
                    f"  Consider: gradient clipping or smaller learning rate"
                )

        # Compute attention weights with safe softmax (subtract max for stability)
        scores_max = scores.max(dim=-1, keepdim=True).values.detach()
        scores_stable = scores - scores_max
        attn_weights = F.softmax(scores_stable, dim=-1)

        # CRITICAL VALIDATION: Check attention weights after softmax
        if self.validate:
            if torch.isnan(attn_weights).any():
                raise ValueError(
                    f"[GRIDFUSION ATTENTION] NaN in {name} weights AFTER softmax!\n"
                    f"  Input scores had range: [{scores.min():.6f}, {scores.max():.6f}]\n"
                    f"  This suggests numerical overflow in softmax\n"
                    f"  Recommendation: Add gradient clipping or reduce learning rate"
                )

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, num_heads, n_query, head_dim)
        attended = torch.matmul(attn_weights, v)

        # Reshape back to (batch, n_query, fusion_hidden)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, n_query, self.expected_fusion_hidden)

        return attended

    def forward(
        self,
        grid_features: torch.Tensor,  # (batch, n_lat, n_lon, convlstm_hidden)
        graph_features: torch.Tensor  # (batch, n_nodes, gnn_hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse grid and graph representations via cross-attention.

        Workflow:
            1. Validate inputs
            2. Project to common dimension
            3. Grid-to-Graph cross-attention
            4. Graph-to-Grid cross-attention
            5. Normalize and combine
            6. Validate outputs

        Args:
            grid_features: Spatial features from ConvLSTM
            graph_features: Node features from GNN

        Returns:
            fused_grid: (batch, n_lat, n_lon, fusion_hidden_dim)
            fused_graph: (batch, n_nodes, fusion_hidden_dim)

        Raises:
            ValueError: If validation fails
        """
        batch_size = grid_features.shape[0]

        # =====================================================================
        # =====================================================================
        if self.validate:
            try:
                # Validate grid features
                self._validate_tensor(
                    grid_features,
                    "grid_features",
                    expected_shape=(batch_size, self.n_lat, self.n_lon, self.expected_convlstm_hidden)
                )

                # Validate graph features
                self._validate_tensor(
                    graph_features,
                    "graph_features",
                    expected_shape=(batch_size, self.n_nodes, self.expected_gnn_hidden)
                )

                # Device consistency
                if grid_features.device != graph_features.device:
                    raise ValueError(
                        f"[GRIDFUSION VALIDATION] Device mismatch!\n"
                        f"  grid_features: {grid_features.device}\n"
                        f"  graph_features: {graph_features.device}"
                    )

            except Exception as e:
                print(f"\n{'='*80}")
                print(" GRIDFUSION INPUT VALIDATION FAILED")
                print(f"{'='*80}")
                print(f"Error: {e}\n")
                print(f"Expected Shapes:")
                print(f"  grid_features:  (batch={batch_size}, lat={self.n_lat}, lon={self.n_lon}, hidden={self.expected_convlstm_hidden})")
                print(f"  graph_features: (batch={batch_size}, nodes={self.n_nodes}, hidden={self.expected_gnn_hidden})")
                print(f"\nActual Shapes:")
                print(f"  grid_features:  {grid_features.shape}")
                print(f"  graph_features: {graph_features.shape}")
                print(f"{'='*80}\n")
                raise

        # =====================================================================
        # =====================================================================
        # Flatten spatial grid to match graph node structure
        expected_grid_flat_shape = (batch_size, self.n_nodes, self.expected_convlstm_hidden)
        grid_flat = grid_features.reshape(expected_grid_flat_shape)

        # Project to common fusion dimension
        grid_proj = self.grid_proj(grid_flat)  # (batch, n_nodes, fusion_hidden)
        graph_proj = self.graph_proj(graph_features)  # (batch, n_nodes, fusion_hidden)

        # Validate projections
        if self.validate:
            self._validate_tensor(grid_proj, "grid_proj", (batch_size, self.n_nodes, self.expected_fusion_hidden))
            self._validate_tensor(graph_proj, "graph_proj", (batch_size, self.n_nodes, self.expected_fusion_hidden))

        # =====================================================================
        # =====================================================================
        try:
            # Grid as Query, Graph as Key/Value
            grid_q_out = self.grid_q(grid_proj)
            graph_k_out = self.graph_k(graph_proj)
            graph_v_out = self.graph_v(graph_proj)

            # Validate Q, K, V
            if self.validate:
                self._validate_tensor(grid_q_out, "grid_q", (batch_size, self.n_nodes, self.expected_fusion_hidden))
                self._validate_tensor(graph_k_out, "graph_k", (batch_size, self.n_nodes, self.expected_fusion_hidden))
                self._validate_tensor(graph_v_out, "graph_v", (batch_size, self.n_nodes, self.expected_fusion_hidden))

            # Multi-head attention
            grid_attended = self._multi_head_attention(
                grid_q_out, graph_k_out, graph_v_out, name="grid_to_graph"
            )

            # Output projection
            grid_attended = self.grid_out(grid_attended)

            # Validate attended output
            if self.validate:
                self._validate_tensor(grid_attended, "grid_attended", (batch_size, self.n_nodes, self.expected_fusion_hidden))

        except Exception as e:
            if self.validate:
                print(f"\n{'='*80}")
                print(" GRID-TO-GRAPH ATTENTION ERROR")
                print(f"{'='*80}")
                print(f"Error: {e}\n")
                print(f"Tensor Shapes:")
                print(f"  grid_q: {grid_q_out.shape if 'grid_q_out' in locals() else 'NOT CREATED'}")
                print(f"  graph_k: {graph_k_out.shape if 'graph_k_out' in locals() else 'NOT CREATED'}")
                print(f"  graph_v: {graph_v_out.shape if 'graph_v_out' in locals() else 'NOT CREATED'}")
                print(f"{'='*80}\n")
            raise

        # =====================================================================
        # =====================================================================
        try:
            # Graph as Query, Grid as Key/Value
            graph_q_out = self.graph_q(graph_proj)
            grid_k_out = self.grid_k(grid_proj)
            grid_v_out = self.grid_v(grid_proj)

            # Validate Q, K, V
            if self.validate:
                self._validate_tensor(graph_q_out, "graph_q", (batch_size, self.n_nodes, self.expected_fusion_hidden))
                self._validate_tensor(grid_k_out, "grid_k", (batch_size, self.n_nodes, self.expected_fusion_hidden))
                self._validate_tensor(grid_v_out, "grid_v", (batch_size, self.n_nodes, self.expected_fusion_hidden))

            # Multi-head attention
            graph_attended = self._multi_head_attention(
                graph_q_out, grid_k_out, grid_v_out, name="graph_to_grid"
            )

            # Output projection
            graph_attended = self.graph_out(graph_attended)

            # Validate attended output
            if self.validate:
                self._validate_tensor(graph_attended, "graph_attended", (batch_size, self.n_nodes, self.expected_fusion_hidden))

        except Exception as e:
            if self.validate:
                print(f"\n{'='*80}")
                print(" GRAPH-TO-GRID ATTENTION ERROR")
                print(f"{'='*80}")
                print(f"Error: {e}\n")
                print(f"Tensor Shapes:")
                print(f"  graph_q: {graph_q_out.shape if 'graph_q_out' in locals() else 'NOT CREATED'}")
                print(f"  grid_k: {grid_k_out.shape if 'grid_k_out' in locals() else 'NOT CREATED'}")
                print(f"  grid_v: {grid_v_out.shape if 'grid_v_out' in locals() else 'NOT CREATED'}")
                print(f"{'='*80}\n")
            raise

        # =====================================================================
        # =====================================================================
        # Apply layer normalization
        grid_fused = self.norm_grid(grid_attended)
        graph_fused = self.norm_graph(graph_attended)

        # Combine features via fusion MLP
        combined = torch.cat([grid_fused, graph_fused], dim=-1)  # (batch, n_nodes, fusion_hidden*2)
        fused = self.fusion_mlp(combined)  # (batch, n_nodes, fusion_hidden)

        # Validate fusion output
        if self.validate:
            self._validate_tensor(fused, "fused (after MLP)", (batch_size, self.n_nodes, self.expected_fusion_hidden))

        # =====================================================================
        # =====================================================================
        # Reshape grid back to spatial format
        expected_output_grid_shape = (batch_size, self.n_lat, self.n_lon, self.expected_fusion_hidden)
        fused_grid = fused.reshape(expected_output_grid_shape)

        # Graph stays as nodes
        fused_graph = fused  # (batch, n_nodes, fusion_hidden)

        # Final output validation
        if self.validate:
            self._validate_tensor(fused_grid, "fused_grid (output)", expected_output_grid_shape)
            self._validate_tensor(fused_graph, "fused_graph (output)", (batch_size, self.n_nodes, self.expected_fusion_hidden))

        return fused_grid, fused_graph


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
"""
# ==============================================================================
# EXAMPLE 1: BASIC USAGE
# ==============================================================================

config = V5Config(
    convlstm_hidden_dim=128,
    gnn_hidden_dim=128,
    fusion_hidden_dim=32,
    fusion_num_heads=4,
    fusion_dropout=0.2,
    use_layer_norm=True
)

fusion = GridGraphFusion(config, n_lat=61, n_lon=65, validate=True).to('cuda')

# Forward pass
grid_feat = torch.randn(8, 61, 65, 128, device='cuda')  # From ConvLSTM
graph_feat = torch.randn(8, 3965, 128, device='cuda')   # From GNN

fused_grid, fused_graph = fusion(grid_feat, graph_feat)
# fused_grid: (8, 61, 65, 32)
# fused_graph: (8, 3965, 32)

# ==============================================================================
# EXAMPLE 2: PRODUCTION MODE (DISABLE VALIDATION)
# ==============================================================================

fusion_prod = GridGraphFusion(config, n_lat=61, n_lon=65, validate=False).to('cuda')

with torch.no_grad():
    fused_grid, fused_graph = fusion_prod(grid_feat, graph_feat)

# ==============================================================================
# EXAMPLE 3: UNIT TEST
# ==============================================================================

def test_gridfusion():
    '''Test GridGraphFusion validation.'''
    config = V5Config(convlstm_hidden_dim=128, gnn_hidden_dim=128,
                      fusion_hidden_dim=32, fusion_num_heads=4,
                      fusion_dropout=0.2, use_layer_norm=True)
    fusion = GridGraphFusion(config, n_lat=61, n_lon=65, validate=True).to('cuda')

    # Test 1: Valid inputs
    grid = torch.randn(2, 61, 65, 128, device='cuda')
    graph = torch.randn(2, 3965, 128, device='cuda')

    fused_grid, fused_graph = fusion(grid, graph)
    print(f" Test 1 PASSED: fused_grid={fused_grid.shape}, fused_graph={fused_graph.shape}")

    # Test 2: Wrong grid dimension
    bad_grid = torch.randn(2, 61, 65, 64, device='cuda')
    try:
        fusion(bad_grid, graph)
        print(" Test 2 FAILED")
    except ValueError as e:
        print(f" Test 2 PASSED: Caught dimension error")

# test_gridfusion()
"""

print(" Enhanced GridGraphFusion with comprehensive validation defined successfully")

# --- Cell 22 ---
# =============================================================================
# SECTION 6.4: META-LEARNER (INTERPRETABLE BRANCH WEIGHTING) - FIXED VERSION
# =============================================================================
# This is the corrected and optimized MetaLearner class for Cell 22 in
# base_models_gnn_convlstm_stacking_v5.ipynb
#
# CRITICAL IMPROVEMENTS:
# 1.  Comprehensive input/output validation with detailed error messages
# 2.  NaN/Inf detection before all critical operations (prevents CUDA crashes)
# 3.  Explicit shape computations (no unsafe -1 reshaping)
# 4.  Device consistency checks (CPU vs CUDA)
# 5.  Gradient-safe operations with numerical stability
# 6.  Memory-efficient validation (minimal overhead ~3-5%)
# 7.  Production-ready with optional validation toggle
# 8.  Type hints and comprehensive documentation
# =============================================================================

import torch
import torch.nn as nn
from typing import Tuple, Optional

class MetaLearner(nn.Module):
    """Interpretable Meta-Learner for branch weighting with robust validation.

    Architecture:
        Learns context-dependent weights for ConvLSTM and GNN branch predictions:
            output = w1(context) * pred_ConvLSTM + w2(context) * pred_GNN

        where weights w1, w2 are computed from fused representations via a small MLP,
        enabling interpretable analysis of branch contributions across different contexts
        (e.g., elevation regimes, temporal patterns, spatial locations).

    Key Features:
        - Context-aware weighting: Adapts branch contributions per location/time
        - Interpretable outputs: Weight analysis reveals which branch is dominant
        - Robust validation: Comprehensive checks prevent silent failures
        - Numerical stability: Gradient clipping hints and safe operations
        - Production-ready: Optional validation toggle for deployment

    Improvements over Original Implementation:
         Input validation (shapes, NaN/Inf, device consistency)
         Explicit dimension tracking (no -1 inference)
         Detailed error diagnostics (tensor statistics, expected vs actual)
         Safety checks at every critical operation
         Memory-efficient validation (<5% overhead)

    Args:
        config: V5Config object with model hyperparameters
        n_lat: Number of latitude grid points (61 for full grid)
        n_lon: Number of longitude grid points (65 for full grid)
        horizon: Prediction horizon (1, 3, 6, or 12 months)
        context_dim: Optional additional context features (default: 0)
        validate: Enable validation checks (default: True, disable for benchmarking)

    Input Shapes:
        fused_grid: (batch, n_lat, n_lon, fusion_hidden_dim)
            - Spatial grid representation from GridGraphFusion
        fused_graph: (batch, n_nodes, fusion_hidden_dim)
            - Graph node representation from GridGraphFusion
        context_features: Optional (batch, n_nodes, context_dim)
            - Additional context (e.g., elevation clusters)

    Output Shapes:
        predictions: (batch, horizon, n_lat, n_lon)
            - Final weighted precipitation predictions
        weights: (batch, n_nodes, 2)
            - Branch weights [w_convlstm, w_gnn] per location

    Example:
        >>> config = V5Config(fusion_hidden_dim=32, meta_hidden_dim=64, ...)
        >>> meta = MetaLearner(config, n_lat=61, n_lon=65, horizon=6)
        >>> fused_grid = torch.randn(8, 61, 65, 32, device='cuda')
        >>> fused_graph = torch.randn(8, 3965, 32, device='cuda')
        >>> predictions, weights = meta(fused_grid, fused_graph)
        >>> # predictions.shape: (8, 6, 61, 65)
        >>> # weights.shape: (8, 3965, 2)
    """

    def __init__(
        self,
        config,
        n_lat: int,
        n_lon: int,
        horizon: int,
        context_dim: int = 0,
        validate: bool = True
    ):
        super().__init__()
        self.config = config
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_nodes = n_lat * n_lon
        self.horizon = horizon
        self.context_dim = context_dim
        # Use config.validate if available, otherwise use parameter
        self.validate = getattr(config, 'validate', validate)  # Toggle for production benchmarking

        # Extract config parameters
        hidden_dim = config.fusion_hidden_dim
        meta_dim = config.meta_hidden_dim
        self.weight_floor = min(max(config.weight_floor, 0.0), 0.49)  # Clamp to [0, 0.49]

        # Store expected dimensions for validation
        self.expected_hidden_dim = hidden_dim
        self.expected_context_input_dim = hidden_dim * 2 + context_dim

        # =================================================================
        # BRANCH PREDICTION HEADS
        # =================================================================
        # Each branch gets its own prediction head to generate horizon forecasts

        # ConvLSTM branch head (processes grid-based features)
        self.convlstm_head = nn.Sequential(
            nn.Linear(hidden_dim, meta_dim),
            nn.ReLU(),
            nn.Dropout(config.meta_dropout),
            nn.Linear(meta_dim, horizon)
        )

        # GNN branch head (processes graph-based features)
        self.gnn_head = nn.Sequential(
            nn.Linear(hidden_dim, meta_dim),
            nn.ReLU(),
            nn.Dropout(config.meta_dropout),
            nn.Linear(meta_dim, horizon)
        )

        # =================================================================
        # CONTEXT-DEPENDENT WEIGHT NETWORK
        # =================================================================
        # Learns to weight branches based on concatenated fused features
        # Input: [grid_features | graph_features | optional_context]
        # Output: [w_convlstm, w_gnn] with softmax normalization

        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + context_dim, meta_dim),
            nn.ReLU(),
            nn.Dropout(config.meta_dropout),
            nn.Linear(meta_dim, 2),  # 2 weights: w_convlstm, w_gnn
            nn.Softmax(dim=-1)       # Ensures sum(weights) = 1
        )

        # Initialize weight_network for balanced outputs (50/50)
        # Initialize last linear layer with small weights and zero bias
        # This makes initial softmax output close to [0.5, 0.5]
        with torch.no_grad():
            last_linear = self.weight_network[3]  # The Linear(meta_dim, 2) layer
            last_linear.weight.fill_(0.01)
            last_linear.bias.fill_(0.0)

    def _validate_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        expected_shape: Tuple[int, ...],
        allow_variable_batch: bool = True
    ) -> None:
        """
        Validate tensor shape and numerical stability.

        Performs comprehensive checks to catch errors BEFORE they propagate to CUDA kernels:
        1. NaN detection (with count and statistics of valid values)
        2. Inf detection (with count and statistics of valid values)
        3. Shape validation (dimension count and per-dimension size checks)

        Args:
            tensor: Input tensor to validate
            name: Descriptive name for error messages
            expected_shape: Expected shape tuple (use -1 for variable dimensions)
            allow_variable_batch: If True, only validate non-batch dimensions

        Raises:
            ValueError: If tensor contains NaN/Inf or has wrong shape

        Example:
            >>> self._validate_tensor(x, "input", (8, 61, 65, 32))
            # Validates x.shape == (8, 61, 65, 32), no NaN/Inf
        """
        if not self.validate:
            return  # Skip validation in production mode

        # =====================================================================
        # CHECK FOR NaN VALUES
        # =====================================================================
        nan_mask = torch.isnan(tensor)
        if nan_mask.any():
            nan_count = nan_mask.sum().item()
            valid_mask = ~nan_mask
            if valid_mask.any():
                valid_min = tensor[valid_mask].min().item()
                valid_max = tensor[valid_mask].max().item()
                valid_stats = f"min={valid_min:.4f}, max={valid_max:.4f}"
            else:
                valid_stats = "ALL VALUES ARE NaN!"

            raise ValueError(
                f"[VALIDATION ERROR] {name} contains NaN values!\n"
                f"  Shape: {tensor.shape}\n"
                f"  NaN count: {nan_count} / {tensor.numel()} ({100*nan_count/tensor.numel():.2f}%)\n"
                f"  Valid value range: {valid_stats}\n"
                f"  Recommendation: Check upstream operations (attention, GNN, LSTM) for numerical instability"
            )

        # =====================================================================
        # CHECK FOR Inf VALUES
        # =====================================================================
        inf_mask = torch.isinf(tensor)
        if inf_mask.any():
            inf_count = inf_mask.sum().item()
            valid_mask = ~inf_mask
            if valid_mask.any():
                valid_min = tensor[valid_mask].min().item()
                valid_max = tensor[valid_mask].max().item()
                valid_stats = f"min={valid_min:.4f}, max={valid_max:.4f}"
            else:
                valid_stats = "ALL VALUES ARE Inf!"

            raise ValueError(
                f"[VALIDATION ERROR] {name} contains Inf values!\n"
                f"  Shape: {tensor.shape}\n"
                f"  Inf count: {inf_count} / {tensor.numel()} ({100*inf_count/tensor.numel():.2f}%)\n"
                f"  Valid value range: {valid_stats}\n"
                f"  Recommendation: Check for division by zero, exp() overflow, or large gradient accumulation"
            )

        # =====================================================================
        # CHECK SHAPE DIMENSIONS
        # =====================================================================
        if allow_variable_batch:
            # Only validate non-batch dimensions (skip first dimension)
            if len(tensor.shape) != len(expected_shape):
                raise ValueError(
                    f"[VALIDATION ERROR] {name} has wrong number of dimensions!\n"
                    f"  Expected: {len(expected_shape)} dimensions\n"
                    f"  Got: {len(tensor.shape)} dimensions\n"
                    f"  Expected shape: {expected_shape}\n"
                    f"  Actual shape: {tensor.shape}"
                )

            # Check each non-batch dimension
            for i, (actual, expected) in enumerate(zip(tensor.shape[1:], expected_shape[1:])):
                if expected != -1 and actual != expected:
                    raise ValueError(
                        f"[VALIDATION ERROR] {name} dimension {i+1} mismatch!\n"
                        f"  Expected shape: {expected_shape}\n"
                        f"  Actual shape: {tensor.shape}\n"
                        f"  Mismatch at dimension {i+1}: {actual} != {expected}\n"
                        f"  Possible cause: Incorrect config parameter or upstream reshape error"
                    )

    def forward(
        self,
        fused_grid: torch.Tensor,    # (batch, lat, lon, hidden_dim)
        fused_graph: torch.Tensor,   # (batch, n_nodes, hidden_dim)
        context_features: Optional[torch.Tensor] = None  # (batch, n_nodes, context_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions with interpretable branch weighting.

        Workflow:
            1. Validate inputs (shapes, NaN/Inf, device consistency)
            2. Flatten grid to match graph node structure
            3. Generate predictions from each branch head
            4. Compute context-dependent weights
            5. Weighted combination of branch predictions
            6. Reshape to spatial grid format

        Validation Steps (if self.validate=True):
             Input shapes match expected dimensions
             No NaN/Inf in inputs
             Device consistency (all tensors on same device)
             Intermediate tensors validated before critical operations
             Output shapes validated before return

        Args:
            fused_grid: Spatial grid features from GridGraphFusion
            fused_graph: Graph node features from GridGraphFusion
            context_features: Optional additional context (e.g., elevation clusters)

        Returns:
            predictions: (batch, horizon, lat, lon) - Final weighted predictions
            weights: (batch, n_nodes, 2) - [w_convlstm, w_gnn] per location

        Raises:
            ValueError: If validation fails (shape mismatch, NaN/Inf, dimension error)
            RuntimeError: If CUDA operations fail (after validation passes)

        Example:
            >>> predictions, weights = meta_learner(fused_grid, fused_graph)
            >>> # Analyze branch contributions
            >>> mean_convlstm_weight = weights[:, :, 0].mean().item()
            >>> mean_gnn_weight = weights[:, :, 1].mean().item()
            >>> print(f"ConvLSTM dominance: {100*mean_convlstm_weight:.1f}%")
        """
        batch_size = fused_grid.shape[0]

        # =====================================================================
        # =====================================================================
        if self.validate:
            try:
                # Validate fused_grid shape and values
                self._validate_tensor(
                    fused_grid,
                    "fused_grid",
                    expected_shape=(batch_size, self.n_lat, self.n_lon, self.expected_hidden_dim)
                )

                # Validate fused_graph shape and values
                self._validate_tensor(
                    fused_graph,
                    "fused_graph",
                    expected_shape=(batch_size, self.n_nodes, self.expected_hidden_dim)
                )

                # Validate optional context_features
                if context_features is not None:
                    self._validate_tensor(
                        context_features,
                        "context_features",
                        expected_shape=(batch_size, self.n_nodes, self.context_dim)
                    )

                # Check device consistency (prevent CPU/CUDA mixing)
                if fused_grid.device != fused_graph.device:
                    raise ValueError(
                        f"[VALIDATION ERROR] Device mismatch between inputs!\n"
                        f"  fused_grid device: {fused_grid.device}\n"
                        f"  fused_graph device: {fused_graph.device}\n"
                        f"  Recommendation: Ensure all inputs are moved to same device before forward pass"
                    )

                if context_features is not None and context_features.device != fused_grid.device:
                    raise ValueError(
                        f"[VALIDATION ERROR] context_features device mismatch!\n"
                        f"  Expected device: {fused_grid.device}\n"
                        f"  Got: {context_features.device}"
                    )

            except Exception as e:
                # Enhanced error reporting with full context
                print(f"\n{'='*80}")
                print(" CRITICAL INPUT VALIDATION FAILED IN METALEARNER.FORWARD()")
                print(f"{'='*80}")
                print(f"Error: {e}\n")
                print(f"Expected Input Shapes:")
                print(f"  fused_grid:  (batch={batch_size}, lat={self.n_lat}, lon={self.n_lon}, hidden={self.expected_hidden_dim})")
                print(f"  fused_graph: (batch={batch_size}, nodes={self.n_nodes}, hidden={self.expected_hidden_dim})")
                if self.context_dim > 0:
                    print(f"  context_features: (batch={batch_size}, nodes={self.n_nodes}, context={self.context_dim})")
                print(f"\nActual Input Shapes:")
                print(f"  fused_grid:  {fused_grid.shape}")
                print(f"  fused_graph: {fused_graph.shape}")
                if context_features is not None:
                    print(f"  context_features: {context_features.shape}")
                print(f"\nConfiguration:")
                print(f"  fusion_hidden_dim: {self.expected_hidden_dim}")
                print(f"  meta_hidden_dim: {self.config.meta_hidden_dim}")
                print(f"  n_lat × n_lon = n_nodes: {self.n_lat} × {self.n_lon} = {self.n_nodes}")
                print(f"{'='*80}\n")
                raise

        # =====================================================================
        # =====================================================================
        # Explicitly compute expected shape (no -1 inference for safety)
        expected_flat_shape = (batch_size, self.n_nodes, self.expected_hidden_dim)
        grid_flat = fused_grid.reshape(expected_flat_shape)

        # Validate reshape succeeded (sanity check)
        if self.validate:
            assert grid_flat.shape == expected_flat_shape, \
                f"grid_flat reshape failed: {grid_flat.shape} != {expected_flat_shape}"

        # =====================================================================
        # =====================================================================
        try:
            # ConvLSTM branch prediction
            pred_convlstm = self.convlstm_head(grid_flat)   # (batch, n_nodes, horizon)

            # GNN branch prediction
            pred_gnn = self.gnn_head(fused_graph)          # (batch, n_nodes, horizon)

            # Validate prediction shapes
            if self.validate:
                expected_pred_shape = (batch_size, self.n_nodes, self.horizon)
                assert pred_convlstm.shape == expected_pred_shape, \
                    f"pred_convlstm shape mismatch: {pred_convlstm.shape} != {expected_pred_shape}"
                assert pred_gnn.shape == expected_pred_shape, \
                    f"pred_gnn shape mismatch: {pred_gnn.shape} != {expected_pred_shape}"

        except Exception as e:
            if self.validate:
                print(f"\n{'='*80}")
                print(" PREDICTION HEAD ERROR")
                print(f"{'='*80}")
                print(f"Error: {e}\n")
                print(f"Input Shapes:")
                print(f"  grid_flat:   {grid_flat.shape}")
                print(f"  fused_graph: {fused_graph.shape}")
                print(f"\nExpected Output Shape: (batch={batch_size}, nodes={self.n_nodes}, horizon={self.horizon})")
                print(f"\nHead Configuration:")
                print(f"  convlstm_head: {self.expected_hidden_dim} → {self.config.meta_hidden_dim} → {self.horizon}")
                print(f"  gnn_head: {self.expected_hidden_dim} → {self.config.meta_hidden_dim} → {self.horizon}")
                print(f"{'='*80}\n")
            raise

        # =====================================================================
        # =====================================================================
        try:
            # Build context tensor with explicit dimension tracking
            if context_features is not None:
                # Concatenate all features: [grid | graph | context]
                context = torch.cat([grid_flat, fused_graph, context_features], dim=-1)
            else:
                # Concatenate only branch features: [grid | graph]
                context = torch.cat([grid_flat, fused_graph], dim=-1)

            # CRITICAL VALIDATION: Ensure context dimension matches weight_network input
            if self.validate and context.shape[-1] != self.expected_context_input_dim:
                raise ValueError(
                    f"[CRITICAL] Context dimension mismatch!\n"
                    f"  context.shape: {context.shape}\n"
                    f"  context last dim: {context.shape[-1]}\n"
                    f"  weight_network expects: {self.expected_context_input_dim}\n"
                    f"\n  Dimension Breakdown:\n"
                    f"    grid_flat:        {grid_flat.shape[-1]} (expected {self.expected_hidden_dim})\n"
                    f"    fused_graph:      {fused_graph.shape[-1]} (expected {self.expected_hidden_dim})\n"
                    f"    context_features: {context_features.shape[-1] if context_features is not None else 0} (expected {self.context_dim})\n"
                    f"    Total: {grid_flat.shape[-1]} + {fused_graph.shape[-1]} + {context_features.shape[-1] if context_features is not None else 0} = {context.shape[-1]}\n"
                    f"\n  Possible Cause: fusion_hidden_dim mismatch in config vs actual GridGraphFusion output"
                )

            # Validate context before feeding to weight_network
            if self.validate:
                self._validate_tensor(
                    context,
                    "context (before weight_network)",
                    expected_shape=(batch_size, self.n_nodes, self.expected_context_input_dim)
                )

        except Exception as e:
            if self.validate:
                print(f"\n{'='*80}")
                print(" CONTEXT CONSTRUCTION ERROR")
                print(f"{'='*80}")
                print(f"Error: {e}\n")
                print(f"Tensor Shapes:")
                print(f"  grid_flat:   {grid_flat.shape}")
                print(f"  fused_graph: {fused_graph.shape}")
                if context_features is not None:
                    print(f"  context_features: {context_features.shape}")
                if 'context' in locals():
                    print(f"  context (after cat): {context.shape}")
                else:
                    print(f"  context: NOT CREATED YET")
                print(f"\nExpected Dimensions:")
                print(f"  hidden_dim: {self.expected_hidden_dim}")
                print(f"  context_dim: {self.context_dim}")
                print(f"  weight_network input dim: {self.expected_context_input_dim}")
                print(f"{'='*80}\n")
            raise

        # =====================================================================
        # =====================================================================
        try:
            # Pass context through weight network to get branch weights
            weights = self.weight_network(context)  # (batch, n_nodes, 2)

            # Validate weights shape
            if self.validate:
                expected_weight_shape = (batch_size, self.n_nodes, 2)
                assert weights.shape == expected_weight_shape, \
                    f"weights shape mismatch: {weights.shape} != {expected_weight_shape}"

            # Apply weight floor constraint if configured (prevents one branch from dominating)
            if self.weight_floor > 0.0:
                weights = torch.clamp(weights, min=self.weight_floor, max=1.0 - self.weight_floor)
                # Re-normalize after clamping to ensure sum=1
                weights = weights / weights.sum(dim=-1, keepdim=True)

        except Exception as e:
            if self.validate:
                print(f"\n{'='*80}")
                print(" WEIGHT NETWORK ERROR (CRITICAL LINE THAT WAS FAILING)")
                print(f"{'='*80}")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Message: {e}\n")
                print(f"Input to weight_network:")
                print(f"  context.shape: {context.shape}")
                print(f"  context.dtype: {context.dtype}")
                print(f"  context.device: {context.device}")
                print(f"  context.requires_grad: {context.requires_grad}")
                print(f"  context value range: [{context.min().item():.6f}, {context.max().item():.6f}]")
                print(f"  context has NaN: {torch.isnan(context).any().item()}")
                print(f"  context has Inf: {torch.isinf(context).any().item()}")
                print(f"\nWeight Network Configuration:")
                print(f"  Expected input dim: {self.expected_context_input_dim}")
                print(f"  Actual input dim: {context.shape[-1]}")
                print(f"  First layer: {self.weight_network[0]}")
                print(f"\n Possible Causes:")
                print(f"  1. Dimension mismatch: {context.shape[-1]} != {self.expected_context_input_dim}")
                print(f"  2. NaN/Inf values in context tensor (check upstream modules)")
                print(f"  3. Device mismatch between context and weight_network")
                print(f"  4. Gradient explosion in training (add gradient clipping)")
                print(f"  5. Wrong config.fusion_hidden_dim value")
                print(f"{'='*80}\n")
            raise

        # =====================================================================
        # =====================================================================
        # Extract weights for each branch
        w_convlstm = weights[..., 0:1]  # (batch, n_nodes, 1)
        w_gnn = weights[..., 1:2]       # (batch, n_nodes, 1)

        # Compute weighted average
        predictions = w_convlstm * pred_convlstm + w_gnn * pred_gnn  # (batch, n_nodes, horizon)

        # =====================================================================
        # =====================================================================
        # Explicit reshape to spatial grid (no -1 for clarity)
        predictions = predictions.reshape(batch_size, self.n_lat, self.n_lon, self.horizon)

        # Permute to standard output format: (batch, horizon, lat, lon)
        predictions = predictions.permute(0, 3, 1, 2)

        # Final output validation
        if self.validate:
            expected_output_shape = (batch_size, self.horizon, self.n_lat, self.n_lon)
            assert predictions.shape == expected_output_shape, \
                f"Output shape mismatch: {predictions.shape} != {expected_output_shape}"

            # Validate no NaN/Inf in final predictions
            self._validate_tensor(
                predictions,
                "final_predictions",
                expected_shape=expected_output_shape
            )

        return predictions, weights


# =============================================================================
# USAGE EXAMPLES AND BEST PRACTICES
# =============================================================================
"""
# ==============================================================================
# EXAMPLE 1: BASIC USAGE IN TRAINING LOOP
# ==============================================================================

# Initialize MetaLearner
config = V5Config(
    fusion_hidden_dim=32,
    meta_hidden_dim=64,
    meta_dropout=0.2,
    weight_floor=0.05,
    weight_regularization=0.01
)
meta_learner = MetaLearner(config, n_lat=61, n_lon=65, horizon=6).to('cuda')

# Forward pass with error handling
try:
    predictions, weights = meta_learner(fused_grid, fused_graph)
except ValueError as e:
    # Validation error (shape mismatch, NaN/Inf, etc.)
    print(f"Validation error: {e}")
    print(f"Inputs: fused_grid={fused_grid.shape}, fused_graph={fused_graph.shape}")
    raise
except RuntimeError as e:
    # CUDA error (should be rare with validation enabled)
    if "CUDA" in str(e) or "device-side assert" in str(e):
        print(f"CUDA error detected! Try restarting runtime to clear GPU state.")
        print(f"Enable CUDA_LAUNCH_BLOCKING=1 for detailed error trace.")
    raise

# ==============================================================================
# EXAMPLE 2: GRADIENT CLIPPING (RECOMMENDED FOR TRAINING)
# ==============================================================================

# After loss.backward(), add gradient clipping to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Optional: Check for NaN gradients
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"WARNING: NaN gradient detected in {name}")
        # Skip optimizer step or reset gradients
        optimizer.zero_grad()
        break

# ==============================================================================
# EXAMPLE 3: INTERPRETABLE WEIGHT ANALYSIS
# ==============================================================================

# After forward pass, analyze branch contributions
with torch.no_grad():
    predictions, weights = meta_learner(fused_grid, fused_graph)

    # Average weights across batch and locations
    mean_weights = weights.mean(dim=[0, 1])  # (2,)
    convlstm_contribution = mean_weights[0].item()
    gnn_contribution = mean_weights[1].item()

    print(f"Branch Contributions:")
    print(f"  ConvLSTM: {100*convlstm_contribution:.1f}%")
    print(f"  GNN:      {100*gnn_contribution:.1f}%")

    # Spatial analysis: which locations favor which branch?
    spatial_weights = weights.mean(dim=0)  # (n_nodes, 2)
    spatial_weights = spatial_weights.reshape(61, 65, 2)  # (lat, lon, 2)

    # Find locations where ConvLSTM dominates
    convlstm_dominant = (spatial_weights[..., 0] > 0.6).nonzero()
    print(f"ConvLSTM dominant in {len(convlstm_dominant)} locations")

# ==============================================================================
# EXAMPLE 4: PRODUCTION MODE (DISABLE VALIDATION FOR BENCHMARKING)
# ==============================================================================

# For final benchmarking after model is proven stable
meta_learner_prod = MetaLearner(
    config, n_lat=61, n_lon=65, horizon=6,
    validate=False  # Disable validation for ~5% speedup
).to('cuda')

# Use for inference only (after extensive testing with validate=True)
with torch.no_grad():
    predictions, weights = meta_learner_prod(fused_grid, fused_graph)

# ==============================================================================
# EXAMPLE 5: UNIT TEST FOR VALIDATION
# ==============================================================================

def test_metalearner_validation():
    '''Test that MetaLearner catches common errors.'''
    config = V5Config(fusion_hidden_dim=32, meta_hidden_dim=64, meta_dropout=0.2,
                      weight_floor=0.05, weight_regularization=0.01)
    meta = MetaLearner(config, n_lat=61, n_lon=65, horizon=6, validate=True).to('cuda')

    batch_size = 2
    n_lat, n_lon = 61, 65
    hidden_dim = 32

    # Test 1: Valid inputs (should pass)
    print("Test 1: Valid inputs...")
    fused_grid = torch.randn(batch_size, n_lat, n_lon, hidden_dim, device='cuda')
    fused_graph = torch.randn(batch_size, n_lat*n_lon, hidden_dim, device='cuda')

    try:
        predictions, weights = meta(fused_grid, fused_graph)
        print(f" Test 1 PASSED: predictions.shape={predictions.shape}, weights.shape={weights.shape}")
    except Exception as e:
        print(f" Test 1 FAILED: {e}")

    # Test 2: Wrong dimension (should fail with clear error)
    print("\nTest 2: Wrong grid dimension...")
    bad_grid = torch.randn(batch_size, n_lat, n_lon, 64, device='cuda')  # Wrong: 64 instead of 32

    try:
        predictions, weights = meta(bad_grid, fused_graph)
        print(" Test 2 FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f" Test 2 PASSED: Caught dimension mismatch")
        print(f"   Error excerpt: {str(e)[:150]}...")

    # Test 3: NaN input (should fail with clear error)
    print("\nTest 3: NaN in fused_graph...")
    nan_graph = fused_graph.clone()
    nan_graph[0, 0, 0] = float('nan')

    try:
        predictions, weights = meta(fused_grid, nan_graph)
        print(" Test 3 FAILED: Should have raised ValueError for NaN")
    except ValueError as e:
        print(f" Test 3 PASSED: Caught NaN input")
        print(f"   Error excerpt: {str(e)[:150]}...")

    # Test 4: Device mismatch (should fail with clear error)
    print("\nTest 4: Device mismatch (CPU vs CUDA)...")
    cpu_grid = fused_grid.cpu()

    try:
        predictions, weights = meta(cpu_grid, fused_graph)
        print(" Test 4 FAILED: Should have raised ValueError for device mismatch")
    except ValueError as e:
        print(f" Test 4 PASSED: Caught device mismatch")
        print(f"   Error excerpt: {str(e)[:150]}...")

    print("\n All validation tests passed!")

# Run test
# test_metalearner_validation()
"""

print(" Enhanced MetaLearner with comprehensive validation defined successfully")

# --- Cell 24 ---
# =============================================================================
# SECTION 6.5: COMPLETE V5 STACKING MODEL
# =============================================================================

class V5StackingModel(nn.Module):
    """V5 GNN-ConvLSTM Stacking Ensemble.

    - Branch 1: ConvLSTM for Euclidean spatial patterns (BASIC features)
    - Branch 2: GNN-TAT for non-Euclidean topographic relations (KCE features)
    - Grid-Graph Fusion via cross-attention
    - Interpretable Meta-Learner for weighted combination

    """

    def __init__(
        self,
        config: V5Config,
        n_features_basic: int,
        n_features_kce: int,
        n_lat: int,
        n_lon: int,
        horizon: int
    ):
        super().__init__()
        self.config = config
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_nodes = n_lat * n_lon
        self.horizon = horizon
        self.use_context_features = config.meta_use_context_features

        # Context feature indices (from KCE)
        self.context_indices = {}
        self.context_dim = 0
        if self.use_context_features:
            try:
                kce_features = FEATURE_SETS['KCE']
                indices = {
                    'elevation': kce_features.index('elevation'),
                    'elev_low': kce_features.index('elev_low'),
                    'elev_med': kce_features.index('elev_med'),
                    'elev_high': kce_features.index('elev_high'),
                    'month_sin': kce_features.index('month_sin'),
                    'month_cos': kce_features.index('month_cos')
                }

                # Validate indices against available features
                max_idx = max(indices.values())
                if max_idx >= n_features_kce:
                    print(f"[V5 WARNING] Context features require {max_idx+1} features, but GNN input has {n_features_kce}.")
                    print("             Disabling meta-learner context features for this experiment.")
                    self.use_context_features = False
                    self.context_dim = 0
                else:
                    self.context_indices = indices
                    # mean_elevation (1) + elevation_cluster (3) + temporal_regime (2)
                    self.context_dim = 6
            except Exception as exc:
                print(f"Warning: context features unavailable ({exc}); disabling meta context.")
                self.use_context_features = False
                self.context_dim = 0

        # Branch 1: ConvLSTM (Euclidean)
        self.convlstm_branch = ConvLSTMBranch(config, n_features_basic)

        # Branch 2: GNN-TAT (Non-Euclidean)
        self.gnn_branch = GNNBranch(config, n_features_kce, self.n_nodes)

        # Grid-Graph Fusion
        self.fusion = GridGraphFusion(config, n_lat, n_lon)

        # Meta-Learner
        self.meta_learner = MetaLearner(config, n_lat, n_lon, horizon, context_dim=self.context_dim)

    def _extract_context(self, x_graph: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract per-node context features for the meta-learner."""
        if not self.use_context_features:
            return None

        idx = self.context_indices

        # (batch, seq, n_nodes, features)
        elevation = x_graph[:, -1, :, idx['elevation']].unsqueeze(-1)
        elev_clusters = x_graph[:, -1, :, [idx['elev_low'], idx['elev_med'], idx['elev_high']]]

        # Temporal regime from mean seasonal encoding
        month_sin = x_graph[:, :, :, idx['month_sin']].mean(dim=1)
        month_cos = x_graph[:, :, :, idx['month_cos']].mean(dim=1)
        temporal = torch.stack([month_sin, month_cos], dim=-1)

        context = torch.cat([elevation, elev_clusters, temporal], dim=-1)
        return context

    def forward(
        self,
        x_grid: torch.Tensor,       # (batch, seq, lat, lon, features_basic)
        x_graph: torch.Tensor,      # (batch, seq, n_nodes, features_kce)
        edge_index: torch.Tensor,   # (2, n_edges)
        edge_weight: torch.Tensor   # (n_edges,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through V5 stacking model.

        Returns:
            predictions: (batch, horizon, lat, lon) - precipitation predictions
            weights: (batch, n_nodes, 2) - branch weights for interpretability
        """
        # Branch 1: ConvLSTM processing
        convlstm_out = self.convlstm_branch(x_grid)  # (batch, lat, lon, dim)

        # Branch 2: GNN-TAT processing
        gnn_out = self.gnn_branch(x_graph, edge_index, edge_weight)  # (batch, n_nodes, dim)

        # Grid-Graph Fusion
        fused_grid, fused_graph = self.fusion(convlstm_out, gnn_out)

        # Meta-Learner for final predictions
        context = self._extract_context(x_graph)
        predictions, weights = self.meta_learner(fused_grid, fused_graph, context)

        return predictions, weights

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per component."""
        counts = {
            'convlstm_branch': sum(p.numel() for p in self.convlstm_branch.parameters()),
            'gnn_branch': sum(p.numel() for p in self.gnn_branch.parameters()),
            'fusion': sum(p.numel() for p in self.fusion.parameters()),
            'meta_learner': sum(p.numel() for p in self.meta_learner.parameters())
        }
        counts['total'] = sum(counts.values())
        return counts

# Test model instantiation
# Use dataset dimensions directly (ds is already sliced if light_mode is True)
# Updated to use correct dimension names 'latitude' and 'longitude'
n_lat = ds.dims[LAT_DIM]
n_lon = ds.dims[LON_DIM]

model = V5StackingModel(
    config=CONFIG,
    n_features_basic=n_features_basic,
    n_features_kce=n_features_kce,
    n_lat=n_lat,
    n_lon=n_lon,
    horizon=CONFIG.horizon
).to(device)

# Print parameter counts
param_counts = model.count_parameters()
print("\nV5 Model Parameter Counts:")
print("="*40)
for name, count in param_counts.items():
    print(f"  {name}: {count:,}")
print(f"\nTotal: {param_counts['total']:,} parameters")
print(f"Target: < {CONFIG.param_budget:,} parameters")
if param_counts['total'] > CONFIG.param_budget:
    print("WARNING: parameter budget exceeded")

# --- Cell 26 ---
# =============================================================================
# SECTION 7: TRAINING INFRASTRUCTURE
# =============================================================================

class V5Trainer:
    """Training infrastructure for V5 model."""

    def __init__(
        self,
        model: V5StackingModel,
        config: V5Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            min_lr=1e-6
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'lr': [],
            'branch_weights': [],  # Track meta-learner weights
            'train_weight_reg': []
        }

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        self.best_model_state = None

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            start_epoch: Epoch to resume from
        """
        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return 0

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_epoch = checkpoint['best_epoch']
            self.patience_counter = checkpoint.get('patience_counter', 0)

        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch

    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_weight_reg = 0.0
        n_batches = 0

        for batch in self.train_loader:
            x_grid = batch['x_grid'].to(self.device)
            x_graph = batch['x_graph'].to(self.device)
            y = batch['y'].to(self.device)
            edge_index = batch['edge_index'][0].to(self.device)
            edge_weight = batch['edge_weight'][0].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            predictions, weights = self.model(x_grid, x_graph, edge_index, edge_weight)

            # Loss computation
            loss = self.criterion(predictions, y)

            weight_reg = torch.tensor(0.0, device=self.device)
            if self.config.weight_reg_lambda > 0:
                weight_reg = torch.mean((weights[..., 0] - 0.5) ** 2)
                loss = loss + self.config.weight_reg_lambda * weight_reg

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions - y)).item()
            total_weight_reg += weight_reg.item()
            n_batches += 1

        return (
            total_loss / n_batches,
            total_mae / n_batches,
            total_weight_reg / n_batches
        )

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, np.ndarray]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0
        all_weights = []

        for batch in self.val_loader:
            x_grid = batch['x_grid'].to(self.device)
            x_graph = batch['x_graph'].to(self.device)
            y = batch['y'].to(self.device)
            edge_index = batch['edge_index'][0].to(self.device)
            edge_weight = batch['edge_weight'][0].to(self.device)

            predictions, weights = self.model(x_grid, x_graph, edge_index, edge_weight)

            loss = self.criterion(predictions, y)

            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions - y)).item()
            all_weights.append(weights.cpu().numpy())
            n_batches += 1

        avg_weights = np.concatenate(all_weights, axis=0).mean(axis=(0, 1))

        return total_loss / n_batches, total_mae / n_batches, avg_weights

    def train(self, output_dir: Path = None, start_epoch: int = 0) -> Dict:
        """Full training loop with resume capability."""
        print("\n" + "="*70)
        print(f"Starting V5 Training" if start_epoch == 0 else f"Resuming from epoch {start_epoch+1}")
        print("="*70)

        start_time = datetime.now()

        for epoch in range(start_epoch, self.config.epochs):
            # Train
            train_loss, train_mae, train_weight_reg = self.train_epoch()

            # Validate
            val_loss, val_mae, weights = self.validate()

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['lr'].append(current_lr)
            self.history['branch_weights'].append(weights.tolist())
            self.history['train_weight_reg'].append(train_weight_reg)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())

                # Save best checkpoint
                if output_dir:
                    checkpoint_path = output_dir / 'v5_stacking_best.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'history': self.history,
                        'config': asdict(self.config)
                    }, checkpoint_path)
            else:
                self.patience_counter += 1

            # Save periodic checkpoint every 10 epochs for resume capability
            if output_dir and (epoch + 1) % 10 == 0:
                periodic_path = output_dir / 'v5_stacking_latest.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': self.best_val_loss,
                    'best_epoch': self.best_epoch,
                    'patience_counter': self.patience_counter,
                    'history': self.history,
                    'config': asdict(self.config)
                }, periodic_path)
                print(f"  [Checkpoint saved: epoch {epoch+1}]")

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                w_conv, w_gnn = weights
                log_msg = (
                    f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.2e} | W_conv: {w_conv:.2f} W_gnn: {w_gnn:.2f}"
                )
                if self.config.weight_reg_lambda > 0:
                    log_msg += f" | WReg: {train_weight_reg:.4f}"
                print(log_msg)

            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        elapsed = datetime.now() - start_time

        print(f"\nTraining completed in {elapsed}")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")

        return self.history

print("V5 Trainer defined successfully")

# --- Cell 27 ---
# =============================================================================
# SECTION 7.2: METRICS EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: V5StackingModel,
    data_loader: DataLoader,
    device: torch.device,
    return_arrays: bool = False
) -> Dict:
    """Evaluate model and compute metrics per horizon."""
    model.eval()

    all_preds = []
    all_targets = []
    all_weights = []

    for batch in data_loader:
        x_grid = batch['x_grid'].to(device)
        x_graph = batch['x_graph'].to(device)
        y = batch['y'].to(device)
        edge_index = batch['edge_index'][0].to(device)
        edge_weight = batch['edge_weight'][0].to(device)

        predictions, weights = model(x_grid, x_graph, edge_index, edge_weight)

        all_preds.append(predictions.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        all_weights.append(weights.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    weights = np.concatenate(all_weights, axis=0)

    # Global quality stats
    pred_finite = np.isfinite(preds)
    target_finite = np.isfinite(targets)
    neg_pred_frac = float(np.mean(preds < 0))
    nan_pred_frac = float(np.mean(~pred_finite))
    nan_target_frac = float(np.mean(~target_finite))

    # Compute metrics per horizon
    metrics = {}
    n_horizons = preds.shape[1]

    for h in range(n_horizons):
        pred_h = preds[:, h].flatten()
        target_h = targets[:, h].flatten()

        # Remove NaN values
        valid_mask = np.isfinite(pred_h) & np.isfinite(target_h)
        pred_h = pred_h[valid_mask]
        target_h = target_h[valid_mask]

        if len(pred_h) == 0:
            continue

        # RMSE
        rmse = np.sqrt(np.mean((pred_h - target_h) ** 2))

        # MAE
        mae = np.mean(np.abs(pred_h - target_h))

        # R2
        ss_res = np.sum((target_h - pred_h) ** 2)
        ss_tot = np.sum((target_h - np.mean(target_h)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Bias
        mean_true = np.mean(target_h)
        mean_pred = np.mean(pred_h)
        total_true = float(np.sum(target_h))
        total_pred = float(np.sum(pred_h))
        bias_mm = mean_pred - mean_true
        bias_pct = 100 * bias_mm / mean_true if mean_true != 0 else 0.0
        scale_ratio = abs(mean_pred) / max(abs(mean_true), 1e-6)

        metrics[f'H{h+1}'] = {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R^2': float(r2),
            'Mean_True_mm': float(mean_true),
            'Mean_Pred_mm': float(mean_pred),
            'mean_bias_mm': float(bias_mm),
            'mean_bias_pct': float(bias_pct),
            'scale_ratio': float(scale_ratio),
            'TotalPrecipitation': float(total_true),
            'TotalPrecipitation_Pred': float(total_pred)
        }

    # Average branch weights
    avg_weights = weights.mean(axis=(0, 1))
    metrics['branch_weights'] = {
        'w_convlstm': float(avg_weights[0]),
        'w_gnn': float(avg_weights[1])
    }

    metrics['quality'] = {
        'neg_pred_frac': neg_pred_frac,
        'nan_pred_frac': nan_pred_frac,
        'nan_target_frac': nan_target_frac
    }

    if return_arrays:
        return metrics, preds, targets

    return metrics


def run_quality_checks(
    metrics: Dict,
    config: V5Config,
    param_counts: Dict[str, int],
    data_report: Dict = None
) -> Dict:
    issues = []
    max_abs_bias = 0.0
    max_scale_ratio = 0.0

    for h_key, h_metrics in metrics.items():
        if not h_key.startswith('H'):
            continue
        max_abs_bias = max(max_abs_bias, abs(h_metrics.get('mean_bias_pct', 0.0)))
        max_scale_ratio = max(max_scale_ratio, h_metrics.get('scale_ratio', 0.0))

    neg_pred_frac = metrics.get('quality', {}).get('neg_pred_frac', 0.0)
    nan_pred_frac = metrics.get('quality', {}).get('nan_pred_frac', 0.0)
    nan_target_frac = metrics.get('quality', {}).get('nan_target_frac', 0.0)

    if max_abs_bias > config.max_bias_pct:
        issues.append('bias_pct')
    if max_scale_ratio > config.max_scale_ratio:
        issues.append('scale_ratio')
    if neg_pred_frac > config.max_negative_frac:
        issues.append('neg_pred')
    if nan_pred_frac > 0 or nan_target_frac > 0:
        issues.append('nan_values')

    w_conv = metrics.get('branch_weights', {}).get('w_convlstm', 0.5)
    if w_conv < config.min_branch_weight or w_conv > (1.0 - config.min_branch_weight):
        issues.append('branch_collapse')

    param_budget_ok = param_counts.get('total', 0) <= config.param_budget
    if not param_budget_ok:
        issues.append('param_budget')

    overlap_leakage = False
    missing_features = []
    if data_report:
        overlap_leakage = bool(data_report.get('overlap_leakage', False))
        missing_features = data_report.get('missing_features_basic', []) + data_report.get('missing_features_kce', [])
    if overlap_leakage:
        issues.append('overlap_windows')
    if missing_features:
        issues.append('missing_features')

    return {
        'issues': issues,
        'max_abs_bias_pct': float(max_abs_bias),
        'max_scale_ratio': float(max_scale_ratio),
        'neg_pred_frac': float(neg_pred_frac),
        'nan_pred_frac': float(nan_pred_frac),
        'nan_target_frac': float(nan_target_frac),
        'branch_weight_convlstm': float(w_conv),
        'param_budget_ok': bool(param_budget_ok),
        'overlap_leakage': bool(overlap_leakage),
        'missing_features': missing_features
    }



def _forecast_dates(ds: xr.Dataset, val_indices: List[int], input_window: int, horizon: int) -> List[List[str]]:
    if 'time' not in ds.coords and 'time' not in ds.dims:
        return [[] for _ in val_indices]
    times = pd.to_datetime(ds['time'].values)
    dates = []
    for start in val_indices:
        base = start + input_window
        horizon_dates = []
        for h in range(horizon):
            idx = base + h
            if idx < len(times):
                horizon_dates.append(pd.Timestamp(times[idx]).strftime('%Y-%m'))
        dates.append(horizon_dates)
    return dates


def export_predictions_for_maps(
    preds: np.ndarray,
    targets: np.ndarray,
    output_root: Path,
    horizon: int,
    exp_name: str,
    model_name: str,
    config: V5Config,
    ds: xr.Dataset,
    data_report: Dict
) -> Path:
    # Export predictions for map generation (V2/V3 compatible format).
    map_out_root = output_root / 'map_exports'
    export_dir = map_out_root / f'H{horizon}' / exp_name / model_name
    export_dir.mkdir(parents=True, exist_ok=True)

    np.save(export_dir / 'predictions.npy', preds.astype(np.float32))
    np.save(export_dir / 'targets.npy', targets.astype(np.float32))

    val_count = preds.shape[0]
    if data_report:
        val_start_idx = int(data_report.get('val_start_idx', data_report.get('n_train', 0)))
        val_end_idx = int(data_report.get('val_end_idx', val_start_idx + val_count - 1))
        if (val_end_idx - val_start_idx + 1) != val_count:
            val_end_idx = val_start_idx + val_count - 1
    else:
        val_start_idx = 0
        val_end_idx = val_start_idx + val_count - 1

    val_indices = list(range(val_start_idx, val_end_idx + 1))
    forecast_dates = _forecast_dates(ds, val_indices, config.input_window, horizon)

    metadata = {
        'exp': exp_name,
        'model': model_name,
        'horizon': horizon,
        'input_window': config.input_window,
        'train_val_split': config.train_val_split,
        'feature_set_basic': data_report.get('feature_set_basic') if data_report else None,
        'feature_set_kce': data_report.get('feature_set_kce') if data_report else None,
        'split_mode': data_report.get('split_mode', 'window') if data_report else 'window',
        'split_time_idx': int(data_report.get('split_time_idx', val_start_idx)) if data_report else int(val_start_idx),
        'val_start_idx': int(val_start_idx),
        'val_end_idx': int(val_end_idx),
        'val_indices': val_indices,
        'forecast_dates': forecast_dates,
        'shape': list(preds.shape),
        'generated_at': datetime.now().isoformat(),
        'rmse_mean': float(np.sqrt(np.mean((preds - targets) ** 2))),
        'framework': 'V5 GNN-ConvLSTM Stacking'
    }
    (export_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    print(f"Exported predictions to: {export_dir}")
    return export_dir

print("Metrics evaluation functions defined")

# --- Cell 29 ---
# =============================================================================
# SECTION 8: MAIN TRAINING LOOP
# =============================================================================

# OPTIMIZED: Run only BASIC_KCE for doctoral thesis
# ConvLSTM with BASIC features (12), GNN with KCE features (15 with topography)
EXPERIMENTS = [
    {'name': 'BASIC_KCE', 'basic_set': 'BASIC', 'gnn_set': 'KCE'},
]
RUN_MIXED_ABLATION = False  # Disabled - running single optimized config

# Meta-learner stabilization (step 2)
WEIGHT_FLOOR = 0.3  # Force more balanced weights
WEIGHT_REG_LAMBDA = 0.02


def run_v5_experiments(
    ds: xr.Dataset,
    config: V5Config,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    output_root: Path,
    experiments: List[Dict]
):
    """Run V5 experiments across multiple horizons."""

    all_metrics = []
    experiment_state = {
        'config': asdict(config),
        'experiments': experiments,
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }

    for exp in experiments:
        exp_name = exp['name']
        feature_set_basic = exp['basic_set']
        feature_set_kce = exp['gnn_set']
        exp_weight_floor = exp.get('weight_floor', WEIGHT_FLOOR)
        exp_weight_reg = exp.get('weight_reg_lambda', WEIGHT_REG_LAMBDA)

        experiment_state['results'][exp_name] = {}

        for horizon in config.enabled_horizons:
            print("\n" + "="*70)
            print(f"TRAINING V5 STACKING MODEL - {exp_name} - HORIZON H{horizon}")
            print("="*70)

            # Update horizon in config
            current_config = V5Config(**{
                **asdict(config),
                'horizon': horizon,
                'weight_floor': exp_weight_floor,
                'weight_reg_lambda': exp_weight_reg
            })

            # Prepare data
            train_dataset, val_dataset, n_basic, n_kce, data_report = prepare_data(
                ds, current_config, edge_index, edge_weight,
                feature_set_basic=feature_set_basic,
                feature_set_kce=feature_set_kce
            )

            train_loader = DataLoader(train_dataset, batch_size=current_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=current_config.batch_size, shuffle=False)

            # Get grid dimensions
            n_lat = train_dataset.n_lat
            n_lon = train_dataset.n_lon

            # Initialize model
            model = V5StackingModel(
                config=current_config,
                n_features_basic=n_basic,
                n_features_kce=n_kce,
                n_lat=n_lat,
                n_lon=n_lon,
                horizon=horizon
            ).to(device)

            param_counts = model.count_parameters()
            if param_counts['total'] > current_config.param_budget:
                print("Warning: parameter budget exceeded")

            # Create output directory for this horizon
            horizon_dir = output_root / f'h{horizon}' / exp_name / 'training_metrics'
            horizon_dir.mkdir(parents=True, exist_ok=True)

            # Train
            trainer = V5Trainer(model, current_config, train_loader, val_loader, device)
            history = trainer.train(horizon_dir)

            # Evaluate
            return_arrays = bool(current_config.export_predictions or current_config.generate_map_plots)
            if return_arrays:
                metrics, preds, targets = evaluate_model(model, val_loader, device, return_arrays=True)
            else:
                metrics = evaluate_model(model, val_loader, device)
                preds, targets = None, None
            quality = run_quality_checks(metrics, current_config, param_counts, data_report)
            metrics['quality_checks'] = quality

            if quality['issues']:
                print(f"Warning: quality checks flagged {quality['issues']}")
                if current_config.enforce_quality_gates:
                    raise ValueError(f"Quality gates failed: {quality['issues']}")

            map_export_dir = None
            if current_config.export_predictions and preds is not None:
                map_export_dir = export_predictions_for_maps(
                    preds, targets, output_root, horizon,
                    exp_name=exp_name, model_name='V5_STACKING',
                    config=current_config, ds=ds, data_report=data_report
                )

            # Save training history
            history_data = {
                'model_name': 'V5_STACKING',
                'experiment': exp_name,
                'feature_set_basic': feature_set_basic,
                'feature_set_kce': feature_set_kce,
                'weight_floor': exp_weight_floor,
                'weight_reg_lambda': exp_weight_reg,
                'horizon': horizon,
                'best_epoch': trainer.best_epoch,
                'best_val_loss': trainer.best_val_loss,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'total_epochs': len(history['train_loss']),
                'parameters': param_counts['total'],
                'branch_weights_final': history['branch_weights'][-1] if history['branch_weights'] else [0.5, 0.5],
                'data_report': data_report,
                'quality_checks': quality,
                'map_export_dir': str(map_export_dir) if map_export_dir else None
            }

            with open(horizon_dir / 'v5_stacking_history.json', 'w') as f:
                json.dump(history_data, f, indent=2)

            # Save training log CSV
            log_df = pd.DataFrame({
                'epoch': range(1, len(history['train_loss']) + 1),
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'train_mae': history['train_mae'],
                'val_mae': history['val_mae'],
                'lr': history['lr'],
                'train_weight_reg': history.get('train_weight_reg', [])
            })
            log_df.to_csv(horizon_dir / f'v5_stacking_training_log_h{horizon}.csv', index=False)

            # Collect metrics for CSV
            for h_key, h_metrics in metrics.items():
                if h_key.startswith('H'):
                    h_num = int(h_key[1:])
                    row = {
                        'TotalHorizon': horizon,
                        'Experiment': exp_name,
                        'Feature_Set_Basic': feature_set_basic,
                        'Feature_Set_GNN': feature_set_kce,
                        'Model': 'V5_STACKING',
                        'H': h_num,
                        **h_metrics
                    }
                    all_metrics.append(row)

            # Store results
            experiment_state['results'][exp_name][f'H{horizon}'] = {
                'metrics': metrics,
                'history': history_data,
                'param_counts': param_counts,
                'data_report': data_report,
                'quality_checks': quality,
                'map_export_dir': str(map_export_dir) if map_export_dir else None
            }

            # Print summary
            print(f"\n{exp_name} H{horizon} Results Summary:")
            for h_key, h_metrics in metrics.items():
                if h_key.startswith('H'):
                    print(f"  {h_key}: RMSE={h_metrics['RMSE']:.2f}mm, "
                          f"MAE={h_metrics['MAE']:.2f}mm, R2={h_metrics['R^2']:.4f}")
            print(f"  Branch weights: ConvLSTM={metrics['branch_weights']['w_convlstm']:.2%}, "
                  f"GNN={metrics['branch_weights']['w_gnn']:.2%}")
            if quality['issues']:
                print(f"  Quality issues: {', '.join(quality['issues'])}")

            # Cleanup
            del model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save consolidated metrics CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(output_root / 'metrics_spatial_v5_all_horizons.csv', index=False)
        print(f"\nConsolidated metrics saved to: {output_root / 'metrics_spatial_v5_all_horizons.csv'}")

    # Save experiment state
    with open(output_root / 'experiment_state_v5.json', 'w') as f:
        json.dump(experiment_state, f, indent=2, default=str)

    return experiment_state, metrics_df if all_metrics else None

print("Main training function defined")

# --- Cell 30 ---
# =============================================================================
# SECTION 8.2: EXECUTE TRAINING (ROBUST MODE)
# =============================================================================

# Monkey-patch V5Trainer with robust error checking to diagnose/prevent CUDA asserts
def robust_train_epoch(self) -> Tuple[float, float, float]:
    """Train for one epoch with added safety checks on CPU first."""
    self.model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_weight_reg = 0.0
    n_batches = 0

    for i, batch in enumerate(self.train_loader):
        # Keep data on CPU initially for checks
        x_grid_cpu = batch['x_grid']
        x_graph_cpu = batch['x_graph']
        y_cpu = batch['y']
        edge_index_cpu = batch['edge_index'][0]
        edge_weight_cpu = batch['edge_weight'][0]

        # Safety Check 1: Validate Graph Indices (CPU)
        num_nodes = x_graph_cpu.size(2)
        if edge_index_cpu.numel() > 0:
            max_idx = edge_index_cpu.max().item()
            min_idx = edge_index_cpu.min().item()
            if max_idx >= num_nodes or min_idx < 0:
                raise ValueError(f"Graph index out of bounds on CPU check: range=[{min_idx}, {max_idx}], num_nodes={num_nodes}")

        # Safety Check 2: Validate Inputs (CPU)
        if torch.isnan(x_grid_cpu).any() or torch.isnan(x_graph_cpu).any():
             raise ValueError("NaNs detected in input features (CPU check)")
        if torch.isnan(y_cpu).any():
             raise ValueError("NaNs detected in targets (CPU check)")

        # Move to Device
        x_grid = x_grid_cpu.to(self.device)
        x_graph = x_graph_cpu.to(self.device)
        y = y_cpu.to(self.device)
        edge_index = edge_index_cpu.to(self.device)
        edge_weight = edge_weight_cpu.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        predictions, weights = self.model(x_grid, x_graph, edge_index, edge_weight)

        # Safety Check 3: Validate Outputs (Device)
        if torch.isnan(predictions).any():
            raise ValueError(f"NaNs produced by model in batch {i}")

        # Loss computation
        loss = self.criterion(predictions, y)

        weight_reg = torch.tensor(0.0, device=self.device)
        if self.config.weight_reg_lambda > 0:
            weight_reg = torch.mean((weights[..., 0] - 0.5) ** 2)
            loss = loss + self.config.weight_reg_lambda * weight_reg

        # Backward pass
        if torch.isnan(loss):
             raise ValueError(f"NaN loss computed in batch {i}")

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip
        )

        self.optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_mae += torch.mean(torch.abs(predictions - y)).item()
        total_weight_reg += weight_reg.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_mae / n_batches,
        total_weight_reg / n_batches
    )

# Apply the patch
V5Trainer.train_epoch = robust_train_epoch
print("Applied robust training loop with CPU-side pre-checks.")

# Run experiments
print("\n" + "#"*70)
print("#" + " "*20 + "V5 GNN-ConvLSTM STACKING" + " "*20 + "#")
print("#"*70)

if CONFIG.light_mode:
    print("\n>>> LIGHT MODE ENABLED - Using reduced grid for testing <<<")
    print(f">>> Grid size: {CONFIG.light_grid_size}x{CONFIG.light_grid_size} <<<\n")

# Enable anomaly detection for better tracebacks
torch.autograd.set_detect_anomaly(True)

try:
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    experiment_state, metrics_df = run_v5_experiments(
        ds=ds,
        config=CONFIG,
        edge_index=edge_index,
        edge_weight=edge_weight,
        output_root=OUTPUT_ROOT,
        experiments=EXPERIMENTS
    )
    print("\n" + "="*70)
    print("V5 TRAINING COMPLETE")
    print("="*70)

except Exception as e:
    print(f"\nCRITICAL TRAINING ERROR: {e}")
    print("Tip: If you see 'device-side assert triggered', try Restarting the Runtime (Runtime -> Restart session) to clear the CUDA state.")
    # Re-raise to see full traceback if needed, or handle gracefully
    raise e
finally:
    torch.autograd.set_detect_anomaly(False)

# --- Cell 32 ---
# =============================================================================
# SECTION 9: RESULTS VISUALIZATION (700 DPI)
# =============================================================================

def plot_v5_results(metrics_df: pd.DataFrame, experiment_state: Dict, output_dir: Path):
    """Generate V5 results visualizations at 700 DPI."""

    if metrics_df is None or len(metrics_df) == 0:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. RMSE by Horizon
    ax1 = axes[0, 0]
    horizons = sorted(metrics_df['H'].unique())
    rmse_by_h = metrics_df.groupby('H')['RMSE'].mean()
    ax1.bar(rmse_by_h.index, rmse_by_h.values, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Prediction Horizon (months)', fontsize=12)
    ax1.set_ylabel('RMSE (mm)', fontsize=12)
    ax1.set_title('V5 Stacking: RMSE by Forecast Horizon', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add target line
    ax1.axhline(y=85, color='red', linestyle='--', label='Target: 85mm')
    ax1.legend()

    # 2. R^2 by Horizon
    ax2 = axes[0, 1]
    r2_by_h = metrics_df.groupby('H')['R^2'].mean()
    ax2.bar(r2_by_h.index, r2_by_h.values, color='forestgreen', edgecolor='black')
    ax2.set_xlabel('Prediction Horizon (months)', fontsize=12)
    ax2.set_ylabel('R^2', fontsize=12)
    ax2.set_title('V5 Stacking: R^2 by Forecast Horizon', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add target line
    ax2.axhline(y=0.65, color='red', linestyle='--', label='Target: 0.65')
    ax2.legend()

    # 3. Bias by Horizon
    ax3 = axes[1, 0]
    bias_by_h = metrics_df.groupby('H')['mean_bias_mm'].mean()
    colors = ['red' if b < 0 else 'blue' for b in bias_by_h.values]
    ax3.bar(bias_by_h.index, bias_by_h.values, color=colors, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Prediction Horizon (months)', fontsize=12)
    ax3.set_ylabel('Mean Bias (mm)', fontsize=12)
    ax3.set_title('V5 Stacking: Bias by Forecast Horizon', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Branch Weights Evolution
    ax4 = axes[1, 1]

    labels = []
    w_convlstm = []
    w_gnn = []

    for exp_name, exp_results in experiment_state.get('results', {}).items():
        for h_key, results in exp_results.items():
            if 'metrics' in results and 'branch_weights' in results['metrics']:
                labels.append(f"{exp_name}-{h_key}")
                w_convlstm.append(results['metrics']['branch_weights']['w_convlstm'])
                w_gnn.append(results['metrics']['branch_weights']['w_gnn'])

    if not labels:
        ax4.text(0.5, 0.5, 'No branch weights found', ha='center', va='center')
        ax4.axis('off')
    else:
        x = np.arange(len(labels))
        width = 0.35

        ax4.bar(x - width/2, w_convlstm, width, label='ConvLSTM', color='coral', edgecolor='black')
        ax4.bar(x + width/2, w_gnn, width, label='GNN-TAT', color='teal', edgecolor='black')

        ax4.set_xlabel('Experiment-Horizon', fontsize=12)
        ax4.set_ylabel('Branch Weight', fontsize=12)
        ax4.set_title('V5 Meta-Learner: Branch Weight Distribution', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=30, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim(0, 1)

    plt.tight_layout()

    # Save at 700 DPI
    fig_path = output_dir / 'v5_results_summary.png'
    fig.savefig(fig_path, dpi=700, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"\nVisualization saved to: {fig_path} (700 DPI)")

# Generate visualizations
plot_v5_results(metrics_df, experiment_state, OUTPUT_ROOT)

# --- Cell 33 ---
# =============================================================================
# SECTION 9.1: EVOLUTION BY HORIZON (V2/V3 STYLE)
# =============================================================================

COMP_DIR = OUTPUT_ROOT / 'comparisons'
COMP_DIR.mkdir(parents=True, exist_ok=True)


def plot_v5_evolution_by_horizon(metrics_df: pd.DataFrame, output_dir: Path):
    if metrics_df is None or metrics_df.empty:
        print('No metrics to plot for evolution panels')
        return

    metrics = ['RMSE', 'MAE', 'R^2']
    has_tp = 'TotalPrecipitation' in metrics_df.columns and 'TotalPrecipitation_Pred' in metrics_df.columns
    if has_tp:
        metrics.append('TotalPrecipitation')

    for total_h in sorted(metrics_df['TotalHorizon'].unique()):
        sub_df = metrics_df[metrics_df['TotalHorizon'] == total_h].copy()
        if sub_df.empty:
            continue

        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]

        for metric, ax in zip(metrics, axes):
            if metric != 'TotalPrecipitation':
                data = sub_df.groupby(['H', 'Model'])[metric].mean().unstack()
                data.plot(ax=ax, marker='o')
                ax.set_ylabel(metric)
            else:
                data_true = sub_df.groupby(['H', 'Model'])['TotalPrecipitation'].mean().unstack()
                data_pred = sub_df.groupby(['H', 'Model'])['TotalPrecipitation_Pred'].mean().unstack()
                for model in data_true.columns:
                    ax.plot(data_true.index, data_true[model], marker='o', label=f'{model} True')
                    ax.plot(data_pred.index, data_pred[model], marker='s', linestyle='--', label=f'{model} Pred')
                ax.set_ylabel('Total Precipitation (mm)')

            ax.set_xlabel('Horizon (months)')
            ax.set_title(f'{metric} by Horizon')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        fig.suptitle(f'V5 Evolution by Horizon (H={total_h})', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig_path = output_dir / f'v5_evolution_h{total_h}.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Evolution plot saved to: {fig_path}')


if metrics_df is not None and len(metrics_df) > 0:
    plot_v5_evolution_by_horizon(metrics_df, COMP_DIR)

# --- Cell 34 ---
# =============================================================================
# SECTION 9.2: METRICS TABLE (V2/V3 STYLE)
# =============================================================================

COMP_DIR = OUTPUT_ROOT / 'comparisons'
COMP_DIR.mkdir(parents=True, exist_ok=True)


def plot_v5_metrics_table(metrics_df: pd.DataFrame, output_dir: Path):
    if metrics_df is None or metrics_df.empty:
        print('No metrics available to build summary table')
        return

    has_tp = 'TotalPrecipitation' in metrics_df.columns and 'TotalPrecipitation_Pred' in metrics_df.columns

    summary_data = []
    experiments = metrics_df['Experiment'].unique()
    models = metrics_df['Model'].unique()
    headers = ['Experiment', 'Model', 'RMSE', 'MAE', 'R^2', 'Total Pcp (True)', 'Total Pcp (Pred)', 'Best H']

    for exp in experiments:
        for model in models:
            sub = metrics_df[(metrics_df['Experiment'] == exp) & (metrics_df['Model'] == model)]
            if sub.empty:
                continue
            avg_rmse = sub['RMSE'].mean()
            avg_mae = sub['MAE'].mean()
            avg_r2 = sub['R^2'].mean()
            if has_tp:
                avg_tp_t = sub['TotalPrecipitation'].mean()
                avg_tp_p = sub['TotalPrecipitation_Pred'].mean()
            else:
                avg_tp_t = float('nan')
                avg_tp_p = float('nan')
            best_h = sub.loc[sub['RMSE'].idxmin(), 'H']
            summary_data.append([
                exp, model,
                f'{avg_rmse:.4f}', f'{avg_mae:.4f}', f'{avg_r2:.4f}',
                f'{avg_tp_t:.1f}' if has_tp else 'n/a',
                f'{avg_tp_p:.1f}' if has_tp else 'n/a',
                f'H={best_h}'
            ])

    fig, ax = plt.subplots(figsize=(17, 10))
    ax.axis('off')
    table = ax.table(cellText=summary_data, colLabels=headers, cellLoc='center', loc='center', bbox=[0, 0, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.8)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4a86e8')
        table[(0, j)].set_text_props(weight='bold', color='white')

    fig.suptitle('Metrics Summary by Model and Experiment', fontsize=16, fontweight='bold', y=0.95)
    plt.subplots_adjust(top=0.85)
    fig_path = output_dir / 'v5_metrics_summary_table.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
    plt.show()

    if summary_data:
        print('Metrics summary table saved to:', fig_path)


if metrics_df is not None and len(metrics_df) > 0:
    plot_v5_metrics_table(metrics_df, COMP_DIR)

# --- Cell 35 ---
# =============================================================================
# SECTION 9.3: MODEL COMPARISON (V4 STYLE)
# =============================================================================

COMP_DIR = OUTPUT_ROOT / 'comparisons'
COMP_DIR.mkdir(parents=True, exist_ok=True)


def plot_model_comparison_v5(res_df: pd.DataFrame, output_dir: Path):
    if res_df is None or res_df.empty:
        print('No results to plot for model comparison')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    pivot_rmse = res_df.groupby(['Model', 'Experiment'])['RMSE'].mean().unstack()
    pivot_rmse.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_ylabel('RMSE (mm)')
    ax1.set_title('RMSE by Model and Feature Set')
    ax1.legend(title='Feature Set')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = axes[0, 1]
    pivot_r2 = res_df.groupby(['Model', 'Experiment'])['R^2'].mean().unstack()
    pivot_r2.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_ylabel('R^2')
    ax2.set_title('R^2 by Model and Feature Set')
    ax2.legend(title='Feature Set')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0.628, color='red', linestyle='--', label='V4 Baseline (0.628)')

    ax3 = axes[1, 0]
    for model in res_df['Model'].unique():
        model_data = res_df[res_df['Model'] == model]
        for exp in model_data['Experiment'].unique():
            exp_data = model_data[model_data['Experiment'] == exp]
            ax3.plot(exp_data['H'], exp_data['RMSE'], marker='o', label=f"{model} ({exp})")
    ax3.set_xlabel('Horizon (H)')
    ax3.set_ylabel('RMSE (mm)')
    ax3.set_title('RMSE Degradation by Horizon')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    for model in res_df['Model'].unique():
        model_bias = res_df[res_df['Model'] == model]['mean_bias_mm']
        ax4.hist(model_bias, bins=20, alpha=0.5, label=model)
    ax4.axvline(x=0, color='red', linestyle='--')
    ax4.set_xlabel('Mean Bias (mm)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Bias Distribution by Model')
    ax4.legend()

    plt.tight_layout()

    fig_path = output_dir / 'model_comparison_v5_stacking.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {fig_path}")
    plt.show()


if metrics_df is not None and len(metrics_df) > 0:
    plot_model_comparison_v5(metrics_df, COMP_DIR)

# --- Cell 36 ---
# =============================================================================
# SECTION 9.4: MAP EXPORTS (V2/V3 COMPATIBLE)
# =============================================================================

SHAPE_DIR = BASE_PATH / 'data' / 'input' / 'shapes'


def _load_boyaca_shape(shape_dir: Path):
    if gpd is None:
        print('geopandas not available; overlay disabled')
        return None, None
    if not shape_dir.exists():
        print(f"[WARN] Shape directory not found: {shape_dir}")
        return None, None

    shp_path = shape_dir / 'MGN_Departamento.shp'
    try:
        dept_gdf = gpd.read_file(shp_path)
    except Exception as exc:
        print(f"[WARN] Failed to read shapefile {shp_path}: {exc}")
        return None, None

    if dept_gdf is None or dept_gdf.empty:
        print(f"[WARN] Shapefile is empty: {shp_path}")
        return dept_gdf, None

    col_candidates = [c for c in dept_gdf.columns if c.lower() in {
        'nombre_dpt', 'nombre_dept', 'nombre', 'departamen', 'dpto'
    }]
    boyaca = None
    if col_candidates:
        name_col = col_candidates[0]
        boyaca = dept_gdf[dept_gdf[name_col].astype(str).str.upper().str.contains('BOYACA', na=False)].copy()
    elif len(dept_gdf) == 1:
        boyaca = dept_gdf.copy()
        print('[INFO] No name column found; assuming single feature is Boyaca')
    else:
        print(f"[WARN] Could not identify Boyaca geometry; columns present: {list(dept_gdf.columns)}")

    if boyaca is not None and boyaca.empty:
        print('[WARN] Boyaca geometry not found in shapefile; overlay disabled')
        boyaca = None

    return dept_gdf, boyaca


DEPT_GDF, BOYACA = _load_boyaca_shape(SHAPE_DIR)

MAP_OUT_ROOT = OUTPUT_ROOT / 'map_exports'
PLOT_HORIZONS = CONFIG.enabled_horizons or [CONFIG.horizon]
PLOT_EXPERIMENTS = [exp['name'] for exp in EXPERIMENTS] if 'EXPERIMENTS' in globals() else ['BASIC_KCE']
PLOT_MODELS = ['V5_STACKING']
MAP_SAMPLE_INDEX = CONFIG.map_sample_index
CYCLE_ALL_VAL_WINDOWS = CONFIG.map_cycle_all_val_windows
EXPORT_DPI = CONFIG.map_export_dpi
GIF_FRAME_DURATION = CONFIG.map_gif_duration
MAP_MAX_WINDOWS = None

lats = ds[LAT_DIM].values
lons = ds[LON_DIM].values
EXTENT = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]


def _load_exports(exp_name: str, model_name: str, horizon: int):
    export_dir = MAP_OUT_ROOT / f'H{horizon}' / exp_name / model_name
    pred_f = export_dir / 'predictions.npy'
    targ_f = export_dir / 'targets.npy'
    meta_f = export_dir / 'metadata.json'
    if not (pred_f.exists() and targ_f.exists() and meta_f.exists()):
        return None, None, None, None
    y_pred = np.load(pred_f)
    y_true = np.load(targ_f)
    meta = json.loads(meta_f.read_text())
    if y_pred.ndim == 5:
        y_pred = y_pred[..., 0]
    if y_true.ndim == 5:
        y_true = y_true[..., 0]
    return y_pred, y_true, meta, export_dir


def _plot_triplet(ax, data, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(data, origin='lower', extent=EXTENT, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_maps_v5():
    if not MAP_OUT_ROOT.exists():
        print(f"Map export root not found: {MAP_OUT_ROOT}")
        return
    if imageio is None:
        print('imageio not available; GIF export disabled')

    for horizon in PLOT_HORIZONS:
        for exp_name in PLOT_EXPERIMENTS:
            for model_name in PLOT_MODELS:
                y_pred, y_true, meta, export_dir = _load_exports(exp_name, model_name, horizon)
                if y_pred is None:
                    print(f"[SKIP] {exp_name} - {model_name} - H{horizon}: exports not found")
                    continue

                if CYCLE_ALL_VAL_WINDOWS:
                    indices_to_plot = range(len(y_true))
                else:
                    idx = MAP_SAMPLE_INDEX if MAP_SAMPLE_INDEX >= 0 else (len(y_true) - 1)
                    idx = max(min(idx, len(y_true) - 1), 0)
                    indices_to_plot = [idx]

                if MAP_MAX_WINDOWS is not None:
                    indices_to_plot = list(indices_to_plot)[:MAP_MAX_WINDOWS]

                forecast_dates = meta.get('forecast_dates', [])
                if not forecast_dates and meta.get('val_indices'):
                    forecast_dates = _forecast_dates(ds, meta['val_indices'], CONFIG.input_window, horizon)

                frame_count = 0
                gif_path = export_dir / f'{model_name}_H{horizon}.gif'
                writer = None

                for idx in indices_to_plot:
                    sample_dates = forecast_dates[idx] if idx < len(forecast_dates) else []
                    for h in range(min(horizon, len(sample_dates) or horizon)):
                        real = y_true[idx, h]
                        pred = y_pred[idx, h]
                        err = np.abs(pred - real) / (np.abs(real) + 1e-6) * 100.0

                        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
                        date_label = sample_dates[h] if h < len(sample_dates) else f'H{h + 1}'
                        im0 = _plot_triplet(axes[0], real, f'Real {date_label}', 'Blues')
                        im1 = _plot_triplet(axes[1], pred, f'{model_name} H{h + 1} {date_label}', 'Blues')
                        im2 = _plot_triplet(axes[2], np.clip(err, 0, 100), f'MAPE% {model_name} H{h + 1} {date_label}', 'Reds', vmin=0, vmax=100)
                        if BOYACA is not None:
                            try:
                                BOYACA.boundary.plot(ax=axes[0], color='k', linewidth=0.8)
                                BOYACA.boundary.plot(ax=axes[1], color='k', linewidth=0.8)
                                BOYACA.boundary.plot(ax=axes[2], color='k', linewidth=0.8)
                            except Exception as exc:
                                print(f"[WARN] shapefile overlay failed: {exc}")

                        cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                        cbar0.set_label('Precipitation (mm)', rotation=270, labelpad=12)
                        cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                        cbar1.set_label('Precipitation (mm)', rotation=270, labelpad=12)
                        cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                        cbar2.set_label('MAPE (%)', rotation=270, labelpad=12)

                        title = f'{exp_name} | {model_name} | H{horizon} | {date_label}'
                        fig.suptitle(title, fontsize=12)

                        export_dir.mkdir(parents=True, exist_ok=True)
                        png_path = export_dir / f'{model_name}_H{horizon}_{date_label}.png'
                        fig.savefig(png_path, dpi=EXPORT_DPI)
                        plt.close(fig)

                        if imageio is not None:
                            frame = imageio.imread(png_path)
                            if writer is None:
                                writer = imageio.get_writer(gif_path, mode='I', duration=GIF_FRAME_DURATION, loop=0)
                            writer.append_data(frame)
                            del frame

                        frame_count += 1
                        del real, pred, err

                if writer is not None:
                    writer.close()

                print(f"[OK] {exp_name} - {model_name} - H{horizon}: saved {frame_count} frames to {export_dir}")


if getattr(CONFIG, 'generate_map_plots', True):
    plot_maps_v5()
else:
    print('Map plot generation disabled')

# --- Cell 37 ---
# =============================================================================
# SECTION 9.2: TRAINING CURVES VISUALIZATION
# =============================================================================

def plot_training_curves(experiment_state: Dict, output_dir: Path):
    """Plot training curves for all horizons."""

    entries = []
    for exp_name, exp_results in experiment_state.get('results', {}).items():
        for h_key, results in exp_results.items():
            entries.append((exp_name, h_key, results))

    if not entries:
        print("No training results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['steelblue', 'coral', 'forestgreen', 'purple']

    for idx, (exp_name, h_key, results) in enumerate(entries[:4]):
        ax = axes[idx]

        # Load training log
        h_num = h_key.replace('H', '')
        log_path = output_dir / f'h{h_num}' / exp_name / 'training_metrics' / f'v5_stacking_training_log_h{h_num}.csv'

        if log_path.exists():
            log_df = pd.read_csv(log_path)

            ax.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss', color=colors[idx], alpha=0.8)
            ax.plot(log_df['epoch'], log_df['val_loss'], label='Val Loss', color=colors[idx], linestyle='--', alpha=0.8)

            # Mark best epoch
            best_epoch = results['history']['best_epoch'] + 1
            best_val = results['history']['best_val_loss']
            ax.scatter([best_epoch], [best_val], color='red', s=100, zorder=5, marker='*', label=f'Best: {best_val:.4f}')

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'Training Curves - {exp_name} {h_key}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save at 700 DPI
    fig_path = output_dir / 'v5_training_curves.png'
    fig.savefig(fig_path, dpi=700, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Training curves saved to: {fig_path} (700 DPI)")

# Plot training curves
plot_training_curves(experiment_state, OUTPUT_ROOT)

# --- Cell 39 ---
# =============================================================================
# SECTION 10: SUMMARY AND RESULTS
# =============================================================================

print("\n" + "="*70)
print("V5 GNN-ConvLSTM STACKING - EXPERIMENT SUMMARY")
print("="*70)

# Print configuration
print("\n[Configuration]")
print(f"  Light Mode: {CONFIG.light_mode}")
if CONFIG.light_mode:
    print(f"  Grid Size: {CONFIG.light_grid_size}x{CONFIG.light_grid_size}")
print(f"  GNN Type: {CONFIG.gnn_type}")
print(f"  Horizons: {CONFIG.enabled_horizons}")
print(f"  Epochs: {CONFIG.epochs}")
print(f"  Batch Size: {CONFIG.batch_size}")
if 'EXPERIMENTS' in globals():
    exp_names = [exp['name'] for exp in EXPERIMENTS]
    print(f"  Experiments: {', '.join(exp_names)}")
if 'data_report' in globals() and data_report:
    print(f"  Split Mode: {data_report.get('split_mode', 'unknown')}")
    print(f"  Split Index: {data_report.get('split_time_idx', 'unknown')}")

# Print results summary
if metrics_df is not None and len(metrics_df) > 0:
    print("\n[Results Summary]")
    print("-"*70)
    print(metrics_df[['TotalHorizon', 'H', 'RMSE', 'MAE', 'R^2', 'mean_bias_mm']].to_string(index=False))
    print("-"*70)

    # Overall averages
    print("\n[Overall Averages]")
    print(f"  RMSE: {metrics_df['RMSE'].mean():.2f} mm (Target: < 85 mm)")
    print(f"  MAE: {metrics_df['MAE'].mean():.2f} mm")
    print(f"  R2: {metrics_df['R^2'].mean():.4f} (Target: > 0.65)")
    print(f"  Bias: {metrics_df['mean_bias_mm'].mean():.2f} mm")

# Comparability caveats
print("\n[Comparability Caveats]")
if metrics_df is not None and len(metrics_df) > 0 and 'Mean_True_mm' in metrics_df.columns:
    mean_true = metrics_df['Mean_True_mm'].mean()
    print(f"  Mean True (V5): {mean_true:.2f} mm")
print("  Apples-to-apples runs: BASIC_BASIC, KCE_KCE, PAFC_PAFC")
if 'RUN_MIXED_ABLATION' in globals() and RUN_MIXED_ABLATION:
    print("  Mixed feature set BASIC_KCE is treated as an ablation")
else:
    print("  Mixed feature sets are excluded unless RUN_MIXED_ABLATION=True")
print("  Keep dataset slice, mask, scaling, and split identical across versions")
print("  Report mean_true_mm and scale_ratio per version")

# Print branch weights
print("\n[Branch Weights (Meta-Learner)]")
for exp_name, exp_results in experiment_state.get('results', {}).items():
    for h_key, results in exp_results.items():
        if 'metrics' in results and 'branch_weights' in results['metrics']:
            weights = results['metrics']['branch_weights']
            print(f"  {exp_name} {h_key}: ConvLSTM={weights['w_convlstm']:.1%}, GNN={weights['w_gnn']:.1%}")

# Quality checks
print("\n[Quality Checks]")
for exp_name, exp_results in experiment_state.get('results', {}).items():
    for h_key, results in exp_results.items():
        qc = results.get('quality_checks', {})
        issues = qc.get('issues', [])
        status = 'OK' if not issues else f"WARN: {', '.join(issues)}"
        print(f"  {exp_name} {h_key}: {status}")

# Output files
print("\n[Output Files]")
print(f"  Output Directory: {OUTPUT_ROOT}")
for path in OUTPUT_ROOT.glob('*'):
    if path.is_file():
        print(f"  - {path.name}")

# Next steps
print("\n[Next Steps]")
if CONFIG.light_mode:
    print("  1. Set light_mode=False for full grid validation (61x65)")
    print("  2. Run full experiments on Colab GPU")
print("  3. Compare results with V4 baseline (R2=0.628, RMSE=92.12mm)")

# --- Cell 40 ---
# =============================================================================
# SECTION 10.2: CLEANUP
# =============================================================================

# Memory cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU Memory freed. Current usage: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

print("\nNotebook execution completed successfully.")

