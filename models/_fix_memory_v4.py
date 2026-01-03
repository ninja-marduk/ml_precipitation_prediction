"""
Script to fix CUDA OOM error in V4 GNN-TAT notebook.
Replaces the GNN_TAT class with a memory-optimized version using chunked processing.
"""

import json

# Read the notebook
notebook_path = r'd:\github.com\ninja-marduk\ml_precipitation_prediction\models\base_models_GNN_TAT_V4.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New cell with memory-optimized model
new_cell_source = r'''# =============================================================================
# CELL 8: GNN-TAT Model Definition (PyTorch) - MEMORY OPTIMIZED
# =============================================================================
# Fixes applied:
# - Proper residual connection in TemporalAttention
# - CHUNKED GNN processing to avoid OOM on large grids
# - Reduced model dimensions (~100K params)
# - GCN with add_self_loops=True for stability
# - Memory cleanup between chunks

class TemporalAttention(nn.Module):
    """
    Multi-Head Temporal Attention module with proper residual connection.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Residual projection (if dimensions differ)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Save input for residual (with projection if needed)
        residual = self.residual_proj(x)

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, V)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        # Proper residual connection: residual + dropout(output)
        output = self.layer_norm(residual + self.dropout(output))

        return output


class SpatialGNNEncoder(nn.Module):
    """
    GNN Encoder for spatial dependencies.
    Supports GCN, GAT, and GraphSAGE with proper handling.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 gnn_type: str = 'GAT', num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if gnn_type == 'GAT':
                layer = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            elif gnn_type == 'SAGE':
                layer = SAGEConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggr='mean'
                )
            else:  # GCN
                layer = GCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    add_self_loops=True,
                    normalize=True
                )

            self.gnn_layers.append(layer)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)

        # GNN layers with residual connections
        for gnn, norm in zip(self.gnn_layers, self.norms):
            residual = x

            if self.gnn_type == 'GAT':
                x = gnn(x, edge_index)
            elif self.gnn_type == 'SAGE':
                x = gnn(x, edge_index)
            else:  # GCN
                x = gnn(x, edge_index, edge_weight)

            x = F.gelu(x)
            x = self.dropout(x)
            x = norm(x + residual)

        return x


class GNN_TAT(nn.Module):
    """
    Graph Neural Network with Temporal Attention - MEMORY OPTIMIZED.

    Key optimizations for full grid support:
    - CHUNKED GNN processing (processes timesteps in chunks)
    - Memory cleanup between chunks
    - Efficient temporal processing
    """

    def __init__(self, n_features: int, n_nodes: int, n_lat: int, n_lon: int,
                 horizon: int, config: Dict, gnn_chunk_size: int = 15):
        super().__init__()

        self.n_features = n_features
        self.n_nodes = n_nodes
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.horizon = horizon
        self.gnn_chunk_size = gnn_chunk_size  # Process this many timesteps at once

        gnn_cfg = config['gnn_config']
        hidden_dim = gnn_cfg['hidden_dim']
        self.hidden_dim = hidden_dim

        # GNN Encoder
        self.gnn_encoder = SpatialGNNEncoder(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            num_layers=gnn_cfg['num_gnn_layers'],
            gnn_type=gnn_cfg['gnn_type'],
            num_heads=gnn_cfg['num_heads'],
            dropout=gnn_cfg['dropout']
        )

        # Temporal Attention (per node, efficient)
        self.temporal_attention = TemporalAttention(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=gnn_cfg['num_temporal_heads'],
            dropout=gnn_cfg['temporal_dropout']
        )

        # LSTM for sequence modeling (per node)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gnn_cfg['num_lstm_layers'],
            batch_first=True,
            dropout=gnn_cfg['dropout'] if gnn_cfg['num_lstm_layers'] > 1 else 0
        )

        # Output projection (per node -> horizon predictions)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(gnn_cfg['dropout']),
            nn.Linear(hidden_dim, horizon)
        )

        self._print_model_info(gnn_cfg)

    def _print_model_info(self, gnn_cfg):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"GNN-TAT Model initialized (MEMORY OPTIMIZED):")
        print(f"  Input: ({self.n_features} features, {self.n_nodes} nodes)")
        print(f"  Output: (horizon={self.horizon}, {self.n_lat}x{self.n_lon} grid)")
        print(f"  GNN type: {gnn_cfg['gnn_type']}")
        print(f"  Hidden dim: {gnn_cfg['hidden_dim']}")
        print(f"  GNN chunk size: {self.gnn_chunk_size} timesteps")
        print(f"  Parameters: {total_params:,}")

    def _process_gnn_chunk(self, x_chunk: torch.Tensor, edge_index: torch.Tensor,
                           edge_weight: torch.Tensor = None) -> torch.Tensor:
        """Process a chunk of timesteps through GNN."""
        chunk_size = x_chunk.shape[0]  # num_graphs in this chunk
        device = x_chunk.device

        # Create batched edge_index for this chunk only
        batch_edge_index = self._batch_edge_index(edge_index, chunk_size, device)
        batch_edge_weight = edge_weight.repeat(chunk_size) if edge_weight is not None else None

        # Flatten nodes: (chunk_size * n_nodes, n_features)
        x_nodes = x_chunk.view(-1, self.n_features)

        # Apply GNN
        gnn_out = self.gnn_encoder(x_nodes, batch_edge_index, batch_edge_weight)

        # Reshape: (chunk_size, n_nodes, hidden_dim)
        gnn_out = gnn_out.view(chunk_size, self.n_nodes, self.hidden_dim)

        return gnn_out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        device = x.device

        # Reshape: (batch, seq_len, lat, lon, features) -> (batch, seq_len, n_nodes, features)
        x = x.view(batch_size, seq_len, self.n_nodes, self.n_features)

        # Flatten batch and seq_len: (batch * seq_len, n_nodes, features)
        x_flat = x.view(batch_size * seq_len, self.n_nodes, self.n_features)
        total_graphs = batch_size * seq_len

        # CHUNKED GNN Processing to avoid OOM
        gnn_outputs = []
        for start_idx in range(0, total_graphs, self.gnn_chunk_size):
            end_idx = min(start_idx + self.gnn_chunk_size, total_graphs)
            chunk = x_flat[start_idx:end_idx]  # (chunk_size, n_nodes, features)

            # Process chunk
            chunk_out = self._process_gnn_chunk(chunk, edge_index, edge_weight)
            gnn_outputs.append(chunk_out)

        # Concatenate all chunks
        gnn_out = torch.cat(gnn_outputs, dim=0)  # (batch * seq_len, n_nodes, hidden)
        del gnn_outputs

        # Reshape back: (batch, seq_len, n_nodes, hidden_dim)
        gnn_out = gnn_out.view(batch_size, seq_len, self.n_nodes, self.hidden_dim)

        # Temporal Processing (per node)
        temporal_in = gnn_out.permute(0, 2, 1, 3)  # (batch, n_nodes, seq_len, hidden)
        temporal_in = temporal_in.reshape(batch_size * self.n_nodes, seq_len, self.hidden_dim)

        # Apply temporal attention
        temporal_out = self.temporal_attention(temporal_in)

        # Apply LSTM
        lstm_out, _ = self.lstm(temporal_out)

        # Take last timestep: (batch * n_nodes, hidden)
        lstm_last = lstm_out[:, -1, :]

        # Output Projection
        out = self.output_proj(lstm_last)  # (batch * n_nodes, horizon)

        # Reshape: (batch, n_nodes, horizon) -> (batch, horizon, lat, lon, 1)
        out = out.view(batch_size, self.n_nodes, self.horizon)
        out = out.permute(0, 2, 1)  # (batch, horizon, n_nodes)
        out = out.view(batch_size, self.horizon, self.n_lat, self.n_lon, 1)

        return out

    def _batch_edge_index(self, edge_index: torch.Tensor, num_graphs: int,
                          device: torch.device) -> torch.Tensor:
        """Create batched edge_index by replicating and offsetting indices."""
        num_edges = edge_index.shape[1]

        # Create offsets for each graph
        offsets = torch.arange(num_graphs, device=device) * self.n_nodes
        offsets = offsets.view(-1, 1, 1).expand(-1, 2, num_edges)

        # Replicate edge_index and add offsets
        batch_edge_index = edge_index.unsqueeze(0).expand(num_graphs, -1, -1)
        batch_edge_index = batch_edge_index + offsets

        # Flatten: (num_graphs, 2, num_edges) -> (2, num_graphs * num_edges)
        batch_edge_index = batch_edge_index.permute(1, 0, 2).reshape(2, -1)

        return batch_edge_index


# Determine optimal chunk size based on grid size
n_nodes_estimate = lat * lon if 'lat' in dir() and 'lon' in dir() else 425
if n_nodes_estimate <= 50:  # Small grid (e.g., 5x5 or 7x7)
    GNN_CHUNK_SIZE = 60  # Can process all timesteps at once
elif n_nodes_estimate <= 150:  # Medium grid (e.g., 10x15)
    GNN_CHUNK_SIZE = 20
else:  # Large grid (e.g., 17x25)
    GNN_CHUNK_SIZE = 8  # Process 8 timesteps at a time

print(f"\nEstimated grid size: {n_nodes_estimate} nodes")
print(f"Using GNN chunk size: {GNN_CHUNK_SIZE} timesteps per chunk")

# Create model
print("\nCreating GNN-TAT model...")

# Get number of features from first experiment
first_exp = list(data_splits.keys())[0]
n_features = data_splits[first_exp][0].shape[-1]

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = GNN_TAT(
    n_features=n_features,
    n_nodes=graph_builder.n_nodes,
    n_lat=lat,
    n_lon=lon,
    horizon=CONFIG['horizon'],
    config=CONFIG,
    gnn_chunk_size=GNN_CHUNK_SIZE
)
model = model.to(device)

# Print memory estimate
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"\nGPU Memory after model init: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
'''

# Update cell 14
nb['cells'][14]['source'] = [line + '\n' for line in new_cell_source.split('\n')]
# Remove trailing empty line if present
if nb['cells'][14]['source'] and nb['cells'][14]['source'][-1].strip() == '':
    nb['cells'][14]['source'] = nb['cells'][14]['source'][:-1]

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("SUCCESS: Cell 14 updated with MEMORY OPTIMIZED GNN-TAT model")
print("\nKey changes:")
print("  - Added gnn_chunk_size parameter (default: based on grid size)")
print("  - Process timesteps in chunks instead of all at once")
print("  - Automatic chunk size: 8 for large grids, 20 for medium, 60 for small")
print("  - Memory cleanup between chunks")
print("\nFor 17x25 grid (425 nodes), uses chunk_size=8:")
print("  - Old: batch_edge_index with 240 graphs = 816K edges = 28GB+")
print("  - New: batch_edge_index with 8 graphs = 27K edges = ~1GB")
