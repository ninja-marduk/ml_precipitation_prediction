# -*- coding: utf-8 -*-
"""Smoke test for the GNN-TAT training loop as it is defined in the notebook.

The training code lives inside base_models_gnn_tat_v4.ipynb, so a bug in it is
only discovered after a GPU session has already been spent. This test extracts the
model classes and the training function straight from the notebook, stubs the
PyTorch Geometric convolutions so it runs on CPU without PyG, and trains for a few
epochs on a tiny synthetic grid.

It checks the plumbing, not the numerics: that the chunk loop handles a remainder
chunk under gradient checkpointing, that the batched edge_index is cached and is
only built with edge weights for GCN, that the epoch loop and early stopping run,
and that the compute-cost fields written to the history JSON are populated.

Usage: python models/scripts/tests/test_gnn_tat_training_loop.py
"""
import io, json, re, sys, tempfile, types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union


# ---------------------------------------------------------------- PyG stubs
class _MsgConv(nn.Module):
    """Minimal mean-aggregation stand-in for GATConv/SAGEConv/GCNConv."""

    def __init__(self, in_channels, out_channels, heads=1, concat=False, **kw):
        super().__init__()
        self.out = out_channels * heads if concat else out_channels
        self.lin = nn.Linear(in_channels, self.out)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.lin(x)
        src, dst = edge_index[0], edge_index[1]
        m = h[src]
        if edge_weight is not None:
            m = m * edge_weight.view(-1, 1)
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        return h + agg


GATConv = SAGEConv = GCNConv = _MsgConv

REPO = Path(__file__).resolve().parents[3]
NB = REPO / "models" / "base_models_gnn_tat_v4.ipynb"
nb = json.load(io.open(NB, encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"]]

# cell 18: keep only the class definitions (drop the module-level model construction)
model_src = cells[18].split("# Determine optimal chunk size")[0]
# cell 21: everything (helper + dataset + train function)
train_src = cells[21]

ns = dict(globals())
exec(compile(model_src, "cell18", "exec"), ns)
exec(compile(train_src, "cell21", "exec"), ns)
GNN_TAT = ns["GNN_TAT"]
train_pytorch_model = ns["train_pytorch_model"]
_fmt_hms = ns["_fmt_hms"]

# ---------------------------------------------------------------- tiny problem
torch.manual_seed(0)
np.random.seed(0)
LAT, LON, SEQ, HOR, NF = 6, 5, 12, 3, 4
N_NODES = LAT * LON
N_TR, N_VA = 20, 6

CONFIG = {
    'epochs': 12, 'batch_size': 3, 'learning_rate': 1e-3, 'patience': 5,
    'weight_decay': 1e-5, 'input_window': SEQ, 'horizon': HOR,
    'gnn_chunk_size': 7,           # deliberately not a divisor: exercises the remainder chunk
    'gnn_grad_checkpoint': True,
    'prediction_batch_size': 4,
    'gpu_resident_data': True, 'use_amp': False, 'amp_dtype': 'bf16',
    'gnn_config': {
        'hidden_dim': 16, 'num_gnn_layers': 2, 'gnn_type': 'GAT', 'num_heads': 4,
        'dropout': 0.1, 'temporal_hidden_dim': 16, 'num_temporal_heads': 4,
        'temporal_dropout': 0.1, 'lstm_hidden_dim': 16, 'num_lstm_layers': 2,
    },
}
ns["CONFIG"] = CONFIG

X_tr = np.random.randn(N_TR, SEQ, LAT, LON, NF).astype(np.float32)
y_tr = np.random.randn(N_TR, HOR, LAT, LON, 1).astype(np.float32)
X_va = np.random.randn(N_VA, SEQ, LAT, LON, NF).astype(np.float32)
y_va = np.random.randn(N_VA, HOR, LAT, LON, 1).astype(np.float32)

n_edges = 300
edge_index = torch.randint(0, N_NODES, (2, n_edges))
edge_weight = torch.rand(n_edges)
device = torch.device('cpu')

model = GNN_TAT(n_features=NF, n_nodes=N_NODES, n_lat=LAT, n_lon=LON,
                horizon=HOR, config=CONFIG, gnn_chunk_size=CONFIG['gnn_chunk_size']).to(device)

print("\n-- needs_edge_weight for GAT:", model._needs_edge_weight, "(expected False)")

out_dir = Path(tempfile.mkdtemp())
model, summary = train_pytorch_model(
    model, X_tr, y_tr, X_va, y_va, edge_index, edge_weight,
    CONFIG, "GNN_TAT_GAT", "BASIC", out_dir, HOR, device)

print("\n-- edge cache keys:", list(model._edge_cache.keys()) if model._edge_cache else "(cleared)")
print("-- summary:")
for k, v in summary.items():
    print(f"     {k}: {v}")

log = pd.read_csv(out_dir / f"h{HOR}" / "BASIC" / "training_metrics" / f"GNN_TAT_GAT_training_log_h{HOR}.csv")
print("\n-- training log columns:", list(log.columns))
assert 'epoch_seconds' in log.columns
assert summary['wall_seconds'] > 0 and summary['sec_per_epoch_mean'] > 0
assert summary['n_edges'] == n_edges and summary['batch_size'] == 3
assert summary['amp'] == 'fp32'

# GCN must still receive the batched edge_weight
cfg_gcn = {**CONFIG, 'gnn_config': {**CONFIG['gnn_config'], 'gnn_type': 'GCN'}}
m2 = GNN_TAT(n_features=NF, n_nodes=N_NODES, n_lat=LAT, n_lon=LON, horizon=HOR,
             config=cfg_gcn, gnn_chunk_size=7)
print("\n-- needs_edge_weight for GCN:", m2._needs_edge_weight, "(expected True)")
assert m2._needs_edge_weight
bidx, bw = m2._batched_edges(edge_index, edge_weight, 7, device)
assert bidx.shape == (2, n_edges * 7) and bw is not None and bw.numel() == n_edges * 7
assert m2._batched_edges(edge_index, edge_weight, 7, device)[0] is bidx, "cache miss"
print("-- batched edges cached and correctly shaped:", tuple(bidx.shape))

# eval path uses prediction_batch_size
ev_src = cells[22]
exec(compile(ev_src, "cell22", "exec"), ns)
print("\nAll checks passed.")
