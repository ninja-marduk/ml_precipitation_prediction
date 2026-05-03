"""Prerequisite check for feature 002 (full-horizon multi-seed reproducibility).

Validates that every on-disk artefact the pipeline depends on is present and
has the expected shape. Prints a go/no-go report and exits non-zero on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'models' / 'output'


def _check_npy(path: Path, expected_shape: tuple, label: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f'MISSING: {label} at {path}'
    try:
        arr = np.load(path, mmap_mode='r')
    except Exception as e:
        return False, f'LOAD_FAIL: {label} at {path} ({e})'
    if arr.shape != expected_shape:
        return False, f'SHAPE: {label} at {path} has {arr.shape}, expected {expected_shape}'
    return True, f'OK    : {label} {arr.shape}'


def _check_file(path: Path, label: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f'MISSING: {label} at {path}'
    size = path.stat().st_size
    if size == 0:
        return False, f'EMPTY : {label} at {path}'
    return True, f'OK    : {label} ({size} bytes)'


def _check_dir(path: Path, label: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f'MISSING: {label} at {path}'
    if not path.is_dir():
        return False, f'NOT_DIR: {label} at {path}'
    return True, f'OK    : {label} exists'


def main() -> int:
    results: list[tuple[bool, str]] = []

    # V2 Keras checkpoint (source for seed-42 inference)
    results.append(_check_file(
        OUT / 'V2_Enhanced_Models' / 'h12' / 'BASIC' / 'training_metrics' / 'ConvLSTM_best_h12.h5',
        'V2 ConvLSTM Keras checkpoint (seed-42 source)',
    ))

    # V2 seeds 123 and 456 predictions (already full-grid)
    for seed in (123, 456):
        results.append(_check_npy(
            OUT / 'V2_Enhanced_Models' / f'SEED{seed}' / 'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM' / 'predictions.npy',
            (33, 12, 61, 65, 1),
            f'V2 ConvLSTM seed {seed} predictions',
        ))
        results.append(_check_npy(
            OUT / 'V2_Enhanced_Models' / f'SEED{seed}' / 'map_exports' / 'H12' / 'BASIC' / 'ConvLSTM' / 'targets.npy',
            (33, 12, 61, 65, 1),
            f'V2 ConvLSTM seed {seed} targets',
        ))

    # V4 GNN-TAT predictions (legacy full-grid + seeds 123/456)
    results.append(_check_npy(
        OUT / 'V4_GNN_TAT_Models' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT' / 'predictions.npy',
        (33, 12, 61, 65, 1),
        'V4 GNN-TAT-GAT seed 42 predictions (legacy full-grid)',
    ))
    for seed in (123, 456):
        results.append(_check_npy(
            OUT / 'V4_GNN_TAT_Models' / f'SEED{seed}' / 'map_exports' / 'H12' / 'BASIC' / 'GNN_TAT_GAT' / 'predictions.npy',
            (33, 12, 61, 65, 1),
            f'V4 GNN-TAT-GAT seed {seed} predictions',
        ))

    # V10 SEED* directories (expected populated for all three seeds)
    for seed in (42, 123, 456):
        results.append(_check_dir(
            OUT / 'V10_Late_Fusion' / f'SEED{seed}',
            f'V10 Late Fusion seed {seed} directory',
        ))

    ok = all(r[0] for r in results)

    print('Prerequisite verification for feature 002 (full-horizon multi-seed)')
    print('=' * 72)
    for _, msg in results:
        print('  ', msg)
    print('=' * 72)
    print(f'Status: {"GO" if ok else "NO-GO"}  ({sum(1 for r in results if r[0])}/{len(results)} checks passed)')

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
