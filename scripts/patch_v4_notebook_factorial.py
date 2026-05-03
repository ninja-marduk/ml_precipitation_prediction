"""Patch V4 notebook to support factorial (features x variants x seeds) runs.

Adds two CONFIG keys:
  - 'enabled_variants': list of GNN variants to train, subset of ['GAT','GCN','SAGE']
  - 'enabled_features': list of feature bundles to run, subset of ['BASIC','KCE','PAFC']

Replaces the hardcoded PAPER 5 FILTER (`['GNN_TAT_GAT']`) in the main training loop
with a config-driven version. Defaults preserve backward compatibility (variants=['GAT'],
features=all three).

Run this once, locally. The patch is idempotent: re-running is a no-op.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB = REPO / 'models' / 'base_models_gnn_tat_v4.ipynb'

# Marker the patch inserts so re-runs become a no-op.
PATCH_MARKER = "# PATCH: factorial (enabled_variants x enabled_features)"


def _src(cell: dict) -> str:
    return ''.join(cell.get('source', []))


def _set_src(cell: dict, text: str) -> None:
    # Preserve Jupyter convention: list of strings with trailing newlines.
    lines = text.splitlines(keepends=True)
    cell['source'] = lines


def patch_cell6_config(cell: dict) -> bool:
    """Add enabled_variants / enabled_features / factorial marker to CONFIG."""
    src = _src(cell)
    if PATCH_MARKER in src:
        print('  cell 6: already patched, skipping')
        return False

    # Insert new keys right after the 'feature_sets' block closing '},'
    anchor = "    # -------------------------------------------------------------------------\n    # GNN-TAT Specific Configuration"
    if anchor not in src:
        raise RuntimeError('cell 6: anchor not found (GNN-TAT Specific Configuration header)')

    insertion = (
        "    # -------------------------------------------------------------------------\n"
        "    # Factorial filters (feature x variant)\n"
        f"    {PATCH_MARKER}\n"
        "    # 'enabled_variants' : list of GNN variants to train (subset of ['GAT','GCN','SAGE']).\n"
        "    #                     Defaults to ['GAT'] for backward compatibility with the\n"
        "    #                     original Paper 5 filter.\n"
        "    # 'enabled_features' : list of feature bundles to run (subset of feature_sets keys).\n"
        "    #                     Defaults to all three. Set to a subset to run only a subset\n"
        "    #                     of the factorial grid.\n"
        "    # -------------------------------------------------------------------------\n"
        "    'enabled_variants': ['GAT'],\n"
        "    'enabled_features': ['BASIC', 'KCE', 'PAFC'],\n\n"
    )
    new_src = src.replace(anchor, insertion + anchor, 1)
    _set_src(cell, new_src)
    print('  cell 6: added enabled_variants and enabled_features')
    return True


def patch_cell23_training_loop(cell: dict) -> bool:
    """Replace the hardcoded PAPER 5 FILTER with config-driven iteration."""
    src = _src(cell)
    if PATCH_MARKER in src:
        print('  cell 23: already patched, skipping')
        return False

    old_feature_loop = (
        "        for exp_name in CONFIG['feature_sets'].keys():\n"
        "            if exp_name not in data_splits_h:"
    )
    new_feature_loop = (
        f"        {PATCH_MARKER}\n"
        "        _enabled_features = CONFIG.get('enabled_features', list(CONFIG['feature_sets'].keys()))\n"
        "        for exp_name in CONFIG['feature_sets'].keys():\n"
        "            if exp_name not in _enabled_features:\n"
        "                print(f\"  [SKIP feature] {exp_name} not in enabled_features={_enabled_features}\")\n"
        "                continue\n"
        "            if exp_name not in data_splits_h:"
    )
    if old_feature_loop not in src:
        raise RuntimeError('cell 23: feature-loop anchor not found')
    src = src.replace(old_feature_loop, new_feature_loop, 1)

    old_variant_block = (
        "            # PAPER 5 FILTER: Only train GAT variant (used in the paper).\n"
        "            # To train all variants, use: ['GNN_TAT_GAT', 'GNN_TAT_SAGE', 'GNN_TAT_GCN']\n"
        "            for model_name in ['GNN_TAT_GAT']:"
    )
    new_variant_block = (
        "            # Config-driven variant iteration (replaces former PAPER 5 FILTER).\n"
        "            _enabled_variants = CONFIG.get('enabled_variants', ['GAT'])\n"
        "            _variant_models = [f'GNN_TAT_{v}' for v in _enabled_variants]\n"
        "            for model_name in _variant_models:"
    )
    if old_variant_block not in src:
        raise RuntimeError('cell 23: variant-block anchor not found')
    src = src.replace(old_variant_block, new_variant_block, 1)

    _set_src(cell, src)
    print('  cell 23: replaced PAPER 5 FILTER with config-driven loop')
    return True


def main() -> int:
    with NB.open('r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f'Patching: {NB}')
    changed = False
    changed |= patch_cell6_config(nb['cells'][6])
    changed |= patch_cell23_training_loop(nb['cells'][23])

    if not changed:
        print('\nNo changes (already patched).')
        return 0

    with NB.open('w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f'\nSaved: {NB}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
