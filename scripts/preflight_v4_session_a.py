"""Pre-flight check for V4 factorial sweep A (seed 123).

Paste this output into the Colab notebook before starting training.
Locally this script just validates the notebook state and the Drive-side paths.

It answers three questions:
  1. Is the V4 notebook correctly patched (enabled_variants / enabled_features)?
  2. With SEEDS=[123] + feats=['BASIC','PAFC'] + variants=['GCN','SAGE'],
     which of the 18 factorial cells will the loop TRAIN vs SKIP?
  3. Are the 2 pre-existing seed-123 cells (GAT-BASIC, GAT-PAFC) on disk so
     SKIP_EXISTING can detect them?

Usage:
  python scripts/preflight_v4_session_a.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB = REPO / 'models' / 'base_models_gnn_tat_v4.ipynb'
V4_ROOT = REPO / 'models' / 'output' / 'V4_GNN_TAT_Models'

SESSION_A = {
    'SEEDS': [123],
    'enabled_features': ['BASIC', 'PAFC'],
    'enabled_variants': ['GCN', 'SAGE'],
}

OK = '[ OK ]'
WARN = '[WARN]'
FAIL = '[FAIL]'


def check_patch() -> tuple[bool, list[str]]:
    """Verify the V4 notebook has the factorial patch applied and is
    configured for sweep A (either via default values or user overrides).
    """
    messages = []
    ok = True
    with NB.open('r', encoding='utf-8') as f:
        nb = json.load(f)

    import re

    src6 = ''.join(nb['cells'][6].get('source', []))
    src7 = ''.join(nb['cells'][7].get('source', []))
    src23 = ''.join(nb['cells'][23].get('source', []))

    # Normalise to a whitespace-stripped form for tolerant comparisons.
    src6_ns = re.sub(r'\s+', '', src6)
    src7_ns = re.sub(r'\s+', '', src7)

    # Accept either the original defaults or session-A overrides in cell 6.
    variants_ok = (
        "'enabled_variants':['GAT']" in src6_ns or
        "'enabled_variants':['GCN','SAGE']" in src6_ns
    )
    features_ok = (
        "'enabled_features':['BASIC','KCE','PAFC']" in src6_ns or
        "'enabled_features':['BASIC','PAFC']" in src6_ns
    )
    # Session A needs MULTI_SEED=True + SEEDS=[123] + SKIP_EXISTING=True in cell 7
    sweep_a_seeds = bool(re.search(r'SEEDS\s*=\s*\[\s*123\s*\]', src7))
    sweep_a_multi = 'MULTI_SEED=True' in src7_ns
    sweep_a_skip = 'SKIP_EXISTING=True' in src7_ns
    # Session-A variants/features directly set in cell 6.
    sweep_a_variants = "'enabled_variants':['GCN','SAGE']" in src6_ns
    sweep_a_features = "'enabled_features':['BASIC','PAFC']" in src6_ns

    checks = [
        ('cell 6 has PATCH marker',
         'PATCH: factorial' in src6),
        ('cell 6 enabled_variants value is valid',
         variants_ok),
        ('cell 6 enabled_features value is valid',
         features_ok),
        ('cell 23 no longer has hardcoded GAT-only list',
         "for model_name in ['GNN_TAT_GAT']" not in src23),
        ('cell 23 has config-driven variant loop',
         '_variant_models = [f' in src23),
        ('cell 23 has enabled_features filter',
         'if exp_name not in _enabled_features' in src23),
        ('cell 7 MULTI_SEED = True',
         sweep_a_multi),
        ('cell 7 SEEDS = [123] (sweep A only)',
         sweep_a_seeds),
        ('cell 7 SKIP_EXISTING = True',
         sweep_a_skip),
        ('cell 6 enabled_variants configured for sweep A (GCN+SAGE)',
         sweep_a_variants),
        ('cell 6 enabled_features configured for sweep A (BASIC+PAFC)',
         sweep_a_features),
    ]
    for label, passed in checks:
        messages.append(f'  {OK if passed else FAIL}  {label}')
        ok &= passed
    return ok, messages


def predictions_path(seed: int, feat: str, variant: str) -> Path:
    if seed == 42:
        base = V4_ROOT / 'map_exports'
    else:
        base = V4_ROOT / f'SEED{seed}' / 'map_exports'
    return base / 'H12' / feat / f'GNN_TAT_{variant}' / 'predictions.npy'


def simulate_loop() -> list[dict]:
    """Simulate the training loop for session A.

    Order matches cell 23: for seed in SEEDS: for exp in feats: for model in variants.
    """
    rows = []
    for seed in SESSION_A['SEEDS']:
        for feat in SESSION_A['enabled_features']:
            for variant in SESSION_A['enabled_variants']:
                p = predictions_path(seed, feat, variant)
                exists = p.exists() and p.stat().st_size > 0
                rows.append({
                    'seed': seed,
                    'feat': feat,
                    'variant': variant,
                    'path': p,
                    'exists_before': exists,
                    'action': 'SKIP' if exists else 'TRAIN',
                })
    return rows


def check_seed123_gat_cells() -> tuple[bool, list[str]]:
    """The 2 pre-existing cells (GAT-BASIC, GAT-PAFC for seed 123) must exist
    on disk so SKIP_EXISTING can detect them when the loop iterates. With
    Session A filtering variants=['GCN','SAGE'], GAT cells won't be touched,
    but it's still worth showing they are present to confirm the inventory.
    """
    messages = []
    ok = True
    for feat in ('BASIC', 'PAFC'):
        p = predictions_path(123, feat, 'GAT')
        exists = p.exists() and p.stat().st_size > 0
        size_mb = round(p.stat().st_size / 1024 / 1024, 2) if exists else 0
        mark = OK if exists else WARN
        messages.append(f'  {mark}  seed 123 GAT {feat}: {p.name} ({size_mb} MB)')
        ok &= exists
    return ok, messages


def emit_colab_block() -> str:
    return f'''
# --- FEATURE 003 FACTORIAL: SWEEP A (seed 123) ---
# Validated by scripts/preflight_v4_session_a.py on {__import__("datetime").datetime.now().date()}
# Expected: 4 new predictions.npy under V4_GNN_TAT_Models/SEED123/map_exports/H12/...
# Compute: ~40h A100 (~471 Colab Pro+ CU). Expect 1 mid-sweep disconnect.
MULTI_SEED = True
SEEDS = [123]
SKIP_EXISTING = True
CONFIG['enabled_features'] = ['BASIC', 'PAFC']
CONFIG['enabled_variants'] = ['GCN', 'SAGE']

# Sanity print
_n = len(SEEDS) * len(CONFIG['enabled_features']) * len(CONFIG['enabled_variants'])
print(f"[PREFLIGHT] seed(s) = {{SEEDS}}")
print(f"[PREFLIGHT] feats   = {{CONFIG['enabled_features']}}")
print(f"[PREFLIGHT] vars    = {{CONFIG['enabled_variants']}}")
print(f"[PREFLIGHT] planned = {{_n}} cells  (SKIP_EXISTING will auto-skip completed ones)")
'''.strip()


def main() -> int:
    print('=' * 72)
    print('V4 SWEEP A PRE-FLIGHT (seed 123)')
    print('=' * 72)

    # Check 1: notebook patch
    print('\n[1] Notebook patch state')
    patch_ok, msgs = check_patch()
    for m in msgs:
        print(m)

    # Check 2: simulate the loop
    print('\n[2] Session A loop simulation (in training order)')
    rows = simulate_loop()
    train_count = sum(1 for r in rows if r['action'] == 'TRAIN')
    skip_count = sum(1 for r in rows if r['action'] == 'SKIP')
    header = f'  {"#":>2}  {"seed":>4}  {"feat":<6}  {"variant":<6}  action   path'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for i, r in enumerate(rows, 1):
        marker = '[TRAIN]' if r['action'] == 'TRAIN' else '[SKIP ]'
        path_rel = r['path'].relative_to(REPO)
        print(f'  {i:>2}  {r["seed"]:>4}  {r["feat"]:<6}  {r["variant"]:<6}  {marker}  {path_rel}')
    print(f'\n  TRAIN: {train_count} cells')
    print(f'  SKIP : {skip_count} cells')

    # Check 3: pre-existing GAT cells (not touched by this sweep but shown for context)
    print('\n[3] Pre-existing seed-123 GAT cells (untouched by sweep A)')
    gat_ok, msgs = check_seed123_gat_cells()
    for m in msgs:
        print(m)

    # Summary
    print('\n' + '=' * 72)
    ready = patch_ok and train_count == 4 and skip_count == 0
    if ready:
        print(f'{OK}  READY FOR SWEEP A.')
        print(f'      Will train exactly 4 new cells: GCN/SAGE x BASIC/PAFC at seed 123.')
        print(f'      Estimated compute: ~40h A100 (~471 CU).')
    else:
        print(f'{FAIL}  NOT READY.')
        if not patch_ok:
            print('      Fix: run `python scripts/patch_v4_notebook_factorial.py` first.')
        if train_count != 4:
            print(f'      Unexpected train_count={train_count} (expected 4).')
        if skip_count > 0:
            print(f'      Unexpected skip_count={skip_count} (expected 0 for sweep A).')

    print('=' * 72)
    print('\nPASTE THIS INTO COLAB AFTER CELL 7 (MULTI_SEED config):')
    print('-' * 72)
    print(emit_colab_block())
    print('-' * 72)

    return 0 if ready else 1


if __name__ == '__main__':
    sys.exit(main())
