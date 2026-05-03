"""Factorial run planner for V4 (2 feats x 3 GNN x 3 seeds = 18 runs).

Scans the existing V4 artefacts and prints:
  - what is already on disk
  - what is missing to complete the factorial target
  - two ready-to-paste Colab CONFIG blocks (one per remaining seed)
  - SLURM-ready BSC command lines (if BSC access materialises)

Usage:
  python scripts/factorial_v4_plan.py            # status + paste blocks
  python scripts/factorial_v4_plan.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V4_ROOT = REPO / 'models' / 'output' / 'V4_GNN_TAT_Models'

TARGET_SEEDS = [42, 123, 456]
TARGET_FEATURES = ['BASIC', 'PAFC']
TARGET_VARIANTS = ['GAT', 'GCN', 'SAGE']
HORIZON = 12


def predictions_path(seed: int, feat: str, variant: str) -> Path:
    """Canonical location of predictions.npy for a factorial cell."""
    if seed == 42:
        base = V4_ROOT / 'map_exports'
    else:
        base = V4_ROOT / f'SEED{seed}' / 'map_exports'
    return base / f'H{HORIZON}' / feat / f'GNN_TAT_{variant}' / 'predictions.npy'


def inventory() -> list[dict]:
    rows = []
    for seed in TARGET_SEEDS:
        for feat in TARGET_FEATURES:
            for variant in TARGET_VARIANTS:
                p = predictions_path(seed, feat, variant)
                rows.append({
                    'seed': seed,
                    'feat': feat,
                    'variant': variant,
                    'path': str(p.relative_to(REPO)),
                    'exists': p.exists() and p.stat().st_size > 0,
                    'size_mb': round(p.stat().st_size / 1024 / 1024, 2) if p.exists() else 0,
                })
    return rows


def print_human(rows: list[dict]) -> None:
    print('=' * 72)
    print(f'V4 factorial: {len(TARGET_FEATURES)} feats x {len(TARGET_VARIANTS)} GNN x '
          f'{len(TARGET_SEEDS)} seeds = {len(rows)} runs  H={HORIZON}')
    print('=' * 72)
    print(f'{"seed":>5}  {"feat":<6}  {"variant":<6}  {"status":<10}  path')
    for r in rows:
        status = 'ON DISK' if r['exists'] else 'MISSING'
        marker = ' [OK] ' if r['exists'] else ' [??] '
        print(f'{r["seed"]:>5}  {r["feat"]:<6}  {r["variant"]:<6}  {marker}{status:<4}  {r["path"]}')

    missing = [r for r in rows if not r['exists']]
    done = [r for r in rows if r['exists']]
    print('-' * 72)
    print(f'Done:     {len(done):>2}/18')
    print(f'Missing:  {len(missing):>2}/18')

    if not missing:
        print('\n[ALL DONE] factorial already complete.')
        return

    # Group missing by seed
    by_seed: dict[int, list[dict]] = {}
    for r in missing:
        by_seed.setdefault(r['seed'], []).append(r)

    print('\n' + '=' * 72)
    print('READY-TO-PASTE COLAB CONFIGS  (one sweep per seed)')
    print('=' * 72)
    for seed, items in by_seed.items():
        feats = sorted({r['feat'] for r in items})
        variants = sorted({r['variant'] for r in items})
        est_hours = len(items) * 10  # ~10 h/run on A100
        est_cu = round(est_hours * 11.77)  # A100 Colab Pro+ burn rate
        print(f'\n# --- seed {seed}: {len(items)} runs ({est_hours}h A100, ~{est_cu} CU) ---')
        print('MULTI_SEED = True')
        print(f'SEEDS = [{seed}]')
        print('SKIP_EXISTING = True         # protects the 10 already-trained cells')
        print("CONFIG['enabled_features'] = " + json.dumps(feats))
        print("CONFIG['enabled_variants'] = " + json.dumps(variants))
        # Show what cells this will generate
        print(f'# Will produce:')
        for r in items:
            print(f'#   - {r["path"]}')

    print('\n' + '=' * 72)
    print('BSC SLURM TEMPLATE  (one script per seed)')
    print('=' * 72)
    for seed, items in by_seed.items():
        print(f'\n# sbatch run_v4_factorial_seed{seed}.sh')
        print(f'# 1x H100, est. {len(items)*5}h wall-clock')
        print(f'#SBATCH --job-name=v4_seed{seed}')
        print(f'#SBATCH --gres=gpu:h100:1')
        print(f'#SBATCH --time={len(items)*5+2}:00:00')
        print(f'python run_v4_factorial.py --seed {seed} \\')
        print(f'    --features {" ".join(sorted({r["feat"] for r in items}))} \\')
        print(f'    --variants {" ".join(sorted({r["variant"] for r in items}))}')

    print('\n' + '=' * 72)
    print('TOTAL REMAINING: '
          f'{len(missing)} runs, ~{len(missing)*10} A100-h, ~{round(len(missing)*10*11.77)} CU')
    print('=' * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', action='store_true', help='emit JSON instead of human-readable')
    args = ap.parse_args()

    rows = inventory()
    if args.json:
        print(json.dumps({
            'target_cells': len(rows),
            'done': sum(1 for r in rows if r['exists']),
            'missing': sum(1 for r in rows if not r['exists']),
            'cells': rows,
        }, indent=2))
    else:
        print_human(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
