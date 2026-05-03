"""Migrate intracell DEM outputs to new model/feature_set/date_run structure.

Old structure:
  V2_Enhanced_Models_intracell_dem/h12/BASIC_D10/training_metrics/...
  V2_Enhanced_Models_intracell_dem/map_exports/H12/BASIC_D10/ConvLSTM/...
  V4_GNN_TAT_Models_intracell_dem/h12/BASIC_D10/training_metrics/...
  V4_GNN_TAT_Models_intracell_dem/map_exports/H12/BASIC_D10/GNN_TAT_GAT/...
  V10_Late_Fusion_intracell_dem/BASIC_D10/...

New structure:
  intracell_dem/ConvLSTM/BASIC_D10/20260301_1/training_metrics/...
  intracell_dem/ConvLSTM/BASIC_D10/20260301_1/predictions.npy
  intracell_dem/GNN_TAT_GAT/BASIC_D10/20260301_1/training_metrics/...
  intracell_dem/GNN_TAT_GAT/BASIC_D10/20260301_1/predictions.npy
  intracell_dem/Late_Fusion/BASIC_D10/20260302_1/predictions.npy

Usage:
  python scripts/migrate_intracell_outputs.py [--dry-run]
"""
import shutil
import json
import sys
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent / 'models' / 'output'
NEW_ROOT = BASE / 'intracell_dem'

DRY_RUN = '--dry-run' in sys.argv

# Migration mapping: (old_dir_pattern, model_name, run_date)
MIGRATIONS = [
    # V2 ConvLSTM
    {
        'model': 'ConvLSTM',
        'bundles': ['BASIC_D10', 'BASIC_PCA6'],
        'run_id': '20260301_1',
        'training_src': lambda b: BASE / 'V2_Enhanced_Models_intracell_dem' / 'h12' / b / 'training_metrics',
        'predictions_src': lambda b: BASE / 'V2_Enhanced_Models_intracell_dem' / 'map_exports' / 'H12' / b / 'ConvLSTM',
        'metrics_csv': BASE / 'V2_Enhanced_Models_intracell_dem' / 'metrics_spatial_v2_intracell_dem_h12.csv',
    },
    # V4 GNN-TAT
    {
        'model': 'GNN_TAT_GAT',
        'bundles': ['BASIC_D10', 'BASIC_PCA6'],
        'run_id': '20260301_1',
        'training_src': lambda b: BASE / 'V4_GNN_TAT_Models_intracell_dem' / 'h12' / b / 'training_metrics',
        'predictions_src': lambda b: BASE / 'V4_GNN_TAT_Models_intracell_dem' / 'map_exports' / 'H12' / b / 'GNN_TAT_GAT',
        'metrics_csv': BASE / 'V4_GNN_TAT_Models_intracell_dem' / 'metrics_spatial_v4_intracell_dem_h12.csv',
    },
    # V10 Late Fusion
    {
        'model': 'Late_Fusion',
        'bundles': ['BASIC', 'BASIC_D10', 'BASIC_PCA6'],
        'run_id': '20260302_1',
        'training_src': None,
        'predictions_src': lambda b: BASE / 'V10_Late_Fusion_intracell_dem' / b,
        'metrics_csv': None,
    },
]


def copy_tree(src, dst):
    """Copy directory contents, creating dst if needed."""
    if not src.exists():
        print(f'  SKIP (not found): {src}')
        return 0
    count = 0
    for item in src.iterdir():
        target = dst / item.name
        if DRY_RUN:
            print(f'  [DRY] {item} -> {target}')
        else:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
        count += 1
    return count


def copy_file(src, dst_dir, new_name=None):
    """Copy a single file."""
    if not src.exists():
        print(f'  SKIP (not found): {src}')
        return 0
    name = new_name or src.name
    target = dst_dir / name
    if DRY_RUN:
        print(f'  [DRY] {src} -> {target}')
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return 1


def main():
    total_files = 0
    print(f'Migration {"(DRY RUN)" if DRY_RUN else "(LIVE)"}\n')
    print(f'Source: {BASE}')
    print(f'Target: {NEW_ROOT}\n')

    for m in MIGRATIONS:
        model = m['model']
        run_id = m['run_id']

        for bundle in m['bundles']:
            dst = NEW_ROOT / model / bundle / run_id
            print(f'\n--- {model} / {bundle} / {run_id} ---')

            # Training metrics
            if m['training_src']:
                src = m['training_src'](bundle)
                tm_dst = dst / 'training_metrics'
                n = copy_tree(src, tm_dst)
                total_files += n
                print(f'  training_metrics: {n} files')

            # Predictions (predictions.npy, targets.npy, metadata.json)
            pred_src = m['predictions_src'](bundle)
            n = copy_tree(pred_src, dst)
            total_files += n
            print(f'  predictions: {n} files')

            # Metrics CSV (copy to run dir)
            if m['metrics_csv'] and m['metrics_csv'].exists():
                n = copy_file(m['metrics_csv'], dst)
                total_files += n

            # Generate migration metadata
            meta = {
                'model': model,
                'feature_set': bundle,
                'run_id': run_id,
                'migrated_from': {
                    'training': str(m['training_src'](bundle)) if m['training_src'] else None,
                    'predictions': str(m['predictions_src'](bundle)),
                },
                'migrated_at': datetime.now().isoformat(),
            }
            if not DRY_RUN:
                dst.mkdir(parents=True, exist_ok=True)
                with open(dst / 'migration_info.json', 'w') as f:
                    json.dump(meta, f, indent=2)
                total_files += 1

    print(f'\n{"="*50}')
    print(f'Total files: {total_files}')

    if not DRY_RUN:
        # Rename old directories
        old_dirs = [
            'V2_Enhanced_Models_intracell_dem',
            'V4_GNN_TAT_Models_intracell_dem',
            'V10_Late_Fusion_intracell_dem',
        ]
        for d in old_dirs:
            old = BASE / d
            renamed = BASE / f'{d}_old'
            if old.exists():
                if renamed.exists():
                    print(f'  WARNING: {renamed} already exists, skipping rename')
                else:
                    old.rename(renamed)
                    print(f'  Renamed: {old.name} -> {renamed.name}')

        print(f'\nMigration complete! Old dirs renamed to *_old.')
        print('Verify results, then delete *_old dirs manually.')
    else:
        print('\nDry run complete. Run without --dry-run to execute.')


if __name__ == '__main__':
    main()
