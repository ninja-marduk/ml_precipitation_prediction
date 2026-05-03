"""
ML Precipitation Prediction - End-to-End Pipeline Orchestrator

Runs the complete pipeline from data download to benchmark evaluation.
Each stage can be run independently or as part of the full pipeline.

Usage:
    python workflows/run_pipeline.py                    # Run all stages
    python workflows/run_pipeline.py --stages 7 8 9     # Run specific stages
    python workflows/run_pipeline.py --from 7            # Run from stage 7 onwards
    python workflows/run_pipeline.py --dry-run           # Show what would run
    python workflows/run_pipeline.py --mini              # Use mini dataset (demo)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import logging

import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = Path(__file__).resolve().parent

# Pipeline stages in execution order
STAGES = {
    1: {
        'script': '01_download_chirps.py',
        'name': 'Download CHIRPS',
        'description': 'Download and crop CHIRPS daily precipitation data',
        'gpu': False,
    },
    2: {
        'script': '02_aggregate_monthly.py',
        'name': 'Aggregate Monthly',
        'description': 'Aggregate daily precipitation to monthly totals',
        'gpu': False,
    },
    3: {
        'script': '03_merge_dem.py',
        'name': 'Merge DEM',
        'description': 'Merge DEM elevation data with precipitation grid',
        'gpu': False,
    },
    4: {
        'script': '04_feature_engineering.py',
        'name': 'Feature Engineering',
        'description': 'Elevation and precipitation clustering',
        'gpu': False,
    },
    5: {
        'script': '05_train_v2_convlstm.py',
        'name': 'Train V2 ConvLSTM',
        'description': 'Train V2 Enhanced ConvLSTM (requires GPU)',
        'gpu': True,
    },
    6: {
        'script': '06_train_v4_gnn_tat.py',
        'name': 'Train V4 GNN-TAT',
        'description': 'Train V4 GNN with Temporal Attention (requires GPU)',
        'gpu': True,
    },
    7: {
        'script': '07_late_fusion_v10.py',
        'name': 'V10 Late Fusion',
        'description': 'Ridge regression fusion of V2 + V4 predictions',
        'gpu': False,
    },
    8: {
        'script': '08_benchmark_metrics.py',
        'name': 'Benchmark Metrics',
        'description': 'Compute ACC, FSS, elevation-stratified metrics',
        'gpu': False,
    },
    9: {
        'script': '09_generate_figures.py',
        'name': 'Generate Figures',
        'description': 'Generate publication-quality figures and LaTeX tables',
        'gpu': False,
    },
}


def run_stage(stage_num, stage_info, config_path=None, extra_args=None, dry_run=False):
    """Run a single pipeline stage."""
    script_path = WORKFLOWS_DIR / stage_info['script']

    if not script_path.exists():
        logger.error(f"Stage {stage_num} script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if config_path:
        cmd.extend(['--config', config_path])
    if extra_args:
        cmd.extend(extra_args)

    gpu_tag = ' [GPU]' if stage_info['gpu'] else ''
    logger.info(f"\n{'='*60}")
    logger.info(f"  STAGE {stage_num}: {stage_info['name']}{gpu_tag}")
    logger.info(f"  {stage_info['description']}")
    logger.info(f"{'='*60}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(f"  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    logger.info(f"  COMPLETED in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='ML Precipitation Prediction - Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1  Download CHIRPS daily data
  2  Aggregate to monthly totals
  3  Merge DEM elevation data
  4  Feature engineering (clustering)
  5  Train V2 ConvLSTM [GPU]
  6  Train V4 GNN-TAT [GPU]
  7  V10 Late Fusion (Ridge)
  8  Benchmark metrics (ACC, FSS, elevation)
  9  Generate figures and LaTeX tables

Examples:
  python workflows/run_pipeline.py --from 7          # V10 + benchmarks only
  python workflows/run_pipeline.py --stages 8 9      # Metrics + figures only
  python workflows/run_pipeline.py --skip-gpu         # Skip GPU stages
        """
    )
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--stages', type=int, nargs='+', default=None,
                        help='Specific stages to run (e.g., --stages 7 8 9)')
    parser.add_argument('--from', type=int, default=None, dest='from_stage',
                        help='Run from this stage onwards')
    parser.add_argument('--skip-gpu', action='store_true',
                        help='Skip stages that require GPU (5, 6)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without executing')
    parser.add_argument('--mini', action='store_true',
                        help='Use mini dataset for testing')
    parser.add_argument('--intracell-dem', action='store_true',
                        help='Use intra-cell DEM models (Paper 5) instead of Paper 4 defaults')
    parser.add_argument('--bundle', type=str, default='BASIC_D10',
                        choices=['BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS'],
                        help='Feature bundle for --intracell-dem (default: BASIC_D10)')
    args = parser.parse_args()

    # Determine which stages to run
    if args.stages:
        stages_to_run = sorted(args.stages)
    elif args.from_stage:
        stages_to_run = [s for s in STAGES if s >= args.from_stage]
    else:
        stages_to_run = list(STAGES.keys())

    if args.skip_gpu:
        stages_to_run = [s for s in stages_to_run if not STAGES[s]['gpu']]

    config_path = args.config
    if config_path is None:
        default_config = WORKFLOWS_DIR / 'config.yaml'
        if default_config.exists():
            config_path = str(default_config)

    # Build intracell-dem extra args (only for stages >= 7)
    intracell_extra = []
    if args.intracell_dem:
        intracell_extra = ['--intracell-dem', '--bundle', args.bundle]

    # Print pipeline plan
    print('\n' + '=' * 60)
    print('  ML PRECIPITATION PREDICTION PIPELINE')
    print('=' * 60)
    print(f'  Config: {config_path or "defaults"}')
    print(f'  Stages: {stages_to_run}')
    if args.intracell_dem:
        print(f'  Mode: INTRACELL DEM (Paper 5)')
        print(f'  Bundle: {args.bundle}')
    if args.dry_run:
        print('  Mode: DRY RUN')
    if args.mini:
        print('  Dataset: MINI (demo)')
    print('=' * 60)

    # Execute stages
    results = {}
    pipeline_start = time.time()

    for stage_num in stages_to_run:
        stage_info = STAGES[stage_num]
        extra_args = ['--dry-run'] if (args.mini and stage_info['gpu']) else []

        # Pass intracell-dem flags to stages 7+ (fusion, benchmarks, figures)
        if stage_num >= 5 and intracell_extra:
            extra_args.extend(intracell_extra)

        success = run_stage(stage_num, stage_info, config_path, extra_args or None, args.dry_run)
        results[stage_num] = success

        if not success and not args.dry_run:
            logger.error(f"Pipeline stopped at stage {stage_num}")
            break

    pipeline_elapsed = time.time() - pipeline_start

    # Summary
    print('\n' + '=' * 60)
    print('  PIPELINE SUMMARY')
    print('=' * 60)
    for stage_num, success in results.items():
        status = 'PASS' if success else 'FAIL'
        name = STAGES[stage_num]['name']
        print(f"  [{status}] Stage {stage_num}: {name}")
    print(f"\n  Total time: {pipeline_elapsed:.1f}s")
    print('=' * 60)

    # Exit code
    if all(results.values()):
        logger.info("Pipeline completed successfully.")
        sys.exit(0)
    else:
        logger.error("Pipeline completed with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
