"""
Pipeline Stage 08: Benchmark Metrics

Runs all benchmark evaluation scripts on model predictions:
- Consolidated metrics (R2, RMSE, MAE per horizon)
- Statistical tests (Friedman + Nemenyi)
- Spatiotemporal metrics (ACC + FSS)
- Elevation-stratified analysis

Source: scripts/benchmark/02, 04, 13, 14, 15

Usage:
    python workflows/08_benchmark_metrics.py
    python workflows/08_benchmark_metrics.py --config workflows/config.yaml
    python workflows/08_benchmark_metrics.py --skip-statistical
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import subprocess
from pathlib import Path
import logging

import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_SCRIPTS = {
    'spatiotemporal': {
        'script': 'scripts/benchmark/14_spatiotemporal_metrics.py',
        'description': 'ACC + FSS metrics',
    },
    'elevation': {
        'script': 'scripts/benchmark/15_elevation_stratified.py',
        'description': 'Elevation-stratified R2/RMSE',
    },
    'tables_figures': {
        'script': 'scripts/benchmark/16_generate_new_metrics_figures.py',
        'description': 'LaTeX tables + PDF figures',
    },
}

# Optional scripts (may not exist or may require additional data)
OPTIONAL_SCRIPTS = {
    'consolidate': {
        'script': 'scripts/benchmark/02_consolidate_metrics.py',
        'description': 'Consolidate V2/V3/V4 metrics',
    },
    'statistical': {
        'script': 'scripts/benchmark/04_statistical_tests.py',
        'description': 'Paired t-tests + Cohen\'s d',
    },
    'friedman': {
        'script': 'scripts/benchmark/13_statistical_analysis.py',
        'description': 'Friedman + Nemenyi tests',
    },
}


def run_script(script_path, description, extra_args=None):
    """Run a Python script and report success/failure."""
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        logger.warning(f"Script not found: {full_path}")
        return False

    cmd = [sys.executable, str(full_path)]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Running: {description} ({script_path})")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )

    if result.returncode != 0:
        logger.error(f"FAILED: {description}")
        logger.error(result.stderr[-500:] if result.stderr else "No error output")
        return False

    logger.info(f"OK: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run benchmark metrics')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--skip-statistical', action='store_true',
                        help='Skip statistical tests')
    parser.add_argument('--only', type=str, default=None,
                        help='Run only specific benchmark: spatiotemporal, elevation, tables_figures')
    parser.add_argument('--intracell-dem', action='store_true',
                        help='Evaluate intra-cell DEM predictions (Paper 5)')
    parser.add_argument('--bundle', type=str, default='BASIC_D10',
                        choices=['BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS'],
                        help='Feature bundle for --intracell-dem (default: BASIC_D10)')
    args = parser.parse_args()

    # Forward intracell-dem flags to benchmark scripts
    forward_args = []
    if args.intracell_dem:
        forward_args = ['--intracell-dem', '--bundle', args.bundle]

    logger.info("=" * 60)
    logger.info("  BENCHMARK METRICS PIPELINE")
    if args.intracell_dem:
        logger.info(f"  Mode: INTRACELL DEM ({args.bundle})")
    logger.info("=" * 60)

    results = {}

    # Run main benchmark scripts
    if args.only:
        if args.only in BENCHMARK_SCRIPTS:
            info = BENCHMARK_SCRIPTS[args.only]
            results[args.only] = run_script(info['script'], info['description'], forward_args or None)
        else:
            logger.error(f"Unknown benchmark: {args.only}")
            logger.info(f"Available: {list(BENCHMARK_SCRIPTS.keys())}")
            sys.exit(1)
    else:
        for name, info in BENCHMARK_SCRIPTS.items():
            results[name] = run_script(info['script'], info['description'], forward_args or None)

        # Optional statistical tests
        if not args.skip_statistical:
            for name, info in OPTIONAL_SCRIPTS.items():
                results[name] = run_script(info['script'], info['description'])

    # Summary
    print('\n' + '=' * 60)
    print('  BENCHMARK RESULTS')
    print('=' * 60)
    for name, success in results.items():
        status = 'PASS' if success else 'FAIL'
        print(f"  [{status}] {name}")
    print('=' * 60)

    n_pass = sum(results.values())
    n_total = len(results)
    logger.info(f"Completed: {n_pass}/{n_total} scripts passed")


if __name__ == "__main__":
    main()
