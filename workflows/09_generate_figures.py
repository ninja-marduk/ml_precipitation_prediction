"""
Pipeline Stage 09: Generate Publication Figures

Generates all publication-quality figures from benchmark results.
Wraps scripts/benchmark/12 and 16 into a single CLI entry point.

Source: scripts/benchmark/12_generate_benchmark_figures.py
        scripts/benchmark/16_generate_new_metrics_figures.py

Usage:
    python workflows/09_generate_figures.py
    python workflows/09_generate_figures.py --config workflows/config.yaml
    python workflows/09_generate_figures.py --only new-metrics
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _versions import log_environment, log_script_version

FIGURE_SCRIPTS = {
    'benchmark': {
        'script': 'scripts/benchmark/12_generate_benchmark_figures.py',
        'description': 'Benchmark comparison figures (horizon, heatmap, radar, Pareto)',
    },
    'new-metrics': {
        'script': 'scripts/benchmark/16_generate_new_metrics_figures.py',
        'description': 'ACC, FSS, elevation-stratified figures + LaTeX tables',
    },
}


def run_script(script_path, description, extra_args=None):
    """Run a figure generation script."""
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        logger.warning(f"Script not found: {full_path}")
        return False

    cmd = [sys.executable, str(full_path)]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Generating: {description}")
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
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--only', type=str, default=None,
                        help='Generate only: benchmark, new-metrics')
    parser.add_argument('--intracell-dem', action='store_true',
                        help='Generate figures for intra-cell DEM predictions (Paper 5)')
    parser.add_argument('--bundle', type=str, default='BASIC_D10',
                        choices=['BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS'],
                        help='Feature bundle for --intracell-dem (default: BASIC_D10)')
    args = parser.parse_args()

    log_environment(logger, ['numpy', 'pandas', 'matplotlib', 'seaborn'])
    log_script_version(logger, __file__)

    # Forward intracell-dem flags to figure scripts
    forward_args = []
    if args.intracell_dem:
        forward_args = ['--intracell-dem', '--bundle', args.bundle]

    logger.info("=" * 60)
    logger.info("  FIGURE GENERATION PIPELINE")
    if args.intracell_dem:
        logger.info(f"  Mode: INTRACELL DEM ({args.bundle})")
    logger.info("=" * 60)

    if args.only:
        if args.only in FIGURE_SCRIPTS:
            info = FIGURE_SCRIPTS[args.only]
            run_script(info['script'], info['description'], forward_args or None)
        else:
            logger.error(f"Unknown figure set: {args.only}")
            logger.info(f"Available: {list(FIGURE_SCRIPTS.keys())}")
            sys.exit(1)
    else:
        for name, info in FIGURE_SCRIPTS.items():
            run_script(info['script'], info['description'], forward_args or None)

    logger.info("Figure generation complete.")
    logger.info(f"Output dirs:")
    logger.info(f"  Benchmark: docs/papers/4/figures/")
    logger.info(f"  New metrics: scripts/benchmark/output/figures/")
    logger.info(f"  LaTeX tables: scripts/benchmark/output/tables/")


if __name__ == "__main__":
    main()
