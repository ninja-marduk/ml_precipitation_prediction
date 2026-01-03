"""
Master Benchmark Analysis Script

Runs complete V2 vs V3 comparative analysis pipeline.

Usage:
    python scripts/benchmark/run_full_benchmark_analysis.py --verbose
    python scripts/benchmark/run_full_benchmark_analysis.py --skip-extraction
"""

import subprocess
import argparse
import time
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def print_header(message: str):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {message}")
    print(f"{'='*70}\n")


def run_script(script_name: str, verbose: bool = False) -> dict:
    """
    Execute a benchmark script and log results.

    Args:
        script_name: Name of the script file
        verbose: Print detailed output

    Returns:
        Dictionary with execution results
    """
    start = time.time()

    print(f"\n{'-'*70}")
    print(f"Running: {script_name}")
    print(f"{'-'*70}")

    script_path = PROJECT_ROOT / 'scripts' / 'benchmark' / script_name

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return {'success': False, 'elapsed': 0, 'error': 'Script not found'}

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=not verbose,
            text=True,
            cwd=str(PROJECT_ROOT)
        )

        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"[OK] Completed in {elapsed:.1f}s")
            return {'success': True, 'elapsed': elapsed, 'error': None}
        else:
            print(f"[FAIL] Failed with exit code {result.returncode}")
            if not verbose and result.stderr:
                print(f"Error output:\n{result.stderr}")
            return {'success': False, 'elapsed': elapsed, 'error': result.stderr}

    except Exception as e:
        elapsed = time.time() - start
        print(f"[ERROR] Exception occurred: {e}")
        return {'success': False, 'elapsed': elapsed, 'error': str(e)}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run complete V2 vs V3 benchmark analysis pipeline'
    )
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output from each script')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip notebook extraction (use existing data)')
    parser.add_argument('--skip-visualizations', action='store_true',
                        help='Skip visualization generation (faster for testing)')
    parser.add_argument('--core-only', action='store_true',
                        help='Run only core analysis (scripts 1-4), skip viz and tables')

    args = parser.parse_args()

    # Define pipeline scripts
    scripts = [
        ('01_extract_notebook_outputs.py', not args.skip_extraction, 'Data Extraction'),
        ('02_consolidate_metrics.py', True, 'Metrics Consolidation'),
        ('03_training_dynamics_analysis.py', True, 'Training Dynamics'),
        ('04_statistical_tests.py', True, 'Statistical Tests'),
    ]

    # Start pipeline
    print_header("V2 vs V3 BENCHMARK ANALYSIS PIPELINE")
    print(f"Configuration:")
    print(f"  Verbose: {args.verbose}")
    print(f"  Skip extraction: {args.skip_extraction}")
    print(f"  Skip visualizations: {args.skip_visualizations}")
    print(f"  Core only: {args.core_only}")

    total_start = time.time()
    results = []

    # Execute pipeline
    for script_name, should_run, description in scripts:
        if should_run:
            result = run_script(script_name, args.verbose)
            results.append({
                'script': script_name,
                'description': description,
                **result
            })

            if not result['success']:
                print(f"\n[WARNING] Pipeline stopped due to error in {script_name}")
                print(f"Fix the error and re-run the pipeline.")
                sys.exit(1)
        else:
            print(f"\n[SKIP] Skipping: {script_name} ({description})")

    total_elapsed = time.time() - total_start

    # Print summary
    print_header("PIPELINE EXECUTION SUMMARY")

    print("Results:")
    for r in results:
        status = "[OK] Success" if r['success'] else "[FAIL] Failed"
        print(f"  {r['description']:30s} - {status:15s} ({r['elapsed']:.1f}s)")

    print(f"\nTotal execution time: {total_elapsed/60:.1f} minutes")

    # Output locations
    print("\nOutput Locations:")
    print(f"  Data:    {PROJECT_ROOT / 'docs' / 'models' / 'comparative' / 'data'}")
    print(f"  Tables:  {PROJECT_ROOT / 'docs' / 'models' / 'comparative' / 'tables'}")
    print(f"  Figures: {PROJECT_ROOT / 'docs' / 'models' / 'comparative' / 'figures'}")

    print("\n" + "="*70)
    print("  [OK] BENCHMARK ANALYSIS COMPLETE")
    print("="*70)

    # Success summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)

    if successful == total:
        print(f"\nAll {total} scripts completed successfully!")
        sys.exit(0)
    else:
        print(f"\nCompleted {successful}/{total} scripts successfully.")
        sys.exit(1)


if __name__ == '__main__':
    main()
