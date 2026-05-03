"""Reproducibility pipeline for feature 002 (full-horizon multi-seed).

Single entry point for:
  1. V2 seed-42 inference (from h5 checkpoint)
  2. V10 seed-42 Ridge fusion re-run
  3. Drift audit gate
  4. Per-seed per-horizon metrics (V2, V4, V10 x {42,123,456})
  5. Per-model consolidation (mean/std/count)
  6. Cross-model unification
  7. Horizon-degradation figure
  8. Paper 5 literal substitution (root + delivery)
  9. EGU26 poster literal substitution
 10. Overleaf ZIP rebuild

Phase-flag CLI: `--phase <name>` runs a single phase. `--phase all` runs the full
sequence, stopping at the drift gate unless `--accept-drift` is set.

See `.specify/specs/002-full-horizon-multiseed/` for the full spec.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parent.parent
# Make `scripts` importable as a package regardless of CWD
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = REPO / 'models' / 'output'
V2_DIR = OUT / 'V2_Enhanced_Models'
V4_DIR = OUT / 'V4_GNN_TAT_Models'
V10_DIR = OUT / 'V10_Late_Fusion'

DOCS_P5 = REPO / '.docs' / 'papers' / '5'
DOCS_P5_DELIVERY = DOCS_P5 / 'delivery'
DOCS_P5_FIGURES = DOCS_P5 / 'figures'
DOCS_P5_DATA = DOCS_P5 / 'data'
POSTER_TEX = REPO / '.docs' / 'conferences' / 'EGU26' / 'poster' / 'poster.tex'

SPEC_DIR = REPO / '.specify' / 'specs' / '002-full-horizon-multiseed'
LITERALS_JSON = SPEC_DIR / 'literals_to_refresh.json'
DRIFT_LOG = SPEC_DIR / 'drift_audit.log'

ZIP_OUT = Path(r'd:/tmp/paper5_joh_overleaf.zip')

SEEDS = (42, 123, 456)
LEGACY_R2 = 0.668
SANITY_BOUND_PCT = 20.0

# --------------------------------------------------------------------------- #
# Phase registry
# --------------------------------------------------------------------------- #

PHASES = [
    'v2-inference',
    'v10-fusion',
    'drift-audit',
    'per-seed-metrics',
    'consolidate',
    'unify',
    'figure',
    'paper-update',
    'poster-update',
    'zip',
    # Feature 003 factorial (spec/003-factorial-feat-variant)
    'factorial-per-cell',
    'factorial-consolidate',
    'factorial-aggregate',
    'factorial-stats',
    'factorial-figure',
]


# --------------------------------------------------------------------------- #
# Phase implementations - populated by subsequent tasks (T010, T012, T014, ...)
# --------------------------------------------------------------------------- #

def phase_v2_inference(args: argparse.Namespace) -> int:
    """T010: regenerate V2 seed-42 full-grid predictions via h5 inference."""
    from scripts._phases.v2_inference import run  # type: ignore
    return run(args)


def phase_v10_fusion(args: argparse.Namespace) -> int:
    """T012: re-run V10 Ridge fusion for seed 42 with new V2 inputs."""
    from scripts._phases.v10_fusion import run  # type: ignore
    return run(args)


def phase_drift_audit(args: argparse.Namespace) -> int:
    """T014: compare refreshed R² to legacy; halt if |drift|>±20%."""
    from scripts._phases.drift_audit import run  # type: ignore
    return run(args)


def phase_per_seed_metrics(args: argparse.Namespace) -> int:
    """T021: compute per-horizon metrics for every (model, seed)."""
    from scripts._phases.per_seed_metrics import run  # type: ignore
    return run(args)


def phase_consolidate(args: argparse.Namespace) -> int:
    """T025: aggregate per-seed CSVs to mean/std/count per model."""
    from scripts._phases.consolidate import run  # type: ignore
    return run(args)


def phase_unify(args: argparse.Namespace) -> int:
    """T029: merge per-model consolidated CSVs into cross-model unified table."""
    from scripts._phases.unify import run  # type: ignore
    return run(args)


def phase_figure(args: argparse.Namespace) -> int:
    """T030: horizon-degradation figure with ±std shaded bands."""
    from scripts._phases.figure import run  # type: ignore
    return run(args)


def phase_paper_update(args: argparse.Namespace) -> int:
    """T040/T042: literal substitution in Paper 5 root+delivery (lockstep)."""
    from scripts._phases.paper_update import run  # type: ignore
    return run(args)


def phase_poster_update(args: argparse.Namespace) -> int:
    """T060: literal substitution in EGU26 poster."""
    from scripts._phases.poster_update import run  # type: ignore
    return run(args)


def phase_zip(args: argparse.Namespace) -> int:
    """T048: regenerate Overleaf ZIP from delivery/."""
    from scripts._phases.zip_bundle import run  # type: ignore
    return run(args)


# -- Feature 003 factorial phases -----------------------------------------

def phase_factorial_per_cell(args: argparse.Namespace) -> int:
    """F003 T020: per-horizon metrics for each of the 18 factorial cells."""
    from scripts._phases.factorial_per_cell_metrics import run  # type: ignore
    return run(args)


def phase_factorial_consolidate(args: argparse.Namespace) -> int:
    """F003 T021: consolidate 18 per-cell CSVs into one 216-row long table."""
    from scripts._phases.factorial_consolidate import run  # type: ignore
    return run(args)


def phase_factorial_aggregate(args: argparse.Namespace) -> int:
    """F003 T022: aggregate across seeds -> factorial_feat_variant.csv + _byhorizon."""
    from scripts._phases.factorial_aggregate import run  # type: ignore
    return run(args)


def phase_factorial_stats(args: argparse.Namespace) -> int:
    """F003 T030: 2-way ANOVA (feat x variant) + Tukey HSD."""
    from scripts._phases.factorial_stats import run  # type: ignore
    return run(args)


def phase_factorial_figure(args: argparse.Namespace) -> int:
    """F003 T031: grouped bar plot (Okabe-Ito, 700 DPI PNG + PDF)."""
    from scripts._phases.factorial_figure import run  # type: ignore
    return run(args)


PHASE_DISPATCH = {
    'v2-inference':     phase_v2_inference,
    'v10-fusion':       phase_v10_fusion,
    'drift-audit':      phase_drift_audit,
    'per-seed-metrics': phase_per_seed_metrics,
    'consolidate':      phase_consolidate,
    'unify':            phase_unify,
    'figure':           phase_figure,
    'paper-update':     phase_paper_update,
    'poster-update':    phase_poster_update,
    'zip':              phase_zip,
    # Feature 003 factorial
    'factorial-per-cell':    phase_factorial_per_cell,
    'factorial-consolidate': phase_factorial_consolidate,
    'factorial-aggregate':   phase_factorial_aggregate,
    'factorial-stats':       phase_factorial_stats,
    'factorial-figure':      phase_factorial_figure,
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Full-horizon multi-seed reproducibility pipeline (feature 002).',
    )
    p.add_argument(
        '--phase', choices=[*PHASES, 'all'], default='all',
        help='Phase to execute (default: all; sequential through the registry).',
    )
    p.add_argument(
        '--from-scratch', action='store_true',
        help='Wipe derived artefacts (regenerated SEED42 dirs, CSVs, figure, ZIP) before running.',
    )
    p.add_argument(
        '--accept-drift', action='store_true',
        help='Proceed past the drift gate even if |R² drift| > 20%%.',
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Audit mode for paper-update (prints hits without writing).',
    )
    p.add_argument(
        '--model', choices=('v2', 'v4', 'v10'), default=None,
        help='Limit per-seed-metrics and consolidate phases to a single model.',
    )
    return p.parse_args(argv)


def run_all(args: argparse.Namespace) -> int:
    """Execute every phase sequentially. Halts on first non-zero."""
    for phase in PHASES:
        args.phase = phase
        print(f'\n━━━ phase: {phase} ━━━')
        rc = PHASE_DISPATCH[phase](args)
        if rc != 0:
            print(f'phase {phase} returned {rc}; halting.')
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.phase == 'all':
        return run_all(args)
    return PHASE_DISPATCH[args.phase](args)


if __name__ == '__main__':
    sys.exit(main())
