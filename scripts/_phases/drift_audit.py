"""T014 - Drift audit gate.

Compares the refreshed V10 seed-42 Ridge OOF R² against the legacy literal
`0.668` reported in Paper 5. Exits non-zero if |relative drift| > 20% unless
`--accept-drift` is set. Writes the audit outcome to
`.specify/specs/002-full-horizon-multiseed/drift_audit.log` for FR-013
traceability (T076 consumes it).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V10_SEED42_SUMMARY = REPO / 'models' / 'output' / 'V10_Late_Fusion' / 'SEED42' / 'v10_summary.json'
DRIFT_LOG = REPO / '.specify' / 'specs' / '002-full-horizon-multiseed' / 'drift_audit.log'

LEGACY_R2 = 0.668
BOUND_PCT = 20.0


def run(args: argparse.Namespace) -> int:
    print('[T014] Drift audit (V10 seed-42 R^2 vs legacy 0.668)')

    if not V10_SEED42_SUMMARY.exists():
        print(f'  MISSING: {V10_SEED42_SUMMARY}')
        print('  Run phase v10-fusion first.')
        return 1

    s = json.loads(V10_SEED42_SUMMARY.read_text(encoding='utf-8'))
    refreshed_r2 = float(s['results']['ridge_oof']['R2'])
    refreshed_rmse = float(s['results']['ridge_oof']['RMSE'])
    w_v2 = float(s['learned_weights']['w_v2'])
    w_v4 = float(s['learned_weights']['w_v4'])
    delta_abs = refreshed_r2 - LEGACY_R2
    delta_pct = delta_abs / LEGACY_R2 * 100.0

    lines = [
        f'timestamp: {datetime.now().isoformat()}',
        f'legacy_R2:    {LEGACY_R2:.4f}',
        f'refreshed_R2: {refreshed_r2:.4f}',
        f'delta_abs:    {delta_abs:+.4f}',
        f'delta_pct:    {delta_pct:+.2f}%',
        f'bound:        +/-{BOUND_PCT}%',
        f'refreshed_RMSE: {refreshed_rmse:.2f} mm',
        f'refreshed_weights: w_v2={w_v2:.4f}  w_v4={w_v4:.4f}',
    ]
    within_bound = abs(delta_pct) <= BOUND_PCT
    lines.append(f'status: {"WITHIN_BOUND" if within_bound else "DRIFT_ALERT"}')
    if not within_bound and not args.accept_drift:
        lines.append('action:  HALT - re-invoke with --accept-drift to proceed')
    elif not within_bound and args.accept_drift:
        lines.append('action:  PROCEED (--accept-drift)')
    else:
        lines.append('action:  PROCEED')

    for l in lines:
        print('  ' + l)

    DRIFT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_LOG.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'  wrote: {DRIFT_LOG}')

    if not within_bound and not args.accept_drift:
        print('')
        print('  DRIFT_ALERT')
        print(f'  legacy_R2    = {LEGACY_R2}')
        print(f'  refreshed_R2 = {refreshed_r2}')
        print(f'  delta_pct    = {delta_pct:+.2f}%')
        print('  Bound exceeded. Halting before paper artefacts are modified.')
        print('  Re-invoke with --accept-drift to proceed anyway.')
        return 10

    print('[T014] Drift audit passed.')
    return 0


if __name__ == '__main__':
    import argparse as ap
    p = ap.ArgumentParser()
    p.add_argument('--accept-drift', action='store_true')
    sys.exit(run(p.parse_args()))
