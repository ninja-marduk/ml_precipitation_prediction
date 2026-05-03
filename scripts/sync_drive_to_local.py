"""Sync Colab training outputs from Google Drive (local mirror) to the repo.

When Colab runs V2/V4/V10 training, outputs land in
  /content/drive/MyDrive/ml_precipitation_prediction/models/output/...
which on Windows is mirrored at
  G:\\Mi unidad\\ml_precipitation_prediction\\models\\output\\...

This script copies NEW-or-NEWER files from that mirror into the local repo at
  d:\\github.com\\ninja-marduk\\ml_precipitation_prediction\\models\\output\\...

Guarantees:
  - NEVER deletes anything on either side.
  - Only touches files under `models/output/` (training artefacts).
  - Default: copies only training-output suffixes (.npy, .json, .csv, .pt, .h5).
  - Compares mtime + size; files already in sync are skipped silently.

Usage:
  python scripts/sync_drive_to_local.py                        # V2 + V4 + V10 all seeds
  python scripts/sync_drive_to_local.py --model V4             # V4 only
  python scripts/sync_drive_to_local.py --model V4 --seed 123  # V4 SEED123 only
  python scripts/sync_drive_to_local.py --dry-run              # show what would copy
  python scripts/sync_drive_to_local.py --verbose              # print every skipped file
  python scripts/sync_drive_to_local.py --drive-root "G:\\Mi unidad\\..."  # override
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DRIVE_ROOT = Path(r'G:\Mi unidad\ml_precipitation_prediction')
DEFAULT_LOCAL_ROOT = REPO_ROOT

MODEL_DIRS = {
    'V2':  'models/output/V2_Enhanced_Models',
    'V4':  'models/output/V4_GNN_TAT_Models',
    'V10': 'models/output/V10_Late_Fusion',
}

COPIED_SUFFIXES = {'.npy', '.json', '.csv', '.pt', '.h5', '.png', '.txt'}


@dataclass
class SyncAction:
    src: Path
    dst: Path
    reason: str      # 'new', 'newer', 'size-diff'
    bytes: int


@dataclass
class SyncReport:
    planned: list[SyncAction]
    skipped: int
    errors: list[str]

    @property
    def total_bytes(self) -> int:
        return sum(a.bytes for a in self.planned)


def should_sync(src: Path, dst: Path) -> tuple[bool, str]:
    """Return (should_copy, reason)."""
    if not dst.exists():
        return True, 'new'
    src_stat = src.stat()
    dst_stat = dst.stat()
    if src_stat.st_size != dst_stat.st_size:
        return True, 'size-diff'
    if src_stat.st_mtime > dst_stat.st_mtime + 1.0:  # 1 sec tolerance
        return True, 'newer'
    return False, 'up-to-date'


def plan_sync(drive_subdir: Path, local_subdir: Path, seed_filter: int | None,
              verbose: bool) -> SyncReport:
    """Walk drive_subdir and plan actions relative to local_subdir."""
    planned: list[SyncAction] = []
    errors: list[str] = []
    skipped = 0

    if not drive_subdir.exists():
        errors.append(f'Drive path missing: {drive_subdir}')
        return SyncReport(planned=[], skipped=0, errors=errors)

    for src in drive_subdir.rglob('*'):
        if not src.is_file():
            continue
        if src.suffix.lower() not in COPIED_SUFFIXES:
            if verbose:
                print(f'    [SKIP suffix] {src.name}')
            skipped += 1
            continue

        rel = src.relative_to(drive_subdir)

        # Optional seed filter: keep only files whose path contains SEED{seed}
        # or files in the SEED42 legacy root (map_exports/ directly).
        if seed_filter is not None:
            seed_token = f'SEED{seed_filter}'
            rel_parts = set(rel.parts)
            if seed_filter == 42:
                # seed 42 lives at the root map_exports/ for V4 (legacy)
                if seed_token not in rel_parts and 'map_exports' not in rel_parts:
                    skipped += 1
                    continue
            else:
                if seed_token not in rel_parts:
                    skipped += 1
                    continue

        dst = local_subdir / rel
        copy_flag, reason = should_sync(src, dst)
        if copy_flag:
            planned.append(SyncAction(src=src, dst=dst, reason=reason,
                                       bytes=src.stat().st_size))
        else:
            if verbose:
                print(f'    [up-to-date] {rel}')
            skipped += 1

    return SyncReport(planned=planned, skipped=skipped, errors=errors)


def apply_plan(report: SyncReport, dry_run: bool) -> int:
    """Execute the sync plan. Returns count of files actually copied."""
    copied = 0
    for action in report.planned:
        action.dst.parent.mkdir(parents=True, exist_ok=True)
        marker = '[DRY]' if dry_run else '[COPY]'
        size_mb = action.bytes / 1024 / 1024
        rel = action.src.relative_to(action.src.parents[len(action.src.parents) - 1])
        print(f'  {marker} ({action.reason}, {size_mb:.2f} MB) {action.dst.relative_to(REPO_ROOT)}')
        if not dry_run:
            try:
                shutil.copy2(action.src, action.dst)
                copied += 1
            except Exception as exc:
                print(f'    ERROR copying: {exc}')
    return copied


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description='Sync Colab Drive outputs to local repo.')
    ap.add_argument('--model', choices=['V2', 'V4', 'V10', 'all'], default='all',
                    help='Which model family to sync. Default: all.')
    ap.add_argument('--seed', type=int, choices=[42, 123, 456], default=None,
                    help='Seed filter (42/123/456). Default: all seeds.')
    ap.add_argument('--drive-root', type=Path, default=DEFAULT_DRIVE_ROOT,
                    help=f'Drive mirror root. Default: {DEFAULT_DRIVE_ROOT}')
    ap.add_argument('--local-root', type=Path, default=DEFAULT_LOCAL_ROOT,
                    help=f'Local repo root. Default: {DEFAULT_LOCAL_ROOT}')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print planned actions but do not copy.')
    ap.add_argument('--verbose', action='store_true',
                    help='Print per-file details (including skipped).')
    args = ap.parse_args(argv)

    if not args.drive_root.exists():
        print(f'[ERROR] Drive root missing: {args.drive_root}')
        print('  Is Google Drive installed and signed in?  Is the drive letter correct?')
        return 2

    targets = [args.model] if args.model != 'all' else list(MODEL_DIRS.keys())

    print('=' * 72)
    print(f'DRIVE -> LOCAL SYNC  mode={"DRY" if args.dry_run else "APPLY"}')
    print(f'  Drive root: {args.drive_root}')
    print(f'  Local root: {args.local_root}')
    print(f'  Models    : {targets}')
    print(f'  Seed filter: {args.seed if args.seed else "all"}')
    print('=' * 72)

    grand_planned = 0
    grand_skipped = 0
    grand_bytes = 0
    grand_errors: list[str] = []
    grand_copied = 0

    for model in targets:
        rel = MODEL_DIRS[model]
        drive_subdir = args.drive_root / rel
        local_subdir = args.local_root / rel
        print(f'\n--- {model} ({rel}) ---')

        report = plan_sync(drive_subdir, local_subdir, args.seed, args.verbose)
        grand_errors.extend(report.errors)
        grand_skipped += report.skipped
        grand_planned += len(report.planned)
        grand_bytes += report.total_bytes

        if report.errors:
            for e in report.errors:
                print(f'  [ERR] {e}')
            continue

        if not report.planned:
            print(f'  up-to-date ({report.skipped} files already synced)')
            continue

        grand_copied += apply_plan(report, args.dry_run)

    print('\n' + '=' * 72)
    print(f'SUMMARY  planned={grand_planned}  skipped={grand_skipped}  '
          f'errors={len(grand_errors)}  size={grand_bytes/1024/1024:.1f} MB')
    if args.dry_run:
        print('  (dry run: no files were copied. Re-run without --dry-run to apply.)')
    else:
        print(f'  actually copied: {grand_copied}')
    print('=' * 72)

    return 1 if grand_errors else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
