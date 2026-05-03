"""T048-T050 - Overleaf ZIP bundle.

Mirrors `.docs/papers/5/delivery/` into a zip with MDPI residues excluded and
build artefacts filtered out. Output: d:/tmp/paper5_joh_overleaf.zip.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / '.docs' / 'papers' / '5' / 'delivery'
OUT = Path(r'd:/tmp/paper5_joh_overleaf.zip')

# Skip these when walking the delivery directory
EXCLUDE_DIRS = {
    'Definitions',           # leftover MDPI class bundle
    '__pycache__',
    '_minted-paper',
}
EXCLUDE_FILE_NAMES = {
    'Phd Manuel Perez - Paper 3 - MDPI Hydrology.pdf',
    # JoH submission package files: belong on the EM portal directly,
    # not inside the Overleaf project ZIP.
    'cover_letter.txt',
    'highlights.txt',
    'suggested_reviewers.txt',
    'submission_checklist.md',
}
EXCLUDE_SUFFIXES = {
    '.aux', '.log', '.out', '.fls', '.fdb_latexmk', '.bbl', '.blg', '.abs', '.pdf',
    '.synctex.gz', '.toc',
}


def _should_skip(path: Path) -> bool:
    rel = path.relative_to(SRC)
    if any(p in EXCLUDE_DIRS for p in rel.parts):
        return True
    if path.name in EXCLUDE_FILE_NAMES:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        # Exception: paper.pdf must be included for reviewer convenience
        if path.name == 'paper.pdf':
            return False
        return True
    return False


def run(args: argparse.Namespace) -> int:
    print(f'[T048] Overleaf ZIP bundle  src={SRC.relative_to(REPO)}  out={OUT}')

    if not SRC.exists():
        print(f'  MISSING: {SRC}')
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite
    if OUT.exists():
        OUT.unlink()

    n_files = 0
    n_skipped = 0
    total_bytes = 0
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(SRC):
            # prune excluded dirs from traversal
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fname in files:
                src_path = Path(root) / fname
                if _should_skip(src_path):
                    n_skipped += 1
                    continue
                arcname = src_path.relative_to(SRC).as_posix()
                zf.write(src_path, arcname)
                n_files += 1
                total_bytes += src_path.stat().st_size

    zip_size = OUT.stat().st_size
    print(f'  files included: {n_files}')
    print(f'  files skipped:  {n_skipped}  (build artefacts + MDPI residues)')
    print(f'  source bytes:   {total_bytes/1024/1024:.1f} MB')
    print(f'  zip size:       {zip_size/1024/1024:.1f} MB')

    # Sanity: verify ZIP opens and count matches
    with zipfile.ZipFile(OUT, 'r') as zf:
        names = zf.namelist()
    assert len(names) == n_files, f'ZIP file count mismatch {len(names)} != {n_files}'

    # Print top-level contents preview
    print('  top-level contents (first 20):')
    seen = set()
    for name in names:
        top = name.split('/', 1)[0]
        if top not in seen:
            seen.add(top)
            marker = '/' if '/' in name else ''
            print(f'    {top}{marker}')
            if len(seen) >= 20:
                break

    # Confirm key files present
    required = {
        'paper.tex', 'paper.pdf', 'refs.bib',
        'cas-sc.cls', 'cas-common.sty', 'cas-model2-names.bst', 'stfloats.sty',
        'figures/horizon_degradation_multiseed.png',
    }
    missing = [r for r in required if r not in names]
    if missing:
        print(f'  WARN: required files missing from ZIP: {missing}')
        return 2
    print(f'  all {len(required)} required files present in ZIP')
    return 0


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace()))
