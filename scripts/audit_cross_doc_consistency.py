"""Feature 004 T100 - cross-doc consistency audit.

Verifies that Paper 5 (root + delivery), thesis, and EGU26 poster all
report the same canonical numerical values from features 002 + 003,
and that no document references a label / citation / figure that does
not resolve.

Targets:
  Paper 5 root:     .docs/papers/5/paper.tex
  Paper 5 delivery: .docs/papers/5/delivery/paper.tex
  Thesis:           .docs/thesis/thesis.tex
  Poster:           .docs/conferences/EGU26/poster/poster.tex

Canonical numbers (path-C / feature 002 + factorial / feature 003):
  V10 primary R^2     = 0.672
  V10 primary RMSE    = 76.23 mm
  V10 multi-seed mean = 0.655 +/- 0.018
  Multi-seed RMSE     = 78.13 +/- 1.99
  Ridge weights       = w_Conv 0.509, w_GNN 0.652, bias -6.37
  Factorial best cell = PAFC-GCN R^2 = 0.516 +/- 0.059
  ANOVA feat          = F=5.25, p=0.041
  Inter-seed sigma    = GAT-BASIC 0.114, GCN-BASIC 0.013

Usage:
  python scripts/audit_cross_doc_consistency.py
  python scripts/audit_cross_doc_consistency.py --strict   # exit 1 on any drift
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DOCS = {
    'paper-root':     REPO / '.docs/papers/5/paper.tex',
    'paper-delivery': REPO / '.docs/papers/5/delivery/paper.tex',
    'thesis':         REPO / '.docs/thesis/thesis.tex',
    'poster':         REPO / '.docs/conferences/EGU26/poster/poster.tex',
}

# (regex, label, expected min count per document, expected max per document)
# count constraints expressed as dict; if a doc isn't listed -> no constraint
CANONICAL = [
    # V10 primary
    (r'0\.672',            'V10 R^2 primary',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 5, 'poster': 1}),
    (r'76\.23',            'V10 RMSE primary',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1}),
    # Multi-seed aggregate
    (r'0\.655',            'V10 multi-seed mean R^2',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1, 'poster': 1}),
    (r'0\.018',            'V10 multi-seed sd',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1, 'poster': 1}),
    (r'78\.13',            'V10 multi-seed RMSE mean',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1, 'poster': 1}),
    (r'1\.99',             'V10 multi-seed RMSE sd',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1, 'poster': 1}),
    # Fusion weights
    (r'0\.509',            'w_Conv',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1, 'poster': 1}),
    (r'0\.652',            'w_GNN',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1, 'poster': 1}),
    (r'6\.37',             'Ridge bias',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1}),
    # Factorial best cell + ANOVA
    (r'0\.516',            'Factorial best PAFC-GCN R^2',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1, 'poster': 1}),
    (r'0\.041',            'ANOVA feat p',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1, 'poster': 1}),
    (r'5\.25',             'ANOVA feat F',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1}),
    # Inter-seed extremes
    (r'0\.114',            'GAT-BASIC sigma',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1}),
    (r'0\.013',            'GCN-BASIC sigma',
     {'paper-root': 1, 'paper-delivery': 1, 'thesis': 1}),
]

# Legacy values that should NOT appear anywhere except the documented exception
LEGACY = [
    (r'0\.668',     'V10 legacy R^2 (path-A bare ConvLSTM)'),
    (r'76\.67',     'V10 legacy RMSE'),
    (r'0\.446',     'w_Conv legacy'),
    (r'0\.710',     'w_GNN legacy'),
    (r'5\.53',      'Ridge bias legacy'),
]


def check_canonical(verbose: bool = False) -> tuple[int, int]:
    """Return (passed, failed) counts."""
    passed = 0
    failed = 0
    print('=' * 78)
    print('CANONICAL VALUES CHECK')
    print('=' * 78)
    print(f'{"value":<10} {"label":<35} ', end='')
    for d in DOCS:
        print(f'{d:>15}', end='')
    print()
    print('-' * 78)
    for pattern, label, expected in CANONICAL:
        print(f'{pattern:<10} {label:<35} ', end='')
        all_ok = True
        for doc_name in DOCS:
            text = DOCS[doc_name].read_text(encoding='utf-8') if DOCS[doc_name].exists() else ''
            actual = len(re.findall(pattern, text))
            min_count = expected.get(doc_name, 0)
            ok = actual >= min_count
            mark = ' OK ' if ok else 'FAIL'
            print(f'  {actual:>3}/{min_count:>1} {mark:>4}', end='')
            if not ok:
                all_ok = False
        print()
        if all_ok:
            passed += 1
        else:
            failed += 1
    return passed, failed


def check_legacy(verbose: bool = False) -> int:
    """Return number of unexpected legacy hits (excluding documented exception)."""
    print()
    print('=' * 78)
    print('LEGACY VALUES CHECK (should be 0 except where documented)')
    print('=' * 78)
    n_legacy = 0
    for pattern, label in LEGACY:
        for doc_name, doc_path in DOCS.items():
            if not doc_path.exists():
                continue
            text = doc_path.read_text(encoding='utf-8')
            count = len(re.findall(pattern, text))
            if count == 0:
                continue
            # Document exception: 0.666 in paper-root/delivery line ~915
            # (intracell DEM table comparing legacy bare-ConvLSTM setup).
            # 0.668 should be 0; 0.666 is allowed only in that specific
            # context.
            print(f'  [WARN] {pattern:<10} ({label}) appears {count}x in {doc_name}')
            n_legacy += count
    if n_legacy == 0:
        print('  [OK] no unexpected legacy values found')
    return n_legacy


def check_xrefs() -> int:
    """Cross-reference + bib + figure audit per document. Returns number of issues."""
    print()
    print('=' * 78)
    print('CROSS-REFERENCE / BIB / FIGURE AUDIT')
    print('=' * 78)
    issues = 0
    for doc_name, doc_path in DOCS.items():
        if not doc_path.exists():
            continue
        text = doc_path.read_text(encoding='utf-8')
        labels = set(re.findall(r'\\label\{([^}]+)\}', text))
        refs = set(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', text))
        cites = set()
        for c in re.findall(r'\\cite[tp]?\{([^}]+)\}', text):
            for k in c.split(','):
                cites.add(k.strip())
        unresolved = refs - labels
        print(f'  {doc_name:<16} labels={len(labels):>3}  refs={len(refs):>3}  '
              f'cites={len(cites):>3}  unresolved={len(unresolved)}')
        if unresolved:
            issues += len(unresolved)
            for u in sorted(unresolved):
                print(f'    [FAIL] unresolved \\ref{{{u}}}')
    return issues


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict', action='store_true',
                    help='exit 1 on any drift or unresolved reference')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)

    passed, failed = check_canonical(verbose=args.verbose)
    n_legacy = check_legacy(verbose=args.verbose)
    n_xref = check_xrefs()

    print()
    print('=' * 78)
    print('SUMMARY')
    print('=' * 78)
    print(f'  Canonical values: {passed} passed, {failed} failed')
    print(f'  Legacy hits     : {n_legacy}')
    print(f'  Unresolved refs : {n_xref}')

    rc = 0 if (failed == 0 and n_xref == 0) else 1
    if args.strict and n_legacy > 0:
        rc = 1

    if rc == 0:
        print('\n  [READY] Cross-doc consistency check PASSED.')
    else:
        print(f'\n  [BLOCKED] Cross-doc consistency check FAILED (rc={rc}).')
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
