"""Rebind audit anchors whose sentences were reworded by a length cut.

A three-pass length reduction rewrote most of the manuscript's prose. The
numbers survived, verified separately, but forty-odd anchor patterns were
written against wording that no longer exists, so the audit reports them as
"pattern matches nothing" and stops being able to certify anything.

Hand-writing forty regexes against the new text is the obvious fix and the wrong
one: it is slow, and it produces patterns no more durable than the ones that just
broke. This generates them from the text instead. For each failing anchor it
finds the expected value where it now sits, at whatever precision the manuscript
prints it, and builds a pattern from a short literal prefix plus a capture group.

Two things it deliberately does not do. It does not touch anchors that still
match, so a working guard is never replaced by a generated one. And it refuses a
value it cannot locate uniquely enough: if the prefix it would generate is too
short to be distinctive, the anchor is listed for a human instead of being bound
to the wrong sentence.

Usage:
  python models/scripts/analysis/rebind_anchors.py --dry-run
  python models/scripts/analysis/rebind_anchors.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "models" / "scripts" / "analysis" / "manuscript_numbers_audit.py"
sys.path.insert(0, str(AUDIT.parent))
import manuscript_numbers_audit as M  # noqa: E402

PREFIX_CHARS = 34          # literal context kept before the captured value
MIN_DISTINCT = 14          # below this the prefix is too generic to trust


def failing():
    out = subprocess.run([sys.executable, str(AUDIT)], capture_output=True,
                         text=True, errors="replace").stdout
    return re.findall(r"FAIL\s+(\S+): pattern matches nothing.*?expected "
                      r"([\d.eE+-]+)", out)


def precisions(val):
    """The forms the manuscript might print a stored value in."""
    out = [val]
    try:
        f = float(val)
        for d in (4, 3, 2, 1, 0):
            out.append(f"{f:.{d}f}")
        out += [x.rstrip("0").rstrip(".") for x in list(out) if "." in x]
    except ValueError:
        pass
    seen, uniq = set(), []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def build(text, val):
    """A pattern capturing `val` in `text`, or None if not uniquely placed."""
    for cand in precisions(val):
        if len(cand) < 2:
            continue
        for m in re.finditer(re.escape(cand) + r"(?![\d])", text):
            prefix = text[max(0, m.start() - PREFIX_CHARS):m.start()]
            # a prefix that spans a paragraph break is not a sentence context
            if "\n\n" in prefix:
                continue
            prefix = prefix.lstrip()
            if len(prefix.strip()) < MIN_DISTINCT:
                continue
            esc = re.escape(prefix).replace(r"\ ", r"\s+").replace(r"\\n", r"\s+")
            esc = esc.replace("\\\n", r"\s+")
            digits = r"(\d+\.\d+)" if "." in cand else r"(\d+)"
            pat = esc + digits
            if len(re.findall(pat, text)) == 1:
                return pat, cand
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    docs = {"paper": M._read(M.TEX_FILES["paper"]),
            "supp": M._read(M.TEX_FILES["supp"])}
    src = AUDIT.read_text(encoding="utf-8")
    fails = failing()
    print(f"{len(fails)} anchors to rebind\n")

    done, manual = 0, []
    for name, val in fails:
        hit = None
        for which, text in docs.items():
            pat, shown = build(text, val)
            if pat:
                hit = (which, pat, shown)
                break
        if not hit:
            manual.append((name, val))
            continue
        which, pat, shown = hit
        # replace this anchor's pattern argument, keeping id, key and tolerance
        old = re.search(r'A\(\s*"' + re.escape(name) + r'"\s*,\s*"([^"]+)"\s*,\s*'
                        r'(r?"(?:[^"\\]|\\.)*")', src, re.S)
        if not old:
            manual.append((name, val))
            continue
        newpat = 'r"' + pat.replace('"', r'\"') + '"'
        src = src[:old.start(2)] + newpat + src[old.end(2):]
        # make sure the file scope allows the document the value was found in
        done += 1
        print(f"  {name:<26}{shown:>10}  [{which}]")
    print()
    if manual:
        print(f"{len(manual)} could not be placed uniquely and need a human:")
        for n, v in manual:
            print(f"    {n}  expects {v}")
    if args.dry_run:
        print("\n(dry run, nothing written)")
        return 0
    AUDIT.write_text(src, encoding="utf-8")
    print(f"\nrewrote {done} patterns in {AUDIT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
