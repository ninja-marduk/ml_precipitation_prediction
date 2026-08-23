"""Measure verbatim overlap between a manuscript and the author's own prior work.

The pre-submission plagiarism check for this project was a careful read plus a
guess. A careful read finds the passages you remember writing twice; it does not
find the ones you forgot, and it cannot tell you what a similarity report will
show. This measures it.

The risk here is not classic plagiarism, it is text recycling: a doctoral thesis
and a published companion article by the same author, sharing methods with the
manuscript. Publishers permit that reuse when it is disclosed and cited, so the
useful output is not a percentage but a list: which passages are recycled, how
long each is, and whether the disclosure covers it. A number alone cannot be
acted on. A ranked list of matched passages can.

Shingling at eight words approximates what similarity software matches. Reporting
several sizes matters because the shape of the overlap says what kind it is:
overlap that survives at n=15 is copied text, overlap present at n=6 and gone by
n=10 is shared vocabulary, which is not recycling and should not be reported as
if it were.

Two things this cannot do, stated so the number is not over-read. It compares
only against documents you give it, so it says nothing about sources you do not
own. And it matches strings, so a passage rewritten by a paraphraser scores zero
here while remaining recycling in substance.

Usage:
  python models/scripts/analysis/text_overlap.py
  python models/scripts/analysis/text_overlap.py --target x.tex --against a.tex b.tex
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P5 = ROOT / ".docs" / "papers" / "5"
DEFAULT_TARGET = P5 / "paper_gmd.tex"
DEFAULT_SOURCES = [ROOT / ".docs" / "thesis" / "thesis.tex"]
SIZES = (6, 8, 10, 15)
REPORT_AT = 8          # the size whose matches are listed
MIN_RUN = 12           # a run shorter than this is a phrase, not a passage


def strip_latex(tex: str) -> str:
    body = tex[tex.index(r"\begin{document}"):] if r"\begin{document}" in tex else tex
    body = re.sub(r"(?<!\\)%.*", "", body)
    body = re.sub(r"\\begin\{(tikzpicture|equation\*?|align\*?|tabular\*?)\}"
                  r".*?\\end\{\1\}", " ", body, flags=re.S)
    body = re.sub(r"\$[^$]*\$", " ", body)
    body = re.sub(r"\\(cite[a-z]*|ref|label|url|includegraphics)\s*\{[^}]*\}", " ", body)
    body = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", body)
    body = re.sub(r"[{}\\&~^_#]", " ", body)
    return body


def tokens(text):
    return re.findall(r"[a-z][a-z'-]*", text.lower())


def shingles(toks, n):
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def runs(t_toks, src_sets, n):
    """Maximal stretches of the target whose n-grams all appear in a source."""
    hits = [False] * max(0, len(t_toks) - n + 1)
    for i in range(len(hits)):
        g = " ".join(t_toks[i:i + n])
        hits[i] = any(g in s for s in src_sets)
    out, i = [], 0
    while i < len(hits):
        if not hits[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(hits) and hits[j + 1]:
            j += 1
        out.append((i, j + n))          # token span
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--against", type=Path, nargs="*", default=DEFAULT_SOURCES)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    if not args.target.exists():
        print(f"no target at {args.target}")
        return 1
    sources = [p for p in args.against if p.exists()]
    missing = [p for p in args.against if not p.exists()]
    for p in missing:
        print(f"source not found, skipped: {p}")
    if not sources:
        print("no source documents to compare against")
        return 1

    t_toks = tokens(strip_latex(args.target.read_text(encoding="utf-8",
                                                      errors="replace")))
    src = {}
    for p in sources:
        src[p] = tokens(strip_latex(p.read_text(encoding="utf-8",
                                                errors="replace")))

    print(f"target : {args.target.name}  ({len(t_toks):,} words)")
    for p, toks in src.items():
        print(f"against: {p.name}  ({len(toks):,} words)")
    print()

    print("COVERAGE  (share of the manuscript whose n-grams appear in a source)")
    print(f"  {'n':>4}  {'covered':>9}   what overlap at this size means")
    print("  " + "-" * 74)
    notes = {6: "shared vocabulary and stock phrasing; not recycling",
             8: "the size similarity software matches on",
             10: "sentence-level reuse",
             15: "copied passages; nothing else survives here"}
    cov = {}
    for n in SIZES:
        sets = [shingles(toks, n) for toks in src.values()]
        tg = [" ".join(t_toks[i:i + n]) for i in range(len(t_toks) - n + 1)]
        hit = sum(1 for g in tg if any(g in s for s in sets))
        cov[n] = hit / max(1, len(tg))
        print(f"  {n:>4}  {cov[n]:>8.1%}   {notes[n]}")

    if cov[15] < 0.002:
        print("\n  Overlap collapses by n=15, so what remains at n=8 is shared "
              "wording\n  rather than copied passages.")
    print()

    sets = [shingles(toks, REPORT_AT) for toks in src.values()]
    spans = [(a, b) for a, b in runs(t_toks, sets, REPORT_AT) if b - a >= MIN_RUN]
    spans.sort(key=lambda ab: ab[0] - ab[1])
    total = sum(b - a for a, b in spans)
    print(f"PASSAGES  ({len(spans)} runs of {MIN_RUN}+ words, "
          f"{total:,} words, {total / max(1, len(t_toks)):.1%} of the manuscript)")
    print("  These are what a reviewer would see quoted side by side.\n")
    for a, b in spans[:args.top]:
        print(f"  {b - a:>3} words  ...{' '.join(t_toks[a:b])[:150]}...")
    if len(spans) > args.top:
        print(f"  ... and {len(spans) - args.top} more")

    print("\nWhat to do with this. Reuse of methods text is permitted by every "
          "major\npublisher when it is disclosed and the prior work is cited at "
          "the point of\nuse. So the question each passage above raises is not "
          "whether it may be\nreused, but whether the manuscript says where it "
          "came from. Check that the\ndisclosure names the document these "
          "passages come from, and that the citation\nsits beside the passage "
          "and not only in a paragraph fifteen pages earlier.")
    print("\nThis matched strings against the documents given. It says nothing "
          "about\nsources not supplied, and a paraphrased passage scores zero "
          "here while\nremaining recycling in substance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
