"""Flatten the heading tree: fewer headings, more paragraphs.

The manuscript carried 76 headings across 28 pages, one every 240 words, with
28 subsubsections, six headings whose only content was a single child heading,
and four whose own body was one word because they existed purely to group.

That costs twice. Every heading is two or three lines of vertical space, so 76 of
them is several pages of nothing. And a division into one, or into two, is a
promise of parallel structure that the text does not keep: what reads as a
sequence of labelled fragments is usually one argument that has been chopped.

Three operations, chosen per heading rather than by rule:

  DROP     remove the heading; its prose joins what precedes it. Used for a
           grouping shell with no body of its own, and for a lone child whose
           parent already introduces it.
  INLINE   subsubsection becomes paragraph. The heading survives as a run-in
           label and stops costing a vertical break. Used where the headings
           really are parallel and named, as the five protocol components are.
  PROMOTE  subsubsection becomes subsection, used when its shell parent is
           dropped and it inherits that level.

Nothing is deleted except heading commands: no sentence, number or citation is
touched, and the audit is expected to pass unchanged afterwards.

Usage:
  python models/scripts/analysis/flatten_headings.py --dry-run
  python models/scripts/analysis/flatten_headings.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / ".docs" / "papers" / "5" / "paper_gmd.tex"

# (opening words of the heading title, operation)
PLAN = [
    # -- Background: five subsections of 76 to 242 words are one argument -----
    ("DL architectures for spatiotemporal", "DROP"),
    ("Hybrid architectures for precipitation", "DROP"),
    ("Ensemble methods and their limitations", "DROP"),
    ("Emerging paradigms: mixture of experts", "DROP"),

    # -- Methods: the five components are genuinely parallel, so keep the
    #    labels and stop paying for the breaks
    ("Component 1.", "INLINE"),
    ("Component 2.", "INLINE"),
    ("Component 3.", "INLINE"),
    ("Component 4.", "INLINE"),
    ("Component 5.", "INLINE"),
    ("Precipitation-elevation relationship", "DROP"),
    ("Feature engineering", "DROP"),
    ("Data-driven development methodology", "DROP"),
    # architecture families: five fragments of 35 to 173 words
    ("Convolutional-recurrent hybrids", "DROP"),
    ("Spectral-temporal hybrids", "DROP"),
    ("Graph-attention-LSTM hybrids", "DROP"),
    ("Ensemble strategies", "DROP"),
    ("Emerging paradigms", "DROP"),

    # -- Results: four grouping shells with one-word bodies -------------------
    ("Base learners and the anchoring step", "DROP"),
    ("Base learners and the zero-cost references", "PROMOTE"),
    ("Graph-based architecture performance", "PROMOTE"),
    ("Horizon degradation analysis", "DROP"),
    ("Combination: calibration or complementarity", "DROP"),
    ("Ensemble strategy analysis", "PROMOTE"),
    ("Stratified ensemble analysis", "PROMOTE"),
    ("Late fusion: combination at the prediction level", "PROMOTE"),
    ("What survives a purged split", "PROMOTE"),
    ("Decomposing the fusion gain", "PROMOTE"),
    ("Feature-architecture interaction", "PROMOTE"),
    ("Convergence between families", "PROMOTE"),
    ("Spatially explicit diagnostics", "DROP"),
    ("Elevation-stratified performance analysis", "PROMOTE"),
    ("Spatial coherence: variogram analysis", "PROMOTE"),
    ("Sub-cell topographic feature engineering", "PROMOTE"),
    ("Visual analysis", "DROP"),
    ("Anomaly skill, the spatial prior, and generality", "DROP"),
    ("Beyond aggregate skill", "PROMOTE"),
    ("A leakage audit of the graph construction", "PROMOTE"),
    ("Does the result generalise", "PROMOTE"),

    # -- Discussion: a 205-word parent whose only child is 821 words ----------
    ("The target is built on a climatology", "DROP"),

    # -- Conclusions: a 68-word run-in label carrying nothing -----------------
    ("Methodological contributions", "DROP"),

    # -- second pass, on what the first pass exposed --------------------------
    # Background became a 435-word section with one 242-word child; that is the
    # same only-child problem one level up.
    ("Forecast verification and what is new here", "DROP"),
    # Two run-in labels shorter than the paragraphs they head.
    ("Order and cost", "DROP"),
    ("Spatial autocorrelation of residuals", "DROP"),
    # Two adjacent subsections on the same object, 189 and 208 words.
    ("Stratified ensemble analysis", "DROP"),
]

HEAD = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\{")


def balanced(text, start):
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i
        i += 1
    return "", start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tex = PAPER.read_text(encoding="utf-8")

    applied, missed = [], []
    for opening, op in PLAN:
        found = False
        for m in list(HEAD.finditer(tex)):
            title, close = balanced(tex, m.end() - 1)
            if not re.sub(r"\s+", " ", title).strip().startswith(opening):
                continue
            found = True
            kind = m.group(1)
            # a \label on the next line must survive a DROP, or every \ref to it
            # breaks; it is re-emitted bare.
            after = tex[close + 1:close + 200]
            lab = re.match(r"\s*\\label\{[^}]*\}", after)
            label_txt = lab.group(0).strip() if lab else ""
            cut_to = close + 1 + (len(lab.group(0)) if lab else 0)

            if op == "DROP":
                repl = (label_txt + "\n") if label_txt else ""
                tex = tex[:m.start()] + repl + tex[cut_to:]
            elif op == "INLINE":
                tex = (tex[:m.start()] + "\\paragraph{" + title + "}"
                       + tex[close + 1:])
            elif op == "PROMOTE":
                tex = (tex[:m.start()] + "\\subsection{" + title + "}"
                       + tex[close + 1:])
            applied.append((op, kind, re.sub(r"\s+", " ", title)[:50]))
            break
        if not found:
            missed.append(opening)

    for op, kind, t in applied:
        print(f"  {op:<8}{kind:<15}{t}")
    if missed:
        print("\nnot found: " + "; ".join(missed))
    print(f"\n{len(applied)} headings changed, {len(missed)} not found")
    if args.dry_run:
        print("(dry run, nothing written)")
        return 0
    PAPER.write_text(tex, encoding="utf-8")
    print(f"wrote {PAPER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
