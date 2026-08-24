"""Check that text, floats and cross-references still agree with each other.

Three compression passes, ten floats relocated to the supplement, and a heading
tree cut from 76 to 52. Every one of those operations breaks references for a
living, so this looks for the wreckage that compiles cleanly and reads wrong.

LaTeX catches a `\\ref` to a label that does not exist. It does not catch a float
nobody references, a reference to a panel the figure no longer has, a roadmap
sentence naming sections that were merged away, or a supplement pointer left
behind at the number the item used to carry. Those all compile, and a reader
finds them instead.

Seven checks:

  ORPHAN FLOATS     a float the text never points at
  DANGLING REFS     a \\ref whose label is defined nowhere
  ORDER             floats numbered far from the order they are first cited
  PANELS            text naming a panel letter the figure does not define
  SUPPLEMENT        a Table S<n> or Fig. S<n> beyond what the supplement has
  ROADMAP           a section named in the introduction's roadmap that is gone
  DANGLING PROSE    "as shown below", "the previous section", left without a
                    referent by a cut

Nothing here is a judgement about wording. Every finding is a mismatch between
two artefacts, so each is either true or a bug in this script, and both are
worth knowing.

Usage: python models/scripts/analysis/coherence_check.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P5 = ROOT / ".docs" / "papers" / "5"
PAPER = P5 / "paper_gmd.tex"
SUPP = P5 / "supplement.tex"

FLOAT = re.compile(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}", re.S)


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def main():
    paper = strip_comments(PAPER.read_text(encoding="utf-8", errors="replace"))
    supp = strip_comments(SUPP.read_text(encoding="utf-8", errors="replace"))
    issues = 0

    def line_of(text, pos):
        return text.count("\n", 0, pos) + 1

    # ---- float inventory -------------------------------------------------
    floats = []           # (kind, label, start, body)
    for m in FLOAT.finditer(paper):
        lab = re.search(r"\\label\{([^}]*)\}", m.group(2))
        floats.append((m.group(1), lab.group(1) if lab else None,
                       m.start(), m.group(2)))
    labels = {lab for _, lab, _, _ in floats if lab}
    all_labels = set(re.findall(r"\\label\{([^}]*)\}", paper))
    refs = [(m.group(1), m.start()) for m in re.finditer(r"\\ref\{([^}]*)\}", paper)]
    refset = {r for r, _ in refs}

    print("ORPHAN FLOATS  (present but never referenced)")
    orphans = [(k, l) for k, l, _, _ in floats if l and l not in refset]
    for k, l in orphans:
        issues += 1
        print(f"  FAIL  {k} {l}")
    if not orphans:
        print("  none")
    unlabelled = [k for k, l, _, _ in floats if not l]
    if unlabelled:
        issues += len(unlabelled)
        print(f"  FAIL  {len(unlabelled)} float(s) carry no \\label at all")
    print()

    print("DANGLING REFS  (\\ref to a label defined nowhere in this file)")
    bad = sorted({r for r, _ in refs if r not in all_labels})
    for r in bad:
        issues += 1
        pos = next(p for rr, p in refs if rr == r)
        print(f"  FAIL  \\ref{{{r}}} at line {line_of(paper, pos)}")
    if not bad:
        print("  none")
    print()

    # ---- citation order --------------------------------------------------
    print("ORDER  (float position against the order the text first cites it)")
    first_cite = {}
    for r, pos in refs:
        if r in labels and r not in first_cite:
            first_cite[r] = pos
    seq = [l for _, l, _, _ in floats if l in first_cite]
    by_cite = sorted(seq, key=lambda l: first_cite[l])
    out_of_order = [(a, b) for a, b in zip(seq, by_cite) if a != b]
    if out_of_order:
        print(f"  {len(out_of_order)} float(s) appear in a different order than "
              f"they are cited:")
        print(f"    document order: {', '.join(seq)}")
        print(f"    citation order: {', '.join(by_cite)}")
        print("  Not an error in itself; journals renumber. Worth a look only if "
              "a\n  reader would meet a float long before its first mention.")
    else:
        print("  document order matches citation order")
    print()

    # ---- panels ----------------------------------------------------------
    print("PANELS  (text naming a panel the figure does not define)")
    found = False
    for kind, lab, start, body in floats:
        if kind != "figure" or not lab:
            continue
        # panels the caption or the graphic defines, e.g. "(a)" in the caption
        cap = re.search(r"\\caption\{", body)
        capstart = cap.end() if cap else 0
        defined = set(re.findall(r"\(([a-e])\)", body[capstart:]))
        # panels the body text claims when referring to this float
        for m in re.finditer(r"\\ref\{" + re.escape(lab) + r"\}", paper):
            ctx = paper[max(0, m.start() - 220):m.start() + 220]
            claimed = set(re.findall(r"[Pp]anel[s]?~?\s*\(?([a-e])\)?", ctx))
            missing = claimed - defined
            if missing and defined:
                issues += 1
                found = True
                print(f"  FAIL  {lab}: text near line {line_of(paper, m.start())} "
                      f"names panel(s) {sorted(missing)}; "
                      f"caption defines {sorted(defined)}")
    if not found:
        print("  no mismatch found")
    print()

    # ---- supplement pointers ---------------------------------------------
    print("SUPPLEMENT  (pointers against what the supplement actually holds)")
    s_tab = len(re.findall(r"\\begin\{table\*?\}", supp))
    s_fig = len(re.findall(r"\\begin\{figure\*?\}", supp))
    ptr_t = {int(n) for n in re.findall(r"Table~?S(\d+)", paper)}
    ptr_f = {int(n) for n in re.findall(r"(?:Fig\.|Figure)~?S(\d+)", paper)}
    print(f"  supplement has {s_tab} tables and {s_fig} figures")
    over_t = sorted(n for n in ptr_t if n > s_tab)
    over_f = sorted(n for n in ptr_f if n > s_fig)
    for n in over_t:
        issues += 1
        print(f"  FAIL  manuscript cites Table S{n}; supplement has only {s_tab}")
    for n in over_f:
        issues += 1
        print(f"  FAIL  manuscript cites Fig. S{n}; supplement has only {s_fig}")
    unused_t = sorted(set(range(1, s_tab + 1)) - ptr_t)
    unused_f = sorted(set(range(1, s_fig + 1)) - ptr_f)
    if unused_t:
        print(f"  note  supplement tables never cited: "
              f"{', '.join('S' + str(n) for n in unused_t)}")
    if unused_f:
        print(f"  note  supplement figures never cited: "
              f"{', '.join('S' + str(n) for n in unused_f)}")
    if not (over_t or over_f):
        print("  every pointer is within range")
    print()

    # ---- roadmap ---------------------------------------------------------
    print("ROADMAP  (sections the introduction promises against those that exist)")
    titles = [re.sub(r"\s+", " ", t).strip().lower()
              for t in re.findall(r"\\section\*?\{([^}]*)\}", paper)]
    road = re.search(r"remainder of (?:the|this) (?:paper|manuscript)"
                     r"(.{0,900})", paper, re.S)
    if not road:
        print("  no roadmap sentence found")
    else:
        named = re.findall(r"Sect\.~\\ref\{([^}]*)\}|Section~\\ref\{([^}]*)\}",
                           road.group(1))
        named = [a or b for a, b in named]
        missing = [n for n in named if n not in all_labels]
        for n in missing:
            issues += 1
            print(f"  FAIL  roadmap points at {n}, which no longer exists")
        if not missing:
            print(f"  roadmap names {len(named)} section labels, all defined")
        print(f"  document has {len(titles)} sections: "
              f"{'; '.join(t[:28] for t in titles)}")
    print()

    # ---- dangling prose ---------------------------------------------------
    print("LOST BACKSLASH  (a command that became text and still compiles)")
    # A shell heredoc reads the \r of \ref as a carriage return and the \t of
    # \texttt as a tab, so an edit passed through one can strip the backslash
    # from a command. LaTeX raises nothing: what is left is ordinary text and a
    # group, the document compiles, no reference is undefined, and the page
    # prints "Sect. efsec:graph-structure" where a section number belongs.
    STUBS = ("ref", "cite", "citep", "citet", "label", "texttt", "textbf",
             "emph", "textit", "caption", "citealp")
    lost = []
    for name, tex in (("paper", paper), ("supp", supp)):
        for m in re.finditer(r"(?<![A-Za-z\\])(" + "|".join(STUBS) +
                             r")\{[^}]{1,60}\}", tex):
            # a real command is preceded by a backslash; this one is not
            lost.append((name, line_of(tex, m.start()), m.group(0)[:44]))
    for name, ln, frag in lost:
        issues += 1
        print(f"  FAIL  {name} line {ln}: {frag!r} has no backslash")
    if not lost:
        print("  every command still carries its backslash")
    print()

    print("DANGLING PROSE  (deixis a cut may have left without a referent)")
    pats = [r"as (?:shown|discussed|described) (?:below|above)",
            r"the (?:previous|preceding|following|next) (?:section|subsection)",
            r"the (?:table|figure) below", r"see below", r"in what follows"]
    hits = []
    for p in pats:
        for m in re.finditer(p, paper, re.I):
            hits.append((line_of(paper, m.start()), m.group(0)))
    if hits:
        print(f"  {len(hits)} phrase(s) rely on position rather than a "
              f"reference:")
        for ln, txt in sorted(hits)[:12]:
            print(f"    line {ln}: {txt}")
        print("  Each needs checking: the thing referred to may have moved to "
              "the\n  supplement or been cut.")
    else:
        print("  none")

    print()
    print("=" * 72)
    print(f"{issues} hard mismatch(es).")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
