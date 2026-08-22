"""Move named floats from the manuscript to the supplement, and rewire the text.

The manuscript refers to supplement floats as literal text, "Table~S10", not
through \\ref, because the two documents compile separately. So a float that
moves has to be renumbered by hand, and a hand-numbered cross-reference is the
kind of thing that silently rots.

This does it from the compiled numbering instead: the float is appended to the
supplement, the supplement is compiled, its .aux is read for the number LaTeX
actually assigned, and the manuscript's \\ref calls are replaced with that
number. If the supplement's ordering changes later, rerunning fixes every
reference at once.

Usage:
  python models/scripts/analysis/move_floats_to_supplement.py --dry-run
  python models/scripts/analysis/move_floats_to_supplement.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P5 = ROOT / ".docs" / "papers" / "5"
PAPER = P5 / "paper_gmd.tex"
SUPP = P5 / "supplement.tex"

# Floats whose evidence is secondary to the protocol's five components. Each
# leaves a sentence behind in the manuscript; that text is edited separately.
MOVE = [
    "fig:study-area",
    "fig:architecture-families",
    "fig:gnn-architecture",
    "fig:early-vs-late-fusion",
    "tab:multiseed-ridge",
    "fig:horizon_degradation_multiseed",
    "fig:factorial-r2",
    "fig:parameter-efficiency",
    "fig:beyond-aggregate",
    "tab:multiregion",
]

FLOAT = re.compile(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", re.S)
MARKER = "% ==== floats relocated from the manuscript ===="


def extract(tex, label):
    for m in FLOAT.finditer(tex):
        if re.search(r"\\label\{" + re.escape(label) + r"\}", m.group(0)):
            return m.group(0), m.start(), m.end()
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paper = PAPER.read_text(encoding="utf-8")
    supp = SUPP.read_text(encoding="utf-8")

    moved, missing = [], []
    for label in MOVE:
        blk, a, b = extract(paper, label)
        if blk is None:
            missing.append(label)
            continue
        moved.append((label, blk))
        paper = paper[:a] + paper[b:]
    if missing:
        print("not found, skipped: " + ", ".join(missing))
    if not moved:
        print("nothing to move")
        return 1
    print(f"moving {len(moved)} floats")

    # Collapse the blank lines the removals leave behind.
    paper = re.sub(r"\n{4,}", "\n\n\n", paper)

    if MARKER in supp:
        print("supplement already carries relocated floats; not moving again")
        return 1
    tail = "\n\\end{document}"
    assert supp.rstrip().endswith(r"\end{document}")
    block = ("\n\n" + MARKER + "\n"
             "\\section{Material referenced from the manuscript}\n"
             "These floats were moved here from the manuscript for length. They "
             "are unchanged;\nthe manuscript cites each by the number it carries "
             "below.\n\n"
             + "\n\n".join(b for _, b in moved) + "\n")
    supp = supp.rstrip()[: -len(r"\end{document}")] + block + tail

    if args.dry_run:
        for label, blk in moved:
            print(f"  {label:<36}{len(blk.split()):>6} words")
        return 0

    SUPP.write_text(supp, encoding="utf-8")
    PAPER.write_text(paper, encoding="utf-8")

    # Compile the supplement so LaTeX assigns the numbers, then read them.
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", SUPP.name],
                       cwd=P5, capture_output=True)
    aux = (P5 / "supplement.aux").read_text(encoding="utf-8", errors="replace")
    numbers = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux):
        numbers[m.group(1)] = m.group(2)

    paper = PAPER.read_text(encoding="utf-8")
    unresolved = []
    for label, _ in moved:
        num = numbers.get(label)
        if not num:
            unresolved.append(label)
            continue
        kind = "Table" if label.startswith("tab:") else "Fig."
        # \ref{x} -> the literal number; the surrounding "Table~"/"Fig.~" the
        # manuscript already writes stays, so only the argument is replaced.
        paper = re.sub(r"\\ref\{" + re.escape(label) + r"\}", num, paper)
        print(f"  {label:<36}-> {kind} {num}")
    PAPER.write_text(paper, encoding="utf-8")
    if unresolved:
        print("\nNO NUMBER FOUND for: " + ", ".join(unresolved))
        print("Those \\ref calls are still live and will render as ??.")
        return 1
    print("\nReferences rewired. Check that each reads 'Table S<n>' or "
          "'Fig. S<n>' in\ncontext: a bare 'Table~5' left over from the "
          "manuscript numbering is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
