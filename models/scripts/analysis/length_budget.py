"""Where the pages go, so a cut can be aimed rather than guessed.

Reports the manuscript's composition: prose, captions, tables and floats, per
section. A length cut made by deleting whatever looks long tends to remove the
careful qualifications first, because those are the long sentences; this exists
so the cut is aimed at what actually occupies pages.

Usage:
  python models/scripts/analysis/length_budget.py
  python models/scripts/analysis/length_budget.py --file path/to/x.tex
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / ".docs" / "papers" / "5" / "paper_gmd.tex"

FLOAT = re.compile(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", re.S)
CAPTION = re.compile(r"\\caption\{", re.S)
SECTION = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}")


def balanced(text, start):
    """Text of a braced group whose opening brace is at `start`."""
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    return ""


def words(s):
    s = re.sub(r"\\[a-zA-Z@]+\s*", " ", s)
    s = re.sub(r"[{}$\\&~^_%]", " ", s)
    return len([w for w in s.split() if any(c.isalpha() for c in w)])


def inventory(path):
    """One line per float: kind, size, label, opening of the caption.

    The size counts the whole environment, so a TikZ figure reads large and a
    one-line graphics include reads small. What matters for a length cut is the
    caption, which is prose, and the label, which says how many places would
    need rewiring if the float moved to the supplement.
    """
    tex = path.read_text(encoding="utf-8")
    body = re.sub(r"(?<!\\)%.*", "", tex[tex.index(r"\begin{document}"):])
    print(f"{'kind':<7}{'words':>6}  {'label':<32} refs  caption")
    print("-" * 96)
    for m in FLOAT.finditer(body):
        blk, kind = m.group(0), m.group(1)
        lab = re.search(r"\\label\{([^}]*)\}", blk)
        label = lab.group(1) if lab else "?"
        ci = blk.find(r"\caption{")
        cap = balanced(blk, ci + len(r"\caption")) if ci >= 0 else ""
        n_ref = len(re.findall(r"\\ref\{" + re.escape(label) + r"\}", body))
        print(f"{kind:<7}{words(blk):>6}  {label:<32}{n_ref:>5}  "
              f"{re.sub(chr(92) + 's+', ' ', cap).strip()[:44]}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT)
    ap.add_argument("--floats", action="store_true",
                    help="one line per float, for deciding what moves out")
    args = ap.parse_args()
    if args.floats:
        return inventory(args.file)
    tex = args.file.read_text(encoding="utf-8")
    body = tex[tex.index(r"\begin{document}"):]
    body = re.sub(r"(?<!\\)%.*", "", body)

    floats = FLOAT.findall(body)
    n_fig = len(re.findall(r"\\begin\{figure\*?\}", body))
    n_tab = len(re.findall(r"\\begin\{table\*?\}", body))

    caps = []
    for m in CAPTION.finditer(body):
        caps.append(balanced(body, m.end() - 1))

    stripped = FLOAT.sub(" ", body)
    stripped = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ",
                      stripped, flags=re.S)

    print(f"{args.file.name}\n")
    print(f"  figures            {n_fig}")
    print(f"  tables             {n_tab}")
    print(f"  caption words      {sum(words(c) for c in caps):>6}")
    print(f"  prose words        {words(stripped):>6}   (outside floats)")
    print()
    print("  longest captions")
    for n, c in sorted(((words(c), c) for c in caps), reverse=True)[:8]:
        first = re.sub(r"\s+", " ", c).strip()[:62]
        print(f"    {n:>4}  {first}...")

    print("\n  prose by section")
    marks = [(m.start(), m.group(1), m.group(2)) for m in SECTION.finditer(stripped)]
    for i, (pos, kind, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(stripped)
        n = words(stripped[pos:end])
        if kind == "section" or n >= 250:
            indent = {"section": "", "subsection": "  ",
                      "subsubsection": "    ", "paragraph": "      "}[kind]
            print(f"    {n:>5}  {indent}{re.sub(r'\\s+', ' ', name)[:58]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
