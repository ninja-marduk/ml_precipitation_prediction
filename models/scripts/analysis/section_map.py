"""The heading tree, with the structural smells marked.

A heading is a promise that what follows is one of several parallel things. A
section with a single subsection breaks that promise: the subsection is the
section, and the heading is a line of vertical space buying nothing. Two
subsections is usually the same problem wearing a disguise, because a division
into exactly two is nearly always better carried by a paragraph break and a
topic sentence.

This prints the tree with word counts and flags:

  ONLY CHILD    a heading with exactly one child heading; flatten it
  TWO CHILDREN  candidate for the same treatment
  THIN          a heading with less prose than a normal paragraph
  DEEP          subsubsection or paragraph nested three levels down

It counts prose per heading excluding floats, so a heading that looks large
because it contains a table is not mistaken for one that carries argument.

Usage: python models/scripts/analysis/section_map.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / ".docs" / "papers" / "5" / "paper_gmd.tex"

LEVEL = {"section": 0, "subsection": 1, "subsubsection": 2, "paragraph": 3}
HEAD = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\{")
FLOAT = re.compile(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", re.S)
THIN_AT = 90


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


def words(s):
    s = FLOAT.sub(" ", s)
    s = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", s, flags=re.S)
    s = re.sub(r"\\[a-zA-Z@]+\s*", " ", s)
    s = re.sub(r"[{}$\\&~^_%]", " ", s)
    return len([w for w in s.split() if any(c.isalpha() for c in w)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT)
    args = ap.parse_args()
    tex = args.file.read_text(encoding="utf-8")
    body = re.sub(r"(?<!\\)%.*", "", tex[tex.index(r"\begin{document}"):])

    nodes = []
    for m in HEAD.finditer(body):
        title, close = balanced(body, m.end() - 1)
        nodes.append({"lvl": LEVEL[m.group(1)], "kind": m.group(1),
                      "title": re.sub(r"\s+", " ", title).strip(),
                      "start": m.start(), "end": close})
    for i, n in enumerate(nodes):
        stop = nodes[i + 1]["start"] if i + 1 < len(nodes) else len(body)
        n["own"] = words(body[n["end"]:stop])

    # children: the next headings at exactly one level deeper, before the tree
    # returns to this level or shallower
    for i, n in enumerate(nodes):
        kids = 0
        for m in nodes[i + 1:]:
            if m["lvl"] <= n["lvl"]:
                break
            if m["lvl"] == n["lvl"] + 1:
                kids += 1
        n["kids"] = kids

    print(f"{args.file.name}\n")
    print(f"{'words':>6}  {'kids':>4}  heading")
    print("-" * 86)
    smells = []
    for n in nodes:
        flags = []
        if n["kids"] == 1:
            flags.append("ONLY CHILD")
        elif n["kids"] == 2 and n["lvl"] <= 1:
            flags.append("TWO CHILDREN")
        if n["kids"] == 0 and n["own"] < THIN_AT:
            flags.append("THIN")
        if n["lvl"] >= 2 and n["kind"] != "paragraph":
            flags.append("DEEP")
        if flags:
            smells.append((n, flags))
        mark = ("  <- " + ", ".join(flags)) if flags else ""
        print(f"{n['own']:>6}  {n['kids']:>4}  {'  ' * n['lvl']}"
              f"{n['title'][:52]}{mark}")

    print()
    print("=" * 86)
    tot = {k: sum(1 for n in nodes if n["kind"] == k) for k in LEVEL}
    print("  ".join(f"{k}: {v}" for k, v in tot.items() if v))
    print(f"\n{len(smells)} headings flagged. Flattening an ONLY CHILD or a "
          f"THIN heading turns a\nline of vertical space back into prose, which "
          f"is what a length cut wants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
