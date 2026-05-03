"""Revert caption-above to caption-below for all figures (standard convention).

Pattern handled:
  \\begin{figure}...\\centering \\caption{...}\\label{fig:...} <body> \\end{figure}
becomes:
  \\begin{figure}...\\centering <body> \\caption{...}\\label{fig:...} \\end{figure}

Body can be: \\includegraphics, \\begin{tikzpicture}, \\resizebox{...}{...}{...},
or a \\begin{axis}/pgfplots block.

Usage:
    python scripts/caption_below.py                       # operates on paper 5 by default
    python scripts/caption_below.py .docs/papers/4/paper.tex  # explicit path
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Repo root = parent of `scripts/` (where this file lives).
REPO = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = REPO / '.docs' / 'papers' / '5' / 'paper.tex'


def transform(m: re.Match) -> str:
    inner = m.group(1)
    cap = re.search(r'\\caption\{(?:[^{}]|\{[^{}]*\})*\}', inner)
    lab = re.search(r'\\label\{fig:[^}]+\}', inner)
    if not cap or not lab:
        return m.group(0)
    cap_text = cap.group(0)
    lab_text = lab.group(0)
    body_markers = [r'\includegraphics', r'\begin{tikzpicture}',
                    r'\begin{axis}', r'\resizebox']
    cap_pos = cap.start()
    earliest_body = min(
        (inner.find(mk) for mk in body_markers if inner.find(mk) >= 0),
        default=len(inner)
    )
    if cap_pos > earliest_body:
        return m.group(0)  # already caption-below
    # Caption is BEFORE body, need to move it to AFTER body.
    new_inner = inner.replace(cap_text + '\n', '', 1).replace(lab_text + '\n', '', 1)
    new_inner = new_inner.replace(cap_text, '', 1).replace(lab_text, '', 1)
    new_inner = new_inner.rstrip() + '\n' + cap_text + '\n' + lab_text + '\n'
    return r'\begin{figure}' + new_inner + r'\end{figure}'


def main(target_path: Path) -> int:
    if not target_path.exists():
        print(f'ERROR: target file not found: {target_path}', file=sys.stderr)
        return 1

    text = target_path.read_text(encoding='utf-8')
    figure_pat = re.compile(r'\\begin\{figure\}((?:.|\n)*?)\\end\{figure\}')
    new_text = figure_pat.sub(transform, text)
    target_path.write_text(new_text, encoding='utf-8')

    # Verify
    n_above = 0
    for m in figure_pat.finditer(new_text):
        inner = m.group(1)
        cap = re.search(r'\\caption\{', inner)
        body = min(
            (inner.find(mk) for mk in [r'\includegraphics', r'\begin{tikzpicture}',
                                       r'\begin{axis}', r'\resizebox'] if inner.find(mk) >= 0),
            default=999999
        )
        if cap and cap.start() < body:
            n_above += 1
    print(f'Done. Target: {target_path.relative_to(REPO)}')
    print(f'Figures with caption-above remaining: {n_above}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Allow absolute or relative-to-repo path.
        arg = Path(sys.argv[1])
        target = arg if arg.is_absolute() else (REPO / arg)
    else:
        target = DEFAULT_TARGET
    sys.exit(main(target))
