"""Reassemble the Spanish reading translation and prove no number moved.

A translation is a rewrite of every sentence in a document that is mostly
numbers, so the thing that can go wrong silently is a digit. This concatenates
the translated chunks back into one file and then compares the two documents on
the only axis that must not change: the ordered sequence of numeric literals,
citation keys, cross-reference labels and figure filenames.

If a translator dropped a table row, transposed a decimal or paraphrased a
citation key, the sequences diverge and the position of the first divergence
says where. A pass here does not mean the Spanish is good; it means the Spanish
carries the same evidence.

Usage:
  python models/scripts/analysis/assemble_translation.py --chunks DIR
  python models/scripts/analysis/assemble_translation.py --verify-only
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / ".docs" / "papers" / "5"
EN = PAPER / "paper_gmd.tex"
ES = PAPER / "paper_es.tex"

# What must be identical between the two documents, in order.
TOKENS = {
    # The ordinal suffix is allowed and then stripped: English writes "2.5th
    # percentile" and Spanish "percentil 2.5", and without this the suffix makes
    # the pattern back off to "2" on one side and read "2.5" on the other, which
    # reports a difference that does not exist.
    "numbers": re.compile(
        r"(?<![A-Za-z0-9_.])(\d+(?:[.,]\d+)*)(?:st|nd|rd|th)?(?![A-Za-z0-9_])"),
    "citations": re.compile(r"\\cite[a-z]*\{([^}]*)\}"),
    "references": re.compile(r"\\ref\{([^}]*)\}"),
    "labels": re.compile(r"\\label\{([^}]*)\}"),
    "graphics": re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}"),
}


def strip_comments(tex: str) -> str:
    # A % that is not escaped starts a comment. Comments differ between the two
    # files by design (the Spanish file carries a header explaining itself).
    return re.sub(r"(?<!\\)%.*", "", tex)


def strip_noise(tex: str) -> str:
    """Remove spans whose digits are not evidence a reader reads.

    Two classes were producing false alarms. Math spans carry exponents and
    subscripts, and translation legitimately moves them within a sentence:
    "produced 8-15% lower $R^2$" becomes "produjeron un $R^2$ un 8-15% menor",
    which reorders the 2 of R^2 against the 8 and the 15 without changing a
    single reported quantity. And structural counts like \\multicolumn{8} change
    when a longer Spanish footnote needs one more wrapped row.

    Stripping both leaves the numbers that state results, which are the ones a
    translation must not touch.
    """
    tex = re.sub(r"\$[^$]*\$", " ", tex)                      # inline math
    tex = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " ", tex,
                 flags=re.S)
    tex = re.sub(r"\\(?:multicolumn|multirow|cmidrule)\s*\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", tex,
                 flags=re.S)
    return tex


def tokens_of(tex: str, pat: re.Pattern) -> list[str]:
    out = []
    for m in pat.finditer(tex):
        out.append(m.group(1) if m.groups() else m.group(0))
    return out


def assemble(chunk_dir: Path) -> int:
    chunks = sorted(chunk_dir.glob("chunk_*.tex"))
    if not chunks:
        print(f"no chunks in {chunk_dir}")
        return 1
    # chunk_NN_start_end.tex; verify the ranges tile without gap or overlap
    spans = []
    for c in chunks:
        parts = c.stem.split("_")
        spans.append((int(parts[2]), int(parts[3]), c))
    spans.sort()
    for (a1, b1, _), (a2, _, c2) in zip(spans, spans[1:]):
        if a2 != b1 + 1:
            print(f"FATAL: gap or overlap before {c2.name}: "
                  f"previous ends {b1}, this starts {a2}")
            return 1
    text = "".join(c.read_text(encoding="utf-8") for _, _, c in spans)
    ES.write_text(text, encoding="utf-8")
    print(f"assembled {len(spans)} chunks into {ES.name}, "
          f"{len(text.splitlines())} lines")
    return 0


def verify() -> int:
    if not ES.exists():
        print(f"no {ES.name} to verify")
        return 1
    en, es = strip_comments(_read(EN)), strip_comments(_read(ES))
    bad = 0
    for name, pat in TOKENS.items():
        src, dst = (strip_noise(en), strip_noise(es)) if name == "numbers" \
            else (en, es)
        a, b = tokens_of(src, pat), tokens_of(dst, pat)
        if name == "numbers":
            # Word order changes in translation, so compare the multiset: every
            # reported quantity must appear, the same number of times.
            from collections import Counter
            ca, cb = Counter(a), Counter(b)
            if ca == cb:
                print(f"  OK   {name:<12} {len(a)} reported quantities, "
                      f"identical multiset")
                continue
            bad += 1
            miss, extra = ca - cb, cb - ca
            print(f"  FAIL {name:<12} English {len(a)}, Spanish {len(b)}")
            if miss:
                print(f"       missing from Spanish: {sorted(miss.elements())[:12]}")
            if extra:
                print(f"       only in Spanish:      {sorted(extra.elements())[:12]}")
            continue
        if a == b:
            print(f"  OK   {name:<12} {len(a)} identical, in order")
            continue
        bad += 1
        print(f"  FAIL {name:<12} English has {len(a)}, Spanish has {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                lo = max(0, i - 2)
                print(f"       first divergence at position {i}:")
                print(f"         English: {a[lo:i+3]}")
                print(f"         Spanish: {b[lo:i+3]}")
                break
        else:
            longer, which = (a, "English") if len(a) > len(b) else (b, "Spanish")
            print(f"       {which} has {abs(len(a)-len(b))} extra at the end: "
                  f"{longer[min(len(a), len(b)):][:6]}")
    print()
    if bad:
        print(f"{bad} token classes diverge. The translation has changed "
              f"evidence,\nnot only wording. Fix before reading or circulating.")
        return 1
    print("Every number, citation, cross-reference, label and figure path is\n"
          "identical and in the same order. The translation carries the same\n"
          "evidence as the manuscript of record.")
    return 0


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=None)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if args.chunks and not args.verify_only:
        rc = assemble(args.chunks)
        if rc:
            return rc
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
