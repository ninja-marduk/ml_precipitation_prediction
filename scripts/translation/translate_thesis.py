"""Thesis ES → EN translation pipeline (P8.1).

Translates `.docs/thesis/thesis.tex` paragraph-by-paragraph using Anthropic's
Claude API. Preserves LaTeX structure (chapters, sections, labels, refs,
math, citations).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/translation/translate_thesis.py [--start CHAPTER] [--end CHAPTER] [--dry-run]

Output:
    .docs/thesis/thesis_en.tex            (parallel English version)
    scripts/translation/output/log.csv    (paragraph-level translation log)

Estimated cost: ~$8-15 USD for full thesis (~123 pages, ~80 input/output token windows).
Estimated time: ~30-45 minutes wall clock with rate limiting.
"""
from pathlib import Path
import sys
import re
import time
import csv
import os

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / ".docs/thesis/thesis.tex"
DST = ROOT / ".docs/thesis/thesis_en.tex"
LOG = Path(__file__).parent / "output" / "log.csv"
LOG.parent.mkdir(parents=True, exist_ok=True)


# Patterns that should NOT be translated (preserved verbatim)
PRESERVE_PATTERNS = [
    re.compile(r"\\(begin|end)\{[^}]+\}"),                # environment delimiters
    re.compile(r"\\label\{[^}]+\}"),                       # labels
    re.compile(r"\\(ref|cref|Cref|autoref|nameref|eqref)\{[^}]+\}"),  # references
    re.compile(r"\\cite[a-z]*\{[^}]+\}"),                  # citations
    re.compile(r"\$\$.*?\$\$", re.DOTALL),                 # display math
    re.compile(r"\$[^$]+\$"),                              # inline math
    re.compile(r"%[^\n]*"),                                # comments
]

TRANSLATION_PROMPT = """You are a professional academic translator specialising in machine learning and hydrology. Translate the following Spanish LaTeX paragraph into rigorous, formal academic English suitable for a doctoral thesis at a Q1-journal-level standard.

CRITICAL RULES:
1. Preserve EVERY LaTeX command verbatim: \\section{...}, \\citep{...}, \\ref{...}, \\label{...}, math environments ($...$, $$...$$, \\begin{equation}...\\end{equation}), itemize/enumerate, etc.
2. Preserve all citation keys (e.g., \\citep{Poveda2011}) WITHOUT change.
3. Preserve all \\label{...} unchanged.
4. Translate ONLY the natural-language Spanish prose.
5. Do NOT add any new content, explanations, or commentary.
6. Keep technical terms consistent: ConvLSTM, GNN-TAT, KCE, PAFC, R², RMSE, NSE, CHIRPS, SRTM (always English).
7. Use US English spelling ("modeling", "behavior", "characterize").
8. Output ONLY the translated LaTeX paragraph, nothing else.

Spanish input:
---
{spanish}
---

English output (LaTeX, preserve all commands):"""


def split_paragraphs(text):
    """Split LaTeX into paragraph-sized translation units, preserving comments
    and environment boundaries as their own units."""
    units = []
    current = []
    in_env_skip = 0  # nesting level inside don't-translate envs
    skip_envs = {"figure", "table", "equation", "align", "tabular", "tikzpicture",
                 "lstlisting", "verbatim", "minted"}

    for line in text.split("\n"):
        s = line.strip()
        # Detect env begin/end
        m_begin = re.match(r"\\begin\{([a-zA-Z*]+)\}", s)
        m_end = re.match(r"\\end\{([a-zA-Z*]+)\}", s)
        if m_begin and m_begin.group(1) in skip_envs:
            if current:
                units.append(("translate", "\n".join(current)))
                current = []
            in_env_skip += 1
            current.append(line)
            continue
        if m_end and m_end.group(1) in skip_envs:
            current.append(line)
            in_env_skip -= 1
            if in_env_skip == 0:
                units.append(("preserve", "\n".join(current)))
                current = []
            continue
        if in_env_skip > 0:
            current.append(line)
            continue
        # Empty line = paragraph break
        if not line.strip():
            if current:
                units.append(("translate", "\n".join(current)))
                current = []
            else:
                units.append(("preserve", ""))
        else:
            current.append(line)
    if current:
        units.append(("translate", "\n".join(current)))
    return units


def translate_paragraph(spanish_text, api_key):
    """Call Claude API for translation. Stub implementation — replace with
    real Anthropic SDK call when running."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-5",  # or claude-sonnet-4-5 for cheaper
        max_tokens=4000,
        messages=[
            {"role": "user", "content": TRANSLATION_PROMPT.format(spanish=spanish_text)}
        ]
    )
    return response.content[0].text.strip()


def main(args):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set. Use --dry-run to test parsing only.")
        sys.exit(1)

    print(f"Reading {SRC}...")
    text = SRC.read_text(encoding="utf-8")
    units = split_paragraphs(text)

    n_translate = sum(1 for kind, _ in units if kind == "translate" and len(_.strip()) > 30)
    n_preserve = len(units) - n_translate
    print(f"  Parsed {len(units)} units: {n_translate} to translate, {n_preserve} to preserve")
    print(f"  Estimated tokens: ~{n_translate * 500:,} input + ~{n_translate * 500:,} output")
    print(f"  Estimated cost (Opus): ~${n_translate * 500 * 1.5e-5 + n_translate * 500 * 7.5e-5:.2f}")
    print(f"  Estimated cost (Sonnet): ~${n_translate * 500 * 3e-6 + n_translate * 500 * 1.5e-5:.2f}")

    if args.dry_run:
        print("\n[DRY-RUN] No API calls made.")
        # Save units to log for inspection
        with LOG.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "kind", "first_50_chars"])
            for i, (kind, content) in enumerate(units):
                w.writerow([i, kind, content[:50].replace("\n", " ")])
        print(f"  Dry-run log: {LOG}")
        return

    out = []
    log_rows = []
    for i, (kind, content) in enumerate(units):
        if kind == "preserve" or len(content.strip()) < 30:
            out.append(content)
            log_rows.append((i, kind, "skipped", len(content)))
            continue
        try:
            translated = translate_paragraph(content, api_key)
            out.append(translated)
            log_rows.append((i, "translate", "ok", len(content)))
            print(f"  [{i+1}/{len(units)}] translated ({len(content)} → {len(translated)} chars)")
            time.sleep(0.5)  # rate limit
        except Exception as e:
            print(f"  [{i+1}/{len(units)}] ERROR: {e}")
            out.append(content)  # fallback: keep original
            log_rows.append((i, "translate", f"error: {e}", len(content)))

    DST.write_text("\n".join(out), encoding="utf-8")
    print(f"\nSaved: {DST}")

    with LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "kind", "status", "chars"])
        for row in log_rows:
            w.writerow(row)
    print(f"Log: {LOG}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Parse only, no API calls")
    main(p.parse_args())
