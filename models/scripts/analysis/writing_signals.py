"""Measure the writing signals that survive a de-AI editing pass.

A banned-word list is the wrong tool for this manuscript, and its failure here is
instructive. An earlier review swept it clean: zero em-dashes, zero "delve", zero
"Moreover". The text still read as machine-smoothed, and what gave it away was
not a word from any list but a ratio: one contrastive construction, "rather
than", appearing at ten times normal academic frequency. A list cannot find that,
because the construction is unremarkable; only its density is wrong.

So this measures the shape of the prose rather than its vocabulary. Nothing here
outputs a verdict, and the script never says "AI". Each signal is a number with a
reference range, and every one of them has an innocent explanation as well as a
guilty one; the report gives both. What the tool is for is finding the place
where a human should look.

Six signals:

  BURSTINESS      Human sentence length varies more than machine sentence
                  length, and more than lightly-edited machine length. Reported
                  as the coefficient of variation and as the share of sentences
                  clustered near the mean.
  TICS            Any n-gram whose rate is far above what a document of this
                  length should show. Self-calibrating: it needs no list and
                  finds whatever this text overuses.
  OPENERS         Repeated sentence-opening frames, which is how templated
                  structure survives paraphrase.
  PUNCTUATION     Rates per thousand words, including the markers a de-AI pass
                  removes and the ones it substitutes in.
  PARAGRAPHS      Uniformity of paragraph length. Machine drafting produces
                  paragraphs of suspiciously even size.
  RICHNESS        Hapax rate and type-token ratio over fixed windows, which fall
                  when text is regenerated toward the centre of a distribution.

Usage:
  python models/scripts/analysis/writing_signals.py
  python models/scripts/analysis/writing_signals.py --file path/to/x.tex --top 25
"""
from __future__ import annotations

import argparse
import collections
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / ".docs" / "papers" / "5" / "paper_gmd.tex"

# Reference ranges. These are working figures for formal academic English, used
# to say "unusual" rather than "wrong"; where a value has no defensible
# reference the script says so instead of inventing one.
REF = {
    "cv_sentence_len": (0.45, 0.75),      # coefficient of variation
    "clustered_share": (0.0, 0.45),       # share within +-25% of the mean
    "hapax_rate": (0.35, 0.55),           # share of types occurring once
    "semicolon_per_1k": (0.0, 4.0),
    "colon_per_1k": (0.5, 6.0),
}

DISCOURSE = ("however", "moreover", "furthermore", "additionally", "notably",
             "importantly", "consequently", "therefore", "thus", "hence",
             "nevertheless", "nonetheless", "in addition", "in conclusion",
             "in summary", "overall")

STOP = set("""a an the of for and or in on to with at by from as is are was were
be been being that this these those it its which who whom whose not no nor but
we our us they their them he she his her i you your than then so such can could
may might will would shall should must do does did have has had if when while
because since although though both each any all more most less least one two
three there here what how why into over under between within without across""".split())


def load(path):
    """Prose only: comments, floats, math, tikz and commands removed."""
    tex = path.read_text(encoding="utf-8", errors="replace")
    body = tex[tex.index(r"\begin{document}"):] if r"\begin{document}" in tex else tex
    body = re.sub(r"(?<!\\)%.*", "", body)
    body = re.sub(r"\\begin\{(figure|table|tikzpicture|equation|tabular\*?)\*?\}"
                  r".*?\\end\{\1\*?\}", " ", body, flags=re.S)
    body = re.sub(r"\$[^$]*\$", " NUM ", body)
    body = re.sub(r"\\(cite[a-z]*|ref|label|url|texttt|citep|citet)\{[^}]*\}",
                  " ", body)
    body = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", body)
    body = re.sub(r"[{}\\&~^_]", " ", body)
    return re.sub(r"[ \t]+", " ", body)


def sentences(text):
    flat = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", flat)
    return [p.strip() for p in parts if len(p.split()) >= 3]


def words(text):
    return re.findall(r"[a-z][a-z'-]+", text.lower())


def band(value, key):
    lo, hi = REF[key]
    if value < lo:
        return "below the usual range"
    if value > hi:
        return "ABOVE the usual range"
    return "within the usual range"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    text = load(args.file)
    sents = sentences(text)
    wl = words(text)
    n = len(wl)
    if n < 500:
        print(f"only {n} words of prose; the signals below need more")
        return 1
    lens = [len(s.split()) for s in sents]

    print(f"{args.file.name}: {n:,} words of prose, {len(sents):,} sentences\n")

    # ---- burstiness ----------------------------------------------------
    mean, sd = statistics.mean(lens), statistics.stdev(lens)
    cv = sd / mean
    clustered = sum(1 for x in lens if abs(x - mean) <= 0.25 * mean) / len(lens)
    print("BURSTINESS")
    print(f"  sentence length      mean {mean:.1f}, s.d. {sd:.1f}, "
          f"median {statistics.median(lens):.0f}, range {min(lens)} to {max(lens)}")
    print(f"  coefficient of var.  {cv:.2f}   {band(cv, 'cv_sentence_len')}")
    print(f"  within +-25% of mean {clustered:.0%}   "
          f"{band(clustered, 'clustered_share')}")
    print("  Low variation is the classic machine signature, but a house style "
          "that\n  favours one clause per sentence produces it too.\n")

    # ---- tics ------------------------------------------------------------
    print("TICS  (n-grams far above the rate a text this long should show)")
    per_1k = {}
    for size in (2, 3, 4):
        counts = collections.Counter()
        for i in range(len(wl) - size + 1):
            g = wl[i:i + size]
            if all(w in STOP for w in g) or g[0] in {"num"}:
                continue
            counts[" ".join(g)] += 1
        for g, c in counts.items():
            if c < 4:
                continue
            rate = 1000 * c / n
            # a content n-gram above this rate is repeated far more than topic
            # vocabulary alone explains
            floor = {2: 1.2, 3: 0.6, 4: 0.4}[size]
            if rate >= floor:
                per_1k[g] = (rate, c, size)
    if not per_1k:
        print("  nothing above the flagging rate\n")
    else:
        for g, (rate, c, size) in sorted(per_1k.items(),
                                         key=lambda kv: -kv[1][0])[:args.top]:
            print(f"  {rate:5.2f}/1k  x{c:<4} {g}")
        print("  A domain term repeats legitimately; a connective or a frame "
              "does not.\n  Judge each by whether it names something or joins "
              "something.\n")

    # ---- openers ---------------------------------------------------------
    print("OPENERS  (repeated sentence-opening frames)")
    op = collections.Counter(" ".join(s.split()[:3]).lower().strip(",")
                             for s in sents)
    rep = [(k, v) for k, v in op.most_common(args.top) if v >= 3]
    if rep:
        for k, v in rep:
            print(f"  x{v:<4} {k}")
    else:
        print("  no opening frame repeated three times or more")
    numeric = sum(v for k, v in op.items()
                  if re.match(r"^(one|two|three|four|five|several) \w+", k))
    print(f"  enumerating openers (\"Three things...\"): {numeric}")
    print("  These survive paraphrase, because a paraphraser rewrites words and "
          "keeps\n  the frame. A handful is voice; a dozen is a template.\n")

    # ---- punctuation ------------------------------------------------------
    print("PUNCTUATION  (per 1,000 words)")
    marks = {"em-dash": len(re.findall(r"—|---|\\textemdash", text)),
             "en-dash": text.count("–"),
             "semicolon": text.count(";"),
             "colon": text.count(":"),
             "parenthesis pair": text.count("("),
             "comma": text.count(",")}
    for k, v in marks.items():
        r = 1000 * v / n
        note = ""
        if k == "semicolon":
            note = "   " + band(r, "semicolon_per_1k")
        elif k == "colon":
            note = "   " + band(r, "colon_per_1k")
        print(f"  {k:<18}{v:>6}   {r:6.2f}/1k{note}")
    print("  An em-dash count of zero in a long manuscript is itself a datum: "
          "it means\n  either a house rule or a sweep. Say which, because a "
          "reviewer may wonder.\n")

    # ---- paragraphs -------------------------------------------------------
    paras = [p for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 25]
    if len(paras) >= 8:
        pl = [len(p.split()) for p in paras]
        pcv = statistics.stdev(pl) / statistics.mean(pl)
        print("PARAGRAPHS")
        print(f"  {len(paras)} paragraphs, mean {statistics.mean(pl):.0f} words, "
              f"coefficient of variation {pcv:.2f}")
        print("  Machine drafting tends to produce paragraphs of even size; "
              "below about\n  0.35 is worth a look.\n")

    # ---- richness ---------------------------------------------------------
    freq = collections.Counter(wl)
    hapax = sum(1 for w, c in freq.items() if c == 1) / len(freq)
    win = 400
    ttrs = [len(set(wl[i:i + win])) / win
            for i in range(0, len(wl) - win, win)]
    print("RICHNESS")
    print(f"  types {len(freq):,}, hapax rate {hapax:.2f}   "
          f"{band(hapax, 'hapax_rate')}")
    if ttrs:
        print(f"  type-token ratio over {win}-word windows: "
              f"mean {statistics.mean(ttrs):.2f}, "
              f"s.d. {statistics.stdev(ttrs) if len(ttrs) > 1 else 0:.3f}")
    dm = sum(len(re.findall(r"\b" + re.escape(d) + r"\b", text.lower()))
             for d in DISCOURSE)
    print(f"  discourse markers: {dm} ({1000 * dm / n:.2f}/1k)")
    print("\nNo score is produced and none should be. Every signal above has an "
          "innocent\nreading; the tool's job is to say where to look, and a "
          "human's is to look.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
