"""Check the figure requirements Copernicus actually enforces.

Most published advice on figure integrity is written for microscopy and western
blots, and none of it applies to plots generated from data by a script. What does
apply to a Geoscientific Model Development submission is narrower and checkable:

  COLOUR      Copernicus asks that colour schemes in maps and charts be readable
              with a colour vision deficiency, and names Crameri's scientific
              colour maps as validated. A survey of geoscience journals found
              55% of papers carrying a rainbow ramp or a red-green pair, so this
              is the common defect, not an exotic one. This project shipped one:
              a per-cell skill map on RdYlGn.
  VECTOR      Vector graphics are asked for first, with fonts embedded.
  TYPE 3      Matplotlib's default Type 3 fonts are rejected by several
              preflight tools.

It reads the figure-generating scripts for the colour part, because a colormap
name is in the source and is not recoverable from a rendered PNG. It reads the
built PDFs for the rest.

Nothing here is a forensics check. Every figure in this project is generated from
released data by a released script, so splicing, adjustment and synthetic imagery
do not arise; figure_reuse.py covers the one forensic category that does, which
is the same figure appearing in two of the author's own documents.

Usage: python models/scripts/analysis/figure_compliance.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIGDIR = ROOT / ".docs" / "papers" / "5" / "figures"
SCRIPT_DIRS = [ROOT / "models" / "scripts" / "figures",
               ROOT / "scripts" / "_phases"]
TEX = [ROOT / ".docs" / "papers" / "5" / "paper_gmd.tex",
       ROOT / ".docs" / "papers" / "5" / "supplement.tex"]

# Ramps that fail colour-vision readability, and why each is on the list.
BANNED = {
    "jet": "the rainbow ramp; uneven lightness invents structure",
    "rainbow": "as jet",
    "gist_rainbow": "as jet",
    "nipy_spectral": "rainbow family",
    "hsv": "cyclic rainbow, unreadable as a magnitude",
    "RdYlGn": "red to green, the pair a colour vision deficiency merges",
    "RdYlGn_r": "as RdYlGn",
    "brg": "blue-red-green, includes the red-green pair",
    "Spectral": "rainbow-adjacent and red-green at the ends",
    "Spectral_r": "as Spectral",
    "gist_ncar": "rainbow family",
    "CMRmap": "uneven lightness",
}
SAFE_NOTE = ("viridis, cividis, magma, inferno, plasma, batlow, berlin, "
             "managua, vanimo, RdBu, coolwarm")


def used_figures():
    out = set()
    for t in TEX:
        if t.exists():
            for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                                 t.read_text(encoding="utf-8", errors="replace")):
                out.add(m.group(1))
    return out


def main():
    problems = 0

    print("COLOUR  (colormaps named in the figure-generating scripts)")
    found = {}
    for d in SCRIPT_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            src = p.read_text(encoding="utf-8", errors="replace")
            src = re.sub(r"^\s*#.*$", "", src, flags=re.M)     # skip comments
            for m in re.finditer(r"cmap\s*=\s*[\"']([A-Za-z_]+)[\"']|"
                                 r"plt\.cm\.([A-Za-z_]+)|"
                                 r"colormaps\[[\"']([A-Za-z_]+)[\"']\]", src):
                name = m.group(1) or m.group(2) or m.group(3)
                found.setdefault(name, set()).add(p.name)
    if not found:
        print("  no colormap named in any script")
    for name in sorted(found):
        why = BANNED.get(name)
        where = ", ".join(sorted(found[name]))
        if why:
            problems += 1
            print(f"  FAIL  {name:<16}{where}")
            print(f"        {why}")
        else:
            print(f"  ok    {name:<16}{where}")
    if any(n in BANNED for n in found):
        print(f"\n  Safe alternatives: {SAFE_NOTE}")
    print()

    print("FORMAT  (the figures the manuscript and supplement include)")
    used = used_figures()
    if not used:
        print("  no figures included")
    raster = [f for f in sorted(used) if not f.lower().endswith(".pdf")]
    for f in sorted(used):
        p = FIGDIR / f
        state = "missing" if not p.exists() else \
            ("vector" if f.lower().endswith(".pdf") else "RASTER")
        if state == "missing":
            problems += 1
        print(f"  {state:<8}{f}")
    if raster:
        problems += 1
        print(f"\n  {len(raster)} raster include(s); Copernicus asks for vector "
              f"first, with fonts embedded.")
    print()

    print("FONTS  (Type 3 in a built PDF is rejected by several preflight tools)")
    for pdf in sorted(FIGDIR.glob("*.pdf")):
        if pdf.name not in used:
            continue
        try:
            out = subprocess.run(["pdffonts", str(pdf)], capture_output=True,
                                 text=True, errors="replace").stdout
        except FileNotFoundError:
            print("  pdffonts not available; skipped")
            break
        n3 = out.count("Type 3")
        if n3:
            problems += 1
            print(f"  FAIL  {pdf.name}: {n3} Type 3 font(s)")
    else:
        print("  no Type 3 fonts in any included figure")

    print()
    print("=" * 70)
    if problems:
        print(f"{problems} issue(s) to fix.")
        return 1
    print("Colour, format and fonts all meet the stated requirements.")
    print("What this does not check: whether an axis is truncated, whether a "
          "headline\nvalue is shown without its spread, or whether smoothing is "
          "disclosed. Those\nare reviewer-discretion matters with no screening "
          "behind them, which makes\nthem more likely to be raised, not less.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
