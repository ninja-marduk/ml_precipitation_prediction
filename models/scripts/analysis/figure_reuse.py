"""Find figures this manuscript shares with the author's own other documents.

Image screening at publishers looks for a figure appearing twice. It cannot tell
reuse by the same author from misappropriation, so a legitimately reused panel
comes back as a flag and the author has to explain it. The cheap defence is to
know the list before the screener does.

This compares figures by perceptual hash rather than by filename, because the
same plot exported twice, or regenerated at a different DPI, is a different file
and the same image. Two hashes are used together: an average hash, which is
robust to resampling, and a difference hash, which is sensitive to horizontal
structure and separates plots that differ only in their data.

The risk here is narrow and worth stating so the output is not over-read. Every
figure in this project is generated from data by a released script, so there is
no question of splicing, adjustment or synthetic imagery, which is what the
screening tools were built for. What can go wrong is reuse: a figure that also
appears in the thesis or the companion article, without the manuscript saying so.

Needs Pillow. If it is missing the script says so rather than guessing.

Usage:
  python models/scripts/analysis/figure_reuse.py
  python models/scripts/analysis/figure_reuse.py --dirs a/figures b/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIRS = [
    ROOT / ".docs" / "papers" / "5" / "figures",
    ROOT / ".docs" / "papers" / "4" / "figures",
    ROOT / ".docs" / "thesis" / "figures",
    ROOT / ".docs" / "thesis",
]
EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
NEAR = 6            # Hamming distance below which two images are the same plot


def hashes(path, size=16):
    from PIL import Image
    with Image.open(path) as im:
        g = im.convert("L").resize((size, size), Image.LANCZOS)
        px = list(g.getdata())
    mean = sum(px) / len(px)
    ahash = "".join("1" if p > mean else "0" for p in px)
    # difference hash: compare each pixel with its right-hand neighbour
    dh = []
    for r in range(size):
        row = px[r * size:(r + 1) * size]
        dh += ["1" if row[c] > row[c + 1] else "0" for c in range(size - 1)]
    return ahash, "".join(dh)


def dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", type=Path, nargs="*", default=DEFAULT_DIRS)
    args = ap.parse_args()
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("Pillow is not installed; this check needs it (pip install Pillow)")
        return 1

    images = []
    for d in args.dirs:
        if not d.exists():
            print(f"  (no directory {d})")
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXT and p.is_file():
                images.append(p)
    if len(images) < 2:
        print("fewer than two images found")
        return 1

    print(f"{len(images)} images across {len({p.parent for p in images})} "
          f"directories\n")
    hs = {}
    for p in images:
        try:
            hs[p] = hashes(p)
        except Exception as e:                                 # noqa: BLE001
            print(f"  unreadable, skipped: {p.name} ({type(e).__name__})")

    keys = list(hs)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a.parent == b.parent:
                continue          # duplicates inside one document are its own
            da = dist(hs[a][0], hs[b][0])
            dd = dist(hs[a][1], hs[b][1])
            if da <= NEAR and dd <= NEAR * 2:
                pairs.append((da + dd, a, b))
    pairs.sort()

    if not pairs:
        print("No figure appears in more than one of these documents.")
        print("Nothing to disclose on this axis.")
        return 0

    print(f"{len(pairs)} figure(s) appear in more than one document:\n")
    for score, a, b in pairs:
        print(f"  distance {score:>3}")
        print(f"    {a.parent.name}/{a.name}")
        print(f"    {b.parent.name}/{b.name}")
    print("\nReuse of one's own figure is permitted when it is disclosed and the "
          "earlier\ndocument is cited in the caption, and it is a problem when it "
          "is not. An\nimage screener cannot tell those apart, so each pair above "
          "needs a caption\nthat names where the figure first appeared, or a "
          "regenerated figure that is\ngenuinely different.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
