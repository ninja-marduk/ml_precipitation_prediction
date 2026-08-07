"""Resample manuscript figures to the project's 700 DPI standard.

Several figures are stored at 970 to 1,260 effective DPI for the width they are
placed at, which costs file size without adding anything a reader or a printer
can resolve. This computes the effective DPI from the pixel width and the
`width=` fraction each figure is given in the manuscript, and rewrites the ones
above the target.

Nothing is overwritten in place: originals are copied to `figures/original/`
first, and a dry run is the default.

Usage:
    python models/scripts/figures/resample_to_target_dpi.py            # report only
    python models/scripts/figures/resample_to_target_dpi.py --apply    # rewrite
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
TEX = ROOT / ".docs" / "papers" / "5" / "paper_gmd.tex"
FIGDIR = ROOT / ".docs" / "papers" / "5" / "figures"
BACKUP = FIGDIR / "original"

TARGET_DPI = 700
TEXTWIDTH_CM = 17.0        # Copernicus single-column manuscript text width


def placements(tex_path: Path) -> dict[str, float]:
    """Map figure file stem to the widest fraction of \\textwidth it is drawn at."""
    t = tex_path.read_text(encoding="utf-8")
    out: dict[str, float] = {}
    for m in re.finditer(r"\\includegraphics(\[[^\]]*\])?\{([^}]*)\}", t):
        opt, name = (m.group(1) or ""), m.group(2)
        w = re.search(r"width\s*=\s*([\d.]*)\s*\\(?:text|line)width", opt)
        frac = float(w.group(1)) if w and w.group(1) else 1.0
        out[name] = max(out.get(name, 0.0), frac)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite the oversampled files")
    ap.add_argument("--target", type=int, default=TARGET_DPI)
    args = ap.parse_args()

    place = placements(TEX)
    if not place:
        print("no \\includegraphics found; is the manuscript path right?")
        return

    print(f"target {args.target} DPI over a {TEXTWIDTH_CM:g} cm text width\n")
    print(f"{'figure':<40}{'placed':>8}{'px':>13}{'DPI':>7}{'MB':>7}   action")
    print("-" * 92)

    todo, before, after = [], 0.0, 0.0
    for name, frac in sorted(place.items()):
        path = next((FIGDIR / (name + e) for e in ("", ".png", ".pdf", ".jpg")
                     if (FIGDIR / (name + e)).exists()), None)
        if path is None:
            print(f"{name[:38]:<40}{'':>8}{'no encontrado':>13}")
            continue
        mb = path.stat().st_size / 1e6
        before += mb
        if path.suffix.lower() != ".png":
            after += mb
            print(f"{name[:38]:<40}{frac:>8.2f}{'(vector)':>13}{'':>7}{mb:>7.2f}   sin cambio")
            continue

        with Image.open(path) as im:
            w, h = im.size
        inches = frac * TEXTWIDTH_CM / 2.54
        dpi = w / inches
        if dpi <= args.target * 1.05:
            after += mb
            print(f"{name[:38]:<40}{frac:>8.2f}{f'{w}x{h}':>13}{dpi:>7.0f}{mb:>7.2f}   ya cumple")
            continue

        new_w = max(1, int(round(args.target * inches)))
        new_h = max(1, int(round(h * new_w / w)))
        todo.append((path, new_w, new_h, mb, dpi))
        print(f"{name[:38]:<40}{frac:>8.2f}{f'{w}x{h}':>13}{dpi:>7.0f}{mb:>7.2f}"
              f"   -> {new_w}x{new_h}")

    if not todo:
        print("\nnada que remuestrear")
        return

    if not args.apply:
        print(f"\n{len(todo)} figuras por encima del objetivo. "
              f"Ejecuta con --apply para reescribirlas.")
        return

    BACKUP.mkdir(parents=True, exist_ok=True)
    for path, w, h, mb, _ in todo:
        keep = BACKUP / path.name
        if not keep.exists():
            shutil.copy2(path, keep)
        with Image.open(path) as im:
            im = im.convert("RGBA") if im.mode == "P" else im
            im.resize((w, h), Image.LANCZOS).save(path, optimize=True)
        after += path.stat().st_size / 1e6

    print(f"\noriginales en {BACKUP.relative_to(ROOT)}")
    print(f"total {before:.2f} MB -> {after:.2f} MB "
          f"({100 * (1 - after / before):.0f}% menos)")


if __name__ == "__main__":
    main()
