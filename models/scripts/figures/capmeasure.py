"""How long is every caption in the package, in words of running prose."""
import pathlib
import re
import sys

CAP = re.compile(r"\\caption\{")
LAB = re.compile(r"\\label\{([^}]*)\}")


def captions(s):
    out = []
    for m in CAP.finditer(s):
        i = m.end()
        d, j = 1, i
        while d:
            if s[j] == "{":
                d += 1
            elif s[j] == "}":
                d -= 1
            j += 1
        cap = s[i:j - 1]
        lab = LAB.search(s[j:j + 400])
        txt = re.sub(r"\\[a-zA-Z]+", " ", cap)
        txt = re.sub(r"[{}$~\\]", " ", txt)
        out.append((len(txt.split()), lab.group(1) if lab else "?"))
    return out


LIMIT = int(sys.argv[1])
for name in sys.argv[2:]:
    p = pathlib.Path(name)
    print(f"== {p.name}")
    over = 0
    for w, lab in captions(p.read_text(encoding="utf-8")):
        flag = "   OVER" if w > LIMIT else ""
        over += w > LIMIT
        print(f"  {w:4d}  {lab}{flag}")
    print(f"  ({over} over {LIMIT})")
