import re
import os

path = os.path.join("d:", os.sep, "github.com", "ninja-marduk", "ml_precipitation_prediction", "docs", "papers", "5", "paper.tex")
with open(path, "r", encoding="utf-8") as fh:
    text = fh.read()

# Extract \citep{} and \citet{} keys
cite_pattern = re.compile(r'\\cite[pt]\{([^}]+)\}')
cite_matches = cite_pattern.findall(text)

cited_keys = set()
for match in cite_matches:
    for key in match.split(','):
        cited_keys.add(key.strip())

# Extract \bibitem{} keys
bibitem_pattern = re.compile(r'\\bibitem\{([^}]+)\}')
bibitem_keys = set(bibitem_pattern.findall(text))

print("=== ALL CITED KEYS ===")
for k in sorted(cited_keys):
    print("  " + k)
print("Total unique cited keys:", len(cited_keys))

print()
print("=== ALL BIBITEM KEYS ===")
for k in sorted(bibitem_keys):
    print("  " + k)
print("Total bibitem keys:", len(bibitem_keys))

print()
print("=== ORPHAN BIBITEMS (in bibliography but NEVER cited) ===")
orphans = sorted(bibitem_keys - cited_keys)
for k in orphans:
    print("  " + k)
if not orphans:
    print("  (none)")
print("Total orphans:", len(orphans))

print()
print("=== MISSING BIBITEMS (cited but NO matching bibitem) ===")
missing = sorted(cited_keys - bibitem_keys)
for k in missing:
    print("  " + k)
if not missing:
    print("  (none)")
print("Total missing:", len(missing))
