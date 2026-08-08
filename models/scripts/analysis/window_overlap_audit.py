# -*- coding: utf-8 -*-
"""Do consecutive validation windows overlap, and does random fold assignment leak?

A referee claims t[s, 1:] equals t[s+1, :-1] for every consecutive pair, i.e. the
windows slide by one month, so window s at leads 2..12 covers the same target
months as window s+1 at leads 1..11. If so, blocking folds by window index but
assigning windows to folds at random still places 11 of 12 leads of each held-out
window inside the training fold, and the refit is not out of sample either.

Also checks the second claim: that two different arrays in the archive are both
used as the seed-42 GNN-TAT prediction.
"""
import numpy as np
from pathlib import Path

ROOT = Path(r"d:/github.com/ninja-marduk/ml_precipitation_prediction")
OUT = ROOT / "models" / "output"


def load(d, what="predictions"):
    a = np.load(Path(d) / f"{what}.npy").astype(np.float64)
    return a[..., 0] if a.ndim == 5 else a


tgt = load(OUT / "V10_Late_Fusion/SEED42", "targets")
S, H = tgt.shape[0], tgt.shape[1]
print(f"targets: {S} windows x {H} leads x {tgt.shape[2]}x{tgt.shape[3]}\n")

print("=== do consecutive windows overlap? ===")
matches = 0
for s in range(S - 1):
    a, b = tgt[s, 1:], tgt[s + 1, :-1]
    m = np.isfinite(a) & np.isfinite(b)
    if np.allclose(a[m], b[m], rtol=0, atol=1e-6):
        matches += 1
print(f"pairs where t[s,1:] == t[s+1,:-1] : {matches} of {S-1}")
if matches == S - 1:
    print("  CONFIRMED: windows slide by one month and overlap by 11 of 12 leads.")
    print("  Blocking by window index with random fold assignment therefore leaves")
    print("  most of each held-out window's target months inside the training fold.")

print("\n=== how much of a held-out window is present in training? ===")
rng = np.random.default_rng(42)
order = rng.permutation(np.arange(S))
folds = np.array_split(order, 5)
leaked = tot = 0
for held in folds:
    tr = set(int(x) for x in order if x not in set(int(y) for y in held))
    for s in held:
        s = int(s)
        for h in range(H):
            tot += 1
            month = s + h + 1          # absolute index of the target month
            # is that month covered by any training window?
            if any(0 <= month - (t + 1) < H for t in tr):
                leaked += 1
print(f"held-out (window, lead) pairs whose target month also appears in a")
print(f"training window: {leaked:,} of {tot:,}  ({100*leaked/tot:.1f}%)")

print("\n=== contiguous blocking, for comparison ===")
leaked_c = tot_c = 0
for held in np.array_split(np.arange(S), 5):
    tr = set(range(S)) - set(int(x) for x in held)
    for s in held:
        s = int(s)
        for h in range(H):
            tot_c += 1
            month = s + h + 1
            if any(0 <= month - (t + 1) < H for t in tr):
                leaked_c += 1
print(f"contiguous folds: {leaked_c:,} of {tot_c:,}  ({100*leaked_c/tot_c:.1f}%)")

print("\n=== two arrays for the same seed-42 GNN-TAT prediction? ===")
a = load(OUT / "V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT")
b = load(OUT / "V4_GNN_TAT_Models/SEED42/map_exports/H12/BASIC/GNN_TAT_GAT")


def r2(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return float(1 - np.sum((y[m] - p[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2))


print(f"  root-level  map_exports/... : R2 = {r2(a, tgt):.4f}")
print(f"  SEED42/     map_exports/... : R2 = {r2(b, tgt):.4f}")
print(f"  identical arrays            : {np.array_equal(a, b)}")
print("  The root-level array is the pre-correction run on the leaked graph; the")
print("  SEED42 one is the corrected retrain. Any script reading the root path is")
print("  using the superseded predictions.")
