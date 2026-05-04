"""Holm-Bonferroni step-down multiple-testing correction for Paper 5 Table 11.

Applies Holm 1979 procedure to control FWER at α=0.05 across 9 pairwise
hypothesis tests + 3 ANOVA p-values. Replaces raw p-value column with
both raw and Holm-adjusted columns.

Output: scripts/benchmark/output/holm_bonferroni.csv
        scripts/benchmark/output/holm_bonferroni.tex (LaTeX-ready table)
"""
from pathlib import Path

import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def holm_bonferroni(p_values):
    """Holm step-down adjusted p-values. Returns same order as input."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, max(running_max, p * (m - rank)))
        adjusted[orig_idx] = adj
        running_max = adj
    return adjusted


# Tests from Paper 5 Table 11 (pairwise) + ANOVA p-values from §4.x
tests = [
    # name, family, raw_p, effect_size_d, notes
    ("ConvLSTM vs GNN-TAT (overall)",       "pairwise", 0.015,  1.03,  "Mann-Whitney U"),
    ("ConvLSTM vs Late Fusion (overall)",   "pairwise", 0.001,  1.45,  "Mann-Whitney U"),
    ("GNN-TAT vs Late Fusion (overall)",    "pairwise", 0.052,  0.71,  "Nemenyi post-hoc"),
    ("KCE vs BASIC (GNN-TAT)",              "pairwise", 0.036,  0.84,  "Wilcoxon signed-rank"),
    ("PAFC vs BASIC (any architecture)",    "pairwise", 0.395,  0.21,  "Wilcoxon signed-rank"),
    ("Stacking Ensemble vs ConvLSTM",       "pairwise", 0.001,  3.21,  "Mann-Whitney U"),
    ("BiMamba vs ConvLSTM",                 "pairwise", 0.001,  2.85,  "Mann-Whitney U"),
    ("FNO-pure vs FNO-ConvLSTM hybrid",     "pairwise", 0.001,  2.11,  "Mann-Whitney U"),
    ("Sub-cell DEM bundles vs BASIC",       "pairwise", 0.001,  1.92,  "Wilcoxon signed-rank"),
    # ANOVA family (factorial design 3 features × 3 variants × 3 seeds)
    ("Feature bundle main effect",          "anova",    0.041,  None,  "Two-way ANOVA"),
    ("Variant main effect",                 "anova",    0.504,  None,  "Two-way ANOVA"),
    ("Feature × variant interaction",       "anova",    0.599,  None,  "Two-way ANOVA"),
]

raw_p = [t[2] for t in tests]
holm_p = holm_bonferroni(raw_p)

rows = []
for (name, family, p_raw, d, notes), p_holm in zip(tests, holm_p):
    sig_raw = p_raw < 0.05
    sig_holm = p_holm < 0.05
    flipped = sig_raw and not sig_holm
    rows.append({
        "test": name,
        "family": family,
        "p_raw": p_raw,
        "p_holm": round(p_holm, 4),
        "sig_raw": "Yes" if sig_raw else "No",
        "sig_holm": "Yes" if sig_holm else "No",
        "flipped": "FLIPPED" if flipped else "",
        "cohen_d": d,
        "notes": notes,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "holm_bonferroni.csv", index=False)

# Print human summary
print("=" * 75)
print("Holm-Bonferroni multiple testing correction (FWER α=0.05)")
print("=" * 75)
for r in rows:
    flag = " * FLIPPED *" if r["flipped"] == "FLIPPED" else ""
    sig = "✓" if r["sig_holm"] == "Yes" else "✗"
    print(f"  {sig} {r['test']:48s}  raw={r['p_raw']:.4f}  Holm={r['p_holm']:.4f}{flag}")

print(f"\nFlipped (no longer significant after correction): "
      f"{sum(1 for r in rows if r['flipped'])}")
print(f"\nSaved: {OUT / 'holm_bonferroni.csv'}")

# LaTeX-ready table
latex_lines = [
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"\textbf{Test} & \textbf{$p$ (raw)} & \textbf{$p$ (Holm)} & \textbf{Cohen's $d$} & \textbf{Sig. ($\alpha$=0.05)} \\",
    r"\midrule",
]
for r in rows:
    d_str = f"{r['cohen_d']:.2f}" if r['cohen_d'] is not None else "--"
    sig = r"\checkmark" if r["sig_holm"] == "Yes" else r"\textendash"
    latex_lines.append(
        f"{r['test']} & {r['p_raw']:.3f} & {r['p_holm']:.3f} & {d_str} & {sig} \\\\")
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")

(OUT / "holm_bonferroni.tex").write_text("\n".join(latex_lines), encoding="utf-8")
print(f"Saved: {OUT / 'holm_bonferroni.tex'}")
