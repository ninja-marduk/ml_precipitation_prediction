"""Feature 003 T030 - 2-way ANOVA + Tukey HSD for the factorial.

Fits `R^2 ~ C(feat) + C(variant) + C(feat):C(variant)` on the 18 H=12 cells
(3 seeds x 2 feats x 3 variants). Reports F-statistics and p-values for the
two main effects and the interaction term, plus Shapiro-Wilk on residuals.

If the interaction is significant (p<0.05), runs Tukey HSD on the combined
`feat_variant` factor (6 groups).

Outputs:
  - `.docs/papers/5/data/factorial_anova.json`       ANOVA table as JSON
  - `.docs/papers/5/data/factorial_tukey.csv`         Tukey HSD pairwise (if run)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / 'models' / 'output' / 'V4_GNN_TAT_Models' / 'metrics_factorial_consolidated.csv'
DST_DIR = REPO / '.docs' / 'papers' / '5' / 'data'
ANOVA_DST = DST_DIR / 'factorial_anova.json'
TUKEY_DST = DST_DIR / 'factorial_tukey.csv'


def run(args: argparse.Namespace) -> int:
    print('[T030] 2-way ANOVA (feat x variant) on H=12 R^2')
    if not SRC.exists():
        print(f'  MISSING: {SRC}')
        return 1

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        from scipy import stats as sp_stats
    except ImportError as e:
        print(f'  MISSING package: {e}')
        print('  pip install statsmodels scipy')
        return 1

    long = pd.read_csv(SRC)
    h12 = long[long['H'] == 12].copy()
    assert len(h12) == 18, f'expected 18 H=12 rows, got {len(h12)}'

    # Fit the two-way ANOVA
    # statsmodels formula API is picky about the '^' char in column names.
    h12 = h12.rename(columns={'R^2': 'R2'})
    model = ols('R2 ~ C(feat) + C(variant) + C(feat):C(variant)', data=h12).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print('\n  ANOVA (Type II sum of squares):')
    print(f'  {"effect":<30}  {"SS":>10}  {"df":>5}  {"F":>8}  {"p":>10}')
    print(f'  {"-"*30}  {"-"*10}  {"-"*5}  {"-"*8}  {"-"*10}')
    for idx, row in anova_table.iterrows():
        print(f'  {idx:<30}  {row["sum_sq"]:>10.5f}  {int(row["df"]):>5}  '
              f'{row["F"] if not np.isnan(row["F"]) else float("nan"):>8.3f}  '
              f'{row["PR(>F)"] if not np.isnan(row["PR(>F)"]) else float("nan"):>10.5f}')

    # Shapiro-Wilk on residuals (normality assumption)
    w, p_sw = sp_stats.shapiro(model.resid)
    print(f'\n  Shapiro-Wilk on residuals: W={w:.4f}  p={p_sw:.4f}  '
          f'({"normality OK" if p_sw >= 0.05 else "non-normal residuals — caveat"})')

    # Write JSON
    DST_DIR.mkdir(parents=True, exist_ok=True)
    anova_json = {
        'model': 'R^2 ~ C(feat) + C(variant) + C(feat):C(variant)',
        'n_observations': int(len(h12)),
        'horizon': 12,
        'terms': {},
        'residuals': {
            'shapiro_w': float(w),
            'shapiro_p': float(p_sw),
            'rss': float((model.resid ** 2).sum()),
            'r_squared': float(model.rsquared),
            'r_squared_adj': float(model.rsquared_adj),
        },
    }
    for idx, row in anova_table.iterrows():
        anova_json['terms'][idx] = {
            'sum_sq': float(row['sum_sq']),
            'df': float(row['df']),
            'F': (float(row['F']) if not np.isnan(row['F']) else None),
            'p': (float(row['PR(>F)']) if not np.isnan(row['PR(>F)']) else None),
        }
    ANOVA_DST.write_text(json.dumps(anova_json, indent=2), encoding='utf-8')
    print(f'\n  wrote: {ANOVA_DST.relative_to(REPO)}')

    # Tukey HSD if interaction or any main effect is significant
    interaction_p = anova_table.loc['C(feat):C(variant)', 'PR(>F)']
    feat_p = anova_table.loc['C(feat)', 'PR(>F)']
    variant_p = anova_table.loc['C(variant)', 'PR(>F)']

    run_tukey = (interaction_p < 0.05) or (feat_p < 0.05) or (variant_p < 0.05)
    if run_tukey:
        h12['feat_variant'] = h12['feat'] + '_' + h12['variant']
        tukey = pairwise_tukeyhsd(endog=h12['R2'], groups=h12['feat_variant'], alpha=0.05)
        # pairwise_tukeyhsd returns an object; convert to DataFrame
        tukey_df = pd.DataFrame(tukey._results_table.data[1:],
                                 columns=tukey._results_table.data[0])
        tukey_df.to_csv(TUKEY_DST, index=False)
        print(f'\n  Tukey HSD (6 groups, {len(tukey_df)} pairwise comparisons):')
        print(f'  wrote: {TUKEY_DST.relative_to(REPO)}')
        n_sig = int(tukey_df['reject'].sum())
        print(f'  significant pairs: {n_sig} / {len(tukey_df)}')
    else:
        print('\n  Tukey HSD skipped (no main effect nor interaction reached p<0.05)')

    print('\n[T030] ANOVA complete')
    return 0


if __name__ == '__main__':
    sys.exit(run(argparse.Namespace()))
