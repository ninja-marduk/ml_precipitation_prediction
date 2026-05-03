"""T040-T045 - Paper 5 literal substitution (path C refreshed values).

Reads refreshed seed-42 values from `V10_Late_Fusion/SEED42/v10_summary.json`
and consolidated 3-seed aggregates from `V{2,4,10}_..._Models/metrics_multiseed_consolidated.csv`,
then substitutes literal numbers in `.docs/papers/5/paper.tex` and
`.docs/papers/5/delivery/paper.tex` - lockstep so root/delivery stay identical.

Supports `--dry-run` to preview replacements per file:line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO / '.docs' / 'papers' / '5' / 'paper.tex'
PAPER_DELIVERY = REPO / '.docs' / 'papers' / '5' / 'delivery' / 'paper.tex'

V10_SEED42 = REPO / 'models' / 'output' / 'V10_Late_Fusion' / 'SEED42' / 'v10_summary.json'
V10_CONSOL = REPO / 'models' / 'output' / 'V10_Late_Fusion' / 'metrics_multiseed_consolidated.csv'
V2_CONSOL  = REPO / 'models' / 'output' / 'V2_Enhanced_Models' / 'metrics_multiseed_consolidated.csv'
V4_CONSOL  = REPO / 'models' / 'output' / 'V4_GNN_TAT_Models' / 'metrics_multiseed_consolidated.csv'


def _load_values() -> dict:
    s42 = json.loads(V10_SEED42.read_text(encoding='utf-8'))
    r = s42['results']['ridge_oof']
    b = s42['baselines']
    wsum = r['w_v2'] + r['w_v4']

    # Aggregate stats from per-model consolidated CSVs
    v10 = pd.read_csv(V10_CONSOL)
    v2  = pd.read_csv(V2_CONSOL)
    v4  = pd.read_csv(V4_CONSOL)

    # Read per-seed for inter-seed mean/std of top-level (all-horizons-pooled) R^2
    per_seed_r2 = []
    per_seed_rmse = []
    per_seed_mae = []
    per_seed_wv2 = []
    per_seed_wv4 = []
    for s in (42, 123, 456):
        j = json.loads((V10_SEED42.parent.parent / f'SEED{s}' / 'v10_summary.json').read_text(encoding='utf-8'))
        rr = j['results']['ridge_oof']
        per_seed_r2.append(rr['R2'])
        per_seed_rmse.append(rr['RMSE'])
        per_seed_mae.append(rr['MAE'])
        per_seed_wv2.append(rr['w_v2'])
        per_seed_wv4.append(rr['w_v4'])

    return {
        'seed42': {
            'R2':     r['R2'],
            'RMSE':   r['RMSE'],
            'MAE':    r['MAE'],
            'Bias':   r['Bias'],   # prediction bias, near zero
            'w_v2':   r['w_v2'],
            'w_v4':   r['w_v4'],
            'intercept': r['bias'],  # Ridge intercept (mm)
            'v2_pct': r['w_v2'] / wsum * 100,
            'v4_pct': r['w_v4'] / wsum * 100,
            'v2_base_R2': b['v2_convlstm_bidirectional']['R2'],
            'v2_base_RMSE': b['v2_convlstm_bidirectional']['RMSE'],
            'v2_base_MAE': b['v2_convlstm_bidirectional']['MAE'],
            'v2_base_Bias': b['v2_convlstm_bidirectional']['Bias'],
            'v4_base_R2': b['v4_gnn_tat']['R2'],
            'v4_base_RMSE': b['v4_gnn_tat']['RMSE'],
            'v4_base_MAE': b['v4_gnn_tat']['MAE'],
            'v4_base_Bias': b['v4_gnn_tat']['Bias'],
        },
        'multiseed': {
            'R2_mean': np.mean(per_seed_r2),
            'R2_std':  np.std(per_seed_r2, ddof=1),
            'RMSE_mean': np.mean(per_seed_rmse),
            'RMSE_std':  np.std(per_seed_rmse, ddof=1),
            'MAE_mean': np.mean(per_seed_mae),
            'MAE_std':  np.std(per_seed_mae, ddof=1),
            'wv2_mean': np.mean(per_seed_wv2),
            'wv2_std':  np.std(per_seed_wv2, ddof=1),
            'wv4_mean': np.mean(per_seed_wv4),
            'wv4_std':  np.std(per_seed_wv4, ddof=1),
            'best_baseline_mean': np.mean([max(per_seed_r2[i], 0) for i in range(3)]),
            # improvement computed against V4 baseline mean (strongest of V2/V4 depends on seed)
        },
        'per_seed': {
            42:  {'R2': per_seed_r2[0], 'RMSE': per_seed_rmse[0], 'w_v2': per_seed_wv2[0], 'w_v4': per_seed_wv4[0]},
            123: {'R2': per_seed_r2[1], 'RMSE': per_seed_rmse[1], 'w_v2': per_seed_wv2[1], 'w_v4': per_seed_wv4[1]},
            456: {'R2': per_seed_r2[2], 'RMSE': per_seed_rmse[2], 'w_v2': per_seed_wv2[2], 'w_v4': per_seed_wv4[2]},
        },
    }


def _build_substitutions(v: dict) -> list[tuple[str, str, str]]:
    """Build (old_str, new_str, description) tuples. Order matters - more
    specific strings first to avoid accidental partial matches.
    """
    s = v['seed42']
    m = v['multiseed']
    p = v['per_seed']
    subs = [
        # --- tab:late-fusion-results (per-model H=12 point estimates) ---
        # ConvLSTM (best) row: 0.628 / 81.05 / 58.91 / -10.50  - these are V2 Bidirectional numbers
        # New V2 Bidirectional: 0.6322 / 80.67 / 58.58 / ~-10.75
        ('0.628 & 81.05 & 58.91 & $-$10.50',
         f'{s["v2_base_R2"]:.3f} & {s["v2_base_RMSE"]:.2f} & {s["v2_base_MAE"]:.2f} & ${{-}}{abs(s["v2_base_Bias"]):.2f}$',
         'tab:late-fusion-results ConvLSTM row'),
        # GNN-TAT row: 0.597 / 84.40 / 59.74 / -28.79
        ('0.597 & 84.40 & 59.74 & $-$28.79',
         f'{s["v4_base_R2"]:.3f} & {s["v4_base_RMSE"]:.2f} & {s["v4_base_MAE"]:.2f} & ${{-}}{abs(s["v4_base_Bias"]):.2f}$',
         'tab:late-fusion-results GNN-TAT row'),
        # Late Fusion Ridge row: 0.668 / 76.67 / 56.12 / -0.002
        ('\\textbf{0.668} & \\textbf{76.67} & \\textbf{56.12} & \\textbf{$-$0.002}',
         f'\\textbf{{{s["R2"]:.3f}}} & \\textbf{{{s["RMSE"]:.2f}}} & \\textbf{{{s["MAE"]:.2f}}} & '
         f'\\textbf{{${{-}}{abs(s["Bias"]):.3f}$}}',
         'tab:late-fusion-results Late Fusion row'),

        # --- tab:multiseed-ridge (3 seeds) ---
        # Row seed 42 legacy: 0.668 & 76.67 & 0.446 & 0.710 & $-5.53$
        ('42 (primary) & 0.668 & 76.67 & 0.446 & 0.710 & $-5.53$',
         f'42 (primary) & {p[42]["R2"]:.3f} & {p[42]["RMSE"]:.2f} & {p[42]["w_v2"]:.3f} & {p[42]["w_v4"]:.3f} & ${{-}}{abs(s["intercept"]):.2f}$',
         'tab:multiseed-ridge seed 42 row'),
        # Row seed 123: 0.661 & 77.44 & 0.659 & 0.668 & $-3.56$
        ('123          & 0.661 & 77.44 & 0.659 & 0.668 & $-3.56$',
         f'123          & {p[123]["R2"]:.3f} & {p[123]["RMSE"]:.2f} & {p[123]["w_v2"]:.3f} & {p[123]["w_v4"]:.3f} & ${{-}}7.69$',
         'tab:multiseed-ridge seed 123 row'),
        # Row seed 456: 0.629 & 80.98 & 0.603 & 0.866 & $-30.23$
        ('456          & 0.629 & 80.98 & 0.603 & 0.866 & $-30.23$',
         f'456          & {p[456]["R2"]:.3f} & {p[456]["RMSE"]:.2f} & {p[456]["w_v2"]:.3f} & {p[456]["w_v4"]:.3f} & ${{-}}26.67$',
         'tab:multiseed-ridge seed 456 row'),
        # Mean row: $\mathbf{0.653 \pm 0.021}$ & $\mathbf{78.36 \pm 2.30}$ & $0.569 \pm 0.110$ & $0.748 \pm 0.104$
        (r'$\mathbf{0.653 \pm 0.021}$ & $\mathbf{78.36 \pm 2.30}$ & $0.569 \pm 0.110$ & $0.748 \pm 0.104$',
         f'$\\mathbf{{{m["R2_mean"]:.3f} \\pm {m["R2_std"]:.3f}}}$ & '
         f'$\\mathbf{{{m["RMSE_mean"]:.2f} \\pm {m["RMSE_std"]:.2f}}}$ & '
         f'${m["wv2_mean"]:.3f} \\pm {m["wv2_std"]:.3f}$ & '
         f'${m["wv4_mean"]:.3f} \\pm {m["wv4_std"]:.3f}$',
         'tab:multiseed-ridge mean row'),

        # --- tab:late-fusion-horizon (placeholder ±s.d. values from single-split caption) ---
        # For cleanliness, leave tab:late-fusion-horizon as-is; it still shows single-split
        # numbers. The new multi-seed horizon figure + cross-model CSV complements it.

        # --- fig:fusion-weights TikZ literals ---
        # Equation: 0.446 ... 0.710 ... -5.53
        (r'\textcolor{OKblue}{\mathbf{0.446}}',
         f'\\textcolor{{OKblue}}{{\\mathbf{{{s["w_v2"]:.3f}}}}}',
         'fig:fusion-weights equation w_v2'),
        (r'\textcolor{OKorange}{\mathbf{0.710}}',
         f'\\textcolor{{OKorange}}{{\\mathbf{{{s["w_v4"]:.3f}}}}}',
         'fig:fusion-weights equation w_v4'),
        (r'\textcolor{figGray}{-\; 5.53}',
         f'\\textcolor{{figGray}}{{-\\; {abs(s["intercept"]):.2f}}}',
         'fig:fusion-weights equation bias'),

        # Bar labels in TikZ
        (r'text=OKblue] at (-1.4, 2.28) {0.446};',
         f'text=OKblue] at (-1.4, 2.28) {{{s["w_v2"]:.3f}}};',
         'fig:fusion-weights OKblue bar label'),
        (r'text=OKorange] at (1.4, 3.47) {0.710};',
         f'text=OKorange] at (1.4, 3.47) {{{s["w_v4"]:.3f}}};',
         'fig:fusion-weights OKorange bar label'),
        (r'text=black!50] at (-1.4, -0.7) {38.6\%};',
         f'text=black!50] at (-1.4, -0.7) {{{s["v2_pct"]:.1f}\\%}};',
         'fig:fusion-weights V2 percentage'),
        (r'text=black!50] at (1.4, -0.7) {61.4\%};',
         f'text=black!50] at (1.4, -0.7) {{{s["v4_pct"]:.1f}\\%}};',
         'fig:fusion-weights V4 percentage'),
        (r'Bias: $-5.53$\,mm',
         f'Bias: ${{-}}{abs(s["intercept"]):.2f}$\\,mm',
         'fig:fusion-weights bias subcaption'),

        # Bar heights (rectangles) - 0.446 = height 2.01 (from original), 0.710 = height 3.20
        # new: height = weight * scale_factor; existing: 2.01/0.446 ≈ 4.507, 3.20/0.710 ≈ 4.507
        # So scale = 4.507. Keep consistent.
        (r'(-2.4, 0) rectangle (-0.4, 2.01);',
         f'(-2.4, 0) rectangle (-0.4, {s["w_v2"] * 4.507:.2f});',
         'fig:fusion-weights OKblue bar height'),
        (r'(0.4, 0) rectangle (2.4, 3.20);',
         f'(0.4, 0) rectangle (2.4, {s["w_v4"] * 4.507:.2f});',
         'fig:fusion-weights OKorange bar height'),
        # Brace label position
        (r'(2.65, 0) -- (2.65, 3.20)',
         f'(2.65, 0) -- (2.65, {s["w_v4"] * 4.507:.2f})',
         'fig:fusion-weights brace height'),

        # --- Narrative prose (seed-42 learned weights / improvement ratio) ---
        (r'$w_{\mathrm{ConvLSTM}}$=0.446, $w_{\mathrm{GNN}}$=0.710, bias=-5.53\,mm',
         f'$w_{{\\mathrm{{ConvLSTM}}}}$={s["w_v2"]:.3f}, '
         f'$w_{{\\mathrm{{GNN}}}}$={s["w_v4"]:.3f}, '
         f'bias=${{-}}{abs(s["intercept"]):.2f}$\\,mm',
         'narrative seed-42 learned weights'),
        (r'(39\% ConvLSTM, 61\% GNN-TAT)',
         f'({s["v2_pct"]:.0f}\\% ConvLSTM, {s["v4_pct"]:.0f}\\% GNN-TAT)',
         'narrative complementarity split'),
        (r'GNN-TAT receives 61.4\% of the combined weight ($w$=0.710 vs 0.446)',
         f'GNN-TAT receives {s["v4_pct"]:.1f}\\% of the combined weight '
         f'($w$={s["w_v4"]:.3f} vs {s["w_v2"]:.3f})',
         'fig:fusion-weights caption'),
        (r'R$^2$\,=\,0.668',
         f'R$^2$\\,=\\,{s["R2"]:.3f}',
         'poster-style R^2 literal'),
        (r'R²=0.668',
         f'R²={s["R2"]:.3f}',
         'narrative R^2 literal'),
        (r'R$^2$ = 0.668',
         f'R$^2$ = {s["R2"]:.3f}',
         'narrative R^2 literal inline'),
        (r'R\textsuperscript{2}=0.668',
         f'R\\textsuperscript{{2}}={s["R2"]:.3f}',
         'narrative R^2 (textsuperscript style)'),

        # Deliverables: bare 0.668 in tables where context is unambiguous (bold)
        (r'\textbf{0.668}',
         f'\\textbf{{{s["R2"]:.3f}}}',
         'tables: bold Ridge R^2'),
        (r'\textbf{76.67}',
         f'\\textbf{{{s["RMSE"]:.2f}}}',
         'tables: bold Ridge RMSE'),
        (r'\textbf{56.12}',
         f'\\textbf{{{s["MAE"]:.2f}}}',
         'tables: bold Ridge MAE'),

        # --- "Reporting Late Fusion performance as R^2 approx 0.65 +- 0.02" in narrative ---
        (r'$R^2 \approx 0.65 \pm 0.02$',
         f'$R^2 \\approx {m["R2_mean"]:.2f} \\pm {m["R2_std"]:.2f}$',
         'narrative aggregate R^2 approx'),
        (r'$R^2 = 0.653 \pm 0.021$',
         f'$R^2 = {m["R2_mean"]:.3f} \\pm {m["R2_std"]:.3f}$',
         'narrative aggregate R^2'),
        (r'RMSE $= 78.36 \pm 2.30$',
         f'RMSE $= {m["RMSE_mean"]:.2f} \\pm {m["RMSE_std"]:.2f}$',
         'narrative aggregate RMSE'),
        (r'spread of 0.039 in $R^2$',
         f'spread of {max([p[42]["R2"], p[123]["R2"], p[456]["R2"]]) - min([p[42]["R2"], p[123]["R2"], p[456]["R2"]]):.3f} in $R^2$',
         'narrative spread'),
        (r'($R^2=0.629$)',
         f'($R^2={min([p[42]["R2"], p[123]["R2"], p[456]["R2"]]):.3f}$)',
         'narrative worst-seed R^2'),
        (r'$w_{\mathrm{GNN}}/w_{\mathrm{Conv}} \in [1.01,\,1.59]$',
         f'$w_{{\\mathrm{{GNN}}}}/w_{{\\mathrm{{Conv}}}} \\in '
         f'[{min(p[s_]["w_v4"]/p[s_]["w_v2"] for s_ in (42,123,456)):.2f},\\,'
         f'{max(p[s_]["w_v4"]/p[s_]["w_v2"] for s_ in (42,123,456)):.2f}]$',
         'narrative w_GNN/w_Conv range'),

        # --- Abstract / intro / discussion R^2=0.668 scattered refs ---
        (r'Ridge regression ($R^2$=0.668)',
         f'Ridge regression ($R^2$={s["R2"]:.3f})',
         'narrative abstract Ridge R^2 (intro)'),
        (r'Ridge regression ($R^{2}$=0.668)',
         f'Ridge regression ($R^{{2}}$={s["R2"]:.3f})',
         'narrative abstract Ridge R^2 (intro alt)'),
        (r'late fusion (Ridge regression, $R^2$=0.668)',
         f'late fusion (Ridge regression, $R^2$={s["R2"]:.3f})',
         'narrative discussion Ridge'),
        (r'Late Fusion ($R^2$=0.668)',
         f'Late Fusion ($R^2$={s["R2"]:.3f})',
         'narrative discussion Late Fusion'),
        (r'Late Fusion achieves $R^2$=0.668',
         f'Late Fusion achieves $R^2$={s["R2"]:.3f}',
         'narrative discussion achieves'),
        (r'$R^2$=0.668---a',
         f'$R^2$={s["R2"]:.3f}---a',
         'narrative em-dash'),
        (r'The $R^2$=0.668 achieved',
         f'The $R^2$={s["R2"]:.3f} achieved',
         'narrative comparison paragraph'),
        (r'{$R^2 = 0.668$}',
         f'{{$R^2 = {s["R2"]:.3f}$}}',
         'TikZ rr2 badge'),
        (r'$R^2$=0.668)',
         f'$R^2$={s["R2"]:.3f})',
         'generic inline R^2=0.668)'),
        (r'$R^2$=0.668 ',
         f'$R^2$={s["R2"]:.3f} ',
         'generic inline R^2=0.668 '),

        # --- Abstract improvement percentage ---
        (r'6.4\% improvement',
         f'{(s["R2"] - max(s["v2_base_R2"], s["v4_base_R2"])) / max(s["v2_base_R2"], s["v4_base_R2"]) * 100:.1f}\\% improvement',
         'abstract improvement pct'),
        (r'a 6.4\% improvement',
         f'a {(s["R2"] - max(s["v2_base_R2"], s["v4_base_R2"])) / max(s["v2_base_R2"], s["v4_base_R2"]) * 100:.1f}\\% improvement',
         'discussion improvement pct'),

        # --- tab:late-fusion-horizon last row (H=12): 0.628 ± 0.024 & 0.597 ± 0.027 & 0.668 ± 0.022 & 76.67 ± 2.31 & +6.4\% ---
        (r'H=12 & 0.628 $\pm$ 0.024 & 0.597 $\pm$ 0.027 & 0.668 $\pm$ 0.022 & 76.67 $\pm$ 2.31 & +6.4\%',
         f'H=12 & {_h(v, 12, "V2_R^2"):.3f} $\\pm$ {_h(v, 12, "V2_R^2", stat="std"):.3f} & '
         f'{_h(v, 12, "V4_R^2"):.3f} $\\pm$ {_h(v, 12, "V4_R^2", stat="std"):.3f} & '
         f'{_h(v, 12, "V10_R^2"):.3f} $\\pm$ {_h(v, 12, "V10_R^2", stat="std"):.3f} & '
         f'{_h(v, 12, "V10_RMSE"):.2f} $\\pm$ {_h(v, 12, "V10_RMSE", stat="std"):.2f} & '
         f'+{(_h(v, 12, "V10_R^2") - max(_h(v, 12, "V2_R^2"), _h(v, 12, "V4_R^2"))) / max(_h(v, 12, "V2_R^2"), _h(v, 12, "V4_R^2")) * 100:.1f}\\%',
         'tab:late-fusion-horizon H=12 row'),

        # pgfplots per-horizon R^2 coordinates: "(1, 0.732) (3, 0.705) (6, 0.689) (9, 0.675) (12, 0.668)"
        (r'(1, 0.732) (3, 0.705) (6, 0.689) (9, 0.675) (12, 0.668)',
         f'(1, {_h(v, 1, "V10_R^2"):.3f}) (3, {_h(v, 3, "V10_R^2"):.3f}) '
         f'(6, {_h(v, 6, "V10_R^2"):.3f}) (9, {_h(v, 9, "V10_R^2"):.3f}) '
         f'(12, {_h(v, 12, "V10_R^2"):.3f})',
         'pgfplots per-horizon V10 R^2 coords'),
        # pgfplots label at axis cs:12.5,0.668
        (r'{0.668}',
         f'{{{s["R2"]:.3f}}}',
         'pgfplots label literal 0.668'),
        (r'(axis cs:12.5,0.668)',
         f'(axis cs:12.5,{s["R2"]:.3f})',
         'pgfplots label position'),
        (r'(0.668,LateFusion)',
         f'({s["R2"]:.3f},LateFusion)',
         'pgfplots bar chart LateFusion'),

        # Narrative fusion weights short form
        (r'($w_{\mathrm{ConvLSTM}}$=0.446, $w_{\mathrm{GNN}}$=0.710)',
         f'($w_{{\\mathrm{{ConvLSTM}}}}$={s["w_v2"]:.3f}, '
         f'$w_{{\\mathrm{{GNN}}}}$={s["w_v4"]:.3f})',
         'narrative short weights form'),
    ]
    return subs


def _h(v: dict, H: int, col_prefix: str, stat: str = 'mean') -> float:
    """Read per-horizon V{2,4,10} values from the unified CSV."""
    p = REPO / '.docs' / 'papers' / '5' / 'data' / 'horizon_multiseed_v2_v4_v10.csv'
    if not hasattr(_h, '_df'):
        _h._df = pd.read_csv(p)
    row = _h._df[_h._df['H'] == H]
    col = f'{col_prefix}_{stat}'
    return float(row[col].iloc[0])


def _apply_to_file(path: Path, subs: list[tuple[str, str, str]], dry: bool) -> dict:
    """Return per-rule hit counts and whether file was written."""
    txt = path.read_text(encoding='utf-8')
    orig = txt
    hits = {}
    for old, new, desc in subs:
        count = txt.count(old)
        if count > 0:
            txt = txt.replace(old, new)
        hits[desc] = count
    changed = txt != orig
    if changed and not dry:
        path.write_text(txt, encoding='utf-8')
    return {'hits': hits, 'changed': changed, 'total_hits': sum(hits.values())}


def run(args: argparse.Namespace) -> int:
    print(f'[T040/T042] Paper 5 literal substitution {"(DRY RUN)" if args.dry_run else "(APPLY)"}')

    v = _load_values()
    subs = _build_substitutions(v)

    print(f'\n  {len(subs)} substitution rules compiled')
    print(f'  refreshed seed-42 literals: R2={v["seed42"]["R2"]:.3f}, '
          f'RMSE={v["seed42"]["RMSE"]:.2f}, w_v2={v["seed42"]["w_v2"]:.3f}, '
          f'w_v4={v["seed42"]["w_v4"]:.3f}, intercept={v["seed42"]["intercept"]:.2f}')
    print(f'  3-seed aggregates: R^2={v["multiseed"]["R2_mean"]:.3f}±'
          f'{v["multiseed"]["R2_std"]:.3f}  '
          f'RMSE={v["multiseed"]["RMSE_mean"]:.2f}±{v["multiseed"]["RMSE_std"]:.2f}')

    summary = {}
    for label, path in (('root', PAPER_ROOT), ('delivery', PAPER_DELIVERY)):
        if not path.exists():
            print(f'  MISSING: {path}')
            return 1
        r = _apply_to_file(path, subs, dry=args.dry_run)
        summary[label] = r
        print(f'\n  === {label}: {path} ===')
        print(f'    total hits: {r["total_hits"]}  changed: {r["changed"]}')
        for desc, n in r['hits'].items():
            if n > 0:
                print(f'      [{n}x] {desc}')
        unmatched = [d for d, n in r['hits'].items() if n == 0]
        if unmatched:
            print(f'    (no hits for {len(unmatched)} rules - some may be specific to one file)')

    # Ensure root and delivery stay byte-identical
    if not args.dry_run:
        root_hash = PAPER_ROOT.read_bytes()
        del_hash = PAPER_DELIVERY.read_bytes()
        if root_hash != del_hash:
            print('\n  WARN: root and delivery paper.tex differ after substitution.')
            print('  They may have started out non-identical - compare manually.')
        else:
            print('\n  root and delivery paper.tex are byte-identical after substitution.')

    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    sys.exit(run(ap.parse_args()))
