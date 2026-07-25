# Figure generation scripts

All scripts that produce figures for any downstream artefact (papers, thesis, posters, slides, dashboards). Organised by **what the script produces or analyses**, never by which document consumes it.

## Layout

| Subdirectory | Contents |
|--------------|----------|
| `_config.py` | Single source of truth for fonts, colours, DPI, panel labels |
| `benchmark/` | Multi-architecture benchmark figures (horizon, radar, ranking, parameter efficiency, suite, reviewer-annotated variants) |
| `analysis/` | Data and feature analysis figures (KCE clusters, K-selection sweep, spatial R², study area, bimodal climograph) |
| `late_fusion/` | V10 Late Fusion family figures (performance, weights, component contributions) |
| `literature_review/` | Systematic review figures (data charts, PRISMA + taxonomy TikZ) |
| `graphical_abstract/` | Graphical abstract generator + diagnostic tool |

## Invocation

All scripts are run from the repo root:

```bash
python models/scripts/figures/<subdir>/<script>.py
```

Each script bootstraps `_config.py` via `sys.path.insert` so imports work regardless of CWD. The pattern is:

```python
FIGURES_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = FIGURES_ROOT.parents[2]
sys.path.insert(0, str(FIGURES_ROOT))
from _config import setup_paper_style, COLORS, OUTPUT_DPI
```

## Adding a new figure script

1. Pick the subdirectory by **purpose** (benchmark / analysis / specific-model-family / etc.)
2. Name the file by **what it produces**, not by which document consumes it. Forbidden tokens in script names: `paperN`, `chapterN`, `thesis`, `defence`, `submission`, `final`, `fix_*`
3. Use the bootstrap pattern above for `_config` imports
4. Add a row to the subdirectory's `README.md` mapping `script → output → input data`
