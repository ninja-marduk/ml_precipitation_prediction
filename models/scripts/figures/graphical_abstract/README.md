# Graphical abstract

Generator + quality diagnostic for the systematic-review graphical abstract.

| Script | Output(s) | Input data |
|--------|-----------|------------|
| `generate.py` | `docs/papers/1/latex/figures/image3.{png,pdf}` (1200 DPI raster + vector) | Hardcoded metrics from Phase 28 corrections (in-script) |
| `diagnose.py` | Console quality report + `image3_debug.png` overlay (bounding boxes + issues) | Imports `draw()` from sibling `generate.py` |

Exit code 0 on the diagnostic = pass (0 HIGH, 0 MEDIUM); 1 = HIGH issues found. Run before committing any change to `generate.py`.
