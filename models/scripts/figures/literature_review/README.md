# Literature-review figures

Systematic-review figures (vector PDF data charts + TikZ taxonomy/PRISMA diagrams).

| Script | Output(s) | Input data |
|--------|-----------|------------|
| `charts.py` | `docs/papers/1/latex/figures/image{10,11,12,13}.pdf` (metric frequency, research trends, R² ranking, R² boxplots) | `docs/papers/1/data/*.csv` (12 CSVs from `convert_excel_to_csv.py`) |
| `prisma_taxonomy.py` | `docs/papers/1/latex/tikz/{fig_prisma, fig_tree_classification, fig_tree_preprocessing, fig_tree_optimization, fig_tree_combination, fig_tree_postprocessing}.tex` (6 TikZ files) | Hardcoded tree structure (in-script) |
