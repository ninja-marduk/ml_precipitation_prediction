import json
import sys
import ast

# Path to the notebook
notebook_path = '/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/models/hybrid_models_GRU-w12.ipynb'

# Load the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Extract Python code from code cells
code_cells = []
for cell in notebook['cells']:
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        code_cells.append(source)

# Check syntax of each code cell
success = True
for i, code in enumerate(code_cells):
    try:
        ast.parse(code)
        print(f"Cell {i+1}: Syntax OK")
    except SyntaxError as e:
        success = False
        print(f"Cell {i+1}: Syntax Error: {e}")
        lines = code.split('\n')
        if e.lineno <= len(lines):
            print(f"Line {e.lineno}: {lines[e.lineno-1]}")
        print()

if success:
    print("\nAll cells have valid syntax!")
    sys.exit(0)
else:
    print("\nSyntax errors found!")
    sys.exit(1) 