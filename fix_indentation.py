import json
import re

# Read the notebook
notebook_path = '/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/models/hybrid_models_GRU-w12.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Get the source code
for cell in notebook['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        # Join all lines, then split to normalize
        full_code = ''.join(source)
        
        # Fix the indentation issue: find blocks where 'return None' is followed by indented code
        full_code = re.sub(r'return None\s+(\s+)feature_data', 'return None\n\1feature_data', full_code)
        
        # Another approach: fix the specific issue at line 614
        full_code = re.sub(r'return None\s+feature_data', 'return None\n\nfeature_data', full_code)
        
        # Remove any unreachable code
        full_code = re.sub(r'return None.*?(\n\s*except)', r'return None\1', full_code, flags=re.DOTALL)
        
        # Split back into lines and update the cell
        cell['source'] = full_code.splitlines(True)  # Keep line endings

# Write back to file
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook indentation fixed")

# Now let's run a syntax check
import ast
try:
    for cell in notebook['cells']:
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            ast.parse(source)
    print("Syntax check: OK")
except SyntaxError as e:
    print(f"Syntax error: {e}")
    lines = source.split('\n')
    if e.lineno <= len(lines):
        print(f"Line {e.lineno}: {lines[e.lineno-1]}") 