import json
import re

# Read the notebook
notebook_path = '/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/models/hybrid_models_GRU-w12.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Convert to string to do the replacements
notebook_str = json.dumps(notebook)

# Replace all instances of the problem
notebook_str = notebook_str.replace('return Nonefeature}")', 'return None')
notebook_str = notebook_str.replace('return Nonefeature}\\")\\n', 'return None\\n')
notebook_str = notebook_str.replace('return Nonefeature}\")\n', 'return None\n')
notebook_str = notebook_str.replace('return Nonefeature}', 'return None')
notebook_str = re.sub(r'return None\w+', 'return None', notebook_str)

# Convert back to object
fixed_notebook = json.loads(notebook_str)

# Write back to file
with open(notebook_path, 'w') as f:
    json.dump(fixed_notebook, f, indent=1)

print("Notebook fixed")

# Let's verify by checking the file for remaining issues
with open(notebook_path, 'r') as f:
    content = f.read()
    remaining = re.findall(r'return None\w+', content)
    if remaining:
        print(f"Warning: Still found {len(remaining)} issues: {remaining}")
    else:
        print("All issues fixed!") 