import json

# Read the notebook file
notebook_path = 'models/hybrid_models_GRU-w12.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Get the source code of the first cell
source = notebook['cells'][0]['source']
content = ''.join(source)

# Find and fix the problematic section
problematic_code = "if feature in available_features:\n            continue\n                \n            if feature in dataset:"
fixed_code = "if feature in available_features:\n            continue\n                \n        if feature in dataset:"

# Apply the fix
fixed_content = content.replace(problematic_code, fixed_code)

# Update the notebook
notebook['cells'][0]['source'] = fixed_content.split('\n')

# Write back to the file
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("Fixed unreachable code in build_dataloaders function.") 