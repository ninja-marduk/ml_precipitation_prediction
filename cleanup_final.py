import json

# Read notebook
with open('models/hybrid_models_GRU-w12.ipynb', 'r') as f:
    notebook = json.load(f)

content = ''.join(notebook['cells'][0]['source'])

# Find the problematic build_dataloaders with multiple returns
start_idx = content.find('def build_dataloaders(model_key, fold_name, dataset, batch_size=64):')
if start_idx == -1:
    print('Function not found')
    exit()

# Find the first return statement
first_return = content.find('return {', start_idx)
if first_return == -1:
    print('Return statement not found')
    exit()

# Find the end of the first return block
return_end = content.find('}', first_return) + 1

# Find the next valid section (safe_is_nan global function)
next_valid = content.find('def safe_is_nan(arr):', return_end)
if next_valid == -1:
    next_valid = len(content)

# Reconstruct content without the duplicate/broken section
before = content[:return_end]
after = content[next_valid:]
new_content = before + '\n\n' + after

# Update notebook
notebook['cells'][0]['source'] = new_content.split('\n')

# Save
with open('models/hybrid_models_GRU-w12.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print('✅ Removed duplicate and unreachable code successfully') 