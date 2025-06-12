import json
with open('models/hybrid_models_GRU-w12.ipynb', 'r') as f:
    notebook = json.load(f)
content = ''.join(notebook['cells'][0]['source'])
start_idx = content.find('def build_dataloaders(model_key, fold_name, dataset, batch_size=64):')
first_return = content.find('return {', start_idx)
return_end = content.find('}', first_return) + 1
next_valid = content.find('def safe_is_nan(arr):', return_end)
if next_valid == -1:
    next_valid = len(content)
before = content[:return_end]
after = content[next_valid:]
new_content = before + '\n\n' + after
notebook['cells'][0]['source'] = new_content.split('\n')
with open('models/hybrid_models_GRU-w12.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
print('✅ Cleaned up code successfully') 