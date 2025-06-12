#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for the input_shape error in train_experiment_complete function
"""

import re
import json
from pathlib import Path

def fix_input_shape_error():
    # Path to the notebook file
    notebook_path = Path('models/hybrid_models_GRU-w12.ipynb')
    
    # Check if file exists
    if not notebook_path.exists():
        print(f"ERROR: File {notebook_path} not found!")
        return False
    
    print(f"Loading notebook: {notebook_path}")
    
    # Read the notebook content
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the cell with the train_experiment_complete function
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            cell_source = ''.join(cell['source'])
            if 'def train_experiment_complete(' in cell_source:
                print(f"Found train_experiment_complete in cell {cell_idx}")
                
                # Find the problematic section
                pattern = r'(x_train = dataloader_config\[\'x_train\'\].*?x_val = scalerX\.transform\s*\(x_val\.reshape\(-1, input_shape\[-1\]\)\)\.reshape\(x_val\.shape\))'
                match = re.search(pattern, cell_source, re.DOTALL)
                
                if match:
                    problematic_section = match.group(1)
                    print(f"Found problematic section at position {match.start()}")
                    
                    # Create the fixed section
                    fixed_section = '''        train_dataset = dataloader_config['train_dataset']
        val_dataset = dataloader_config['val_dataset']
        spatial_config = dataloader_config['spatial_config']
        x_train = dataloader_config['x_train']
        y_train = dataloader_config['y_train']
        x_val = dataloader_config['x_val']
        y_val = dataloader_config['y_val']
        
        # Dimensiones de entrada (definir antes de usarlo en el escalado)
        if 'spatial_height' not in spatial_config or 'spatial_width' not in spatial_config:
            raise ValueError("Configuración espacial incompleta")
            
        spatial_height = spatial_config['spatial_height']
        spatial_width = spatial_config['spatial_width']
        num_features = len(features)
        input_shape = (spatial_height, spatial_width, num_features)

        # escalado
        scalerX = StandardScaler()
        x_train = scalerX.fit_transform(x_train.reshape(-1, num_features)).reshape(x_train.shape)
        x_val = scalerX.transform  (x_val.reshape(-1, num_features)).reshape(x_val.shape)'''
                    
                    # Replace the problematic section
                    fixed_content = cell_source.replace(problematic_section, fixed_section)
                    
                    # Update the cell source
                    notebook['cells'][cell_idx]['source'] = [fixed_content]
                    
                    # Remove the redundant input_shape definition later in the function
                    later_pattern = r'(# Dimensiones de entrada.*?input_shape = \(spatial_height, spatial_width, num_features\))'
                    later_match = re.search(later_pattern, fixed_content, re.DOTALL)
                    
                    if later_match and later_match.start() > match.end():
                        print(f"Removing redundant input_shape definition")
                        redundant_section = later_match.group(1)
                        final_content = fixed_content.replace(redundant_section, "# Using input_shape already defined above")
                        notebook['cells'][cell_idx]['source'] = [final_content]
                    
                    print("Applied fixes to the function")
                    break
                else:
                    print("Could not find the problematic section")
    
    # Save the modified notebook
    backup_path = notebook_path.with_suffix('.ipynb.bak')
    print(f"Creating backup at: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"Saving fixed notebook to: {notebook_path}")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print("Fix complete!")
    return True

if __name__ == "__main__":
    print("Running input_shape error fix...")
    success = fix_input_shape_error()
    if success:
        print("✅ Successfully fixed the input_shape error")
    else:
        print("❌ Failed to fix the input_shape error") 