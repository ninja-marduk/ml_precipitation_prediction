#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for the shapes_paths iteration error in plot_spatial_maps function
"""

import json
import re
from pathlib import Path

def fix_shapes_paths_error():
    """Fix the error with shapes_paths not being iterable"""
    
    # Path to the notebook file
    notebook_path = Path('models/hybrid_models_GRU-w12.ipynb')
    
    # Check if file exists
    if not notebook_path.exists():
        print(f"ERROR: File {notebook_path} not found!")
        return False
    
    print(f"Loading notebook: {notebook_path}")
    
    # Create a backup of the original file
    backup_path = notebook_path.with_suffix('.ipynb.bak2')
    print(f"Creating backup at: {backup_path}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as bf:
                bf.write(notebook_content)
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    # Load the notebook as JSON
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return False
    
    # Find the cell with shapes_paths definition
    shape_paths_modified = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Look for shapes_paths definition
            if "shapes_paths = [" in source:
                print("Found shapes_paths definition")
                
                # Patterns to match and replace
                patterns = [
                    # Pattern 1: Simple Path object
                    (r'shapes_paths\s*=\s*\[\s*DATA_INPUT_DIR\/\'([^\']+)\'', 
                     r'shapes_paths = [str(DATA_INPUT_DIR/\'\1\')'),
                    
                    # Pattern 2: Array of Path objects
                    (r'shapes_paths\s*=\s*\[\s*DATA_INPUT_DIR\/([^\]]+)', 
                     r'shapes_paths = [str(DATA_INPUT_DIR/\1')
                ]
                
                # Try each pattern
                for pattern, replacement in patterns:
                    if re.search(pattern, source):
                        new_source = re.sub(pattern, replacement, source)
                        cell['source'] = [new_source]
                        shape_paths_modified = True
                        print(f"Modified shapes_paths definition: {pattern} -> {replacement}")
                        break
    
    # Find the cell with plot_spatial_maps function that uses shapes_paths
    plot_function_modified = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Look for the function and the line with the error
            if "def plot_spatial_maps(" in source and "for path in shapes_paths:" in source:
                print("Found plot_spatial_maps function")
                
                # Add check for string type
                new_source = source.replace(
                    "for path in shapes_paths:",
                    "# Ensure shapes_paths is iterable and contains strings\n" + 
                    "        if shapes_paths and isinstance(shapes_paths, list):\n" + 
                    "            paths_to_check = shapes_paths\n" +
                    "        else:\n" +
                    "            paths_to_check = [str(shapes_paths)] if shapes_paths else []\n" +
                    "        \n" +
                    "        for path in paths_to_check:"
                )
                
                cell['source'] = [new_source]
                plot_function_modified = True
                print("Modified plot_spatial_maps function to handle non-iterable paths")
                break
    
    # Save the modified notebook
    if shape_paths_modified or plot_function_modified:
        try:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, ensure_ascii=False, indent=1)
            print("✅ Successfully saved modified notebook")
            return True
        except Exception as e:
            print(f"Error saving notebook: {e}")
            return False
    else:
        print("⚠️ No modifications were made to the notebook")
        return False

if __name__ == "__main__":
    print("Running shapes_paths fix...")
    success = fix_shapes_paths_error()
    if success:
        print("✅ Successfully fixed the shapes_paths error")
    else:
        print("❌ Failed to fix the shapes_paths error") 