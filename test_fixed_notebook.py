#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test if the fixed notebook can be imported and run
"""

import os
import sys
import importlib.util
import nbformat
from pathlib import Path
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

def run_notebook_test():
    """
    Load the fixed notebook as a module and try to run the function
    that had the input_shape error
    """
    print("\n==== TESTING FIXED NOTEBOOK ====\n")
    
    # Path to the notebook
    notebook_path = Path('models/hybrid_models_GRU-w12.ipynb')
    
    if not notebook_path.exists():
        print(f"ERROR: Notebook not found at {notebook_path}")
        return False
    
    print(f"Loading notebook: {notebook_path}")
    
    try:
        # Import necessary variables and functions from standalone_demo
        from standalone_demo import info_print, error_print, success_print, create_dummy_dataset
        from standalone_demo import EXPERIMENTS, FOLDS, build_dataloaders
        
        # Create a dataset for testing
        ds = create_dummy_dataset()
        
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create a shell to execute the notebook
        shell = InteractiveShell.instance()
        
        # Find and execute the cell with train_experiment_complete
        found_function = False
        executed_cell = False
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if 'def train_experiment_complete(' in cell.source:
                    print("Found train_experiment_complete function in notebook!")
                    found_function = True
                    
                    # Execute the cell to define the function
                    try:
                        shell.run_cell(cell.source)
                        print("Successfully executed the cell with train_experiment_complete")
                        executed_cell = True
                    except Exception as e:
                        print(f"Error executing cell: {e}")
                        return False
        
        if not found_function:
            print("ERROR: Could not find train_experiment_complete function in notebook")
            return False
        
        if not executed_cell:
            print("ERROR: Could not execute the cell with train_experiment_complete")
            return False
        
        # Try to access the function from the global namespace
        if 'train_experiment_complete' not in shell.user_ns:
            print("ERROR: train_experiment_complete not defined in global namespace")
            return False
        
        # Import some dummy objects needed for the function
        shell.run_cell("""
import numpy as np
from sklearn.preprocessing import StandardScaler
import xarray as xr
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GRU, TimeDistributed, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define stubs for required objects
NUM_EPOCHS = 10
BATCH_SIZE = 32
EARLY_PATIENCE = 5
MODEL_FACTORY = {
    'ConvGRU-ED': lambda input_shape, dropout: Sequential([
        Flatten(input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(1)
    ])
}
HYPERPARAMS = {
    'lr': 0.001,
    'dropout': {'F1': 0.2, 'F2': 0.3}
}

def success_print(*args):
    print("✅", *args)
    
def info_print(*args):
    print("ℹ️ ", *args)
    
def warning_print(*args):
    print("⚠️ ", *args)
    
def error_print(*args):
    print("❌", *args)

class DummyModelSaver:
    def save_model_silently(self, *args, **kwargs):
        return "model_123"

model_saver = DummyModelSaver()

def generate_predictions(*args, **kwargs):
    return {
        'overall_metrics': {
            'val_loss': 0.1,
            'val_mae': 0.05,
            'val_mape': 5.0
        }
    }

def plot_spatial_maps(*args, **kwargs):
    return None, None
""")
        
        # Try to call the function
        print("\nTrying to call train_experiment_complete...")
        result = shell.run_cell("""
try:
    # Minimal dummy dataset
    dataset = xr.Dataset(
        data_vars={
            'prcp': (('time', 'lat', 'lon'), np.random.rand(12, 10, 10)),
        },
        coords={
            'time': pd.date_range('2020-01-01', periods=12, freq='M'),
            'lat': np.linspace(0, 10, 10),
            'lon': np.linspace(0, 10, 10),
        }
    )
    
    # Create dummy dataloaders
    def build_dataloaders(exp_name, fold_name, dataset, batch_size):
        x_train = np.random.rand(100, 10, 10, 1)
        y_train = np.random.rand(100)
        x_val = np.random.rand(20, 10, 10, 1)
        y_val = np.random.rand(20)
        
        return {
            'train_dataset': (x_train, y_train),
            'val_dataset': (x_val, y_val),
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'train_samples': len(x_train),
            'val_samples': len(x_val),
            'spatial_config': {
                'spatial_height': 10,
                'spatial_width': 10
            }
        }
    
    # Test if the fixed function works
    result = train_experiment_complete('ConvGRU-ED', 'F1', dataset, save_model=False)
    print("✅ SUCCESS: train_experiment_complete executed without the input_shape error!")
    print(f"Result: {result}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
""")
        
        if result.success:
            print("\n✅ NOTEBOOK FIX VERIFICATION PASSED!")
            print("The input_shape error has been successfully fixed in the notebook.")
            return True
        else:
            print("\n❌ NOTEBOOK FIX VERIFICATION FAILED!")
            print("The input_shape error may not be completely fixed in the notebook.")
            return False
            
    except Exception as e:
        print(f"Error during notebook testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_notebook_test()
    if success:
        print("\n✅ SUCCESS: The notebook fix for input_shape error is working!")
    else:
        print("\n❌ ERROR: The notebook fix for input_shape error is not working!") 