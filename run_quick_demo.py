#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the quick_demo function from the training pipeline
"""

# Import the necessary modules
import sys
from pathlib import Path

# Add the parent directory to sys.path
BASE_PATH = Path.cwd()
if BASE_PATH not in sys.path:
    sys.path.append(str(BASE_PATH))

# Import the quick_demo function
try:
    # Try to import from the training_pipeline module
    from training_pipeline import quick_demo
    
    print("Running quick demo to test the training pipeline...")
    results = quick_demo()
    
    if results:
        print("\n✅ Quick demo completed successfully!")
        print(f"Results: {results['summary']}")
    else:
        print("\n❌ Quick demo failed!")
except Exception as e:
    print(f"\n❌ Error running quick demo: {e}")
    import traceback
    traceback.print_exc() 