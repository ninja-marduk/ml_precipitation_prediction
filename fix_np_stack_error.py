#!/usr/bin/env python
# Fix for the "all input arrays must have the same shape" error in build_dataloaders function

import numpy as np

def safe_stack(arrays, axis=-1):
    """
    Stack arrays along a new axis with shape verification and correction.
    
    Parameters:
    -----------
    arrays : list of numpy.ndarray
        The arrays to be stacked.
    axis : int, optional
        The axis along which the arrays will be stacked.
        
    Returns:
    --------
    numpy.ndarray
        The stacked array.
    """
    if not arrays:
        raise ValueError("No arrays to stack")
    
    # Check if all shapes are the same
    shapes = [arr.shape for arr in arrays]
    if len(set(str(shape) for shape in shapes)) == 1:
        # All shapes are the same, use normal stack
        return np.stack(arrays, axis=axis)
    
    # Shapes are different, try to resize all to the first shape
    target_shape = arrays[0].shape
    resized_arrays = []
    
    for i, arr in enumerate(arrays):
        if arr.shape != target_shape:
            # Resize to match target shape
            try:
                resized_arr = np.resize(arr, target_shape)
                resized_arrays.append(resized_arr)
                print(f"Resized array {i} from {arr.shape} to {target_shape}")
            except Exception as e:
                raise ValueError(f"Failed to resize array {i} from {arr.shape} to {target_shape}: {str(e)}")
        else:
            resized_arrays.append(arr)
    
    # Stack the resized arrays
    return np.stack(resized_arrays, axis=axis)

# Usage in build_dataloaders:
"""
Replace:
    X_sequences.append(np.stack(X_window, axis=-1))

With:
    X_sequences.append(safe_stack(X_window, axis=-1))
"""

print("✅ Fix ready! Add this function to your notebook and replace np.stack with safe_stack") 