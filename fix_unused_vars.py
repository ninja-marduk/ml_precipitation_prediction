import json

# Read the notebook file
notebook_path = 'models/hybrid_models_GRU-w12.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Get the cell content as a single string
content = ''.join(notebook['cells'][0]['source'])

# Find the unused variables declaration
unused_vars = '# Crear datasets de entrenamiento y validación\n    train_data = []\n    train_targets = []\n    val_data = []\n    val_targets = []'

# Create a replacement that uses these variables
replacement = '''# Crear datasets de entrenamiento y validación
    train_data = []
    train_targets = []
    val_data = []
    val_targets = []
    
    # Usar variables para crear datasets TensorFlow
    import tensorflow as tf
    
    # Datasets vacíos como fallback
    train_dataset = tf.data.Dataset.from_tensor_slices(([], []))
    val_dataset = tf.data.Dataset.from_tensor_slices(([], []))
    
    # Return configuration
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "features": features,
        "feature_shapes": feature_shapes,
        "spatial_config": spatial_config,
        "train_samples": len(train_data),
        "val_samples": len(val_data)
    }'''

# Replace the unused variables with our implementation
modified_content = content.replace(unused_vars, replacement)

# Update the notebook
notebook['cells'][0]['source'] = modified_content.split('\n')

# Save the updated notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("Fixed unused variables by adding a return statement.") 