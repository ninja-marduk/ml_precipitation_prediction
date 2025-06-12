import json

# Read the notebook file
notebook_path = 'models/hybrid_models_GRU-w12.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Get the cell content as a single string
content = ''.join(notebook['cells'][0]['source'])

# Add code to use train_data and other variables
# Look for the end of the build_dataloaders function
idx_end = content.find("def safe_is_nan(arr):")

# Find where to insert our code (before the end of the function)
idx_insert = content.rfind("return", 0, idx_end)
if idx_insert == -1:
    # If no return statement, find where to add it
    idx_insert = content.rfind("warning_print", 0, idx_end)
    idx_insert = content.find("\n", idx_insert)

# Add code to populate datasets and return them
new_code = """
    # Now populate train_data and val_data with features
    time_steps = len(dataset.time)
    
    # Create sequence samples for each time window
    for t in range(time_steps - INPUT_WINDOW - OUTPUT_HORIZON + 1):
        # Check if this sample belongs to training or validation set
        is_train = train_mask[t + INPUT_WINDOW - 1]  # Based on the target's first time step
        
        if not (is_train or val_mask[t + INPUT_WINDOW - 1]):
            continue  # Skip if neither train nor validation
        
        # Create input sequence with shape (INPUT_WINDOW, height, width, features)
        input_seq = np.zeros((INPUT_WINDOW, spatial_height, spatial_width, len(features)))
        
        # Fill input sequence with feature data
        for f_idx, feature in enumerate(features):
            if feature in feature_data:
                for w in range(INPUT_WINDOW):
                    if t + w < time_steps:
                        # Extract feature data for this time step
                        f_data = feature_data[feature][t + w]
                        
                        # Ensure it has the right spatial dimensions
                        if len(f_data.shape) == 2 and f_data.shape == (spatial_height, spatial_width):
                            input_seq[w, :, :, f_idx] = f_data
                        elif len(f_data.shape) == 0:  # Scalar
                            input_seq[w, :, :, f_idx] = f_data
                        else:
                            warning_print(f"⚠️ Skipping feature {feature} with incompatible shape {f_data.shape}")
        
        # Extract target sequence (precipitation values for next OUTPUT_HORIZON time steps)
        target_seq = np.zeros(OUTPUT_HORIZON)
        for h in range(OUTPUT_HORIZON):
            if t + INPUT_WINDOW + h < time_steps:
                # Use spatial average of precipitation as target
                target_data = dataset['total_precipitation'].values[t + INPUT_WINDOW + h]
                target_seq[h] = np.mean(target_data)
        
        # Add to appropriate dataset
        if is_train:
            train_data.append(input_seq)
            train_targets.append(target_seq)
        else:
            val_data.append(input_seq)
            val_targets.append(target_seq)
    
    # Convert to numpy arrays
    if train_data:
        train_data = np.array(train_data)
        train_targets = np.array(train_targets)
    if val_data:
        val_data = np.array(val_data)
        val_targets = np.array(val_targets)
    
    info_print(f"✅ Created datasets: Training={len(train_data)} samples, Validation={len(val_data)} samples")
    
    # Create TensorFlow datasets
    import tensorflow as tf
    
    if train_data:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_targets))
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(([], []))
    
    if val_data:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_targets))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_dataset = tf.data.Dataset.from_tensor_slices(([], []))
    
    # Return configuration
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'features': features,
        'feature_shapes': feature_shapes,
        'spatial_config': spatial_config,
        'train_samples': len(train_data),
        'val_samples': len(val_data)
    }
"""

# Insert our new code
fixed_content = content[:idx_insert] + new_code + content[idx_insert:]

# Update the notebook
notebook['cells'][0]['source'] = fixed_content.split('\n')

# Save the updated notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("Added implementation to use train_data and other dataset variables.") 