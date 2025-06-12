# Solución para el error "all input arrays must have the same shape"

# El problema ocurre en la función build_dataloaders del notebook cuando se intenta apilar 
# características con diferentes formas usando np.stack.

# Para solucionar este error, agrega esta función al principio del notebook:

def safe_stack(arrays, axis=-1):
    """Apila arrays asegurando que todos tengan la misma forma."""
    if not arrays:
        raise ValueError("No hay arrays para apilar")
    
    # Verificar si todas las formas son iguales
    shapes = [arr.shape for arr in arrays]
    if len(set(str(shape) for shape in shapes)) == 1:
        # Todas las formas son iguales, usar stack normal
        return np.stack(arrays, axis=axis)
    
    # Las formas son diferentes, redimensionar todas a la primera forma
    target_shape = arrays[0].shape
    resized_arrays = []
    
    for i, arr in enumerate(arrays):
        if arr.shape != target_shape:
            try:
                resized_arr = np.resize(arr, target_shape)
                resized_arrays.append(resized_arr)
                print(f"Array {i} redimensionado de {arr.shape} a {target_shape}")
            except Exception as e:
                raise ValueError(f"Error al redimensionar array {i} de {arr.shape} a {target_shape}: {str(e)}")
        else:
            resized_arrays.append(arr)
    
    # Apilar los arrays redimensionados
    return np.stack(resized_arrays, axis=axis)

# Luego, encuentra la línea que contiene:
# X_sequences.append(np.stack(X_window, axis=-1))

# Y reemplázala por:
# X_sequences.append(safe_stack(X_window, axis=-1))

print("✅ Sigue las instrucciones para implementar la solución en el notebook.") 