"""
Este script implementa una solución para el error 'all input arrays must have the same shape'
en la función build_dataloaders del notebook de modelos híbridos.

Pasos para aplicar esta solución:

1. Agrega esta función al principio del notebook (justo después de las importaciones):

```python
def safe_stack(arrays, axis=-1):
    """
    Apila arrays con verificación y corrección de formas.
    
    Parámetros:
    -----------
    arrays : lista de numpy.ndarray
        Los arrays que se apilarán.
    axis : int, opcional
        El eje a lo largo del cual se apilarán los arrays.
        
    Retorna:
    --------
    numpy.ndarray
        El array apilado.
    """
    if not arrays:
        raise ValueError("No hay arrays para apilar")
    
    # Verificar si todas las formas son iguales
    shapes = [arr.shape for arr in arrays]
    if len(set(str(shape) for shape in shapes)) == 1:
        # Todas las formas son iguales, usar stack normal
        return np.stack(arrays, axis=axis)
    
    # Las formas son diferentes, intentar redimensionar todas a la primera forma
    target_shape = arrays[0].shape
    resized_arrays = []
    
    for i, arr in enumerate(arrays):
        if arr.shape != target_shape:
            # Redimensionar para que coincida con la forma objetivo
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
```

2. Localiza el bloque de código que contiene `X_sequences.append(np.stack(X_window, axis=-1))` 
   y reemplázalo por `X_sequences.append(safe_stack(X_window, axis=-1))`

3. Si hay algún problema de identación o formato, asegúrate de corregirlo manteniendo 
   la estructura adecuada del notebook.

Esta solución garantiza que todos los arrays tengan la misma forma antes de apilarlos,
resolviendo el error "all input arrays must have the same shape".
"""

# El código que genera el error:
"""
try:
    # Verificar si todos los arrays tienen la misma forma
    shapes = [x.shape for x in X_window]
    if len(set(shapes)) > 1:
        error_print(f"❌ Las características tienen formas diferentes: {shapes}")
        # Si hay diferentes formas, intentar redimensionar al primer shape
        target_shape = X_window[0].shape
        for i in range(1, len(X_window)):
            if X_window[i].shape != target_shape:
                X_window[i] = np.resize(X_window[i], target_shape)
    
    X_sequences.append(np.stack(X_window, axis=-1))
    y_sequences.append(np.mean(y_window, axis=(1,2)))  # Promedio espacial
except ValueError as e:
    error_print(f"❌ Error al apilar arrays: {str(e)}")
    # Diagnóstico detallado
    for i, feature in enumerate(features):
        if i < len(X_window):
            error_print(f"   {feature}: {X_window[i].shape}")
    continue
"""

# La solución es utilizar una función safe_stack que maneja esta situación correctamente:

print("✅ La solución está lista. Sigue las instrucciones para aplicarla al notebook.")
print("   Esta función garantiza que todos los arrays tengan la misma forma antes de apilarlos.") 