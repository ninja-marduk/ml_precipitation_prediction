import json
import os

# Ruta al archivo del notebook
notebook_path = 'models/hybrid_models_GRU-w12.ipynb'

# Cargar el notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Variables para rastrear si se han realizado cambios
changes_made = False

# Recorrer todas las celdas del notebook
for cell_idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        # Convertir el contenido de la celda a una sola cadena
        content = ''.join(cell['source'])
        
        # 1. Reemplazar model_saver por _model_saver
        if 'model_saver.save_model_silently' in content:
            cell['source'] = [line.replace('model_saver.save_model_silently', '_model_saver.save_model_silently') 
                             for line in cell['source']]
            changes_made = True
            print(f"Corregido: model_saver → _model_saver en celda {cell_idx}")
        
        # 2. Mejorar el manejo de std para evitar warnings
        # Buscar líneas específicas que causan warnings
        for i, line in enumerate(cell['source']):
            # Buscar usos problemáticos de np.std
            if 'std_error = float(np.std(' in line:
                # Reemplazar con versión más segura
                cell['source'][i] = line.replace(
                    'std_error = float(np.std(',
                    'std_error = float(np.nanstd('
                )
                changes_made = True
                print(f"Corregido: np.std → np.nanstd en celda {cell_idx}, línea {i}")
            
            # Buscar otros patrones relacionados con el cálculo de desviación estándar
            if 'np.std(' in line and 'with np.errstate' not in line:
                # Envolver en un bloque with np.errstate si no lo está ya
                indentation = len(line) - len(line.lstrip())
                spaces = ' ' * indentation
                
                # Insertar manejo de errores antes de la línea problemática
                cell['source'].insert(i, f"{spaces}with np.errstate(invalid='ignore', divide='ignore'):\n")
                changes_made = True
                print(f"Añadido: manejo de errores para np.std en celda {cell_idx}, línea {i}")

# Guardar el notebook modificado
if changes_made:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Se guardaron los cambios en {notebook_path}")
else:
    print("No se encontraron problemas para corregir.")

print("Proceso completado.") 