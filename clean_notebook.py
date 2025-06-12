#!/usr/bin/env python3
"""
LIMPIEZA AUTOMÁTICA DEL NOTEBOOK - ELIMINAR DUPLICACIONES Y CORREGIR PROBLEMAS
"""

import json
import re
import ast

def clean_notebook():
    """Limpia el notebook eliminando duplicaciones y corrigiendo problemas"""
    
    print("🧹 LIMPIEZA AUTOMÁTICA DEL NOTEBOOK")
    print("=" * 80)
    
    # Cargar notebook
    with open('models/hybrid_models_GRU-w12.ipynb', 'r') as f:
        notebook = json.load(f)
    
    print(f"📋 Notebook original: {len(notebook['cells'])} celdas")
    
    # Concatenar todo el contenido de la primera celda (la principal)
    main_content = ''.join(notebook['cells'][0]['source'])
    
    print("🔍 ELIMINANDO DUPLICACIONES...")
    
    # 1. ELIMINAR PRIMERA DEFINICIÓN DE build_dataloaders (línea 2420)
    print("   🗑️ Eliminando primera definición de build_dataloaders...")
    
    # Buscar la primera definición de build_dataloaders
    first_build_start = main_content.find('def build_dataloaders(model_key, fold_name, dataset, batch_size=64):')
    if first_build_start != -1:
        # Buscar el final de esta función (hasta la siguiente def o final)
        search_start = first_build_start + 100
        next_def_pattern = r'\ndef [a-zA-Z_]'
        next_def_match = re.search(next_def_pattern, main_content[search_start:])
        
        if next_def_match:
            first_build_end = search_start + next_def_match.start()
        else:
            # Si no hay siguiente def, buscar el final lógico
            first_build_end = main_content.find('\n\n# ', first_build_start + 100)
            if first_build_end == -1:
                first_build_end = first_build_start + 2000  # Fallback
        
        # Eliminar la primera definición
        main_content = main_content[:first_build_start] + main_content[first_build_end:]
        print("      ✅ Primera build_dataloaders eliminada")
    
    # 2. ELIMINAR PRIMERAS DEFINICIONES DE VARIABLES
    print("   🗑️ Eliminando definiciones duplicadas de variables...")
    
    variables_to_clean = [
        ('EXPERIMENTS = {', 'HYPERPARAMS = {'),
        ('FOLDS = {', 'EXPERIMENTS = {'),
        ('MODEL_FACTORY = {', 'FOLDS = {'),
        ('HYPERPARAMS = {', 'MODEL_FACTORY = {')
    ]
    
    for var_start, var_end in variables_to_clean:
        first_occurrence = main_content.find(var_start)
        if first_occurrence != -1:
            # Buscar el final de esta variable (hasta la siguiente variable principal)
            search_from = first_occurrence + len(var_start)
            
            # Buscar el cierre de la variable (equilibrando llaves)
            brace_count = 1
            pos = search_from
            
            while pos < len(main_content) and brace_count > 0:
                if main_content[pos] == '{':
                    brace_count += 1
                elif main_content[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            if brace_count == 0:
                # Encontrar el final de la línea después del cierre
                end_pos = main_content.find('\n', pos) + 1
                
                # Verificar si hay una segunda definición
                second_occurrence = main_content.find(var_start, end_pos)
                if second_occurrence != -1:
                    # Eliminar la primera definición
                    main_content = main_content[:first_occurrence] + main_content[end_pos:]
                    print(f"      ✅ Primera definición de {var_start} eliminada")
    
    # 3. CORREGIR INCONSISTENCIAS EN PARÁMETROS
    print("   🔧 Corrigiendo inconsistencias en parámetros...")
    
    # Cambiar model_key por exp_name en definición de build_dataloaders
    main_content = main_content.replace(
        'def build_dataloaders(model_key, fold_name, dataset, batch_size=64):',
        'def build_dataloaders(exp_name, fold_name, dataset, batch_size=64):'
    )
    
    # Cambiar model_key por exp_name en el cuerpo de la función
    main_content = re.sub(
        r'features = EXPERIMENTS\[model_key\]',
        'features = EXPERIMENTS[exp_name]',
        main_content
    )
    
    print("      ✅ Parámetros unificados a exp_name")
    
    # 4. ELIMINAR FUNCIÓN train_experiment ANTIGUA (si existe y hay train_experiment_complete)
    if 'def train_experiment(' in main_content and 'def train_experiment_complete' in main_content:
        print("   🗑️ Eliminando función train_experiment antigua...")
        
        # Buscar y eliminar la función train_experiment antigua
        old_train_start = main_content.find('def train_experiment(exp_name, fold_name, dataset, save_model=True):')
        if old_train_start != -1:
            # Buscar el final de la función
            search_start = old_train_start + 100
            next_def = re.search(r'\ndef [a-zA-Z_]', main_content[search_start:])
            
            if next_def:
                old_train_end = search_start + next_def.start()
                main_content = main_content[:old_train_start] + main_content[old_train_end:]
                print("      ✅ train_experiment antigua eliminada")
    
    # 5. ASEGURAR CONFIGURACIÓN COMPLETA DE EXPERIMENTOS
    print("   🔧 Verificando configuración de experimentos...")
    
    # Verificar que EXPERIMENTS tenga configuración completa
    if "'ConvGRU-ED':" in main_content and "'feature_list':" not in main_content[main_content.rfind("'ConvGRU-ED':"):]:
        print("      ⚠️ EXPERIMENTS parece incompleta, pero se mantiene la definición existente")
    else:
        print("      ✅ Configuración de EXPERIMENTS parece completa")
    
    # 6. LIMPIAR CÓDIGO MUERTO Y COMENTARIOS INNECESARIOS
    print("   🧹 Limpiando código innecesario...")
    
    # Eliminar líneas de run_complete_training_pipeline() al final si están sueltas
    lines = main_content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Eliminar líneas sueltas de ejecución
        if line.strip() in ['run_complete_training_pipeline()', '# run_complete_training_pipeline()']:
            continue
        cleaned_lines.append(line)
    
    main_content = '\n'.join(cleaned_lines)
    
    # 7. ACTUALIZAR NOTEBOOK
    print("\n💾 GUARDANDO NOTEBOOK LIMPIO...")
    
    # Actualizar contenido de la primera celda
    notebook['cells'][0]['source'] = main_content.split('\n')
    
    # Mantener la segunda celda (funciones de validación)
    print(f"   📋 Celdas finales: {len(notebook['cells'])}")
    
    # Guardar notebook limpio
    with open('models/hybrid_models_GRU-w12.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("✅ NOTEBOOK LIMPIO GUARDADO")
    
    # 8. VERIFICAR RESULTADO
    print("\n🔍 VERIFICANDO LIMPIEZA...")
    
    # Recargar y verificar
    with open('models/hybrid_models_GRU-w12.ipynb', 'r') as f:
        clean_notebook = json.load(f)
    
    clean_content = ''.join([''.join(cell['source']) for cell in clean_notebook['cells']])
    
    # Verificar que no haya duplicaciones
    verification_functions = [
        'def build_dataloaders',
        'EXPERIMENTS = {',
        'FOLDS = {',
        'MODEL_FACTORY = {',
        'HYPERPARAMS = {'
    ]
    
    all_clean = True
    for func in verification_functions:
        count = clean_content.count(func)
        if count > 1:
            print(f"   ❌ {func}: todavía {count} definiciones")
            all_clean = False
        else:
            print(f"   ✅ {func}: {count} definición")
    
    # Verificar parámetros
    if 'def build_dataloaders(exp_name, fold_name, dataset' in clean_content:
        print("   ✅ build_dataloaders usa parámetros correctos")
    else:
        print("   ❌ build_dataloaders parámetros incorrectos")
        all_clean = False
    
    # Verificar llamadas
    model_key_calls = clean_content.count('build_dataloaders(model_key')
    exp_name_calls = clean_content.count('build_dataloaders(exp_name')
    
    print(f"   📊 Llamadas con model_key: {model_key_calls}")
    print(f"   📊 Llamadas con exp_name: {exp_name_calls}")
    
    if model_key_calls > 0:
        print("   ⚠️ Todavía hay llamadas con model_key - necesita corrección manual")
    
    return all_clean

if __name__ == "__main__":
    success = clean_notebook()
    
    print(f"\n" + "=" * 80)
    print("🎯 RESULTADO DE LA LIMPIEZA")
    print("=" * 80)
    
    if success:
        print("✅ LIMPIEZA EXITOSA")
        print("📋 Notebook limpio y listo para usar")
        print("🚀 Ahora puedes ejecutar: run_complete_training_pipeline()")
    else:
        print("⚠️ LIMPIEZA PARCIAL")
        print("📋 Algunas correcciones manuales pueden ser necesarias")
        print("🔧 Revisar notebook y ejecutar validate_data_flow()")
    
    print("\n📋 PRÓXIMOS PASOS:")
    print("   1. Ejecutar validate_data_flow() para verificar")
    print("   2. Probar quick_demo() para validar funcionamiento")
    print("   3. Ejecutar run_complete_training_pipeline() si todo está OK")

# Read the notebook
notebook_path = '/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/models/hybrid_models_GRU-w12.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Get the source code
for cell in notebook['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        # Join all lines, then split to normalize
        full_code = ''.join(source)
        
        # Completely remove any code section with "return Nonefeature"
        full_code = re.sub(r'return None\s*?feature.*?(?=\n\s*?except|\n\s*?if|\n\s*?else|\n\s*?#)', 'return None', full_code, flags=re.DOTALL)
        
        # Fix all unreachable code after return statements
        lines = full_code.splitlines()
        cleaned_lines = []
        in_function = False
        indent_level = 0
        current_indent = ""
        
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check for function definition
            if re.match(r'^\s*def\s+\w+', line):
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                current_indent = " " * indent_level
            
            # Check for return statement
            if in_function and re.match(r'^\s*return\s+', line):
                # Add the return statement
                cleaned_lines.append(line)
                
                # Skip lines until we find one with same or less indentation
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    # If we find a line with same or less indentation, stop skipping
                    if next_indent <= indent_level and next_line.strip():
                        i -= 1  # Back up one line to process it in the main loop
                        break
                    
                    # If we find an "except" at the same indentation level, add it
                    if next_indent == indent_level + 4 and re.match(r'^\s*except\s+', next_line):
                        cleaned_lines.append(next_line)
                        break
                        
                    i += 1
            else:
                # Add normal line
                cleaned_lines.append(line)
            
            i += 1
        
        # Join cleaned lines back
        full_code = '\n'.join(cleaned_lines)
        
        # Split back into lines and update the cell
        cell['source'] = [line + '\n' for line in full_code.splitlines()]

# Write back to file
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook cleaned")

# Now let's run a syntax check
try:
    for cell in notebook['cells']:
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            ast.parse(source)
    print("Syntax check: OK")
except SyntaxError as e:
    print(f"Syntax error: {e}")
    lines = source.split('\n')
    if e.lineno <= len(lines):
        print(f"Line {e.lineno}: {lines[e.lineno-1]}") 