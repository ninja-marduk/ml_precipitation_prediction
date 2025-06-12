#!/usr/bin/env python3
"""
ANÁLISIS DETALLADO DEL NOTEBOOK PARA DETECTAR DUPLICACIONES Y PROBLEMAS
"""

import json
import re

def analyze_notebook():
    """Analiza el notebook en detalle para encontrar problemas"""
    
    print("🔍 ANÁLISIS DETALLADO DEL NOTEBOOK")
    print("=" * 80)
    
    # Cargar notebook
    with open('models/hybrid_models_GRU-w12.ipynb', 'r') as f:
        notebook = json.load(f)
    
    print(f"📋 Notebook: {len(notebook['cells'])} celdas")
    
    # Analizar cada celda
    all_content = ""
    for i, cell in enumerate(notebook['cells']):
        content = ''.join(cell['source'])
        all_content += content + "\n"
        print(f"   Celda {i}: {len(content.splitlines())} líneas")
    
    print(f"📋 Total líneas: {len(all_content.splitlines())}")
    
    # 1. DETECTAR FUNCIONES DUPLICADAS
    print("\n🔍 1. FUNCIONES DUPLICADAS:")
    print("-" * 50)
    
    critical_functions = [
        'def build_dataloaders',
        'def run_complete_training_pipeline',
        'def train_all_active_experiments',
        'def train_experiment_complete',
        'def train_experiment(',  # La función original
        'def generate_realistic_predictions',
        'def plot_monthly_predictions',
        'def plot_spatial_maps'
    ]
    
    duplicates_found = []
    for func in critical_functions:
        count = all_content.count(func)
        if count > 1:
            print(f"   🚨 {func}: {count} definiciones (DUPLICADA)")
            duplicates_found.append(func)
            
            # Buscar las posiciones de las duplicaciones
            positions = []
            start = 0
            while True:
                pos = all_content.find(func, start)
                if pos == -1:
                    break
                line_num = all_content[:pos].count('\n') + 1
                positions.append(line_num)
                start = pos + 1
            print(f"      📍 Líneas: {positions}")
        else:
            print(f"   ✅ {func}: {count} definición")
    
    # 2. DETECTAR VARIABLES DUPLICADAS
    print("\n🔍 2. VARIABLES DUPLICADAS:")
    print("-" * 50)
    
    critical_variables = [
        'EXPERIMENTS = {',
        'FOLDS = {',
        'MODEL_FACTORY = {',
        'HYPERPARAMS = {'
    ]
    
    for var in critical_variables:
        count = all_content.count(var)
        if count > 1:
            print(f"   🚨 {var}: {count} definiciones (DUPLICADA)")
            duplicates_found.append(var)
        else:
            print(f"   ✅ {var}: {count} definición")
    
    # 3. ANALIZAR FLUJO DE build_dataloaders
    print("\n🔍 3. ANÁLISIS DE build_dataloaders:")
    print("-" * 50)
    
    # Buscar todas las definiciones de build_dataloaders
    build_patterns = re.findall(
        r'def build_dataloaders\([^)]*\):(.*?)(?=\ndef [^_]|\Z)',
        all_content,
        re.DOTALL
    )
    
    if len(build_patterns) > 1:
        print(f"   🚨 {len(build_patterns)} definiciones de build_dataloaders encontradas")
        
        for i, pattern in enumerate(build_patterns):
            print(f"\n   📋 Definición {i+1}:")
            
            # Verificar parámetros
            if 'model_key' in pattern and 'exp_name' in pattern:
                print("      ⚠️ Usa tanto model_key como exp_name (inconsistente)")
            elif 'model_key' in pattern:
                print("      📝 Usa parámetro: model_key")
            elif 'exp_name' in pattern:
                print("      📝 Usa parámetro: exp_name")
            
            # Verificar extracción de features
            if "EXPERIMENTS[" in pattern and "feature_list" in pattern:
                print("      ✅ Extrae features de EXPERIMENTS")
            else:
                print("      ❌ NO extrae features correctamente")
            
            # Verificar procesamiento de features
            if "for feature in features:" in pattern:
                print("      ✅ Itera sobre features")
            else:
                print("      ❌ NO itera sobre features")
            
            # Verificar return
            if "return {" in pattern:
                print("      ✅ Retorna diccionario")
            else:
                print("      ❌ NO retorna resultado")
    
    # 4. ANALIZAR FLUJO DE ENTRENAMIENTO
    print("\n🔍 4. FLUJO DE ENTRENAMIENTO:")
    print("-" * 50)
    
    # Verificar llamadas a build_dataloaders
    build_calls = re.findall(r'build_dataloaders\([^)]*\)', all_content)
    print(f"   📋 Llamadas a build_dataloaders: {len(build_calls)}")
    
    for i, call in enumerate(build_calls):
        print(f"      {i+1}. {call}")
        
        # Verificar consistencia de parámetros
        if 'exp_name' in call and 'fold_name' in call and 'dataset' in call:
            print("         ✅ Parámetros correctos (exp_name, fold_name, dataset)")
        elif 'model_key' in call:
            print("         ⚠️ Usa model_key (verificar consistencia)")
        else:
            print("         ❌ Parámetros incorrectos")
    
    # 5. VERIFICAR CONFIGURACIÓN DE EXPERIMENTOS
    print("\n🔍 5. CONFIGURACIÓN DE EXPERIMENTOS:")
    print("-" * 50)
    
    # Buscar definiciones de EXPERIMENTS
    exp_matches = re.findall(r"EXPERIMENTS\s*=\s*{([^}]*ConvGRU[^}]*)}", all_content, re.DOTALL)
    
    if len(exp_matches) > 1:
        print(f"   🚨 {len(exp_matches)} definiciones de EXPERIMENTS")
        
        # Analizar la última definición (la que se usará)
        if exp_matches:
            latest_exp = exp_matches[-1]
            
            # Buscar experimentos definidos
            exp_names = re.findall(r"'([^']*GRU[^']*)':", latest_exp)
            print(f"   📋 Experimentos en última definición: {exp_names}")
            
            # Verificar feature_list para cada experimento
            for exp_name in exp_names:
                if f"'{exp_name}'" in latest_exp and 'feature_list' in latest_exp:
                    # Buscar el feature_list específico
                    feature_pattern = rf"'{exp_name}':[^}}]*'feature_list':\s*\[([^\]]*)\]"
                    features_match = re.search(feature_pattern, latest_exp, re.DOTALL)
                    if features_match:
                        features_str = features_match.group(1)
                        feature_count = len(re.findall(r"'[^']*'", features_str))
                        print(f"      • {exp_name}: {feature_count} características")
                    else:
                        print(f"      ❌ {exp_name}: feature_list no encontrada")
                else:
                    print(f"      ⚠️ {exp_name}: configuración incompleta")
    
    # 6. RECOMENDACIONES DE LIMPIEZA
    print("\n🔧 6. RECOMENDACIONES DE LIMPIEZA:")
    print("-" * 50)
    
    if duplicates_found:
        print("   📋 ACCIONES REQUERIDAS:")
        
        for dup in duplicates_found:
            if 'def build_dataloaders' in dup:
                print("   1. ELIMINAR build_dataloaders duplicada - mantener solo la ÚLTIMA")
            elif 'EXPERIMENTS = {' in dup:
                print("   2. ELIMINAR EXPERIMENTS duplicada - mantener solo la ÚLTIMA")
            elif 'def train_experiment(' in dup:
                print("   3. ELIMINAR train_experiment antigua - usar train_experiment_complete")
        
        print("\n   📋 FUNCIONES SEGURAS PARA ELIMINAR:")
        if all_content.count('def train_experiment(') > 0 and all_content.count('def train_experiment_complete') > 0:
            print("      • def train_experiment( - reemplazada por train_experiment_complete")
        
        # Buscar funciones que no se usan
        unused_functions = []
        all_defs = re.findall(r'def ([a-zA-Z_][a-zA-Z0-9_]*)', all_content)
        for func_name in set(all_defs):
            if all_defs.count(func_name) == 1:  # Solo una definición
                # Buscar si se llama en algún lado
                call_pattern = f'{func_name}\\('
                calls = len(re.findall(call_pattern, all_content))
                if calls <= 1:  # Solo la definición
                    unused_functions.append(func_name)
        
        if unused_functions:
            print("      • Funciones posiblemente sin usar:")
            for func in unused_functions[:5]:  # Primeras 5
                print(f"        - {func}")
    else:
        print("   ✅ No se encontraron duplicaciones críticas")
    
    # 7. VERIFICAR PARÁMETROS EN FLUJO COMPLETO
    print("\n🔍 7. VERIFICACIÓN DE PARÁMETROS:")
    print("-" * 50)
    
    # Verificar que run_complete_training_pipeline pase dataset correctamente
    if 'run_complete_training_pipeline' in all_content:
        pipeline_section = re.search(
            r'def run_complete_training_pipeline.*?(?=\ndef|\Z)',
            all_content,
            re.DOTALL
        )
        
        if pipeline_section:
            pipeline_code = pipeline_section.group(0)
            
            if 'train_all_active_experiments(dataset)' in pipeline_code:
                print("   ✅ run_complete_training_pipeline pasa dataset correctamente")
            else:
                print("   ❌ run_complete_training_pipeline NO pasa dataset")
            
            if 'dataset = ds_full' in pipeline_code:
                print("   ✅ Usa ds_full como dataset por defecto")
            else:
                print("   ⚠️ No hay dataset por defecto")
    
    return duplicates_found

if __name__ == "__main__":
    duplicates = analyze_notebook()
    
    print(f"\n" + "=" * 80)
    print("🎯 RESUMEN EJECUTIVO")
    print("=" * 80)
    
    if duplicates:
        print(f"🚨 PROBLEMAS ENCONTRADOS: {len(duplicates)} duplicaciones")
        print("🔧 ACCIÓN REQUERIDA: Limpiar notebook eliminando duplicaciones")
    else:
        print("✅ NOTEBOOK LIMPIO: No se encontraron duplicaciones críticas")
    
    print("📋 SIGUIENTE PASO: Implementar limpieza en el notebook") 