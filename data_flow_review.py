#!/usr/bin/env python3
"""
CODE REVIEW: Validación del flujo de datos en el sistema de entrenamiento
"""

import json
import re

def analyze_data_flow():
    """Analiza el flujo de datos en el notebook de entrenamiento"""
    
    print("🔍 CODE REVIEW: FLUJO DE DATOS EN EL SISTEMA DE ENTRENAMIENTO")
    print("=" * 80)
    
    # Leer el notebook
    try:
        with open('models/hybrid_models_GRU-w12.ipynb', 'r') as f:
            notebook = json.load(f)
        
        # Concatenar todo el código
        full_content = ''.join([''.join(cell['source']) for cell in notebook['cells']])
        
        print(f"📋 Notebook cargado: {len(notebook['cells'])} celdas")
        print(f"📋 Líneas totales de código: {len(full_content.split())}")
        
    except Exception as e:
        print(f"❌ Error leyendo notebook: {e}")
        return
    
    # 1. ANÁLISIS DE FUNCIONES DUPLICADAS
    print("\n🔍 1. ANÁLISIS DE FUNCIONES DUPLICADAS")
    print("-" * 40)
    
    critical_functions = [
        'def run_complete_training_pipeline',
        'def train_all_active_experiments', 
        'def train_experiment_complete',
        'def build_dataloaders',
        'EXPERIMENTS = {'
    ]
    
    for func in critical_functions:
        count = full_content.count(func)
        status = "⚠️" if count > 1 else "✅"
        print(f"   {status} {func}: {count} definiciones")
        
        if count > 1:
            print(f"      🚨 PROBLEMA: Función/variable duplicada puede causar conflictos")
    
    # 2. ANÁLISIS DE CONFIGURACIÓN DE EXPERIMENTOS
    print("\n🔍 2. ANÁLISIS DE CONFIGURACIÓN DE EXPERIMENTOS")
    print("-" * 40)
    
    # Buscar definiciones de EXPERIMENTS
    experiments_matches = re.findall(r"EXPERIMENTS\s*=\s*{[^}]*'ConvGRU-ED'[^}]*}", full_content, re.DOTALL)
    
    if experiments_matches:
        print(f"   ✅ Encontradas {len(experiments_matches)} configuraciones de EXPERIMENTS")
        
        # Analizar la primera configuración
        exp_config = experiments_matches[0]
        
        # Buscar experimentos definidos
        experiment_names = re.findall(r"'([^']*GRU[^']*)':", exp_config)
        print(f"   📋 Experimentos encontrados: {experiment_names}")
        
        # Verificar feature_list para cada experimento
        for exp_name in experiment_names:
            feature_pattern = rf"'{exp_name}':\s*{{[^}}]*'feature_list':\s*\[([^\]]*)\]"
            features_match = re.search(feature_pattern, exp_config, re.DOTALL)
            
            if features_match:
                features_str = features_match.group(1)
                feature_count = len(re.findall(r"'[^']*'", features_str))
                print(f"      • {exp_name}: {feature_count} características")
            else:
                print(f"      ❌ {exp_name}: Sin feature_list definida")
    else:
        print("   ❌ No se encontró configuración de EXPERIMENTS")
    
    # 3. ANÁLISIS DEL FLUJO DE DATOS
    print("\n🔍 3. ANÁLISIS DEL FLUJO DE DATOS")
    print("-" * 40)
    
    # Verificar flujo: pipeline -> train_all -> train_experiment -> build_dataloaders
    flow_checks = [
        ("run_complete_training_pipeline", "train_all_active_experiments(dataset)"),
        ("train_all_active_experiments", "train_experiment_complete(exp_name, fold_name, dataset"),
        ("train_experiment_complete", "build_dataloaders(exp_name, fold_name, dataset"),
        ("build_dataloaders", "EXPERIMENTS[.*]['feature_list']")
    ]
    
    for function, expected_call in flow_checks:
        if function in full_content:
            # Buscar la definición de la función
            func_start = full_content.find(f"def {function}")
            if func_start != -1:
                # Buscar el final de la función (siguiente def o final)
                next_def = full_content.find("\ndef ", func_start + 10)
                if next_def == -1:
                    func_content = full_content[func_start:]
                else:
                    func_content = full_content[func_start:next_def]
                
                # Verificar si hace la llamada esperada
                if re.search(expected_call, func_content):
                    print(f"   ✅ {function}: Pasa datos correctamente")
                else:
                    print(f"   ❌ {function}: NO pasa datos como esperado")
                    print(f"      Buscando: {expected_call}")
            else:
                print(f"   ❌ {function}: Función no encontrada")
        else:
            print(f"   ❌ {function}: No existe en el código")
    
    # 4. ANÁLISIS DE EXTRACCIÓN DE CARACTERÍSTICAS
    print("\n🔍 4. ANÁLISIS DE EXTRACCIÓN DE CARACTERÍSTICAS")
    print("-" * 40)
    
    # Buscar la función build_dataloaders más reciente
    build_dataloaders_matches = re.findall(
        r"def build_dataloaders\([^)]*\):(.*?)(?=\ndef|\Z)", 
        full_content, 
        re.DOTALL
    )
    
    if build_dataloaders_matches:
        latest_build = build_dataloaders_matches[-1]  # Usar la última definición
        
        # Verificar pasos críticos
        critical_steps = [
            ("features = EXPERIMENTS[.*]['feature_list']", "Extrae features del experimento"),
            ("for feature in features:", "Itera sobre features específicas"),
            ("if feature in dataset:", "Verifica feature en dataset"),
            ("feature_data\\[feature\\]", "Almacena datos de feature"),
            ("return", "Retorna datasets construidos")
        ]
        
        for pattern, description in critical_steps:
            if re.search(pattern, latest_build):
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description} - FALTA")
                
        # Verificar manejo de características especiales
        special_features = [
            "cluster_elevation",
            "total_precipitation", 
            "geopotential",
            "temperature",
            "specific_humidity"
        ]
        
        print(f"\n   🔍 Manejo de características especiales:")
        for feature in special_features:
            if feature in latest_build:
                print(f"      ✅ {feature}: Manejada")
            else:
                print(f"      ⚠️ {feature}: No encontrada en build_dataloaders")
    else:
        print("   ❌ build_dataloaders: Función no encontrada")
    
    # 5. ANÁLISIS DE VALIDACIÓN DE ENTRADA
    print("\n🔍 5. ANÁLISIS DE VALIDACIÓN DE ENTRADA")
    print("-" * 40)
    
    validation_patterns = [
        ("if dataset is None:", "Valida dataset de entrada"),
        ("if exp_name not in EXPERIMENTS:", "Valida experimento válido"),
        ("if.*not.*dataset:", "Valida dataset no vacío"),
        ("feature.*in.*dataset", "Valida features en dataset")
    ]
    
    for pattern, description in validation_patterns:
        if re.search(pattern, full_content, re.IGNORECASE):
            print(f"   ✅ {description}")
        else:
            print(f"   ⚠️ {description} - FALTA")
    
    # 6. RECOMENDACIONES
    print("\n🔍 6. RECOMENDACIONES DE MEJORA")
    print("-" * 40)
    
    recommendations = []
    
    # Verificar duplicaciones
    for func in critical_functions:
        if full_content.count(func) > 1:
            recommendations.append(f"Eliminar definiciones duplicadas de {func.replace('def ', '').replace(' = {', '')}")
    
    # Verificar validaciones faltantes
    if not re.search(r"assert.*dataset", full_content):
        recommendations.append("Añadir validaciones assert para dataset")
    
    if not re.search(r"logging.*feature.*extracted", full_content):
        recommendations.append("Añadir logging detallado de extracción de features")
    
    if not re.search(r"shape.*check", full_content):
        recommendations.append("Añadir verificación de shapes de datos")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("   ✅ No se encontraron problemas críticos")
    
    print("\n" + "=" * 80)
    print("🎯 RESUMEN DEL CODE REVIEW COMPLETADO")
    print("=" * 80)

if __name__ == "__main__":
    analyze_data_flow() 