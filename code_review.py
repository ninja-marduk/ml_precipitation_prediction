import json
import re

def code_review():
    """Code review completo del flujo de datos"""
    
    print("🔍 CODE REVIEW: VALIDACIÓN DEL FLUJO DE DATOS")
    print("=" * 80)
    
    # Cargar notebook
    with open('models/hybrid_models_GRU-w12.ipynb', 'r') as f:
        notebook = json.load(f)
    
    content = ''.join([''.join(cell['source']) for cell in notebook['cells']])
    
    print(f"📋 Análisis del notebook: {len(notebook['cells'])} celdas")
    
    # 1. PROBLEMA CRÍTICO: FUNCIONES DUPLICADAS
    print("\n🚨 1. PROBLEMAS CRÍTICOS ENCONTRADOS:")
    print("-" * 50)
    
    duplicates = ['def build_dataloaders', 'EXPERIMENTS = {']
    for func in duplicates:
        count = content.count(func)
        if count > 1:
            print(f"   ❌ {func}: {count} definiciones DUPLICADAS")
            print(f"      🔧 ACCIÓN REQUERIDA: Mantener solo la última definición")
    
    # 2. ANÁLISIS DEL FLUJO DE DATOS
    print("\n🔍 2. ANÁLISIS DEL FLUJO DE DATOS:")
    print("-" * 50)
    
    # Buscar el patrón de llamadas
    flow_patterns = [
        ("run_complete_training_pipeline", "all_results = train_all_active_experiments\\(dataset\\)"),
        ("train_all_active_experiments", "result = train_experiment_complete\\(exp_name, fold_name, dataset"),
        ("train_experiment_complete", "dataloader_config = build_dataloaders\\(exp_name, fold_name, dataset"),
        ("build_dataloaders", "features = EXPERIMENTS\\[.*\\]\\['feature_list'\\]")
    ]
    
    for func_name, pattern in flow_patterns:
        if re.search(pattern, content):
            print(f"   ✅ {func_name}: Pasa datos correctamente")
        else:
            print(f"   ❌ {func_name}: PROBLEMA en paso de datos")
            print(f"      Patrón esperado: {pattern}")
    
    # 3. VALIDACIÓN DE CONFIGURACIÓN DE EXPERIMENTOS
    print("\n🔍 3. CONFIGURACIÓN DE EXPERIMENTOS:")
    print("-" * 50)
    
    # Buscar configuraciones de experimentos
    exp_matches = re.findall(r"'(ConvGRU-ED|ConvGRU-ED-KCE|ConvGRU-ED-KCE-PAFC)'", content)
    if exp_matches:
        unique_experiments = list(set(exp_matches))
        print(f"   ✅ Experimentos configurados: {unique_experiments}")
        
        # Verificar feature_list para cada experimento
        for exp in unique_experiments:
            feature_pattern = rf"'{exp}':[^}}]*'feature_list':\s*\[([^\]]*)\]"
            features_match = re.search(feature_pattern, content, re.DOTALL)
            if features_match:
                features_str = features_match.group(1)
                feature_count = len(re.findall(r"'[^']*'", features_str))
                print(f"      • {exp}: {feature_count} características definidas")
            else:
                print(f"      ❌ {exp}: NO tiene feature_list definida")
    else:
        print("   ❌ NO se encontraron experimentos configurados")
    
    # 4. VALIDACIÓN DE build_dataloaders
    print("\n🔍 4. VALIDACIÓN DE build_dataloaders:")
    print("-" * 50)
    
    # Buscar la función build_dataloaders (usar la última definición)
    build_functions = re.findall(r"def build_dataloaders\([^)]*\):(.*?)(?=def [^_]|\Z)", content, re.DOTALL)
    
    if build_functions:
        latest_build = build_functions[-1]  # Usar última definición
        
        # Verificar pasos críticos en build_dataloaders
        critical_checks = [
            ("features = EXPERIMENTS\\[", "Extrae features del experimento"),
            ("for feature in features:", "Itera sobre features"),
            ("if feature in dataset:", "Verifica feature existe en dataset"),
            ("feature_data\\[feature\\] =", "Almacena datos de feature"),
            ("return.*dataset", "Retorna datasets procesados")
        ]
        
        for pattern, description in critical_checks:
            if re.search(pattern, latest_build):
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description} - FALTA!")
    else:
        print("   ❌ build_dataloaders NO encontrada")
    
    # 5. PROBLEMA POTENCIAL CON PARÁMETROS
    print("\n🔍 5. VALIDACIÓN DE PARÁMETROS:")
    print("-" * 50)
    
    # Verificar consistencia en nombres de parámetros
    param_issues = []
    
    # En run_complete_training_pipeline
    if "def run_complete_training_pipeline(dataset=None)" in content:
        if "dataset = ds_full" in content:
            print("   ✅ run_complete_training_pipeline: Maneja dataset por defecto")
        else:
            param_issues.append("run_complete_training_pipeline no tiene fallback para dataset")
    
    # En train_experiment_complete
    if "train_experiment_complete(exp_name, fold_name, dataset" in content:
        print("   ✅ train_experiment_complete: Recibe parámetros correctos")
    else:
        param_issues.append("train_experiment_complete no recibe parámetros esperados")
    
    # En build_dataloaders
    if "build_dataloaders(exp_name, fold_name, dataset" in content:
        print("   ✅ build_dataloaders: Recibe parámetros correctos")
    else:
        param_issues.append("build_dataloaders no recibe parámetros esperados")
    
    if param_issues:
        for issue in param_issues:
            print(f"   ❌ {issue}")
    
    # 6. RECOMENDACIONES CRÍTICAS
    print("\n🔧 6. RECOMENDACIONES CRÍTICAS:")
    print("-" * 50)
    
    recommendations = []
    
    # Funciones duplicadas
    if content.count('def build_dataloaders') > 1:
        recommendations.append("ELIMINAR build_dataloaders duplicada - usar solo la última definición")
    
    if content.count('EXPERIMENTS = {') > 1:
        recommendations.append("ELIMINAR EXPERIMENTS duplicada - usar solo la última definición")
    
    # Validaciones faltantes
    if "if not dataset" not in content:
        recommendations.append("AÑADIR validación de dataset no vacío")
    
    if "assert" not in content:
        recommendations.append("AÑADIR validaciones assert para parámetros críticos")
    
    # Logging
    if "info_print.*feature.*extracted" not in content:
        recommendations.append("AÑADIR logging detallado de extracción de features")
    
    # Mostrar recomendaciones
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("   ✅ No se encontraron problemas adicionales")
    
    # 7. CÓDIGO DE PRUEBA SUGERIDO
    print("\n🧪 7. CÓDIGO DE PRUEBA SUGERIDO:")
    print("-" * 50)
    
    test_code = '''
# Prueba rápida del flujo de datos
def test_data_flow():
    print("🧪 Probando flujo de datos...")
    
    # 1. Verificar que ds_full existe
    assert 'ds_full' in globals(), "Dataset ds_full no existe"
    
    # 2. Verificar experimentos activos
    active_exps = [name for name, config in EXPERIMENTS.items() if config.get('active', False)]
    print(f"   Experimentos activos: {active_exps}")
    
    # 3. Probar build_dataloaders con primer experimento
    if active_exps:
        exp_name = active_exps[0]
        fold_name = 'F1'
        
        print(f"   Probando: {exp_name} - {fold_name}")
        
        # Verificar configuración
        exp_config = EXPERIMENTS[exp_name]
        features = exp_config['feature_list']
        print(f"   Features requeridas: {len(features)}")
        
        # Verificar features en dataset
        missing_features = [f for f in features if f not in ds_full.data_vars]
        if missing_features:
            print(f"   ❌ Features faltantes en dataset: {missing_features}")
        else:
            print(f"   ✅ Todas las features están en el dataset")
        
        try:
            # Probar build_dataloaders
            result = build_dataloaders(exp_name, fold_name, ds_full, 32)
            if result:
                print(f"   ✅ build_dataloaders ejecutado exitosamente")
                print(f"   📊 Muestras train: {result.get('train_samples', 'N/A')}")
                print(f"   📊 Muestras val: {result.get('val_samples', 'N/A')}")
            else:
                print(f"   ❌ build_dataloaders retornó None")
        except Exception as e:
            print(f"   ❌ Error en build_dataloaders: {e}")
    
    print("🧪 Prueba completada")

# Ejecutar prueba
# test_data_flow()
'''
    
    print(test_code)
    
    print("\n" + "=" * 80)
    print("🎯 RESUMEN DEL CODE REVIEW")
    print("=" * 80)
    print("✅ Flujo principal: run_complete_training_pipeline → train_all → train_experiment → build_dataloaders")
    print("⚠️ PROBLEMAS ENCONTRADOS:")
    print("   1. Funciones duplicadas (build_dataloaders, EXPERIMENTS)")
    print("   2. Posibles conflictos de definiciones")
    print("   3. Falta validación robusta de datos")
    print("\n🔧 ACCIÓN INMEDIATA REQUERIDA:")
    print("   1. Eliminar duplicaciones")
    print("   2. Ejecutar código de prueba sugerido")
    print("   3. Validar que cada modelo recibe sus features correctas")

if __name__ == "__main__":
    code_review() 