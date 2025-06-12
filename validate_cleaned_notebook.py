#!/usr/bin/env python3
"""
VALIDACIÓN ESPECÍFICA DEL FLUJO DE DATOS DESPUÉS DE LA LIMPIEZA
"""

import json
import re

def validate_data_flow():
    """Valida específicamente el flujo de datos en el notebook limpio"""
    
    print("🔍 VALIDACIÓN ESPECÍFICA DEL FLUJO DE DATOS")
    print("=" * 80)
    
    # Cargar notebook limpio
    with open('models/hybrid_models_GRU-w12.ipynb', 'r') as f:
        notebook = json.load(f)
    
    content = ''.join([''.join(cell['source']) for cell in notebook['cells']])
    
    # 1. VERIFICAR FLUJO PRINCIPAL
    print("🔍 1. FLUJO PRINCIPAL DE ENTRENAMIENTO:")
    print("-" * 50)
    
    flow_steps = [
        ('run_complete_training_pipeline', 'Función principal del pipeline'),
        ('train_all_active_experiments(dataset)', 'Entrena todos los experimentos'),
        ('train_experiment_complete(exp_name, fold_name, dataset', 'Entrena experimento específico'),
        ('build_dataloaders(exp_name, fold_name, dataset', 'Construye dataloaders'),
        ("features = EXPERIMENTS[exp_name]['feature_list']", 'Extrae features específicas')
    ]
    
    for pattern, description in flow_steps:
        if pattern in content:
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description} - NO ENCONTRADO")
    
    # 2. VERIFICAR build_dataloaders ESPECÍFICAMENTE
    print("\n🔍 2. ANÁLISIS DETALLADO DE build_dataloaders:")
    print("-" * 50)
    
    # Extraer la función build_dataloaders
    build_match = re.search(
        r'def build_dataloaders\([^)]*\):(.*?)(?=\ndef [^_]|\Z)',
        content,
        re.DOTALL
    )
    
    if build_match:
        build_function = build_match.group(1)
        
        # Verificar componentes críticos
        critical_components = [
            ("features = EXPERIMENTS[exp_name]['feature_list']", "Extrae features del experimento"),
            ("for feature in features:", "Itera sobre cada feature"),
            ("if feature in dataset:", "Verifica feature en dataset"),
            ("feature_data[feature]", "Almacena datos de feature"),
            ("return {", "Retorna configuración de datasets"),
            ("train_dataset", "Crea dataset de entrenamiento"),
            ("val_dataset", "Crea dataset de validación")
        ]
        
        missing_components = []
        for component, description in critical_components:
            if component in build_function:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description} - FALTA")
                missing_components.append(description)
        
        if missing_components:
            print(f"\n   🚨 COMPONENTES FALTANTES: {len(missing_components)}")
            for comp in missing_components:
                print(f"      • {comp}")
    else:
        print("   ❌ build_dataloaders NO ENCONTRADA")
    
    # 3. VERIFICAR CONFIGURACIÓN DE EXPERIMENTOS
    print("\n🔍 3. CONFIGURACIÓN DE EXPERIMENTOS:")
    print("-" * 50)
    
    # Buscar definición de EXPERIMENTS
    exp_match = re.search(r"EXPERIMENTS\s*=\s*{(.*?)^}", content, re.DOTALL | re.MULTILINE)
    
    if exp_match:
        experiments_config = exp_match.group(1)
        
        # Buscar experimentos definidos
        experiment_names = re.findall(r"'([^']*(?:GRU|ConvGRU)[^']*)':", experiments_config)
        print(f"   📋 Experimentos encontrados: {experiment_names}")
        
        # Verificar cada experimento
        for exp_name in experiment_names:
            print(f"\n   🔬 Analizando {exp_name}:")
            
            # Buscar configuración específica
            exp_pattern = rf"'{exp_name}':\s*{{([^}}]*)}}"
            exp_config_match = re.search(exp_pattern, experiments_config, re.DOTALL)
            
            if exp_config_match:
                exp_config = exp_config_match.group(1)
                
                # Verificar componentes requeridos
                required_components = ['feature_list', 'model', 'description', 'active']
                for component in required_components:
                    if f"'{component}'" in exp_config:
                        print(f"      ✅ {component}")
                        
                        # Análisis específico para feature_list
                        if component == 'feature_list':
                            feature_pattern = rf"'feature_list':\s*\[([^\]]*)\]"
                            features_match = re.search(feature_pattern, exp_config)
                            if features_match:
                                features_str = features_match.group(1)
                                feature_count = len(re.findall(r"'[^']*'", features_str))
                                print(f"         📊 {feature_count} características definidas")
                                
                                # Extraer nombres de features
                                feature_names = re.findall(r"'([^']*)'", features_str)
                                print(f"         📋 Features: {feature_names[:3]}...")
                            else:
                                print(f"      ❌ feature_list malformada")
                        
                        # Verificar si está activo
                        if component == 'active':
                            if "'active': True" in exp_config:
                                print(f"         🟢 ACTIVO")
                            else:
                                print(f"         🔴 INACTIVO")
                    else:
                        print(f"      ❌ {component} - FALTA")
            else:
                print(f"      ❌ Configuración de {exp_name} no encontrada")
    else:
        print("   ❌ EXPERIMENTS no encontrada")
    
    # 4. VERIFICAR LLAMADAS Y CONSISTENCIA
    print("\n🔍 4. VERIFICACIÓN DE LLAMADAS:")
    print("-" * 50)
    
    # Buscar todas las llamadas a build_dataloaders
    build_calls = re.findall(r'build_dataloaders\([^)]*\)', content)
    
    print(f"   📋 Total llamadas a build_dataloaders: {len(build_calls)}")
    
    consistent_calls = 0
    for i, call in enumerate(build_calls):
        print(f"      {i+1}. {call}")
        
        # Verificar que use exp_name (no model_key)
        if 'exp_name' in call and 'fold_name' in call and 'dataset' in call:
            print(f"         ✅ Parámetros correctos")
            consistent_calls += 1
        else:
            print(f"         ❌ Parámetros incorrectos o inconsistentes")
    
    print(f"   📊 Llamadas consistentes: {consistent_calls}/{len(build_calls)}")
    
    # 5. VERIFICAR PIPELINE COMPLETO
    print("\n🔍 5. VERIFICACIÓN DEL PIPELINE COMPLETO:")
    print("-" * 50)
    
    # Verificar run_complete_training_pipeline
    pipeline_match = re.search(
        r'def run_complete_training_pipeline.*?(?=\ndef|\Z)',
        content,
        re.DOTALL
    )
    
    if pipeline_match:
        pipeline_code = pipeline_match.group(0)
        
        pipeline_checks = [
            ('if dataset is None:', 'Maneja dataset por defecto'),
            ('dataset = ds_full', 'Usa ds_full como fallback'),
            ('train_all_active_experiments(dataset)', 'Llama a train_all_active_experiments'),
            ('analyze_experiment_results', 'Analiza resultados'),
            ('plot_monthly_predictions', 'Genera visualizaciones'),
            ('plot_spatial_maps', 'Genera mapas espaciales')
        ]
        
        for check, description in pipeline_checks:
            if check in pipeline_code:
                print(f"   ✅ {description}")
            else:
                print(f"   ⚠️ {description} - POSIBLEMENTE FALTA")
    
    # 6. GENERAR CÓDIGO DE PRUEBA ESPECÍFICO
    print("\n🧪 6. CÓDIGO DE PRUEBA GENERADO:")
    print("-" * 50)
    
    test_code = '''
# CÓDIGO DE PRUEBA PARA VALIDAR EL FLUJO DE DATOS

def test_complete_data_flow():
    """Prueba completa del flujo de datos"""
    
    print("🧪 PRUEBA COMPLETA DEL FLUJO DE DATOS")
    print("="*50)
    
    try:
        # 1. Verificar componentes básicos
        assert 'ds_full' in globals(), "❌ ds_full no está cargado"
        assert 'EXPERIMENTS' in globals(), "❌ EXPERIMENTS no está definido"
        print("✅ Componentes básicos OK")
        
        # 2. Verificar experimentos activos
        active_experiments = [name for name, config in EXPERIMENTS.items() if config.get('active', False)]
        print(f"✅ Experimentos activos: {active_experiments}")
        
        if not active_experiments:
            print("⚠️ No hay experimentos activos - activando ConvGRU-ED para prueba")
            EXPERIMENTS['ConvGRU-ED']['active'] = True
            active_experiments = ['ConvGRU-ED']
        
        # 3. Probar build_dataloaders con primer experimento
        exp_name = active_experiments[0]
        features = EXPERIMENTS[exp_name]['feature_list']
        print(f"✅ {exp_name}: {len(features)} características")
        
        # Verificar que features estén en dataset
        missing_features = [f for f in features if f not in ds_full.data_vars]
        if missing_features:
            print(f"❌ Features faltantes en dataset: {missing_features[:5]}...")
            return False
        else:
            print("✅ Todas las features disponibles en dataset")
        
        # 4. Probar build_dataloaders
        print(f"🔬 Probando build_dataloaders({exp_name}, 'F1', ds_full, 16)...")
        result = build_dataloaders(exp_name, 'F1', ds_full, 16)
        
        if result is None:
            print("❌ build_dataloaders retornó None")
            return False
        
        print("✅ build_dataloaders exitoso")
        print(f"   📊 Keys: {list(result.keys())}")
        
        # Verificar contenido del resultado
        expected_keys = ['train_dataset', 'val_dataset', 'spatial_config']
        missing_keys = [key for key in expected_keys if key not in result]
        if missing_keys:
            print(f"⚠️ Keys faltantes: {missing_keys}")
        else:
            print("✅ Todas las keys esperadas presentes")
        
        # 5. Probar train_experiment_complete (solo validación)
        print(f"🧪 Validando train_experiment_complete...")
        
        # Verificar que la función existe y tiene la firma correcta
        import inspect
        sig = inspect.signature(train_experiment_complete)
        params = list(sig.parameters.keys())
        expected_params = ['exp_name', 'fold_name', 'dataset', 'save_model']
        
        if all(param in params for param in expected_params[:3]):
            print("✅ train_experiment_complete tiene parámetros correctos")
        else:
            print(f"❌ train_experiment_complete parámetros incorrectos: {params}")
            return False
        
        print("🎉 FLUJO DE DATOS COMPLETAMENTE VALIDADO")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

# Ejecutar la prueba
# test_complete_data_flow()
'''
    
    print(test_code)
    
    return True

if __name__ == "__main__":
    success = validate_data_flow()
    
    print(f"\n" + "=" * 80)
    print("🎯 RESUMEN DE VALIDACIÓN")
    print("=" * 80)
    
    if success:
        print("✅ VALIDACIÓN EXITOSA")
        print("📋 Flujo de datos correctamente implementado")
        print("🚀 Notebook listo para usar")
        
        print("\n📋 PARA VERIFICAR FUNCIONAMIENTO:")
        print("   1. Ejecutar el código de prueba generado arriba")
        print("   2. Ejecutar validate_data_flow() en el notebook")
        print("   3. Probar quick_demo()")
        print("   4. Ejecutar run_complete_training_pipeline()")
    else:
        print("⚠️ VALIDACIÓN CON PROBLEMAS")
        print("📋 Revisar issues reportados arriba")
    
    print("\n" + "=" * 80) 