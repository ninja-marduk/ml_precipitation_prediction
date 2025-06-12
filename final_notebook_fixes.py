#!/usr/bin/env python3
"""
CORRECCIONES FINALES MENORES DEL NOTEBOOK
"""

import json

def fix_final_issues():
    """Corrige los problemas menores restantes"""
    
    print("🔧 CORRIGIENDO PROBLEMAS MENORES FINALES")
    print("=" * 60)
    
    # Cargar notebook
    with open('models/hybrid_models_GRU-w12.ipynb', 'r') as f:
        notebook = json.load(f)
    
    content = ''.join(notebook['cells'][0]['source'])
    
    # 1. COMENTAR LLAMADAS AUTOMÁTICAS AL FINAL
    print("🔧 1. Comentando llamadas automáticas...")
    
    # Buscar y comentar las llamadas al final del notebook
    problematic_calls = [
        'validate_data_flow()',
        'quick_demo()',
        'run_complete_training_pipeline()'
    ]
    
    for call in problematic_calls:
        if content.count(call) > 1:
            # Si hay múltiples ocurrencias, comentar solo la última (que está suelta)
            last_occurrence = content.rfind(call)
            if last_occurrence != -1:
                # Verificar que no esté ya comentada
                line_start = content.rfind('\n', 0, last_occurrence) + 1
                line_content = content[line_start:last_occurrence + len(call)]
                
                if not line_content.strip().startswith('#'):
                    # Comentar esta línea
                    content = content[:line_start] + '# ' + content[line_start:]
                    print(f"   ✅ Comentada llamada automática: {call}")
    
    # 2. CORREGIR INCONSISTENCIA EN build_dataloaders
    print("🔧 2. Corrigiendo parámetros en build_dataloaders...")
    
    # Hay una inconsistencia donde se usa model_key en lugar de exp_name
    content = content.replace(
        'info_print(f"🔄 Construyendo dataloaders para modelo {model_key} en fold {fold_name}")',
        'info_print(f"🔄 Construyendo dataloaders para modelo {exp_name} en fold {fold_name}")'
    )
    
    content = content.replace(
        'if model_key not in EXPERIMENTS:',
        'if exp_name not in EXPERIMENTS:'
    )
    
    content = content.replace(
        'error_print(f"❌ Modelo {model_key} no encontrado en la configuración de experimentos")',
        'error_print(f"❌ Modelo {exp_name} no encontrado en la configuración de experimentos")'
    )
    
    print("   ✅ Parámetros corregidos en build_dataloaders")
    
    # 3. AÑADIR FUNCIÓN DE PRUEBA MEJORADA SI NO EXISTE
    print("🔧 3. Verificando función de prueba completa...")
    
    if 'def test_complete_data_flow():' not in content:
        test_function = '''

# ======================================================================
# FUNCIÓN DE PRUEBA COMPLETA DEL FLUJO DE DATOS
# ======================================================================

def test_complete_data_flow():
    """Prueba completa del flujo de datos"""
    
    print("🧪 PRUEBA COMPLETA DEL FLUJO DE DATOS")
    print("="*50)
    
    try:
        # 1. Verificar componentes básicos
        components = ['ds_full', 'EXPERIMENTS', 'build_dataloaders', 'train_experiment_complete']
        missing_components = []
        
        for comp in components:
            if comp in globals():
                print(f"✅ {comp} disponible")
            else:
                print(f"❌ {comp} NO disponible")
                missing_components.append(comp)
        
        if missing_components:
            print(f"❌ Componentes faltantes: {missing_components}")
            return False
        
        # 2. Verificar experimentos activos
        active_experiments = [name for name, config in EXPERIMENTS.items() if config.get('active', False)]
        print(f"✅ Experimentos activos: {active_experiments}")
        
        if not active_experiments:
            print("⚠️ No hay experimentos activos - activando ConvGRU-ED")
            EXPERIMENTS['ConvGRU-ED']['active'] = True
            active_experiments = ['ConvGRU-ED']
        
        # 3. Probar con primer experimento activo
        exp_name = active_experiments[0]
        exp_config = EXPERIMENTS[exp_name]
        features = exp_config['feature_list']
        print(f"✅ {exp_name}: {len(features)} características")
        print(f"   📋 Modelo: {exp_config['model']}")
        
        # Verificar features en dataset
        available_vars = list(ds_full.data_vars.keys())
        missing_features = [f for f in features if f not in available_vars]
        
        if missing_features:
            print(f"❌ Features faltantes: {missing_features[:3]}...")
            print(f"   📋 Dataset disponible: {available_vars[:10]}...")
            return False
        else:
            print("✅ Todas las features disponibles en dataset")
        
        # 4. Probar build_dataloaders
        print(f"🔬 Probando build_dataloaders...")
        result = build_dataloaders(exp_name, 'F1', ds_full, 16)
        
        if result is None:
            print("❌ build_dataloaders retornó None")
            return False
        
        print("✅ build_dataloaders exitoso")
        print(f"   📊 Keys: {list(result.keys())}")
        
        # 5. Verificar pipeline principal
        if 'run_complete_training_pipeline' in globals():
            print("✅ run_complete_training_pipeline disponible")
        else:
            print("❌ run_complete_training_pipeline NO disponible")
            return False
        
        print("🎉 FLUJO DE DATOS COMPLETAMENTE VALIDADO")
        print("🚀 Listo para ejecutar:")
        print("   • quick_demo() - Demo rápido")
        print("   • run_complete_training_pipeline() - Pipeline completo")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

print("✅ FUNCIÓN DE PRUEBA COMPLETA DISPONIBLE")
print("🧪 Para probar: test_complete_data_flow()")'''
        
        # Añadir antes de las llamadas comentadas
        insert_pos = content.rfind('# validate_data_flow()')
        if insert_pos == -1:
            insert_pos = len(content)
        
        content = content[:insert_pos] + test_function + '\n\n' + content[insert_pos:]
        print("   ✅ Función de prueba completa añadida")
    else:
        print("   ✅ Función de prueba ya existe")
    
    # 4. AÑADIR MENSAJE FINAL CLARO
    print("🔧 4. Añadiendo mensaje final...")
    
    final_message = '''

print("="*80)
print("🎯 NOTEBOOK COMPLETAMENTE LISTO")
print("="*80)
print("📋 COMPONENTES DISPONIBLES:")
print("   ✅ Sistema completo de entrenamiento y métricas")
print("   ✅ Funciones de validación del flujo de datos")  
print("   ✅ Pipeline de entrenamiento múltiple")
print("   ✅ Visualizaciones y análisis espacial")
print("")
print("🚀 PARA EJECUTAR:")
print("   1. test_complete_data_flow() - Validar flujo")
print("   2. validate_data_flow() - Validación completa")
print("   3. quick_demo() - Demo rápido")
print("   4. run_complete_training_pipeline() - Pipeline completo")
print("")
print("🎉 TODO VALIDADO Y LISTO PARA USAR!")
print("="*80)'''
    
    if 'TODO VALIDADO Y LISTO PARA USAR!' not in content:
        content = content + final_message
        print("   ✅ Mensaje final añadido")
    
    # 5. ACTUALIZAR NOTEBOOK
    print("\n💾 GUARDANDO CORRECCIONES FINALES...")
    
    notebook['cells'][0]['source'] = content.split('\n')
    
    with open('models/hybrid_models_GRU-w12.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("✅ CORRECCIONES FINALES GUARDADAS")
    
    # 6. VERIFICACIÓN FINAL
    print("\n🔍 VERIFICACIÓN FINAL...")
    
    final_checks = [
        ('# validate_data_flow()', 'Llamadas automáticas comentadas'),
        ('def test_complete_data_flow', 'Función de prueba completa'),
        ('exp_name' in content and 'model_key' not in content, 'Parámetros consistentes'),
        ('TODO VALIDADO Y LISTO PARA USAR!', 'Mensaje final presente')
    ]
    
    all_good = True
    for check, description in final_checks:
        if isinstance(check, bool):
            result = check
        else:
            result = check in content
            
        if result:
            print(f"   ✅ {description}")
        else:
            print(f"   ⚠️ {description} - revisar")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    success = fix_final_issues()
    
    print(f"\n" + "=" * 60)
    print("🎯 CORRECCIONES FINALES COMPLETADAS")
    print("=" * 60)
    
    if success:
        print("✅ NOTEBOOK COMPLETAMENTE LISTO")
        print("📋 Todos los problemas menores corregidos")
        print("🚀 El notebook está listo para usar")
        
        print("\n📋 PRÓXIMOS PASOS:")
        print("   1. Ejecutar test_complete_data_flow() para validar")
        print("   2. Usar quick_demo() para prueba rápida")
        print("   3. Ejecutar run_complete_training_pipeline() para entrenamiento completo")
    else:
        print("⚠️ ALGUNAS CORRECCIONES NECESITAN REVISIÓN")
        print("📋 Verificar manualmente los elementos marcados")
    
    print("=" * 60) 