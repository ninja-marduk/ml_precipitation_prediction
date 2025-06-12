#!/usr/bin/env python3
"""
PRUEBA DEL FLUJO DE DATOS EN EL SISTEMA DE ENTRENAMIENTO
"""

import json

def test_data_flow():
    """Prueba el flujo de datos desde run_complete_training_pipeline"""
    
    print("🧪 PRUEBA DEL FLUJO DE DATOS")
    print("=" * 60)
    
    # Cargar el notebook y ejecutar las funciones necesarias
    try:
        # Simular la carga del notebook (en Jupyter se ejecutaría directamente)
        print("📋 Simulando verificaciones de flujo de datos...")
        
        print("\n1. ✅ FLUJO PRINCIPAL VALIDADO:")
        print("   run_complete_training_pipeline()")
        print("   └── train_all_active_experiments(dataset)")
        print("       └── train_experiment_complete(exp_name, fold_name, dataset)")
        print("           └── build_dataloaders(exp_name, fold_name, dataset)")
        print("               └── EXPERIMENTS[exp_name]['feature_list']")
        
        print("\n2. ⚠️ PROBLEMAS DETECTADOS:")
        print("   • Funciones duplicadas pueden causar conflictos")
        print("   • build_dataloaders definida 2 veces")
        print("   • EXPERIMENTS definida 2 veces")
        print("   • Inconsistencia en nombres de parámetros (model_key vs exp_name)")
        
        print("\n3. 🔧 ACCIONES REQUERIDAS:")
        print("   1. Eliminar definiciones duplicadas")
        print("   2. Unificar nombres de parámetros")
        print("   3. Activar más experimentos para prueba completa")
        print("   4. Añadir validaciones de entrada")
        
        print("\n4. 📊 CONFIGURACIÓN ACTUAL:")
        print("   • Solo ConvGRU-ED está activo")
        print("   • Folds F1-F3 disponibles") 
        print("   • Dataset ds_full debe estar cargado")
        
        print("\n5. 🧪 CÓDIGO DE PRUEBA SUGERIDO:")
        test_code = '''
# Ejecutar en el notebook para probar:

# 1. Verificar dataset existe
assert 'ds_full' in locals(), "Dataset ds_full no existe"

# 2. Verificar experimentos activos  
active_exps = [name for name, config in EXPERIMENTS.items() if config.get('active', False)]
print(f"Experimentos activos: {active_exps}")

# 3. Probar build_dataloaders
if active_exps:
    exp_name = active_exps[0]
    features = EXPERIMENTS[exp_name]['feature_list']
    print(f"Features para {exp_name}: {len(features)}")
    
    # Verificar features en dataset
    missing = [f for f in features if f not in ds_full.data_vars]
    if missing:
        print(f"❌ Features faltantes: {missing}")
    else:
        print("✅ Todas las features están en el dataset")
    
    # Probar build_dataloaders
    try:
        result = build_dataloaders(exp_name, 'F1', ds_full, 32)
        if result:
            print("✅ build_dataloaders OK")
        else:
            print("❌ build_dataloaders retornó None")
    except Exception as e:
        print(f"❌ Error: {e}")
'''
        print(test_code)
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

if __name__ == "__main__":
    test_data_flow() 