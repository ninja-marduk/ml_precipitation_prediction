#!/usr/bin/env python3
"""
🔍 VALIDADOR DE CONFIGURACIÓN FNO V3
Ejecutar antes del entrenamiento para anticipar errores
"""

import sys
import os
import gc
import psutil
from pathlib import Path
import tensorflow as tf
import numpy as np
import xarray as xr

def check_tensorflow_version():
    """Verificar versión de TensorFlow"""
    print("🔧 Verificando TensorFlow...")
    tf_version = tf.__version__
    print(f"   • Versión: {tf_version}")
    
    if tf_version < "2.8.0":
        print(f"   ⚠️ WARNING: TensorFlow {tf_version} puede no soportar todas las operaciones FNO")
        print(f"   💡 Recomendado: TensorFlow >= 2.8.0")
        return False
    else:
        print(f"   ✅ Versión compatible")
        return True

def check_gpu_availability():
    """Verificar disponibilidad y configuración de GPU"""
    print("\n🎮 Verificando GPU...")
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("   ⚠️ WARNING: No se detectó GPU")
        print("   💡 El entrenamiento FNO será MUY lento en CPU")
        return False
    
    print(f"   ✅ {len(gpus)} GPU(s) detectada(s)")
    
    for i, gpu in enumerate(gpus):
        try:
            # Configurar memory growth
            tf.config.experimental.set_memory_growth(gpu, True)
            
            # Obtener detalles si es posible
            try:
                details = tf.config.experimental.get_device_details(gpu)
                name = details.get('device_name', 'Unknown')
                print(f"   • GPU {i}: {name}")
            except:
                print(f"   • GPU {i}: Disponible")
                
        except Exception as e:
            print(f"   ⚠️ Error configurando GPU {i}: {e}")
            return False
    
    return True

def check_memory_resources():
    """Verificar recursos de memoria disponibles"""
    print("\n💻 Verificando recursos de memoria...")
    
    # RAM del sistema
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024**3)
    ram_available_gb = memory.available / (1024**3)
    
    print(f"   • RAM Total: {ram_gb:.1f} GB")
    print(f"   • RAM Disponible: {ram_available_gb:.1f} GB")
    print(f"   • RAM en uso: {memory.percent:.1f}%")
    
    if ram_available_gb < 8:
        print("   ⚠️ WARNING: Poca RAM disponible (<8GB)")
        print("   💡 Considerar cerrar otras aplicaciones")
        return False
    elif ram_available_gb < 16:
        print("   ⚠️ CAUTION: RAM limitada (<16GB)")
        print("   💡 Monitorear uso durante entrenamiento")
    else:
        print("   ✅ RAM suficiente")
    
    return True

def check_disk_space():
    """Verificar espacio en disco"""
    print("\n💾 Verificando espacio en disco...")
    
    # Obtener espacio en el directorio actual
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    
    print(f"   • Espacio libre: {free_gb:.1f} GB")
    
    if free_gb < 5:
        print("   🚨 ERROR: Espacio insuficiente (<5GB)")
        print("   💡 Liberar espacio antes de continuar")
        return False
    elif free_gb < 10:
        print("   ⚠️ WARNING: Poco espacio (<10GB)")
        print("   💡 Los modelos y logs pueden ocupar mucho espacio")
    else:
        print("   ✅ Espacio suficiente")
    
    return True

def check_data_files():
    """Verificar archivos de datos necesarios"""
    print("\n📁 Verificando archivos de datos...")
    
    # Detectar si estamos en Colab o local
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        base_path = Path('/content/drive/MyDrive/ml_precipitation_prediction')
    else:
        # Asumir que estamos en el directorio del proyecto
        base_path = Path('.').resolve()
        while not (base_path / 'models').exists() and base_path != base_path.parent:
            base_path = base_path.parent
    
    print(f"   • Base path: {base_path}")
    
    # Verificar archivo principal de datos
    data_file = base_path / 'data' / 'output' / 'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
    
    if not data_file.exists():
        print(f"   🚨 ERROR: Dataset no encontrado")
        print(f"   📍 Buscado en: {data_file}")
        return False
    else:
        print(f"   ✅ Dataset encontrado")
        
        # Verificar que se puede abrir
        try:
            ds = xr.open_dataset(data_file)
            print(f"   • Dimensiones: time={len(ds.time)}, lat={len(ds.latitude)}, lon={len(ds.longitude)}")
            ds.close()
        except Exception as e:
            print(f"   🚨 ERROR: No se puede abrir dataset: {e}")
            return False
    
    # Verificar directorio de salida
    out_dir = base_path / 'models' / 'output' / 'Spatial_CONVRNN'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Directorio de salida: {out_dir}")
    
    return True

def test_fno_operations():
    """Test básico de operaciones FNO"""
    print("\n🔬 Probando operaciones FNO...")
    
    try:
        # Test FFT operations
        print("   • Probando tf.signal.fft2d...")
        test_data = tf.random.normal((2, 32, 32, 4), dtype=tf.float32)
        test_complex = tf.cast(test_data, tf.complex64)
        fft_result = tf.signal.fft2d(test_complex)
        print("   ✅ FFT2D funciona correctamente")
        
        # Test IFFT
        print("   • Probando tf.signal.ifft2d...")
        ifft_result = tf.signal.ifft2d(fft_result)
        print("   ✅ IFFT2D funciona correctamente")
        
        # Test complex operations
        print("   • Probando operaciones complejas...")
        weights = tf.complex(
            tf.random.normal((16, 16, 4, 32)),
            tf.random.normal((16, 16, 4, 32))
        )
        result = tf.einsum("bixy,xyio->boxy", fft_result[:, :16, :16, :], weights)
        print("   ✅ Operaciones complejas funcionan")
        
        print("   ✅ Todas las operaciones FNO funcionan correctamente")
        return True
        
    except Exception as e:
        print(f"   🚨 ERROR en operaciones FNO: {e}")
        print("   💡 Verificar instalación de TensorFlow")
        return False

def estimate_training_time():
    """Estimar tiempo de entrenamiento"""
    print("\n⏱️ Estimando tiempo de entrenamiento...")
    
    # Configuración del Paso 1
    n_models = 3  # FNO models
    n_experiments = 3  # BASIC, KCE, PAFC
    n_combinations = n_models * n_experiments
    
    # Estimar tiempo por combinación
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        time_per_combination = 15  # minutos con GPU
        print("   • Con GPU detectada")
    else:
        time_per_combination = 120  # minutos sin GPU (muy lento)
        print("   • Sin GPU (MUY LENTO)")
    
    total_time_min = n_combinations * time_per_combination
    total_time_hours = total_time_min / 60
    
    print(f"   • Combinaciones: {n_combinations}")
    print(f"   • Tiempo por combinación: ~{time_per_combination} min")
    print(f"   • Tiempo total estimado: ~{total_time_hours:.1f} horas")
    
    if total_time_hours > 8:
        print("   ⚠️ WARNING: Entrenamiento muy largo")
        print("   💡 Considerar ejecutar en lotes más pequeños")

def main():
    """Función principal de validación"""
    print("🔍 VALIDADOR DE CONFIGURACIÓN FNO V3")
    print("=" * 50)
    
    checks = []
    
    # Ejecutar todas las verificaciones
    checks.append(("TensorFlow", check_tensorflow_version()))
    checks.append(("GPU", check_gpu_availability()))
    checks.append(("Memoria", check_memory_resources()))
    checks.append(("Disco", check_disk_space()))
    checks.append(("Datos", check_data_files()))
    checks.append(("FNO Ops", test_fno_operations()))
    
    # Estimar tiempo
    estimate_training_time()
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE VALIDACIÓN")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "🚨 FAIL"
        print(f"   {check_name:12} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 VALIDACIÓN EXITOSA - LISTO PARA ENTRENAR")
        print("💡 Recomendación: Proceder con el Paso 1 (9 combinaciones FNO)")
    else:
        print("🛑 VALIDACIÓN FALLIDA - CORREGIR ERRORES ANTES DE CONTINUAR")
        print("💡 Revisar los errores marcados arriba")
    
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
