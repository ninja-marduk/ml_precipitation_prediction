import os
import sys
import re
import json

# Función para verificar que las características adecuadas están configuradas
def verify_feature_configuration():
    # Definir la estructura esperada
    expected_features = {
        'BASE_FEATURES': [
            'year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
            'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
            'elevation', 'slope', 'aspect'
        ],
        'ELEVATION_CLUSTER_FEATURES': [
            'cluster_high', 'cluster_medium', 'cluster_low'
        ]
    }
    
    # Modelos que deben usar las características cluster
    models_with_clusters = [
        'ConvGRU-ED-KCE',
        'ConvGRU-ED-KCE-PAFC',
        'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA',
        'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA-TopoMask'
    ]
    
    # Características esperadas por modelo
    model_feature_requirements = {
        'ConvGRU-ED': {
            'required': expected_features['BASE_FEATURES'],
            'not_allowed': expected_features['ELEVATION_CLUSTER_FEATURES']
        },
        'ConvGRU-ED-KCE': {
            'required': expected_features['BASE_FEATURES'] + expected_features['ELEVATION_CLUSTER_FEATURES'],
            'not_allowed': []
        },
        'ConvGRU-ED-KCE-PAFC': {
            'required': expected_features['BASE_FEATURES'] + expected_features['ELEVATION_CLUSTER_FEATURES'] + 
                       ['total_precipitation_lag1', 'total_precipitation_lag2', 'total_precipitation_lag12'],
            'not_allowed': []
        },
        'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA': {
            'required': expected_features['BASE_FEATURES'] + expected_features['ELEVATION_CLUSTER_FEATURES'] + 
                       ['total_precipitation_lag1', 'total_precipitation_lag2', 'total_precipitation_lag12'] +
                       ['CEEMDAN_imf_1', 'CEEMDAN_imf_2', 'CEEMDAN_imf_3', 'CEEMDAN_imf_4', 
                        'CEEMDAN_imf_5', 'CEEMDAN_imf_6', 'CEEMDAN_imf_7', 'CEEMDAN_imf_8',
                        'TVFEMD_imf_1', 'TVFEMD_imf_2', 'TVFEMD_imf_3', 'TVFEMD_imf_4',
                        'TVFEMD_imf_5', 'TVFEMD_imf_6', 'TVFEMD_imf_7', 'TVFEMD_imf_8'],
            'not_allowed': []
        },
        'AE-FUSION-ConvGRU-ED-KCE-PAFC-MHA-TopoMask': {
            'required': expected_features['BASE_FEATURES'] + expected_features['ELEVATION_CLUSTER_FEATURES'] + 
                       ['total_precipitation_lag1', 'total_precipitation_lag2', 'total_precipitation_lag12'] +
                       ['CEEMDAN_imf_1', 'CEEMDAN_imf_2', 'CEEMDAN_imf_3', 'CEEMDAN_imf_4', 
                        'CEEMDAN_imf_5', 'CEEMDAN_imf_6', 'CEEMDAN_imf_7', 'CEEMDAN_imf_8',
                        'TVFEMD_imf_1', 'TVFEMD_imf_2', 'TVFEMD_imf_3', 'TVFEMD_imf_4',
                        'TVFEMD_imf_5', 'TVFEMD_imf_6', 'TVFEMD_imf_7', 'TVFEMD_imf_8'],
            'not_allowed': []
        }
    }
    
    # Mostrar configuración esperada
    print("🔍 VERIFICACIÓN DE CONFIGURACIÓN DE CARACTERÍSTICAS POR MODELO")
    print("=" * 70)
    print(f"✅ Características base ({len(expected_features['BASE_FEATURES'])}):")
    for feature in expected_features['BASE_FEATURES']:
        print(f"  - {feature}")
    
    print(f"\n✅ Características cluster ({len(expected_features['ELEVATION_CLUSTER_FEATURES'])}):")
    for feature in expected_features['ELEVATION_CLUSTER_FEATURES']:
        print(f"  - {feature}")
    
    print("\n✅ Requisitos por modelo:")
    for model, req in model_feature_requirements.items():
        print(f"\n  📊 {model}:")
        print(f"    - Características requeridas: {len(req['required'])}")
        print(f"    - Usa clusters one-hot: {'Sí' if model in models_with_clusters else 'No'}")
    
    print("\n🔄 MODIFICACIONES NECESARIAS:")
    print("1. Asegurar que ELEVATION_CLUSTER_FEATURES contenga directamente ['cluster_high', 'cluster_medium', 'cluster_low']")
    print("2. Modificar build_dataloaders para procesar 'cluster_elevation' correctamente:")
    print("""
   # Código actual en build_dataloaders:
   for feature in features:
       if feature in dataset:
           # Manejo especial para cluster_elevation (categórico)
           if feature == 'cluster_elevation':
               # ...procesamiento...
           else:
               # ...procesamiento normal...
               
   # Modificación requerida:
   # 1. Verificar si cualquiera de 'cluster_high', 'cluster_medium', 'cluster_low' está en features
   cluster_features = ['cluster_high', 'cluster_medium', 'cluster_low']
   if any(cf in features for cf in cluster_features):
       # Verificar que 'cluster_elevation' esté en el dataset
       if 'cluster_elevation' in dataset:
           # Generar las características one-hot y añadirlas
           # Eliminar las características cluster de la lista si no existen en el dataset
    """)
    
    print("\n3. Agregar validación antes de crear cada modelo:")
    print("""
   # Agregar antes de crear cada modelo:
   def validate_features_for_model(model_key, available_features):
       # Verificar que las características necesarias estén disponibles
       required_features = model_feature_requirements[model_key]['required']
       missing_features = [f for f in required_features if f not in available_features]
       
       if missing_features:
           warning_print(f"⚠️ Faltan características requeridas para {model_key}: {missing_features}")
           return False
       return True
   
   # Usar al crear el modelo:
   if not validate_features_for_model(model_key, available_features):
       warning_print(f"⚠️ El modelo {model_key} puede no funcionar correctamente por características faltantes")
    """)
    
    return model_feature_requirements

# Ejecutar verificación
if __name__ == "__main__":
    verify_feature_configuration() 