import json
import os
import sys

# Ruta al notebook
notebook_path = 'models/hybrid_models_GRU-w12.ipynb'

# Cargar el notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# El notebook tiene solo una celda con todo el código
code = ''.join(notebook['cells'][0]['source'])

# Encontrar la posición donde inicializar las funciones necesarias para transformar cluster_elevation
# Buscamos después de la inicialización de feature_data, feature_shapes y available_features
init_functions_pos = code.find("feature_data = {}")
init_functions_pos = code.find('\n', init_functions_pos) + 1
init_functions_pos = code.find('\n', init_functions_pos) + 1
init_functions_pos = code.find('\n', init_functions_pos) + 1

# Funciones a insertar - usamos comillas simples para el string multilinea
functions_to_insert = '''    
    # Función para verificar NaN de manera segura
    def safe_is_nan(x):
        """Verifica si un valor es NaN de manera segura para cualquier tipo de dato."""
        try:
            return np.isnan(x)
        except:
            return False
            
    def safe_has_nan(arr):
        """Verifica si un array contiene NaN de manera segura para cualquier tipo de dato."""
        try:
            return np.isnan(arr).any()
        except:
            return False
    
    # Función para one-hot encoding espacial
    def spatial_one_hot_encoding(data, categories=None):
        """
        Convierte una matriz categórica espacial en matrices one-hot manteniendo dimensiones espaciales.
        
        Args:
            data: Array numpy con datos categóricos (height, width)
            categories: Lista de categorías. Si es None, se extraen automáticamente
            
        Returns:
            Dictionary con una matriz por categoría, cada una con forma (height, width)
        """
        if categories is None:
            categories = np.unique(data)
        
        height, width = data.shape
        encoded = {}
        
        for category in categories:
            # Crear máscara binaria donde 1=coincide con la categoría, 0=resto
            mask = (data == category).astype(np.float32)
            encoded[f"cluster_{category}"] = mask
        
        return encoded, categories
    
    # Función para transformar cluster_elevation a variables one-hot
    def transform_cluster_elevation(dataset):
        """
        Transforma la variable categórica 'cluster_elevation' en variables one-hot.
        
        Args:
            dataset: Dataset de xarray con la variable 'cluster_elevation'
            
        Returns:
            Dictionary con matrices one-hot para cada categoría
        """
        if 'cluster_elevation' not in dataset:
            warning_print("⚠️ No se encontró 'cluster_elevation' en el dataset")
            return {}
        
        # Obtener datos categóricos
        cluster_data = dataset['cluster_elevation'].values
        
        # Verificar el tipo de dato
        info_print(f"📊 Tipo de datos cluster_elevation: {cluster_data.dtype}")
        
        try:
            # Aplicar one-hot encoding espacial
            encoded_clusters, categories = spatial_one_hot_encoding(cluster_data)
            
            # Verificar si se generaron todas las categorías esperadas
            expected_categories = ['high', 'medium', 'low']
            missing_categories = [cat for cat in expected_categories 
                                if f"cluster_{cat}" not in encoded_clusters]
            
            if missing_categories:
                warning_print(f"⚠️ Categorías faltantes en cluster_elevation: {missing_categories}")
                # Crear categorías vacías para las faltantes
                for cat in missing_categories:
                    encoded_clusters[f"cluster_{cat}"] = np.zeros_like(cluster_data, dtype=np.float32)
            
            info_print(f"✅ Variable 'cluster_elevation' transformada a {len(encoded_clusters)} variables one-hot")
            return encoded_clusters
        
        except Exception as e:
            error_print(f"❌ Error al transformar cluster_elevation: {e}")
            return {}

    # Verificar si se requieren características de cluster (one-hot)
    needs_cluster_features = any(cf in features for cf in ELEVATION_CLUSTER_FEATURES)
    
    # Si se requieren características de cluster pero no están en el dataset, verificar cluster_elevation
    if needs_cluster_features and not all(cf in dataset for cf in ELEVATION_CLUSTER_FEATURES):
        if 'cluster_elevation' in dataset:
            info_print("⚠️ Características cluster_high, cluster_medium, cluster_low no encontradas")
            info_print("🔄 Transformando 'cluster_elevation' a variables one-hot")
            
            # Transformar cluster_elevation
            encoded_clusters = transform_cluster_elevation(dataset)
            
            # Añadir cada categoría como una característica separada
            for category, encoded_data in encoded_clusters.items():
                feature_data[category] = encoded_data
                feature_shapes[category] = encoded_data.shape
                if category not in available_features:
                    available_features.append(category)
        else:
            warning_print("❌ No se encontró 'cluster_elevation' ni características cluster_*")
'''

# Encontrar el bucle de procesamiento de características para modificar
for_feature_pos = code.find("for feature in features:")

# El nuevo bloque para el bucle for - también con comillas simples
new_for_feature_block = '''    for feature in features:
        # Si ya se procesó como parte de cluster_elevation, saltar
        if feature in available_features:
            continue
            
        if feature in dataset:
            # Manejo especial para cluster_elevation (categórico)
            if feature == 'cluster_elevation':
                # Obtener datos categóricos
                cluster_data = dataset[feature].values
                
                # Aplicar one-hot encoding espacial
                encoded_clusters, _ = spatial_one_hot_encoding(cluster_data)
                
                # Añadir cada categoría como una característica separada
                for category, encoded_data in encoded_clusters.items():
                    feature_data[category] = encoded_data
                    feature_shapes[category] = encoded_data.shape
                    if category not in available_features:
                        available_features.append(category)
                    
                info_print(f"   ✅ Variable categórica '{feature}' convertida a {len(encoded_clusters)} matrices one-hot")
            else:
                # Procesamiento normal para características numéricas
                feature_array = dataset[feature].values
                
                # Verificar NaN de manera segura
                if safe_has_nan(feature_array):
                    warning_print(f"⚠️ La característica '{feature}' contiene valores NaN")
                
                feature_data[feature] = feature_array
                feature_shapes[feature] = feature_array.shape
                available_features.append(feature)
        else:
            # Solo mostrar advertencia si no es una característica de cluster que ya se procesó
            if feature not in ['cluster_high', 'cluster_medium', 'cluster_low'] or 'encoded_clusters' not in locals():
                warning_print(f"⚠️ Característica '{feature}' no encontrada en el dataset")'''

# Encontrar el final del bucle for
for_feature_end_pos = code.find("# Verificar si todas las características", for_feature_pos)

# Reemplazar el código
modified_code = (code[:init_functions_pos] + 
                functions_to_insert + 
                code[init_functions_pos:for_feature_pos] + 
                new_for_feature_block + 
                code[for_feature_end_pos:])

# Actualizar el notebook
notebook['cells'][0]['source'] = [modified_code]

# Guardar el notebook modificado
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Notebook {notebook_path} modificado correctamente.")
print("Se han añadido las siguientes funciones:")
print("1. safe_is_nan() - Verifica NaN de manera segura")
print("2. safe_has_nan() - Verifica NaN en arrays de manera segura")
print("3. spatial_one_hot_encoding() - Convierte datos categóricos a matrices one-hot")
print("4. transform_cluster_elevation() - Transforma cluster_elevation a variables one-hot")
print("\nEl notebook ahora verificará automáticamente:")
print("- Si se requieren características cluster_high, cluster_medium, cluster_low")
print("- Si no están disponibles pero existe cluster_elevation, las generará automáticamente")
print("- Si el modelo requiere estas variables pero no están disponibles, mostrará una advertencia clara") 