import os
import re
from pathlib import Path

# Ruta al notebook
notebook_path = 'models/hybrid_models_GRU-w12.ipynb'

# Mapeo de emojis problemáticos a alternativas seguras
EMOJI_REPLACEMENTS = {
    '🌧️': 'Precip',  # CLOUD WITH RAIN
    '☁️': 'Precip',   # CLOUD WITH RAIN (alternativo)
    '🔮': 'Pred',     # CRYSTAL BALL
    '📊': 'Stats',    # BAR CHART
    '💧': 'H2O',      # DROPLET
}

def replace_in_file(file_path):
    """Reemplaza emojis problemáticos en un archivo"""
    try:
        if not os.path.exists(file_path):
            print(f"No se encontró el archivo: {file_path}")
            return False
        
        # Leer contenido
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Realizar reemplazos
        original_content = content
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            content = content.replace(emoji, replacement)
        
        # Si hubo cambios, guardar el archivo
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Se reemplazaron emojis en: {file_path}")
            return True
        else:
            print(f"No se encontraron emojis problemáticos en: {file_path}")
            return False
    
    except Exception as e:
        print(f"Error al procesar {file_path}: {str(e)}")
        return False

def fix_matplotlib_code():
    """Función principal para corregir los warnings de matplotlib"""
    print("Iniciando corrección de warnings de matplotlib...")
    
    # 1. Corregir archivo principal
    replace_in_file(notebook_path)
    
    # 2. Modificar configuración de matplotlib para todo el proyecto
    mpl_config_code = """
# Configuración de Matplotlib para evitar warnings de glifos
import matplotlib as mpl

# Establecer opciones globales de matplotlib
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
mpl.rcParams['font.family'] = 'sans-serif'

# Desactivar uso de caracteres Unicode en los títulos
mpl.rcParams['axes.unicode_minus'] = False

print("Configuración de matplotlib ajustada para evitar warnings de glifos")
"""
    
    # Crear archivo de configuración
    config_path = 'matplotlib_config.py'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(mpl_config_code)
    print(f"Archivo de configuración creado: {config_path}")
    
    print("\nPara resolver los warnings de matplotlib, haga lo siguiente:")
    print("1. Importe la configuración al inicio de su notebook:")
    print("   %run matplotlib_config.py")
    print("2. Evite usar emojis en títulos y etiquetas de gráficos")
    print("3. Use texto plano en lugar de emojis para mayor compatibilidad")
    
    return True

if __name__ == "__main__":
    fix_matplotlib_code() 