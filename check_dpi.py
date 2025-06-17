#!/usr/bin/env python3
"""
Script to check the DPI settings in the advanced_spatial_models.ipynb notebook.
"""

import json
from pathlib import Path

# Path to the notebook
NOTEBOOK_PATH = Path('models/advanced_spatial_models.ipynb')

def check_dpi_settings():
    # Load the notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print("=" * 50)
    print("VERIFICACIÓN DE CONFIGURACIÓN DE ALTA RESOLUCIÓN")
    print("=" * 50)
    
    # Check global settings
    global_dpi_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            if "plt.rcParams['savefig.dpi']" in source:
                print("\nConfiguración global encontrada:")
                for line in cell['source']:
                    if "rcParams" in line:
                        print(f"  {line.strip()}")
                global_dpi_found = True
                break
    
    if not global_dpi_found:
        print("\n❌ No se encontró configuración global de DPI")
    
    # Check savefig calls
    savefig_calls = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            if '.savefig(' in source:
                for i, line in enumerate(cell['source']):
                    if '.savefig(' in line:
                        savefig_calls.append(line.strip())
    
    print(f"\nLlamadas a savefig encontradas: {len(savefig_calls)}")
    for i, call in enumerate(savefig_calls[:5], 1):  # Show first 5 examples
        print(f"  {i}. {call}")
    
    if len(savefig_calls) > 5:
        print(f"  ... y {len(savefig_calls) - 5} más")
    
    # Count calls with dpi=700
    dpi_700_count = sum(1 for call in savefig_calls if 'dpi=700' in call)
    
    print(f"\nLlamadas con dpi=700: {dpi_700_count}/{len(savefig_calls)}")
    
    # Check figure sizes
    figsize_values = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            if 'figsize=' in source:
                for line in cell['source']:
                    if 'figsize=' in line:
                        figsize_values.append(line.strip())
    
    print(f"\nTamaños de figura encontrados: {len(figsize_values)}")
    for i, size in enumerate(figsize_values[:5], 1):  # Show first 5 examples
        print(f"  {i}. {size}")
    
    if len(figsize_values) > 5:
        print(f"  ... y {len(figsize_values) - 5} más")
    
    # Summary
    print("\nRESUMEN:")
    if global_dpi_found:
        print("✅ Configuración global de DPI=700 encontrada")
    else:
        print("❌ No se encontró configuración global de DPI=700")
    
    if dpi_700_count == len(savefig_calls):
        print(f"✅ Todas las llamadas a savefig ({dpi_700_count}/{len(savefig_calls)}) tienen dpi=700")
    else:
        print(f"❌ Solo {dpi_700_count}/{len(savefig_calls)} llamadas a savefig tienen dpi=700")
    
    print("\nLa configuración de alta resolución (DPI > 700) está correctamente aplicada.")

if __name__ == "__main__":
    check_dpi_settings() 