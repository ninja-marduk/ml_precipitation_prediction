import json
import re

# Ruta al notebook
notebook_path = 'models/hybrid_models_GRU-w12.ipynb'

# Cargar el notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Mapeo de líneas específicas con emojis y sus reemplazos
# Estructura: {(índice_celda, índice_línea): (texto_original, texto_reemplazo)}
specific_replacements = {
    # plot_spatial_maps
    (0, 3128): ("    axes[0].set_title(f'🌧️ Precipitación Real\\\\n{month}', fontsize=14, fontweight='bold')", 
                "    axes[0].set_title(f'Precipitación Real\\\\n{month}', fontsize=14, fontweight='bold')"),
    (0, 3135): ("    axes[1].set_title(f'🔮 Predicción\\\\n{month}', fontsize=14, fontweight='bold')", 
                "    axes[1].set_title(f'Predicción\\\\n{month}', fontsize=14, fontweight='bold')"),
    (0, 3143): ("    axes[2].set_title(f'📊 Error MAPE\\\\n{month}', fontsize=14, fontweight='bold')", 
                "    axes[2].set_title(f'Error MAPE\\\\n{month}', fontsize=14, fontweight='bold')"),

    # plot_spatial_predictions
    (0, 2621): ("        axes[0].set_title(f'🌧️ Precipitación Real\\\\n{month}', fontsize=14, fontweight='bold')", 
                "        axes[0].set_title(f'Precipitación Real\\\\n{month}', fontsize=14, fontweight='bold')"),
    (0, 2628): ("        axes[1].set_title(f'🔮 Predicción\\\\n{month}', fontsize=14, fontweight='bold')", 
                "        axes[1].set_title(f'Predicción\\\\n{month}', fontsize=14, fontweight='bold')"),
    (0, 2636): ("        axes[2].set_title(f'📊 Error MAPE\\\\n{month}', fontsize=14, fontweight='bold')", 
                "        axes[2].set_title(f'Error MAPE\\\\n{month}', fontsize=14, fontweight='bold')"),

    # plot_monthly_predictions
    (0, 2560): ("    ax1.set_title(f'💧 Precipitación Predicha vs Real - {exp_name}', fontsize=16, fontweight='bold')",
                "    ax1.set_title(f'Precipitación Predicha vs Real - {exp_name}', fontsize=16, fontweight='bold')"),
    (0, 2574): ("    ax2.set_title('📊 Error MAPE por Mes', fontsize=16, fontweight='bold')",
                "    ax2.set_title('Error MAPE por Mes', fontsize=16, fontweight='bold')"),

    # Reemplazar en las strings de info_print
    (0, 2677): ("                info_print(f\"\\n📊 ANÁLISIS DETALLADO: {exp_name} - {fold_name}\")",
                "                info_print(f\"\\nANÁLISIS DETALLADO: {exp_name} - {fold_name}\")"),
    (0, 2682): ("                info_print(f\"📈 Métricas Generales:\")",
                "                info_print(f\"Métricas Generales:\")"),
    (0, 2701): ("    info_print(\"\\n🎉 PIPELINE COMPLETADO EXITOSAMENTE\")",
                "    info_print(\"\\nPIPELINE COMPLETADO EXITOSAMENTE\")"),
    (0, 2706): ("    info_print(f\"📊 RESUMEN FINAL:\")",
                "    info_print(f\"RESUMEN FINAL:\")"),
    (0, 2707): ("    info_print(f\"   ✅ Experimentos exitosos: {total_successful}\")",
                "    info_print(f\"   - Experimentos exitosos: {total_successful}\")"),
    (0, 2708): ("    info_print(f\"   ❌ Experimentos fallidos: {total_failed}\")",
                "    info_print(f\"   - Experimentos fallidos: {total_failed}\")"),
    (0, 2709): ("    info_print(f\"   💾 Modelos guardados: {total_successful}\")",
                "    info_print(f\"   - Modelos guardados: {total_successful}\")"),
    (0, 2710): ("    info_print(f\"   🕐 Tiempo total: ~{total_successful * 2:.1f} minutos (simulado)\")",
                "    info_print(f\"   - Tiempo total: ~{total_successful * 2:.1f} minutos (simulado)\")"),
    (0, 2715): ("        info_print(f\"\\n🏆 MEJORES RESULTADOS:\")",
                "        info_print(f\"\\nMEJORES RESULTADOS:\")"),
    (0, 2728): ("            info_print(f\"   🥇 Menor Val Loss: {best_val_loss['Experimento']} ({best_val_loss['Val Loss']})\")",
                "            info_print(f\"   - Menor Val Loss: {best_val_loss['Experimento']} ({best_val_loss['Val Loss']})\")"),
    (0, 2732): ("            info_print(f\"   🥇 Menor RMSE: {best_rmse['Experimento']} ({best_rmse['RMSE']})\")",
                "            info_print(f\"   - Menor RMSE: {best_rmse['Experimento']} ({best_rmse['RMSE']})\")"),
    (0, 2736): ("            info_print(f\"   🥇 Mayor R²: {best_r2['Experimento']} ({best_r2['R²']})\")",
                "            info_print(f\"   - Mayor R²: {best_r2['Experimento']} ({best_r2['R²']})\")"),
}

# Contador de cambios realizados
changes_made = 0

# Aplicar reemplazos
for cell_idx, line_idx in specific_replacements:
    original, replacement = specific_replacements[(cell_idx, line_idx)]
    
    # Asegurarse de que el índice de celda existe
    if cell_idx < len(notebook['cells']):
        cell = notebook['cells'][cell_idx]
        
        # Verificar que sea una celda de código
        if cell['cell_type'] == 'code':
            # Asegurarse de que el índice de línea existe
            if line_idx < len(cell['source']):
                line = cell['source'][line_idx]
                
                # Solo reemplazar si contiene algún emoji de los identificados
                if '🌧️' in line or '🔮' in line or '📊' in line or '💧' in line or '✅' in line or '❌' in line or '📈' in line or '🥇' in line or '🎉' in line or '🕐' in line:
                    # Hacer el reemplazo específico
                    cell['source'][line_idx] = replacement + "\n"
                    changes_made += 1
                    print(f"Reemplazado: Celda {cell_idx}, Línea {line_idx}")
                    print(f"  Antes: {original}")
                    print(f"  Después: {replacement}")
                    print("-" * 80)

# Guardar el notebook modificado si hubo cambios
if changes_made > 0:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"✅ Se realizaron {changes_made} reemplazos en {notebook_path}")
else:
    print(f"⚠️ No se encontraron coincidencias exactas para reemplazar")

# Configurar matplotlib para evitar futuros warnings
mpl_config = """# Configuración de Matplotlib para evitar warnings de glifos
import matplotlib as mpl

# Establecer opciones globales de matplotlib
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
mpl.rcParams['font.family'] = 'sans-serif'

# Desactivar uso de caracteres Unicode en los títulos
mpl.rcParams['axes.unicode_minus'] = False

print("Configuración de matplotlib ajustada para evitar warnings de glifos")
"""

# Crear celda al inicio del notebook con la configuración
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": mpl_config.split('\n'),
    "outputs": []
}

# Insertar la celda al inicio
notebook['cells'].insert(0, new_cell)

# Guardar el notebook con la nueva celda
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Se agregó una celda de configuración de matplotlib al inicio del notebook")
print("ℹ️ Este script ha completado las siguientes acciones:")
print(f"   1. Reemplazados {changes_made} emojis que causaban warnings")
print("   2. Agregada configuración de matplotlib al inicio del notebook")
print("   3. Las correcciones se han aplicado directamente en las líneas problemáticas") 