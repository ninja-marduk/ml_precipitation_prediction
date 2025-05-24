# Script: análisis de promedios de RMSE y MAE por modelo híbrido

import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Paso 1: Cargar el archivo CSV desde el mismo directorio donde se ejecuta el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "hybrid_model_avg_rmse_mae.csv")
    df = pd.read_csv(csv_file, index_col=0)

    # Paso 2: Ordenar por RMSE y MAE para visualización (orden descendente para que los mayores errores estén abajo)
    df_rmse_sorted = df.sort_values(by="RMSE", ascending=False)
    df_mae_sorted = df.sort_values(by="MAE", ascending=False)

    # Paso 3: Crear la figura y los ejes para ambas métricas
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 8))
    
    # Eliminar título principal para coincidir con la imagen de referencia
    # fig.suptitle("Performance of Hybrid Models", fontsize=18, fontweight='bold')
    
    # Configurar un tema de colores apropiado (viridis: azul-verde-amarillo)
    # Usar un mapa de colores que vaya de azul (buen desempeño) a rojo (mal desempeño)
    # plt.style.use('seaborn-colorblind')  # Usar un tema más profesional
    
    # Crear un degradado de colores para las barras RMSE usando 'coolwarm' 
    # Un mapa más temático que va de azul (buen resultado) a rojo (mal resultado)
    rmse_norm = df_rmse_sorted["RMSE"] / df_rmse_sorted["RMSE"].max()
    rmse_colors = plt.cm.coolwarm(rmse_norm)  # coolwarm: azul (menor) a rojo (mayor)
    
    # Crear un degradado de colores para las barras MAE
    # Usando el mismo esquema de colores
    mae_norm = df_mae_sorted["MAE"] / df_mae_sorted["MAE"].max()
    mae_colors = plt.cm.coolwarm(mae_norm)  # coolwarm: azul (menor) a rojo (mayor)

    # Gráfico de barras horizontales para RMSE
    bars_rmse = ax1.barh(df_rmse_sorted.index, df_rmse_sorted["RMSE"], color=rmse_colors)
    ax1.set_title("Average RMSE by Hybrid Model", fontsize=14, pad=10)
    ax1.set_xlabel("RMSE", fontsize=12)
    ax1.set_ylabel("Hybrid Model", fontsize=12)
    
    # Añadir líneas de cuadrícula verticales
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    # Configurar límites del eje X
    ax1.set_xlim(0, 275)
    # Configurar marcas del eje X
    ax1.set_xticks([0, 50, 100, 150, 200, 250])
    
    # Gráfico de barras horizontales para MAE
    bars_mae = ax2.barh(df_mae_sorted.index, df_mae_sorted["MAE"], color=mae_colors)
    ax2.set_title("Average MAE by Hybrid Model", fontsize=14, pad=10)
    ax2.set_xlabel("MAE", fontsize=12)
    # No mostrar etiqueta del eje Y en el segundo gráfico
    # ax2.set_ylabel("Hybrid Model")
    
    # Añadir líneas de cuadrícula verticales
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    # Configurar límites del eje X
    ax2.set_xlim(0, 160)
    # Configurar marcas del eje X
    ax2.set_xticks([0, 50, 100, 150])

    # Añadir colorbar para la interpretación de colores
    cax1 = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=0, vmax=max(df_rmse_sorted["RMSE"].max(), df_mae_sorted["MAE"].max()))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax1)
    cbar.set_label('Error Magnitude', rotation=270, labelpad=20)
    
    # Ajustar diseño teniendo en cuenta la colorbar
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    # Guardar imagen en el mismo directorio
    image_path = os.path.join(script_dir, "hybrid_model_comparison.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado como {image_path}")

    # Mostrar en pantalla
    plt.show()

if __name__ == "__main__":
    main()
