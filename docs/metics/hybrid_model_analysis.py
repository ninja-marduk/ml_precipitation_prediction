# Script: análisis de promedios de RMSE y MAE por modelo híbrido

import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Paso 1: Cargar el archivo CSV desde el mismo directorio donde se ejecuta el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "hybrid_model_avg_rmse_mae.csv")
    df = pd.read_csv(csv_file, index_col=0)

    # Paso 2: Ordenar por RMSE y MAE para visualización
    df_rmse_sorted = df.sort_values(by="RMSE")
    df_mae_sorted = df.sort_values(by="MAE")

    # Paso 3: Crear la figura y los ejes para ambas métricas
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 10), sharey=True)
    fig.suptitle("Performance of Hybrid Models", fontsize=18, fontweight='bold')

    # Gráfico de barras horizontales para RMSE
    ax1.barh(df_rmse_sorted.index, df_rmse_sorted["RMSE"], color="yellowgreen")
    ax1.set_title("Average RMSE by Hybrid Model", fontsize=14)
    ax1.set_xlabel("RMSE")
    ax1.set_ylabel("Hybrid Model")

    # Gráfico de barras horizontales para MAE
    ax2.barh(df_mae_sorted.index, df_mae_sorted["MAE"], color="mediumaquamarine")
    ax2.set_title("Average MAE by Hybrid Model", fontsize=14)
    ax2.set_xlabel("MAE")

    # Ajustar diseño
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Guardar imagen en el mismo directorio
    image_path = os.path.join(script_dir, "hybrid_model_comparison.png")
    plt.savefig(image_path, dpi=300)
    print(f"Gráfico guardado como {image_path}")

    # Mostrar en pantalla
    plt.show()

if __name__ == "__main__":
    main()
