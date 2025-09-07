#!/bin/bash

# Script para generar diagramas PlantUML en alta calidad (600px)
# Asegúrate de tener PlantUML instalado: 
# - brew install plantuml (macOS)
# - apt-get install plantuml (Ubuntu)
# - O descargar plantuml.jar desde http://plantuml.com/download

echo "🚀 Generando diagramas PlantUML en ULTRA ALTA calidad (DPI 800, 1200px+)..."

# Directorio de arquitectura
ARCH_DIR="/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/docs/architecture"
OUTPUT_DIR="$ARCH_DIR/images"

# Crear directorio de salida si no existe
mkdir -p "$OUTPUT_DIR"

# Lista de archivos PlantUML
DIAGRAMS=(
    "ventanas_temporales.puml"
    "ventanas_temporales_detallado.puml"
    "ventanas_temporales_visual.puml"
    "ingenieria_caracteristicas.puml"
    "ingenieria_caracteristicas_flujo.puml"
    "ingenieria_caracteristicas_estructura.puml"
    "modelos_evaluados.puml"
    "modelos_evaluados_detallado.puml"
    "modelos_evaluados_comparacion.puml"
    "modelos_capas_dimensiones.puml"
    "modelos_bloques_3d.puml"
    "modelos_tensores.puml"
    "modelos_arquitectura_visual.puml"
    "modelos_tensores_3d.puml"
)

# Generar cada diagrama
for diagram in "${DIAGRAMS[@]}"; do
    echo "📊 Procesando: $diagram"
    
    # Generar PNG de alta calidad
    if command -v plantuml &> /dev/null; then
        plantuml -tpng -o "$OUTPUT_DIR" "$ARCH_DIR/$diagram"
        echo "✅ Generado: ${diagram%.puml}.png"
    elif [ -f "plantuml.jar" ]; then
        java -jar plantuml.jar -tpng -o "$OUTPUT_DIR" "$ARCH_DIR/$diagram"
        echo "✅ Generado: ${diagram%.puml}.png"
    else
        echo "❌ Error: PlantUML no encontrado. Instala PlantUML o descarga plantuml.jar"
        exit 1
    fi
done

echo ""
echo "🎉 ¡Todos los diagramas generados exitosamente!"
echo "📁 Ubicación: $OUTPUT_DIR"
echo ""
echo "📋 Archivos generados:"
ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || echo "No se encontraron archivos PNG generados"

echo ""
echo "🔧 Configuración aplicada a todos los diagramas:"
echo "   • DPI: 800 (ULTRA ALTA RESOLUCIÓN)"
echo "   • Escala: 3x"
echo "   • Tamaños de fuente optimizados para impresión"
echo "   • Dimensiones mínimas expandidas"
echo "   • Padding aumentado para mejor espaciado"
echo "   • Calidad de imagen: Ultra Alta (1200px+)"
