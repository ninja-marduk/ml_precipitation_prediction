# 📊 Diagramas de Arquitectura - Ultra Alta Calidad

Este directorio contiene los diagramas PlantUML del proyecto de predicción de precipitación, configurados para generar imágenes de **ultra alta calidad (1200px+)** con **800 DPI** para impresión profesional.

## 🎯 Diagramas Disponibles

### Ventanas Temporales
1. **`ventanas_temporales.puml`** - Diagrama básico con packages
2. **`ventanas_temporales_detallado.puml`** - Diagrama de secuencia técnico
3. **`ventanas_temporales_visual.puml`** - Replica exacta del diseño visual

### Ingeniería de Características
4. **`ingenieria_caracteristicas.puml`** - Diagrama visual básico
5. **`ingenieria_caracteristicas_flujo.puml`** - Flujo de procesamiento
6. **`ingenieria_caracteristicas_estructura.puml`** - Estructura de datos UML

### Modelos Evaluados
7. **`modelos_evaluados.puml`** - Diagrama visual de los 3 modelos
8. **`modelos_evaluados_detallado.puml`** - Arquitectura detallada con flujo de datos
9. **`modelos_evaluados_comparacion.puml`** - Comparación lado a lado con métricas

### Arquitecturas por Capas (Dimensiones Detalladas)
10. **`modelos_capas_dimensiones.puml`** - Cada capa como bloque con dimensiones
11. **`modelos_bloques_3d.puml`** - Representación visual 3D con emojis y colores
12. **`modelos_tensores.puml`** - Especificaciones técnicas completas de tensores

## ⚙️ Configuración de Ultra Alta Calidad

Todos los diagramas incluyen las siguientes configuraciones optimizadas para **impresión profesional**:

```plantuml
!define SCALE 3
!define DPI 800
skinparam dpi 800
skinparam defaultFontSize 14-16
skinparam titleFontSize 20-24
skinparam minClassWidth 200-250
skinparam minClassHeight 120-140
skinparam padding 10-12
```

### Especificaciones Técnicas:
- **DPI**: 800 (ultra alta resolución para impresión)
- **Escala**: 3x para máxima nitidez
- **Fuentes**: Optimizadas para legibilidad en impresión
- **Dimensiones**: Expandidas para mejor espaciado
- **Padding**: Aumentado para mejor presentación visual
- **Formato**: PNG de ultra alta calidad

## 🚀 Generación de Imágenes

### Opción 1: Script Automatizado
```bash
./generate_high_quality_diagrams.sh
```

### Opción 2: Manual
```bash
# Instalar PlantUML (si no está instalado)
brew install plantuml  # macOS
# o
apt-get install plantuml  # Ubuntu

# Generar imagen específica
plantuml -tpng -o images/ ventanas_temporales.puml

# Generar todas las imágenes
plantuml -tpng -o images/ *.puml
```

### Opción 3: Usando JAR
```bash
# Descargar plantuml.jar desde http://plantuml.com/download
java -jar plantuml.jar -tpng -o images/ *.puml
```

## 📁 Estructura de Archivos

```
docs/architecture/
├── README_diagramas.md                          # Este archivo
├── generate_high_quality_diagrams.sh           # Script de generación
├── images/                                      # Imágenes generadas
│   ├── ventanas_temporales.png
│   ├── ventanas_temporales_detallado.png
│   ├── ventanas_temporales_visual.png
│   ├── ingenieria_caracteristicas.png
│   ├── ingenieria_caracteristicas_flujo.png
│   ├── ingenieria_caracteristicas_estructura.png
│   ├── modelos_evaluados.png
│   ├── modelos_evaluados_detallado.png
│   ├── modelos_evaluados_comparacion.png
│   ├── modelos_capas_dimensiones.png
│   ├── modelos_bloques_3d.png
│   └── modelos_tensores.png
└── *.puml                                       # Archivos fuente PlantUML
```

## 🎨 Características Visuales

### Ventanas Temporales
- **Colores**: Púrpura (#9966FF) para numeración, colores suaves para contenido
- **Elementos**: Entrada (60 meses), Salida (t+1, t+2, t+3), Partición (train/val/test)
- **Notas**: Explicaciones técnicas detalladas

### Ingeniería de Características
- **Componentes**: KCE (K-means), PAFC (lags), Sets (BASE/BASE+KCE/BASE+KCE+PAFC)
- **Flujo**: Desde datos base hasta características finales
- **Estructura**: Clases UML con dimensiones específicas

### Modelos Evaluados
- **ConvLSTM**: Encoder-decoder con atención CBAM, baseline del proyecto
- **ConvGRU**: Arquitectura residual ligera con normalización reforzada
- **Transformer**: Híbrido CNN+LSTM con self-attention temporal
- **Comparación**: Métricas de rendimiento y complejidad computacional

### Arquitecturas por Capas
- **Dimensiones**: Cada capa mostrada como bloque con shape específico
- **Bloques 3D**: Representación visual con colores y emojis descriptivos
- **Tensores**: Especificaciones técnicas detalladas con número de parámetros
- **Flujo de datos**: Transformaciones de dimensiones paso a paso

## 🔧 Solución de Problemas

### PlantUML no encontrado
```bash
# macOS
brew install plantuml

# Ubuntu/Debian
sudo apt-get install plantuml

# Windows (con Chocolatey)
choco install plantuml
```

### Baja calidad de imagen
- Verificar que `skinparam dpi 800` esté presente
- Asegurar que `!define SCALE 3` esté configurado
- Usar formato PNG en lugar de SVG para mejor compatibilidad
- Para imágenes web, considerar reducir a DPI 300 si el tamaño es muy grande

### Texto muy pequeño
- Incrementar `defaultFontSize` y `titleFontSize`
- Ajustar `minClassWidth` y `minClassHeight`
- Verificar que la escala sea 3x
- Aumentar `padding` para mejor espaciado

## 📈 Uso en Documentación

Las imágenes generadas pueden ser incluidas en:
- Documentación técnica
- Presentaciones
- Papers académicos
- Reportes de proyecto

Ejemplo de inclusión en Markdown:
```markdown
![Ventanas Temporales](images/ventanas_temporales.png)
```

## 🔄 Actualización de Diagramas

1. Editar el archivo `.puml` correspondiente
2. Ejecutar el script de generación o comando manual
3. Verificar la calidad de la imagen generada
4. Actualizar la documentación si es necesario
