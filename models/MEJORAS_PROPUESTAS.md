# 🚀 Mejoras Propuestas para Modelos Espaciales de Precipitación

## 📊 Análisis de Resultados Actuales

### Problemas Identificados:

1. **R² Negativos**: Varios modelos muestran R² < 0, especialmente en H=2
   - ConvLSTM en KCE/PAFC: R² = -0.042 y -0.161
   - ConvRNN en KCE: R² = -0.340
   - ConvGRU en PAFC: R² = -0.052

2. **Degradación en H=3**: RMSE > 100 en muchos casos
   - ConvLSTM: RMSE hasta 111.3
   - ConvGRU: RMSE hasta 106.9
   - ConvRNN: RMSE hasta 87.3

3. **Inestabilidad**: Batch size = 4 es muy pequeño

### Mejores Resultados Actuales:
- **H=1**: ConvRNN BASIC (RMSE=43.49, R²=0.767)
- **H=2**: ConvGRU BASIC (RMSE=32.40, R²=0.401)
- **H=3**: ConvRNN KCE (RMSE=71.47, R²=0.611)

## 🔧 Mejoras Implementadas

### 1. Optimización de Hiperparámetros

| Parámetro | Valor Original | Valor Mejorado | Justificación |
|-----------|----------------|----------------|---------------|
| Batch Size | 4 | **16** | Mayor estabilidad en gradientes |
| Learning Rate | 1e-3 | **5e-4** | Convergencia más suave |
| Epochs | 50 | **100** | Más tiempo con early stopping |
| Patience | 6 | **10** | Evitar detención prematura |
| Dropout | 0 | **0.2** | Regularización |
| L2 Reg | 0 | **1e-5** | Prevenir overfitting |

### 2. Arquitecturas Mejoradas

#### ConvLSTM con Atención (ConvLSTM_Att)
```python
- 3 capas ConvLSTM (64→32→16 filtros)
- CBAM (Channel + Spatial Attention)
- BatchNorm + Dropout en cada capa
- Cabeza multi-escala (1×1, 3×3, 5×5)
```

#### ConvGRU Residual (ConvGRU_Res)
```python
- Skip connections desde input
- BatchNorm mejorado
- 2 bloques ConvGRU (64→32 filtros)
- Conexión residual final
```

#### Transformer Híbrido (Hybrid_Trans)
```python
- Encoder CNN temporal
- Multi-head attention (4 heads)
- LSTM para agregación temporal
- Decoder espacial
```

### 3. Técnicas Avanzadas

#### Learning Rate Scheduling
- **Warmup**: 5 épocas iniciales
- **Cosine Decay**: Reducción suave después del warmup
- **ReduceLROnPlateau**: Reducción adicional si se estanca

#### Data Augmentation
- Ruido gaussiano (σ=0.005)
- Preserva coherencia espacial y temporal

#### Regularización
- Dropout espacial (0.2)
- L2 en todos los pesos
- Batch Normalization

## 📈 Mejoras Esperadas

### Por Horizonte:
- **H=1**: RMSE < 40 (mejora ~8%)
- **H=2**: RMSE < 30, R² > 0.5 (mejora significativa)
- **H=3**: RMSE < 65, R² > 0.65 (mejora ~10%)

### Por Modelo:
1. **ConvLSTM_Att**: Mejor captura de patrones espaciales relevantes
2. **ConvGRU_Res**: Mayor estabilidad y menos degradación temporal
3. **Hybrid_Trans**: Mejor modelado de dependencias largas

## 🚀 Próximos Pasos

### Corto Plazo:
1. Entrenar modelos con configuración mejorada
2. Validar mejoras en métricas
3. Análisis de errores por región

### Medio Plazo:
1. **Ensemble Methods**: Combinar mejores modelos
2. **Multi-Task Learning**: Predecir múltiples variables
3. **Physics-Informed Loss**: Incorporar restricciones físicas

### Largo Plazo:
1. **Modelos 3D**: ConvLSTM3D para capturar altura
2. **Graph Neural Networks**: Para relaciones espaciales irregulares
3. **Uncertainty Quantification**: Intervalos de confianza

## 💻 Uso del Script

```bash
# Entrenar modelos avanzados
python models/train_advanced_models.py

# Con GPU específica
CUDA_VISIBLE_DEVICES=0 python models/train_advanced_models.py
```

## 📊 Monitoreo

Los resultados se guardan en:
- `models/output/Advanced_Spatial/advanced_results.csv`
- Historiales de entrenamiento por experimento
- Modelos guardados en formato .keras

## 🔍 Comparación con Baseline

El script genera automáticamente comparaciones con los modelos originales, mostrando:
- % de mejora en RMSE
- Evolución de R² por horizonte
- Tabla resumen de mejores modelos 