# Gap Analysis: Estado Actual vs. Hipótesis Doctoral
## Predicción Espaciotemporal de Precipitación Mensual con Modelos Híbridos

**Fecha:** 2 de enero de 2026
**Autor:** Manuel R. Pérez
**Objetivo:** Evaluar el estado actual del proyecto vs. hipótesis y objetivos doctorales, identificar oportunidades de innovación de alto impacto para publicación en revistas Q1.

---

## 1. RESUMEN EJECUTIVO

### Estado Actual del Proyecto
| Aspecto | Completado | Pendiente |
|---------|------------|-----------|
| Dataset | CHIRPS-2.0, estaciones, topografía | (Mantener sin cambios) |
| Modelos baseline | V2 ConvLSTM (10 arquitecturas) | - |
| Modelos híbridos | Parcial (FNO falló -4.5% vs V2) | GNN, Wavelet, Physics-Informed |
| Evaluación | Benchmark V2 vs V3 riguroso | Comparación con literatura SOTA |
| Ensemble | No implementado | Stacking, meta-learner |

### Hipótesis Doctoral (Estado de Validación)
> *"Aplicar modelos de machine learning, combinados con análisis de series temporales y métodos de preprocesamiento, mejorará significativamente la precisión de predicciones mensuales de precipitación en áreas montañosas."*

**Estado:** PARCIALMENTE VALIDADA
- ConvLSTM mejora sobre baselines tradicionales
- RMSE ~98mm, R²=0.437 con features básicos
- Hibridación avanzada NO demostrada (FNO falló)
- Decomposición temporal NO implementada
- Ensemble learning NO implementado

---

## 2. GAP ANALYSIS POR OBJETIVO

### Objetivo 1: Generar Dataset Heterogéneo
| Requisito | Estado | Gap |
|-----------|--------|-----|
| Features de fuentes heterogéneas | Completado | CHIRPS + estaciones + topografía |
| Integración para patrones | Parcial | Features KCE/PAFC degradan rendimiento |
| Exploración de features | Completado | Benchmark V2 vs V3 |

**Nota:** Dataset CHIRPS se mantendrá. El gap de KCE/PAFC se puede abordar con mejores arquitecturas (GNN, Wavelet).

### Objetivo 2: Desarrollar Modelos Baseline
**Estado: COMPLETADO** - V2 ConvLSTM es un baseline sólido (RMSE=98mm, R²=0.437)

### Objetivo 3: Proponer Modelos Híbridos y Ensemble
**Estado: GAP CRÍTICO**
| Requisito | Estado | Prioridad |
|-----------|--------|-----------|
| Técnicas de hibridación | FNO falló | CRÍTICA |
| Técnicas ensemble | No implementado | CRÍTICA |

**Técnicas requeridas:**
1. GNN-TAT para dependencias espaciales no-Euclidianas
2. Physics-Informed Loss para restricciones físicas suaves
3. Wavelet + ConvLSTM para decomposición multi-escala
4. Ensemble con meta-learner adaptativo

### Objetivo 4: Evaluar Precisión y Eficiencia
**Estado: PARCIALMENTE COMPLETADO**
- Métricas calculadas (RMSE, MAE, R², NSE)
- Falta comparación con SOTA de literatura

---

## 3. EVALUACIÓN DE TÉCNICAS DE INNOVACIÓN

### Matriz de Evaluación (Escala 1-5, Prioridad: Novedad Científica)

| Técnica | Novedad | Factibilidad | Impacto | Gap Lit. | Publicabilidad | **SCORE** |
|---------|---------|--------------|---------|----------|----------------|-----------|
| **GNN-TAT** | 5 | 4 | 5 | 5 | 5 | **24/25** |
| **Physics-Informed Loss** | 5 | 4 | 4 | 5 | 5 | **23/25** |
| **Wavelet-ConvLSTM** | 4 | 5 | 4 | 4 | 4 | **21/25** |
| **Spatial Attention** | 4 | 5 | 4 | 4 | 4 | **21/25** |
| **Ensemble Meta-Learner** | 4 | 5 | 4 | 3 | 4 | **20/25** |

### Técnica Prioridad 1: GNN-TAT (Graph Neural Networks + Temporal Attention)

**Descripción:** Modelar celdas de precipitación como nodos de un grafo, con edges basados en distancia, elevación y correlación histórica.

**Por qué es la técnica más innovadora:**
- GNN para precipitación espaciotemporal es cutting-edge (< 10 papers en 2024-2025)
- ConvLSTM asume grid regular (Euclidiano), GNN captura dependencias no-Euclidianas
- Permite modelar efectos orográficos (montañas de Boyacá) de forma natural
- Los edges del grafo son interpretables

**Arquitectura propuesta:**
```
Input: CHIRPS grid (T×H×W×C) → Graph Construction → GNN Encoder (GraphSAGE/GAT)
                                                          ↓
                                    Temporal Attention (Multi-Head)
                                                          ↓
                                    LSTM Decoder → Precipitation Forecast
```

**Mejora esperada:** RMSE -10 a -15mm
**Paper target:** Nature Communications / Water Resources Research (Q1)

### Técnica Prioridad 2: Physics-Informed Spatio-Temporal Loss

**Descripción:** Incorporar restricciones físicas como términos de pérdida suaves.

**Loss Function propuesta:**
```python
L_total = L_data + λ₁*L_spatial_smooth + λ₂*L_temporal_causal + λ₃*L_water_balance

donde:
L_data = MSE(y_pred, y_true)
L_spatial_smooth = ||∇P||² (penalizar gradientes espaciales extremos)
L_temporal_causal = ||P(t) - P(t-1)||² si P(t) > threshold
L_water_balance = |∑P_region - P_expected|² (consistencia regional)
```

**Mejora esperada:** RMSE -5 a -12mm
**Paper target:** Journal of Hydrology (Q1)

### Técnica Prioridad 3: Wavelet Multi-Scale ConvLSTM

**Descripción:** DWT decomposition → 4 branches ConvLSTM → Fusion

**Mejora esperada:** RMSE -8 a -12mm, mejor consistencia H1→H12
**Paper target:** Water Resources Research (Q1)

---

## 4. PUNTOS CLAVE DEL REVIEW SISTEMÁTICO (85 estudios)

### Hallazgos Relevantes
1. **Decomposition-based hybrids son los mejores:** RMSE reductions 15-35%
2. **Gap identificado:** "probabilistic post-processing remains underused at monthly horizons"
3. **Técnicas top:**
   - CEEMDAN-SVM-LSTM: 57.54% mejora
   - Bat-ELM: 61.31% mejora
   - SARIMA-ANN: 77.45% mejora
4. **Pitfalls:** Data leakage, station non-representativeness, non-stationarity

### Comparación con SOTA
| Métrica | Tu V2 ConvLSTM | SOTA Review | Nota |
|---------|----------------|-------------|------|
| RMSE | 98.17 mm | ~17-40 mm | Diferente escala (grid vs estación) |
| R² | 0.437 | 0.95-0.98 | Diferente horizonte (H=12 vs H=1-3) |

---

## 5. PLAN DE ACCIÓN (20 semanas)

### Fase 1: GNN-TAT (Semanas 1-4)
1. Construcción del grafo espacial
2. Implementación GNN encoder (PyTorch Geometric)
3. Integración temporal attention
4. Benchmark vs V2

### Fase 2: Physics-Informed Loss (Semanas 5-7)
5. Diseño de loss function híbrida
6. Implementación constraints
7. Ablation studies

### Fase 3: Wavelet Multi-Scale (Semanas 8-10)
8. DWT decomposition
9. Parallel ConvLSTM branches
10. Fusion layer

### Fase 4: Ensemble & Integración (Semanas 11-14)
11. Context-Adaptive Meta-Learner
12. Framework integrado
13. Benchmark Final

### Fase 5: Publicación (Semanas 15-20)
14. Paper 1: GNN-TAT (Nature Communications / WRR)
15. Paper 2: Physics-Informed Loss (J. Hydrology)
16. Paper 3: Framework Integrado (Env. Modelling & Software)

---

## 6. CRITERIOS DE ÉXITO DOCTORAL

| Criterio | Umbral Mínimo | Objetivo Óptimo |
|----------|---------------|-----------------|
| **Novedad científica** | 2 contribuciones | 4 contribuciones |
| R² final | > 0.60 | > 0.75 |
| RMSE final | < 85 mm | < 70 mm |
| Publicaciones Q1 | 2 papers | 3 papers |
| Reproducibilidad | Código abierto | Framework operacional |

---

## 7. CONTRIBUCIONES DOCTORALES ESPERADAS

1. **GNN-TAT para Precipitación Espaciotemporal**
   - Primer modelo GNN para predicción mensual en grid
   - Paper: Nature Communications / Water Resources Research

2. **Physics-Informed Spatio-Temporal Loss**
   - Soft physics constraints sin PDE solvers
   - Paper: Journal of Hydrology

3. **Multi-Scale Wavelet-ConvLSTM**
   - Decomposición temporal + ConvLSTM espacial
   - Paper: Water Resources Research

4. **Framework Integrado + Benchmark**
   - Combinación GNN + Physics + Wavelet + Ensemble
   - Paper: Environmental Modelling & Software

---

## 8. RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| GNN no mejora ConvLSTM | Media | Enfatizar interpretabilidad |
| Physics constraints degradan rendimiento | Baja | λ tunable, ablation studies |
| R² < 0.60 | Baja | Ensemble combina fortalezas |
| Colab Pro timeout | Media | Checkpointing, batch training |

---

## CONCLUSIÓN

El proyecto tiene base sólida pero requiere implementar técnicas de hibridación avanzadas para validar completamente la hipótesis doctoral. La recomendación es priorizar **GNN-TAT** por máxima novedad científica y funcionamiento con dataset existente.

**Siguiente paso:** Diseñar arquitectura GNN-TAT y construir grafo espacial basado en correlaciones CHIRPS + topografía.
