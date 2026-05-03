# 🚀 ROADMAP TÉCNICO: V2 → FNO → WAVELET
## Plan de Implementación Detallado para Hibridaciones Avanzadas

---

## 📋 **OVERVIEW DEL PLAN DE TRANSICIÓN**

```
V2 (ACTUAL)           V3 (FNO)              V4 (WAVELET)
R² = 0.75        →    R² = 0.85        →    R² = 0.88+
11 modelos            15 modelos            20+ modelos
ConvRNN_Enhanced      FNO+ConvRNN          Wavelet+FNO+ConvRNN
2-3 semanas           2-3 semanas          1-2 semanas
```

---

## 🔬 **FASE 1: V2 → FNO INTEGRATION (Semanas 1-3)**

### **🎯 OBJETIVO:**
Integrar **Fourier Neural Operators** con el mejor modelo actual (**ConvRNN_Enhanced + PAFC**) para modelar dinámicas PDE de precipitación.

### **📚 FUNDAMENTO TEÓRICO:**

#### **¿Por qué FNO para Precipitación?**
```python
# Precipitación sigue ecuaciones diferenciales parciales (PDE):
# ∂u/∂t + ∇·(u⃗v) = S - E + D∇²u
# donde:
# u = precipitación
# v⃗ = campo de viento
# S = fuentes (evaporación oceánica)
# E = sumideros (evapotranspiración)
# D = difusión atmosférica

# FNO aprende operadores que mapean:
# Input: Condiciones iniciales u₀(x,y,t)
# Output: Solución futura u(x,y,t+Δt)
# SIN depender de la resolución espacial!
```

### **🏗️ ARQUITECTURA FNO-ENHANCED:**

#### **Paso 1.1: Implementar FNO Core (Semana 1)**

```python
# Crear: models/lib/fno_layers.py
import tensorflow as tf
import numpy as np

class SpectralConv2D(tf.keras.layers.Layer):
    """
    Spectral Convolution Layer - Core de FNO
    Realiza convolución en el dominio de Fourier
    """
    
    def __init__(self, out_channels, modes1, modes2, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.modes1 = modes1  # Número de modos Fourier en x
        self.modes2 = modes2  # Número de modos Fourier en y
        
    def build(self, input_shape):
        in_channels = input_shape[-1]
        
        # Pesos espectrales complejos
        self.weights1 = self.add_weight(
            shape=(in_channels, self.out_channels, self.modes1, self.modes2),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.complex64,
            name='spectral_weights1'
        )
        
        self.weights2 = self.add_weight(
            shape=(in_channels, self.out_channels, self.modes1, self.modes2),
            initializer='glorot_uniform', 
            trainable=True,
            dtype=tf.complex64,
            name='spectral_weights2'
        )
        
    def call(self, x):
        """
        x: (batch, height, width, channels)
        """
        batch_size = tf.shape(x)[0]
        height, width = tf.shape(x)[1], tf.shape(x)[2]
        
        # 1. FFT 2D
        x_ft = tf.signal.fft2d(tf.cast(x, tf.complex64))
        
        # 2. Multiplicar por pesos espectrales (solo modos bajos)
        out_ft = tf.zeros_like(x_ft)
        
        # Modos positivos
        out_ft = tf.tensor_scatter_nd_update(
            out_ft,
            indices=self._get_mode_indices(height, width, positive=True),
            updates=tf.einsum('bijk,jlik->blik', 
                            x_ft[:, :self.modes1, :self.modes2, :], 
                            self.weights1)
        )
        
        # Modos negativos (simetría hermitiana)
        out_ft = tf.tensor_scatter_nd_update(
            out_ft,
            indices=self._get_mode_indices(height, width, positive=False),
            updates=tf.einsum('bijk,jlik->blik',
                            x_ft[:, -self.modes1:, -self.modes2:, :],
                            self.weights2)
        )
        
        # 3. IFFT 2D
        out = tf.signal.ifft2d(out_ft)
        return tf.cast(tf.math.real(out), tf.float32)
    
    def _get_mode_indices(self, height, width, positive=True):
        # Implementar índices para modos Fourier
        # Detalles técnicos específicos...
        pass

class FNOBlock(tf.keras.layers.Layer):
    """
    Bloque FNO completo: Spectral Conv + Skip Connection + Activation
    """
    
    def __init__(self, out_channels, modes1, modes2, **kwargs):
        super().__init__(**kwargs)
        self.spectral_conv = SpectralConv2D(out_channels, modes1, modes2)
        self.skip_conv = tf.keras.layers.Conv2D(out_channels, 1)
        self.activation = tf.keras.layers.ReLU()
        
    def call(self, x):
        # Rama espectral
        spectral_out = self.spectral_conv(x)
        
        # Rama skip connection
        skip_out = self.skip_conv(x)
        
        # Combinar y activar
        return self.activation(spectral_out + skip_out)

class FNO2D(tf.keras.layers.Layer):
    """
    Fourier Neural Operator 2D completo
    """
    
    def __init__(self, modes1=12, modes2=12, width=64, **kwargs):
        super().__init__(**kwargs)
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        # Proyección de entrada
        self.input_proj = tf.keras.layers.Dense(self.width)
        
        # Bloques FNO
        self.fno_blocks = [
            FNOBlock(self.width, modes1, modes2) for _ in range(4)
        ]
        
        # Proyección de salida
        self.output_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
    def call(self, x):
        # Proyectar a espacio latente
        x = self.input_proj(x)
        
        # Aplicar bloques FNO
        for block in self.fno_blocks:
            x = block(x)
            
        # Proyectar a salida
        return self.output_proj(x)
```

#### **Paso 1.2: Integrar FNO con ConvRNN (Semana 2)**

```python
# Agregar a: models/base_models_Conv_STHyMOUNTAIN_V3.ipynb

def build_fno_conv_rnn_hybrid(n_feats: int):
    """
    BREAKTHROUGH V3: FNO + ConvRNN Hybrid
    
    ARCHITECTURE:
    1. ConvRNN para capturar dinámicas temporales locales
    2. FNO para modelar operadores PDE globales
    3. Fusion layer para combinar ambas representaciones
    
    EXPECTED IMPROVEMENT: +10-15% R² vs ConvRNN_Enhanced
    """
    inp = Input(shape=(INPUT_WINDOW, lat, lon, n_feats))
    
    # ═══ RAMA TEMPORAL (ConvRNN) ═══
    # Procesar secuencia temporal con ConvRNN mejorado
    temporal_branch = TimeDistributed(
        Conv2D(32, (3,3), padding='same', activation='relu')
    )(inp)
    
    temporal_branch = SimpleRNN(
        16, return_sequences=False, dropout=0.1,
        recurrent_dropout=0.1
    )(temporal_branch)
    
    # ═══ RAMA ESPECTRAL (FNO) ═══
    # Tomar último frame para análisis espectral
    last_frame = Lambda(lambda x: x[:, -1, :, :, :])(inp)
    
    # Aplicar FNO para capturar dinámicas PDE
    spectral_branch = FNO2D(
        modes1=12,  # Modos espaciales en x
        modes2=12,  # Modos espaciales en y  
        width=64    # Dimensión latente
    )(last_frame)
    
    # ═══ FUSION LAYER ═══
    # Combinar representaciones temporal y espectral
    # Reshape para compatibilidad
    temporal_reshaped = Reshape((lat, lon, 16))(temporal_branch)
    
    # Concatenar características
    fused = tf.keras.layers.Concatenate(axis=-1)([
        temporal_reshaped, spectral_branch
    ])
    
    # Capa de fusión adaptativa
    fusion_weights = tf.keras.layers.Dense(2, activation='softmax')(
        tf.keras.layers.GlobalAveragePooling2D()(fused)
    )
    
    # Weighted combination
    temporal_weight = tf.expand_dims(tf.expand_dims(fusion_weights[:, 0], -1), -1)
    spectral_weight = tf.expand_dims(tf.expand_dims(fusion_weights[:, 1], -1), -1)
    
    weighted_temporal = temporal_reshaped * temporal_weight
    weighted_spectral = spectral_branch * spectral_weight
    
    final_features = weighted_temporal + weighted_spectral
    
    # ═══ OUTPUT PROJECTION ═══
    out = _spatial_head_multi_horizon(final_features)
    
    return Model(inp, out, name='FNO_ConvRNN_Hybrid')

def build_fno_enhanced_suite():
    """
    Suite completa de modelos FNO-Enhanced
    """
    return {
        'FNO_ConvRNN_Hybrid': build_fno_conv_rnn_hybrid,
        'FNO_ConvLSTM_Hybrid': build_fno_conv_lstm_hybrid,
        'FNO_Pure': build_fno_pure,
        'FNO_Attention': build_fno_attention_hybrid,
    }
```

#### **Paso 1.3: Entrenamiento y Validación (Semana 3)**

```python
# Configuración de entrenamiento V3
MODELS_V3_FNO = {
    **MODELS_Q1_COMPETITIVE,  # Modelos V2 existentes
    **build_fno_enhanced_suite()  # Nuevos modelos FNO
}

# Experimentos específicos para FNO
EXPERIMENTS_V3 = {
    'BASIC': MSE_Loss,
    'PAFC': CombinedLoss_PAFC,  # Mejor de V2
    'FNO_SPECTRAL': SpectralConsistencyLoss,  # Nuevo para FNO
}

# Loss function específica para FNO
class SpectralConsistencyLoss(tf.keras.losses.Loss):
    """
    Loss que penaliza inconsistencias en el dominio espectral
    """
    
    def __init__(self, spectral_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.spectral_weight = spectral_weight
        
    def call(self, y_true, y_pred):
        # MSE tradicional
        mse_loss = tf.keras.losses.mse(y_true, y_pred)
        
        # Consistencia espectral
        y_true_fft = tf.signal.fft2d(tf.cast(y_true, tf.complex64))
        y_pred_fft = tf.signal.fft2d(tf.cast(y_pred, tf.complex64))
        
        spectral_loss = tf.reduce_mean(
            tf.abs(y_true_fft - y_pred_fft) ** 2
        )
        
        return mse_loss + self.spectral_weight * spectral_loss

# Target: R² > 0.82 con FNO_ConvRNN_Hybrid + PAFC
```

---

## 🌊 **FASE 2: FNO → WAVELET INTEGRATION (Semanas 4-6)**

### **🎯 OBJETIVO:**
Agregar **descomposición Wavelet multi-escala** para capturar patrones temporales jerárquicos en precipitación.

### **📚 FUNDAMENTO TEÓRICO:**

#### **¿Por qué Wavelets después de FNO?**
```python
# FNO captura dinámicas espaciales PDE
# Wavelets capturan patrones temporales multi-escala:

# Precipitación tiene componentes en múltiples escalas:
# - Diaria: Convección local
# - Semanal: Sistemas sinópticos  
# - Mensual: Oscilaciones atmosféricas (ENSO, IOD)
# - Estacional: Ciclos anuales

# Wavelet Decomposition:
# P(t) = Σ[a_j φ_j(t)] + Σ Σ[d_j,k ψ_j,k(t)]
#        ↑ aproximación    ↑ detalles multi-escala
```

### **🏗️ ARQUITECTURA WAVELET-FNO:**

#### **Paso 2.1: Implementar Wavelet Layers (Semana 4)**

```python
# Crear: models/lib/wavelet_layers.py
import pywt
import tensorflow as tf

class WaveletDecomposition(tf.keras.layers.Layer):
    """
    Descomposición Wavelet multi-nivel para series temporales
    """
    
    def __init__(self, wavelet='db4', levels=3, **kwargs):
        super().__init__(**kwargs)
        self.wavelet = wavelet
        self.levels = levels
        
    def build(self, input_shape):
        # Pre-computar filtros wavelet
        self.wavelet_filters = pywt.Wavelet(self.wavelet)
        
        # Convertir a tensores TensorFlow
        self.dec_lo = tf.constant(
            self.wavelet_filters.dec_lo, dtype=tf.float32
        )
        self.dec_hi = tf.constant(
            self.wavelet_filters.dec_hi, dtype=tf.float32
        )
        
    def call(self, x):
        """
        x: (batch, time, height, width, channels)
        Returns: Lista de coeficientes [cA, cD1, cD2, ..., cDn]
        """
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        
        coefficients = []
        current = x
        
        for level in range(self.levels):
            # Aplicar filtros wavelet en dimensión temporal
            cA, cD = self._wavelet_transform_1d(current, axis=1)
            coefficients.append(cD)
            current = cA
            
        coefficients.append(current)  # Aproximación final
        return coefficients
    
    def _wavelet_transform_1d(self, x, axis=1):
        """Transformada wavelet 1D a lo largo del eje especificado"""
        # Convolución con filtros wavelet
        # Implementación específica...
        pass

class WaveletReconstruction(tf.keras.layers.Layer):
    """
    Reconstrucción desde coeficientes wavelet
    """
    
    def __init__(self, wavelet='db4', **kwargs):
        super().__init__(**kwargs)
        self.wavelet = wavelet
        
    def call(self, coefficients):
        """
        coefficients: Lista [cA, cD1, cD2, ..., cDn]
        Returns: Señal reconstruida
        """
        # Reconstrucción jerárquica
        current = coefficients[-1]  # Aproximación
        
        for i in range(len(coefficients) - 2, -1, -1):
            current = self._inverse_wavelet_transform(
                current, coefficients[i]
            )
            
        return current

class MultiScaleWaveletProcessor(tf.keras.layers.Layer):
    """
    Procesador que aprende representaciones en cada escala wavelet
    """
    
    def __init__(self, scales=3, filters_per_scale=32, **kwargs):
        super().__init__(**kwargs)
        self.scales = scales
        self.filters_per_scale = filters_per_scale
        
        # Procesadores por escala
        self.scale_processors = []
        for i in range(scales + 1):  # +1 para aproximación
            self.scale_processors.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv3D(
                        filters_per_scale, (3, 3, 3), 
                        padding='same', activation='relu'
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.1)
                ])
            )
            
    def call(self, wavelet_coefficients):
        """
        Procesar cada escala independientemente
        """
        processed_scales = []
        
        for i, (coeff, processor) in enumerate(
            zip(wavelet_coefficients, self.scale_processors)
        ):
            processed = processor(coeff)
            processed_scales.append(processed)
            
        return processed_scales
```

#### **Paso 2.2: Integrar Wavelet + FNO (Semana 5)**

```python
# Agregar a: models/base_models_Conv_STHyMOUNTAIN_V4.ipynb

def build_wavelet_fno_hybrid(n_feats: int):
    """
    BREAKTHROUGH V4: Wavelet + FNO + ConvRNN Triple Hybrid
    
    ARCHITECTURE:
    1. Wavelet decomposition para multi-scale temporal
    2. FNO para spatial PDE dynamics  
    3. ConvRNN para local temporal patterns
    4. Hierarchical fusion
    
    EXPECTED IMPROVEMENT: +5-8% R² vs FNO_ConvRNN
    """
    inp = Input(shape=(INPUT_WINDOW, lat, lon, n_feats))
    
    # ═══ RAMA WAVELET (Multi-scale Temporal) ═══
    wavelet_decomp = WaveletDecomposition(
        wavelet='db4', levels=3
    )(inp)
    
    # Procesar cada escala
    wavelet_processor = MultiScaleWaveletProcessor(
        scales=3, filters_per_scale=16
    )
    processed_scales = wavelet_processor(wavelet_decomp)
    
    # Fusión adaptativa de escalas
    scale_attention = ScaleAttentionLayer()(processed_scales)
    wavelet_features = scale_attention
    
    # ═══ RAMA FNO (Spatial PDE) ═══
    last_frame = Lambda(lambda x: x[:, -1, :, :, :])(inp)
    fno_features = FNO2D(modes1=12, modes2=12, width=64)(last_frame)
    
    # ═══ RAMA ConvRNN (Local Temporal) ═══
    conv_rnn_features = build_conv_rnn_enhanced_core(inp)
    
    # ═══ HIERARCHICAL FUSION ═══
    # Nivel 1: Fusión Wavelet + ConvRNN (temporal)
    temporal_fusion = HierarchicalFusion(
        name='temporal_fusion'
    )([wavelet_features, conv_rnn_features])
    
    # Nivel 2: Fusión Temporal + FNO (spatio-temporal)
    final_fusion = HierarchicalFusion(
        name='spatiotemporal_fusion'
    )([temporal_fusion, fno_features])
    
    # ═══ OUTPUT ═══
    out = _spatial_head_multi_horizon(final_fusion)
    
    return Model(inp, out, name='Wavelet_FNO_ConvRNN_Hybrid')

class ScaleAttentionLayer(tf.keras.layers.Layer):
    """
    Attention mechanism para ponderar escalas wavelet
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        num_scales = len(input_shape)
        
        # Attention weights para cada escala
        self.scale_attention = tf.keras.layers.Dense(
            num_scales, activation='softmax'
        )
        
    def call(self, scale_features):
        # Calcular representación global por escala
        scale_representations = []
        for features in scale_features:
            global_repr = tf.keras.layers.GlobalAveragePooling3D()(features)
            scale_representations.append(global_repr)
            
        # Stack y calcular attention
        stacked = tf.stack(scale_representations, axis=1)
        attention_weights = self.scale_attention(
            tf.reduce_mean(stacked, axis=1)
        )
        
        # Weighted combination
        weighted_features = []
        for i, features in enumerate(scale_features):
            weight = tf.expand_dims(
                tf.expand_dims(
                    tf.expand_dims(attention_weights[:, i], -1), -1
                ), -1
            )
            weighted_features.append(features * weight)
            
        return tf.reduce_sum(tf.stack(weighted_features), axis=0)

class HierarchicalFusion(tf.keras.layers.Layer):
    """
    Fusión jerárquica con gating mechanism
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        # Gating network
        self.gate = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])
        
        # Feature alignment
        self.align_layers = [
            tf.keras.layers.Dense(128) for _ in range(len(input_shape))
        ]
        
    def call(self, inputs):
        # Alinear dimensiones
        aligned = [
            align_layer(inp) for align_layer, inp in 
            zip(self.align_layers, inputs)
        ]
        
        # Concatenar para gating
        concat = tf.concat(aligned, axis=-1)
        global_context = tf.keras.layers.GlobalAveragePooling2D()(concat)
        
        # Calcular gates
        gates = self.gate(global_context)
        
        # Aplicar gates
        gated = [
            aligned[i] * tf.expand_dims(tf.expand_dims(gates[:, i], -1), -1)
            for i in range(len(aligned))
        ]
        
        return tf.reduce_sum(tf.stack(gated), axis=0)
```

#### **Paso 2.3: Optimización y Evaluación (Semana 6)**

```python
# Configuración V4 completa
MODELS_V4_WAVELET = {
    **MODELS_V3_FNO,  # Modelos V3 existentes
    'Wavelet_FNO_ConvRNN': build_wavelet_fno_hybrid,
    'Wavelet_ConvRNN': build_wavelet_conv_rnn,
    'Wavelet_FNO_Pure': build_wavelet_fno_pure,
}

# Loss function específica para Wavelet
class MultiScaleLoss(tf.keras.losses.Loss):
    """
    Loss que considera errores en múltiples escalas temporales
    """
    
    def __init__(self, scale_weights=[0.4, 0.3, 0.2, 0.1], **kwargs):
        super().__init__(**kwargs)
        self.scale_weights = scale_weights
        
    def call(self, y_true, y_pred):
        # Descomponer predicción y ground truth
        true_scales = self._wavelet_decompose(y_true)
        pred_scales = self._wavelet_decompose(y_pred)
        
        # Loss por escala
        total_loss = 0
        for i, (true_scale, pred_scale, weight) in enumerate(
            zip(true_scales, pred_scales, self.scale_weights)
        ):
            scale_loss = tf.keras.losses.mse(true_scale, pred_scale)
            total_loss += weight * scale_loss
            
        return total_loss

# Target: R² > 0.88 con Wavelet_FNO_ConvRNN + MultiScaleLoss
```

---

## 📊 **CRONOGRAMA Y MÉTRICAS DE ÉXITO**

### **🗓️ TIMELINE DETALLADO:**

```
SEMANA 1: FNO Core Implementation
├── Día 1-2: SpectralConv2D + FNOBlock
├── Día 3-4: FNO2D completo + testing
├── Día 5-7: Integración con TensorFlow + debugging

SEMANA 2: FNO-ConvRNN Hybrid
├── Día 1-2: build_fno_conv_rnn_hybrid
├── Día 3-4: Fusion layers + testing
├── Día 5-7: Training pipeline + validation

SEMANA 3: FNO Optimization
├── Día 1-2: SpectralConsistencyLoss + hyperparameter tuning
├── Día 3-4: Full training run (15 modelos × 3 experimentos)
├── Día 5-7: Results analysis + V3 validation

SEMANA 4: Wavelet Implementation  
├── Día 1-2: WaveletDecomposition + WaveletReconstruction
├── Día 3-4: MultiScaleWaveletProcessor
├── Día 5-7: Integration testing

SEMANA 5: Wavelet-FNO Integration
├── Día 1-2: build_wavelet_fno_hybrid
├── Día 3-4: HierarchicalFusion + ScaleAttention
├── Día 5-7: End-to-end testing

SEMANA 6: V4 Final Optimization
├── Día 1-2: MultiScaleLoss + training
├── Día 3-4: Full V4 evaluation (20+ modelos)
├── Día 5-7: Results analysis + paper preparation
```

### **🎯 MÉTRICAS DE ÉXITO:**

#### **V3 (FNO) Targets:**
```
PRIMARY METRICS:
├── R² > 0.82 (vs 0.75 actual)
├── RMSE < 40 mm (vs 44.85 actual)
├── H2-H3 consistency > 0.65
└── Training time < 2x V2

SECONDARY METRICS:
├── Spectral consistency (new metric)
├── PDE physics compliance
├── Resolution independence
└── Computational efficiency
```

#### **V4 (Wavelet) Targets:**
```
PRIMARY METRICS:
├── R² > 0.88 (vs 0.82 V3)
├── RMSE < 35 mm
├── Multi-scale pattern capture
└── Temporal consistency improved

SECONDARY METRICS:
├── Scale-specific performance
├── Frequency domain accuracy
├── Long-term prediction stability
└── Interpretability enhanced
```

---

## 🔧 **IMPLEMENTACIÓN PRÁCTICA**

### **📁 ESTRUCTURA DE ARCHIVOS:**

```
models/
├── base_models_Conv_STHyMOUNTAIN_V2.ipynb  ← ACTUAL
├── base_models_Conv_STHyMOUNTAIN_V3.ipynb  ← FNO VERSION
├── base_models_Conv_STHyMOUNTAIN_V4.ipynb  ← WAVELET VERSION
├── lib/
│   ├── fno_layers.py          ← FNO implementation
│   ├── wavelet_layers.py      ← Wavelet implementation
│   ├── fusion_layers.py       ← Fusion mechanisms
│   └── loss_functions.py      ← Advanced losses
└── output/
    ├── V2_results/            ← Baseline results
    ├── V3_FNO_results/        ← FNO comparison
    └── V4_Wavelet_results/    ← Final results
```

### **🚀 COMANDOS DE EJECUCIÓN:**

```bash
# Fase 1: Preparar V3 (FNO)
cp base_models_Conv_STHyMOUNTAIN_V2.ipynb base_models_Conv_STHyMOUNTAIN_V3.ipynb
mkdir -p lib output/V3_FNO_results

# Implementar FNO layers
# (seguir código de fno_layers.py arriba)

# Ejecutar training V3
jupyter nbconvert --execute base_models_Conv_STHyMOUNTAIN_V3.ipynb

# Fase 2: Preparar V4 (Wavelet)
cp base_models_Conv_STHyMOUNTAIN_V3.ipynb base_models_Conv_STHyMOUNTAIN_V4.ipynb
mkdir -p output/V4_Wavelet_results

# Implementar Wavelet layers
# (seguir código de wavelet_layers.py arriba)

# Ejecutar training V4
jupyter nbconvert --execute base_models_Conv_STHyMOUNTAIN_V4.ipynb
```

---

## 📈 **ANÁLISIS DE RESULTADOS ESPERADOS**

### **🔍 COMPARACIÓN PROGRESIVA:**

```
MODELO                    R²     RMSE    MAE     INNOVATION
V2: ConvRNN_Enhanced     0.75   44.85   34.38   7/10
V3: FNO_ConvRNN_Hybrid   0.82   39.20   30.15   8.5/10  
V4: Wavelet_FNO_ConvRNN  0.88   33.75   26.80   9/10

IMPROVEMENT V2→V3:       +9%    -13%    -12%    +1.5
IMPROVEMENT V3→V4:       +7%    -14%    -11%    +0.5
TOTAL IMPROVEMENT:       +17%   -25%    -22%    +2.0
```

### **🎯 VENTAJAS ESPECÍFICAS:**

#### **V3 (FNO) Advantages:**
- ✅ **Resolution independence**: Funciona en cualquier grid
- ✅ **PDE compliance**: Respeta física atmosférica
- ✅ **Spatial efficiency**: Mejor modelado de patrones espaciales
- ✅ **Scalability**: Se escala a regiones más grandes

#### **V4 (Wavelet) Advantages:**
- ✅ **Multi-scale temporal**: Captura patrones en múltiples escalas
- ✅ **Frequency analysis**: Descompone señales complejas
- ✅ **Noise robustness**: Filtra ruido en diferentes frecuencias
- ✅ **Interpretability**: Análisis por escalas temporales

---

## 🏆 **CONCLUSIÓN DEL ROADMAP**

### **✅ PLAN EJECUTABLE:**
1. **Semanas 1-3**: V2 → V3 (FNO) - Target R² = 0.82
2. **Semanas 4-6**: V3 → V4 (Wavelet) - Target R² = 0.88
3. **Total time**: 6 semanas para +17% mejora

### **🚀 BENEFICIOS ESPERADOS:**
- **Performance**: R² 0.75 → 0.88 (+17%)
- **Innovation**: 7/10 → 9/10 (+2 niveles)
- **Publication**: Q1 journal ready
- **Impact**: World-class precipitation prediction

### **⚡ PRÓXIMO PASO:**
**¿Comenzamos con la implementación de FNO (Semana 1)?**

El plan está listo para ejecutar. Solo necesitas confirmar para comenzar con `SpectralConv2D` y el core de FNO.
