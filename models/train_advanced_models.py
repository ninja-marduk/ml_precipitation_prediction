#!/usr/bin/env python3
"""
Script para entrenar modelos espaciales avanzados de predicción de precipitación
Basado en el análisis de resultados previos con mejoras significativas
"""

import os
import sys
import gc
import warnings
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ============================= CONFIGURACIÓN =============================

# Paths
BASE_PATH = Path.cwd()
for p in [BASE_PATH, *BASE_PATH.parents]:
    if (p / '.git').exists():
        BASE_PATH = p
        break

DATA_FILE = BASE_PATH / 'data' / 'output' / 'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
OUT_ROOT = BASE_PATH / 'models' / 'output' / 'Advanced_Spatial'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Hiperparámetros optimizados
CONFIG = {
    'INPUT_WINDOW': 60,
    'HORIZON': 3,
    'EPOCHS': 100,
    'BATCH_SIZE': 16,  # Aumentado de 4 a 16
    'LEARNING_RATE': 5e-4,  # Reducido de 1e-3
    'PATIENCE': 10,
    'DROPOUT': 0.2,
    'L2_REG': 1e-5,
    'WARMUP_EPOCHS': 5
}

# Feature sets
EXPERIMENTS = {
    'BASIC': ['year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
              'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
              'elevation', 'slope', 'aspect'],
    'KCE': ['year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
            'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
            'elevation', 'slope', 'aspect', 'elev_high', 'elev_med', 'elev_low'],
    'PAFC': ['year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
             'max_daily_precipitation', 'min_daily_precipitation', 'daily_precipitation_std',
             'elevation', 'slope', 'aspect', 'elev_high', 'elev_med', 'elev_low',
             'total_precipitation_lag1', 'total_precipitation_lag2', 'total_precipitation_lag12']
}

# ============================= CAPAS PERSONALIZADAS =============================

class SpatialAttention(layers.Layer):
    """Atención espacial para resaltar regiones importantes"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')
        super().build(input_shape)
        
    def call(self, inputs):
        avg_pool = K.mean(inputs, axis=-1, keepdims=True)
        max_pool = K.max(inputs, axis=-1, keepdims=True)
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv(concat)
        return layers.Multiply()([inputs, attention])


class ChannelAttention(layers.Layer):
    """Atención de canal para ponderar features importantes"""
    
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()
        super().build(input_shape)
        
    def call(self, inputs):
        avg_pool = self.gap(inputs)
        max_pool = self.gmp(inputs)
        
        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))
        
        attention = avg_out + max_out
        attention = K.expand_dims(K.expand_dims(attention, 1), 1)
        
        return layers.Multiply()([inputs, attention])


class CBAM(layers.Layer):
    """Convolutional Block Attention Module"""
    
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.channel_attention = ChannelAttention(reduction_ratio)
        self.spatial_attention = SpatialAttention()
        
    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x


# ============================= ARQUITECTURAS MEJORADAS =============================

def build_convlstm_attention(input_shape, n_features, lat, lon):
    """ConvLSTM mejorado con mecanismo de atención"""
    
    inp = layers.Input(shape=input_shape)
    
    # Primera capa con más filtros y regularización
    x = layers.ConvLSTM2D(
        64, (3, 3), 
        padding='same', 
        return_sequences=True,
        kernel_regularizer=regularizers.l1_l2(l1=0, l2=CONFIG['L2_REG'])
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['DROPOUT'])(x)
    
    # Segunda capa
    x = layers.ConvLSTM2D(
        32, (3, 3), 
        padding='same', 
        return_sequences=True,
        kernel_regularizer=regularizers.l1_l2(l1=0, l2=CONFIG['L2_REG'])
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Aplicar atención temporal
    x = layers.TimeDistributed(CBAM())(x)
    
    # Capa final
    x = layers.ConvLSTM2D(
        16, (3, 3), 
        padding='same', 
        return_sequences=False,
        kernel_regularizer=regularizers.l1_l2(l1=0, l2=CONFIG['L2_REG'])
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Cabeza de salida multi-escala
    out = advanced_spatial_head(x, lat, lon)
    
    return models.Model(inp, out, name='ConvLSTM_Attention')


def build_convgru_residual(input_shape, n_features, lat, lon):
    """ConvGRU con conexiones residuales y BatchNorm"""
    
    inp = layers.Input(shape=input_shape)
    
    # Proyección inicial
    x = layers.TimeDistributed(
        layers.Conv2D(32, (1, 1), padding='same')
    )(inp)
    
    # Bloque ConvGRU 1
    gru1 = ConvGRU2D(64, (3, 3), return_sequences=True)(x)
    gru1 = layers.BatchNormalization()(gru1)
    gru1 = layers.Dropout(CONFIG['DROPOUT'])(gru1)
    
    # Bloque ConvGRU 2 con skip connection
    gru2 = ConvGRU2D(32, (3, 3), return_sequences=False)(gru1)
    gru2 = layers.BatchNormalization()(gru2)
    
    # Skip connection desde input
    skip = layers.TimeDistributed(
        layers.Conv2D(32, (1, 1), padding='same')
    )(inp)
    skip = layers.Lambda(lambda x: x[:, -1])(skip)  # Tomar último timestep
    
    # Combinar con residual
    x = layers.Add()([gru2, skip])
    x = layers.Activation('relu')(x)
    
    # Cabeza de salida
    out = advanced_spatial_head(x, lat, lon)
    
    return models.Model(inp, out, name='ConvGRU_Residual')


def build_hybrid_transformer(input_shape, n_features, lat, lon):
    """Modelo híbrido CNN + Transformer para capturar patrones complejos"""
    
    inp = layers.Input(shape=input_shape)
    
    # Encoder convolucional
    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), padding='same', activation='relu')
    )(inp)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), padding='same', activation='relu')
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    
    # Reducir dimensionalidad espacial
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    # Self-attention temporal
    x = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=32,
        dropout=CONFIG['DROPOUT']
    )(x, x)
    x = layers.LayerNormalization()(x)
    
    # Agregación temporal con LSTM
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['DROPOUT'])(x)
    
    # Decodificador espacial
    x = layers.Dense(lat * lon * 16)(x)
    x = layers.Reshape((lat, lon, 16))(x)
    
    # Cabeza de salida
    out = advanced_spatial_head(x, lat, lon)
    
    return models.Model(inp, out, name='Hybrid_Transformer')


def advanced_spatial_head(x, lat, lon):
    """Cabeza de proyección mejorada con procesamiento multi-escala"""
    
    # Aplicar CBAM
    x = CBAM()(x)
    
    # Procesamiento multi-escala
    conv1 = layers.Conv2D(CONFIG['HORIZON'], (1, 1), padding='same')(x)
    conv3 = layers.Conv2D(CONFIG['HORIZON'], (3, 3), padding='same')(x)
    conv5 = layers.Conv2D(CONFIG['HORIZON'], (5, 5), padding='same')(x)
    
    # Combinar escalas
    x = layers.Add()([conv1, conv3, conv5])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('linear')(x)
    
    # Reshape a formato de salida
    x = layers.Lambda(
        lambda t: tf.transpose(t, [0, 3, 1, 2]),
        output_shape=(CONFIG['HORIZON'], lat, lon)
    )(x)
    x = layers.Lambda(
        lambda t: tf.expand_dims(t, -1),
        output_shape=(CONFIG['HORIZON'], lat, lon, 1)
    )(x)
    
    return x


# ============================= UTILIDADES =============================

def cosine_decay_with_warmup(epoch):
    """Learning rate schedule con warmup y cosine decay"""
    lr_base = CONFIG['LEARNING_RATE']
    total_epochs = CONFIG['EPOCHS']
    warmup_epochs = CONFIG['WARMUP_EPOCHS']
    
    if epoch < warmup_epochs:
        return lr_base * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr_base * 0.5 * (1 + np.cos(np.pi * progress))


def create_callbacks(model_name, exp_name, model_path):
    """Crear callbacks optimizados para el entrenamiento"""
    
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['PATIENCE'],
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-4
        ),
        callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=CONFIG['PATIENCE'] // 2,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.LearningRateScheduler(cosine_decay_with_warmup, verbose=0),
        callbacks.CSVLogger(
            model_path.parent / f"{model_name}_training.csv",
            separator=',',
            append=False
        )
    ]


def windowed_arrays(X, y, input_window, horizon):
    """Crear ventanas deslizantes para series temporales"""
    seq_X, seq_y = [], []
    T = len(X)
    
    for start in range(T - input_window - horizon + 1):
        end_w = start + input_window
        end_y = end_w + horizon
        Xw, yw = X[start:end_w], y[end_w:end_y]
        
        if not (np.isnan(Xw).any() or np.isnan(yw).any()):
            seq_X.append(Xw)
            seq_y.append(yw)
    
    return np.asarray(seq_X, dtype=np.float32), np.asarray(seq_y, dtype=np.float32)


# ============================= MAIN =============================

def main():
    """Función principal para entrenar modelos avanzados"""
    
    print("="*70)
    print("🚀 ENTRENAMIENTO DE MODELOS ESPACIALES AVANZADOS")
    print("="*70)
    
    # Configurar GPU
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Cargar datos
    print("\n📂 Cargando dataset...")
    ds = xr.open_dataset(DATA_FILE)
    lat, lon = len(ds.latitude), len(ds.longitude)
    print(f"   Dimensiones: time={len(ds.time)}, lat={lat}, lon={lon}")
    
    # Definir modelos
    MODELS = {
        'ConvLSTM_Att': build_convlstm_attention,
        'ConvGRU_Res': build_convgru_residual,
        'Hybrid_Trans': build_hybrid_transformer
    }
    
    # Resultados
    all_results = []
    
    # Entrenar para cada experimento
    for exp_name, features in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"🔬 EXPERIMENTO: {exp_name} ({len(features)} features)")
        print(f"{'='*60}")
        
        # Preparar datos
        Xarr = ds[features].to_array().transpose('time', 'latitude', 'longitude', 'variable').values
        yarr = ds['total_precipitation'].values[..., None]
        
        X, y = windowed_arrays(Xarr, yarr, CONFIG['INPUT_WINDOW'], CONFIG['HORIZON'])
        
        # Split 80/10/10
        n_train = int(0.8 * len(X))
        n_val = int(0.9 * len(X))
        
        # Normalización
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_flat = X.reshape(-1, len(features))
        y_flat = y.reshape(-1, 1)
        
        scaler_X.fit(X_flat[:n_train])
        scaler_y.fit(y_flat[:n_train])
        
        X_scaled = scaler_X.transform(X_flat).reshape(X.shape)
        y_scaled = scaler_y.transform(y_flat).reshape(y.shape)
        
        # Splits
        X_train = X_scaled[:n_train]
        y_train = y_scaled[:n_train]
        X_val = X_scaled[n_train:n_val]
        y_val = y_scaled[n_train:n_val]
        X_test = X_scaled[n_val:]
        y_test = y_scaled[n_val:]
        
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Crear directorio para experimento
        exp_dir = OUT_ROOT / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Entrenar cada modelo
        for model_name, model_builder in MODELS.items():
            print(f"\n{'─'*50}")
            print(f"Modelo: {model_name}")
            print(f"{'─'*50}")
            
            try:
                # Construir modelo
                if 'ConvGRU' in model_name:
                    # Importar ConvGRU2D del notebook original
                    from tensorflow.keras.layers import ConvLSTM2D as ConvGRU2D
                
                model = model_builder(
                    input_shape=(CONFIG['INPUT_WINDOW'], lat, lon, len(features)),
                    n_features=len(features),
                    lat=lat,
                    lon=lon
                )
                
                # Compilar
                model.compile(
                    optimizer=optimizers.AdamW(
                        learning_rate=CONFIG['LEARNING_RATE'],
                        weight_decay=CONFIG['L2_REG']
                    ),
                    loss='mse',
                    metrics=['mae']
                )
                
                print(f"   Parámetros: {model.count_params():,}")
                
                # Callbacks
                model_path = exp_dir / f"{model_name}_best.keras"
                callbacks_list = create_callbacks(model_name, exp_name, model_path)
                
                # Entrenar
                print(f"\n   🏃 Entrenando...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=CONFIG['EPOCHS'],
                    batch_size=CONFIG['BATCH_SIZE'],
                    callbacks=callbacks_list,
                    verbose=1
                )
                
                # Evaluar en test
                print(f"\n   📊 Evaluando...")
                test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
                
                # Predicciones para métricas
                y_pred_scaled = model.predict(X_test[:10], verbose=0)
                y_pred = scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).reshape(-1, CONFIG['HORIZON'], lat, lon)
                y_true = scaler_y.inverse_transform(
                    y_test[:10].reshape(-1, 1)
                ).reshape(-1, CONFIG['HORIZON'], lat, lon)
                
                # Calcular métricas por horizonte
                for h in range(CONFIG['HORIZON']):
                    rmse = np.sqrt(mean_squared_error(
                        y_true[:, h].ravel(), 
                        y_pred[:, h].ravel()
                    ))
                    mae = mean_absolute_error(
                        y_true[:, h].ravel(), 
                        y_pred[:, h].ravel()
                    )
                    r2 = r2_score(
                        y_true[:, h].ravel(), 
                        y_pred[:, h].ravel()
                    )
                    
                    result = {
                        'Experiment': exp_name,
                        'Model': model_name,
                        'Horizon': h + 1,
                        'RMSE': rmse,
                        'MAE': mae,
                        'R2': r2,
                        'Test_Loss': test_loss,
                        'Parameters': model.count_params(),
                        'Best_Epoch': len(history.history['loss'])
                    }
                    
                    all_results.append(result)
                    print(f"      H={h+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
                
                # Guardar historia
                history_path = exp_dir / f"{model_name}_history.json"
                with open(history_path, 'w') as f:
                    json.dump(history.history, f, indent=2)
                
                # Limpiar memoria
                K.clear_session()
                gc.collect()
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                continue
    
    # Guardar resultados
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_ROOT / 'advanced_results.csv', index=False)
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"📊 Resultados guardados en: {OUT_ROOT / 'advanced_results.csv'}")
    print("="*70)
    
    # Mostrar resumen
    print("\n📈 RESUMEN DE MEJORES MODELOS:")
    print("─"*50)
    
    for exp in EXPERIMENTS.keys():
        exp_data = results_df[results_df['Experiment'] == exp]
        if not exp_data.empty:
            best_idx = exp_data.groupby('Model')['RMSE'].mean().idxmin()
            best_data = exp_data[exp_data['Model'] == best_idx]
            avg_rmse = best_data['RMSE'].mean()
            avg_r2 = best_data['R2'].mean()
            
            print(f"\n{exp}:")
            print(f"  • Mejor modelo: {best_idx}")
            print(f"  • RMSE promedio: {avg_rmse:.4f}")
            print(f"  • R² promedio: {avg_r2:.4f}")


if __name__ == "__main__":
    main() 