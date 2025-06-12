# ======================================================================
# PIPELINE PRINCIPAL DE ENTRENAMIENTO
# ======================================================================

import pandas as pd
from datetime import datetime

# ▶️ ENTRENAMIENTO DE MÚLTIPLES EXPERIMENTOS
def train_all_active_experiments(dataset):
    """
    Entrena todos los experimentos activos en todos los folds activos
    
    Args:
        dataset: Dataset xarray con todos los datos
        
    Returns:
        Dictionary con resultados de todos los entrenamientos
    """
    
    info_print("🚀 INICIANDO ENTRENAMIENTO DE TODOS LOS EXPERIMENTOS ACTIVOS")
    info_print("="*80)
    
    # Obtener experimentos y folds activos
    active_experiments = [name for name, config in EXPERIMENTS.items() if config.get('active', False)]
    active_folds = [name for name, config in FOLDS.items() if config.get('active', False)]
    
    info_print(f"📋 Experimentos activos: {active_experiments}")
    info_print(f"📋 Folds activos: {active_folds}")
    
    all_results = {}
    total_combinations = len(active_experiments) * len(active_folds)
    current_combination = 0
    
    for exp_name in active_experiments:
        all_results[exp_name] = {}
        
        for fold_name in active_folds:
            current_combination += 1
            
            info_print(f"\n🔄 Progreso: {current_combination}/{total_combinations}")
            info_print(f"   Entrenando: {exp_name} - {fold_name}")
            
            # Entrenar experimento
            result = train_experiment(exp_name, fold_name, dataset, save_model=True)
            
            if result:
                all_results[exp_name][fold_name] = result
                success_print(f"✅ Completado: {exp_name} - {fold_name}")
            else:
                error_print(f"❌ Falló: {exp_name} - {fold_name}")
                all_results[exp_name][fold_name] = None
    
    info_print("🎉 ENTRENAMIENTO DE TODOS LOS EXPERIMENTOS COMPLETADO")
    info_print("="*80)
    
    return all_results

# ▶️ ANÁLISIS COMPARATIVO DE EXPERIMENTOS
def analyze_experiment_results(all_results):
    """
    Analiza y compara resultados de todos los experimentos
    
    Args:
        all_results: Resultados de train_all_active_experiments
        
    Returns:
        DataFrame con análisis comparativo
    """
    
    info_print("📊 ANALIZANDO RESULTADOS DE EXPERIMENTOS...")
    
    analysis_data = []
    
    for exp_name, exp_results in all_results.items():
        for fold_name, result in exp_results.items():
            if result:
                row = {
                    'Experimento': exp_name,
                    'Fold': fold_name,
                    'Parámetros': result['model_params'],
                    'Tiempo (s)': f"{result['training_time']:.1f}",
                    'Épocas': result['train_metrics'].get('epochs_trained', 'N/A'),
                    'Val Loss': f"{result['eval_metrics']['val_loss']:.4f}",
                    'Val MAE': f"{result['eval_metrics']['val_mae']:.4f}",
                    'Val MAPE': f"{result['eval_metrics']['val_mape']:.2f}%",
                    'RMSE': f"{result['prediction_results']['overall_metrics']['rmse']:.4f}",
                    'Correlación': f"{result['prediction_results']['overall_metrics']['correlation']:.4f}",
                    'R²': f"{result['prediction_results']['overall_metrics']['r2']:.4f}"
                }
                analysis_data.append(row)
    
    if analysis_data:
        df = pd.DataFrame(analysis_data)
        
        info_print("📈 TABLA COMPARATIVA DE RESULTADOS:")
        display(df)
        
        return df
    else:
        warning_print("⚠️ No hay resultados para analizar")
        return None

# ▶️ ANÁLISIS DETALLADO POR EXPERIMENTO
def detailed_experiment_analysis(all_results):
    """
    Genera análisis detallado de métricas mensuales para cada experimento
    
    Args:
        all_results: Resultados de entrenamientos
    """
    
    info_print("📊 ANÁLISIS DETALLADO DE MÉTRICAS MENSUALES")
    info_print("="*60)
    
    for exp_name, exp_results in all_results.items():
        info_print(f"\n🔬 EXPERIMENTO: {exp_name}")
        info_print("-" * 40)
        
        # Recopilar métricas mensuales de todos los folds
        all_monthly_data = []
        
        for fold_name, result in exp_results.items():
            if result and 'prediction_results' in result:
                monthly_metrics = result['prediction_results']['monthly_metrics']
                
                for month_data in monthly_metrics:
                    month_data_copy = month_data.copy()
                    month_data_copy['fold'] = fold_name
                    month_data_copy['experiment'] = exp_name
                    all_monthly_data.append(month_data_copy)
        
        if all_monthly_data:
            monthly_df = pd.DataFrame(all_monthly_data)
            
            # Estadísticas por mes
            info_print(f"📅 Métricas por mes (promedio de todos los folds):")
            
            monthly_summary = monthly_df.groupby('month').agg({
                'real_mm': 'mean',
                'pred_mm': 'mean', 
                'error_mm': 'mean',
                'mape_percent': 'mean'
            }).round(2)
            
            display(monthly_summary)
            
            # Visualizar métricas mensuales
            plot_monthly_metrics(result['prediction_results'], exp_name)
        else:
            warning_print(f"⚠️ No hay datos mensuales para {exp_name}")

# ▶️ COMPARACIÓN ESPACIAL DE EXPERIMENTOS
def spatial_comparison_analysis(all_results, dataset):
    """
    Realiza análisis espacial comparativo de los experimentos
    
    Args:
        all_results: Resultados de entrenamientos
        dataset: Dataset original
    """
    
    info_print("🗺️ ANÁLISIS ESPACIAL COMPARATIVO")
    info_print("="*60)
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Obtener coordenadas
    lat_coord = 'latitude' if 'latitude' in dataset.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in dataset.coords else 'lon'
    latitudes = dataset[lat_coord].values
    longitudes = dataset[lon_coord].values
    
    # Comparar mapas de error MAPE entre experimentos
    active_experiments = [name for name, results in all_results.items() 
                         if any(r for r in results.values() if r)]
    
    if len(active_experiments) < 2:
        warning_print("⚠️ Se necesitan al menos 2 experimentos para comparación espacial")
        return
    
    # Crear mapa comparativo de errores
    fig, axes = plt.subplots(1, len(active_experiments), figsize=(6*len(active_experiments), 6))
    if len(active_experiments) == 1:
        axes = [axes]
    
    error_cmap = LinearSegmentedColormap.from_list('error_cmap', ['green', 'yellow', 'red'])
    
    for i, exp_name in enumerate(active_experiments):
        exp_results = all_results[exp_name]
        
        # Usar primer fold disponible
        for fold_name, result in exp_results.items():
            if result and 'prediction_results' in result:
                prediction_results = result['prediction_results']
                
                if 'prediction_spatial' in prediction_results:
                    real_spatial = prediction_results['real_spatial']
                    pred_spatial = prediction_results['prediction_spatial']
                    
                    # Calcular error MAPE espacial promedio
                    error_spatial = np.abs(pred_spatial - real_spatial) / real_spatial * 100
                    error_mean = np.mean(error_spatial, axis=0)  # Promedio temporal
                    
                    # Limitar errores extremos
                    error_mean = np.clip(error_mean, 0, 100)
                    
                    # Plotear
                    im = axes[i].pcolormesh(longitudes, latitudes, error_mean, 
                                          cmap=error_cmap, vmin=0, vmax=50, shading='auto')
                    axes[i].set_title(f'{exp_name}\nMAPE Espacial (%)', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel('Longitud')
                    if i == 0:
                        axes[i].set_ylabel('Latitud')
                    
                    # Colorbar
                    cbar = plt.colorbar(im, ax=axes[i])
                    cbar.set_label('Error MAPE (%)')
                    
                    break
    
    plt.tight_layout()
    plt.show()

# ▶️ FUNCIÓN PRINCIPAL DE EJECUCIÓN
def run_complete_training_pipeline(dataset=None):
    """
    Ejecuta el pipeline completo de entrenamiento y análisis
    
    Args:
        dataset: Dataset xarray (si None, usa ds_full global)
        
    Returns:
        Dictionary con todos los resultados
    """
    
    if dataset is None:
        dataset = ds_full
    
    info_print("🎯 EJECUTANDO PIPELINE COMPLETO DE ENTRENAMIENTO")
    info_print("="*80)
    
    # 1. Mostrar configuración actual
    show_current_configuration()
    
    # 2. Entrenar todos los experimentos
    all_results = train_all_active_experiments(dataset)
    
    # 3. Analizar resultados comparativos
    analysis_df = analyze_experiment_results(all_results)
    
    # 4. Análisis detallado por experimento
    detailed_experiment_analysis(all_results)
    
    # 5. Análisis espacial
    spatial_comparison_analysis(all_results, dataset)
    
    # 6. Mostrar resumen final
    info_print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE")
    info_print("📊 Resumen:")
    
    total_experiments = len([r for exp in all_results.values() for r in exp.values() if r])
    failed_experiments = len([r for exp in all_results.values() for r in exp.values() if not r])
    
    info_print(f"   • Experimentos exitosos: {total_experiments}")
    info_print(f"   • Experimentos fallidos: {failed_experiments}")
    info_print(f"   • Modelos guardados: {total_experiments}")
    
    # 7. Generar reporte final
    final_report = generate_final_report(all_results, analysis_df)
    
    return {
        'all_results': all_results,
        'analysis_df': analysis_df,
        'final_report': final_report,
        'summary': {
            'total_successful': total_experiments,
            'total_failed': failed_experiments,
            'completion_time': datetime.now().isoformat()
        }
    }

# ▶️ GENERACIÓN DE REPORTE FINAL
def generate_final_report(all_results, analysis_df):
    """
    Genera un reporte final completo con todas las métricas
    
    Args:
        all_results: Resultados de entrenamientos
        analysis_df: DataFrame con análisis comparativo
        
    Returns:
        Dictionary con reporte final
    """
    
    info_print("📄 GENERANDO REPORTE FINAL COMPLETO")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'input_window': INPUT_WINDOW,
            'output_horizon': OUTPUT_HORIZON,
            'early_patience': EARLY_PATIENCE
        },
        'experiments_summary': {},
        'best_performances': {},
        'monthly_predictions': {}
    }
    
    # Resumen por experimento
    for exp_name, exp_results in all_results.items():
        if any(r for r in exp_results.values() if r):
            successful_folds = [fold for fold, result in exp_results.items() if result]
            
            # Métricas promedio
            avg_metrics = {}
            for metric in ['val_loss', 'val_mae', 'val_mape']:
                values = [exp_results[fold]['eval_metrics'][metric] 
                         for fold in successful_folds]
                avg_metrics[metric] = np.mean(values)
            
            report['experiments_summary'][exp_name] = {
                'successful_folds': successful_folds,
                'total_parameters': exp_results[successful_folds[0]]['model_params'],
                'average_metrics': avg_metrics
            }
    
    # Mejores rendimientos
    if analysis_df is not None and not analysis_df.empty:
        # Convertir métricas a numéricas para comparación
        df_numeric = analysis_df.copy()
        df_numeric['Val Loss'] = pd.to_numeric(df_numeric['Val Loss'])
        df_numeric['RMSE'] = pd.to_numeric(df_numeric['RMSE'])
        df_numeric['R²'] = pd.to_numeric(df_numeric['R²'])
        
        report['best_performances'] = {
            'lowest_val_loss': {
                'experiment': df_numeric.loc[df_numeric['Val Loss'].idxmin(), 'Experimento'],
                'fold': df_numeric.loc[df_numeric['Val Loss'].idxmin(), 'Fold'],
                'value': df_numeric['Val Loss'].min()
            },
            'lowest_rmse': {
                'experiment': df_numeric.loc[df_numeric['RMSE'].idxmin(), 'Experimento'],
                'fold': df_numeric.loc[df_numeric['RMSE'].idxmin(), 'Fold'],
                'value': df_numeric['RMSE'].min()
            },
            'highest_r2': {
                'experiment': df_numeric.loc[df_numeric['R²'].idxmax(), 'Experimento'],
                'fold': df_numeric.loc[df_numeric['R²'].idxmax(), 'Fold'],
                'value': df_numeric['R²'].max()
            }
        }
    
    # Predicciones mensuales consolidadas
    for exp_name, exp_results in all_results.items():
        monthly_data = []
        for fold_name, result in exp_results.items():
            if result and 'prediction_results' in result:
                monthly_metrics = result['prediction_results']['monthly_metrics']
                for month_data in monthly_metrics:
                    month_data_copy = month_data.copy()
                    month_data_copy['fold'] = fold_name
                    monthly_data.append(month_data_copy)
        
        if monthly_data:
            report['monthly_predictions'][exp_name] = monthly_data
    
    info_print(f"✅ Reporte final generado con {len(report['experiments_summary'])} experimentos")
    
    return report

# ▶️ FUNCIÓN DE DEMO RÁPIDO
def quick_demo():
    """
    Ejecuta una demostración rápida del sistema para validar funcionalidad
    """
    
    info_print("🚀 EJECUTANDO DEMO RÁPIDO DEL SISTEMA")
    info_print("="*60)
    
    # Activar solo un experimento y un fold para demo
    original_experiments = {name: config.copy() for name, config in EXPERIMENTS.items()}
    original_folds = {name: config.copy() for name, config in FOLDS.items()}
    
    # Desactivar todos excepto el primero
    for name in EXPERIMENTS:
        EXPERIMENTS[name]['active'] = False
    EXPERIMENTS['ConvGRU-ED']['active'] = True
    
    for name in FOLDS:
        FOLDS[name]['active'] = False
    FOLDS['F1']['active'] = True
    
    try:
        # Ejecutar pipeline con configuración de demo
        results = run_complete_training_pipeline()
        success_print("✅ Demo completado exitosamente")
        return results
        
    finally:
        # Restaurar configuración original
        for name, config in original_experiments.items():
            EXPERIMENTS[name] = config
        for name, config in original_folds.items():
            FOLDS[name] = config
        
        info_print("🔄 Configuración original restaurada")

print("✅ Pipeline principal de entrenamiento cargado - Archivo 3/3")
print("\n" + "="*60)
print("🎯 SISTEMA COMPLETO DE ENTRENAMIENTO Y MÉTRICAS LISTO")
print("="*60)
print("📋 Funciones disponibles:")
print("   • run_complete_training_pipeline() - Pipeline completo")
print("   • quick_demo() - Demo rápido")
print("   • train_experiment(exp_name, fold_name, dataset) - Entrenar un experimento")
print("   • analyze_experiment_results(results) - Analizar resultados")
print("="*60) 