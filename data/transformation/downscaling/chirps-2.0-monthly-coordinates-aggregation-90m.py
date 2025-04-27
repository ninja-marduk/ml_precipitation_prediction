import xarray as xr
import numpy as np
import logging
import os
import gc
import psutil
import traceback
import scipy.interpolate as interp
from datetime import datetime
import glob
import sys
from pathlib import Path
from rasterio.warp import reproject, Resampling

# Agregar la ruta del directorio padre para poder importar módulos personalizados
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
try:
    from utils.memory_utils import print_memory_usage, limit_memory_usage
except ImportError:
    def print_memory_usage():
        process = psutil.Process()
        logging.info(f"Memoria actual: {process.memory_info().rss / (1024 * 1024):.2f} MB")

    def limit_memory_usage(threshold_mb=4000):
        if psutil.Process().memory_info().rss > threshold_mb * 1024 * 1024:
            gc.collect()

# Configuración de logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../../logs")
os.makedirs(LOG_DIR, exist_ok=True)  # Asegurar que el directorio de logs exista

log_filename = datetime.now().strftime("log-%Y-%m-%d.log")
LOG_FILE = os.path.join(LOG_DIR, log_filename)

log_format = "%(asctime)s - %(levelname)s - [%(pathname)s | function: %(funcName)s | line: %(lineno)d] - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # Mostrar logs en la terminal
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")  # Guardar logs en un archivo
    ]
)

logger = logging.getLogger(__name__)

# Rutas de los datasets
AGGREGATED_DATASET_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_monthly_aggregated.nc"
DEM_90M_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/dem_boyaca_90.nc"
OUTPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_monthly_coordinates_aggregation_downscaled_90m.nc"
TEMP_DIR = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/temp"

# Crear directorio temporal si no existe
os.makedirs(TEMP_DIR, exist_ok=True)

def free_memory(message="Liberando memoria"):
    """Liberar memoria de forma agresiva"""
    mem_before = psutil.Process().memory_info().rss / (1024**2)
    gc.collect()  # Primera colección

    # Intentar reducir caché de Python
    try:
        import sys
        if hasattr(sys, 'getsizeof'):
            sys.stderr.flush()
            sys.stdout.flush()
    except:
        pass

    gc.collect()  # Segunda colección
    mem_after = psutil.Process().memory_info().rss / (1024**2)
    logger.info(f"{message}: {mem_before:.2f}MB -> {mem_after:.2f}MB (liberados {mem_before - mem_after:.2f}MB)")
    return mem_before - mem_after

def load_dataset(file_path):
    """
    Cargar un dataset NetCDF como un Dataset de xarray con manejo optimizado de memoria.

    Parameters:
        file_path (str): Ruta al archivo NetCDF.

    Returns:
        xarray.Dataset: Dataset cargado.
    """
    try:
        logger.info(f"Cargando dataset desde: {file_path}")

        # Verificar si el archivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encuentra el archivo: {file_path}")

        # Intentar importar rioxarray para manejar coordenadas geoespaciales
        try:
            import rioxarray
            logger.info("Módulo rioxarray disponible para procesamiento geoespacial")
        except ImportError:
            logger.warning("rioxarray no está instalado. Las transformaciones geoespaciales pueden ser limitadas.")

        # Optimizar uso de memoria con chunks
        free_memory("Antes de cargar dataset")

        try:
            # Intentar cargar con chunks para conjuntos de datos grandes
            ds = xr.open_dataset(file_path, chunks={'time': 1, 'latitude': 50, 'longitude': 50})
        except Exception as e:
            logger.warning(f"No se pudo cargar con chunks: {e}. Intentando sin chunks...")
            ds = xr.open_dataset(file_path)

        # Imprimir información básica del dataset
        logger.info(f"Dataset cargado. Dimensiones: {ds.dims}, Variables: {list(ds.data_vars)}")

        # Si tiene coordenadas geográficas, intentar añadir CRS
        if hasattr(ds, 'rio') and not ds.rio.crs:
            try:
                ds = ds.rio.write_crs("EPSG:4326")  # Asumimos WGS84
                logger.info("Se añadió información CRS (EPSG:4326) al dataset")
            except:
                logger.warning("No se pudo añadir información CRS al dataset")

        return ds
    except Exception as e:
        logger.error(f"Error al cargar dataset desde {file_path}: {e}")
        logger.error(traceback.format_exc())
        raise

def perform_downscaling(aggregated_ds, dem_ds):
    """
    Realizar el downscaling del dataset agregado para que coincida con la resolución del DEM.

    Esta función utiliza métodos de interpolación avanzados y procesa por chunks para optimizar
    el uso de memoria, aplicando la técnica de downscaling más adecuada para datos de precipitación.

    Parameters:
        aggregated_ds (xarray.Dataset): Dataset agregado para hacer downscaling.
        dem_ds (xarray.Dataset): Dataset DEM que sirve como referencia para la resolución.

    Returns:
        xarray.Dataset: Dataset con resolución ajustada.
    """
    try:
        logger.info("Iniciando proceso de downscaling...")

        # Identificar todas las variables de precipitación en el dataset
        precip_vars = []
        for var_name, var in aggregated_ds.data_vars.items():
            # Buscar variables que parezcan de precipitación
            if any(term in var_name.lower() for term in ["precip", "lluvia", "rain", "chirps"]):
                precip_vars.append(var_name)

        if not precip_vars:  # Si no encontramos variables obvias, usar todas
            precip_vars = list(aggregated_ds.data_vars.keys())
            logger.warning(f"No se identificaron variables específicas de precipitación. Usando todas: {precip_vars}")
        else:
            logger.info(f"Variables de precipitación identificadas: {precip_vars}")

        # Preparar el dataset de resultado
        downscaled_ds = xr.Dataset(
            coords={
                'time': aggregated_ds.time,
                'latitude': dem_ds.latitude,
                'longitude': dem_ds.longitude
            },
            attrs={
                'description': 'Dataset de precipitación con downscaling a resolución DEM (90m)',
                'downscaling_method': 'cubic_spline',
                'source': 'CHIRPS-2.0',
                'resolution': '90m',
                'created': datetime.now().strftime("%Y-%m-%d")
            }
        )

        # Determinar la resolución original y objetivo para información
        if 'latitude' in aggregated_ds.dims and 'longitude' in aggregated_ds.dims and len(aggregated_ds.latitude) > 1 and len(aggregated_ds.longitude) > 1:
            orig_res_lat = abs(float(aggregated_ds.latitude[1] - aggregated_ds.latitude[0]))
            orig_res_lon = abs(float(aggregated_ds.longitude[1] - aggregated_ds.longitude[0]))
            logger.info(f"Resolución original aproximada: {orig_res_lat:.6f}° x {orig_res_lon:.6f}° (≈{orig_res_lat*111320:.2f}m x {orig_res_lon*111320*np.cos(np.mean(aggregated_ds.latitude)*np.pi/180):.2f}m)")

        if 'latitude' in dem_ds.dims and 'longitude' in dem_ds.dims and len(dem_ds.latitude) > 1 and len(dem_ds.longitude) > 1:
            target_res_lat = abs(float(dem_ds.latitude[1] - dem_ds.latitude[0]))
            target_res_lon = abs(float(dem_ds.longitude[1] - dem_ds.longitude[0]))
            logger.info(f"Resolución objetivo DEM: {target_res_lat:.10f}° x {target_res_lon:.10f}° (≈{target_res_lat*111320:.2f}m x {target_res_lon*111320*np.cos(np.mean(dem_ds.latitude)*np.pi/180):.2f}m)")

        # Procesamiento por chunks temporales para optimizar memoria
        time_chunks = 4  # Procesar de 4 en 4 timesteps
        n_times = len(aggregated_ds.time)

        for var_name in precip_vars:
            # Verificar que la variable existe
            if var_name not in aggregated_ds:
                logger.warning(f"La variable {var_name} no existe en el dataset. Omitiendo.")
                continue

            logger.info(f"Realizando downscaling de la variable: {var_name}")
            output_var_name = f"{var_name}_downscaled"

            # Procesar por chunks de tiempo
            all_chunks = []
            for i in range(0, n_times, time_chunks):
                chunk_start = i
                chunk_end = min(i + time_chunks, n_times)
                logger.info(f"Procesando chunk temporal {chunk_start+1}-{chunk_end} de {n_times}")

                # Extraer el sub-dataset para este chunk temporal
                time_slice = slice(chunk_start, chunk_end)
                var_chunk = aggregated_ds[var_name].isel(time=time_slice)

                # Realizar la interpolación usando cubic spline para mejor calidad
                try:
                    # Intentar primero con xarray interp
                    downscaled_chunk = var_chunk.interp(
                        latitude=dem_ds.latitude,
                        longitude=dem_ds.longitude,
                        method='cubic'  # cubic spline interpolation
                    )
                    logger.info(f"Downscaling por interpolación cúbica realizado para chunk {chunk_start+1}-{chunk_end}")
                except Exception as e_interp:
                    logger.warning(f"Error en interpolación cúbica: {e_interp}. Intentando método alternativo...")

                    # Si falla, intentar con rioxarray y reproject
                    try:
                        import rioxarray
                        # Asegurar que tiene CRS
                        if not hasattr(var_chunk, 'rio') or not var_chunk.rio.crs:
                            var_chunk = var_chunk.rio.write_crs("EPSG:4326")

                        # Usar reproject para mejor control
                        downscaled_chunk = var_chunk.rio.reproject(
                            dem_ds.rio.crs,
                            shape=(len(dem_ds.latitude), len(dem_ds.longitude)),
                            resampling=Resampling.cubic_spline
                        )
                        logger.info(f"Downscaling por reproyección con cubic_spline realizado para chunk {chunk_start+1}-{chunk_end}")
                    except Exception as e_rio:
                        logger.warning(f"Error en reproject: {e_rio}. Usando método de respaldo (vecino más cercano)...")

                        # Método de fallback: vecino más cercano simple
                        downscaled_chunk = var_chunk.interp(
                            latitude=dem_ds.latitude,
                            longitude=dem_ds.longitude,
                            method='nearest'
                        )
                        logger.info(f"Downscaling por vecino más cercano (fallback) para chunk {chunk_start+1}-{chunk_end}")

                all_chunks.append(downscaled_chunk)
                free_memory(f"Después de procesar chunk {chunk_start+1}-{chunk_end}")

            # Combinar todos los chunks en un solo DataArray
            try:
                logger.info(f"Combinando {len(all_chunks)} chunks temporales para {var_name}")
                combined_data = xr.concat(all_chunks, dim='time')

                # Añadir al dataset resultado
                downscaled_ds[output_var_name] = combined_data

                # Añadir atributos específicos de la variable
                if hasattr(aggregated_ds[var_name], 'attrs'):
                    downscaled_ds[output_var_name].attrs.update(aggregated_ds[var_name].attrs)
                downscaled_ds[output_var_name].attrs['downscaling_method'] = 'cubic_spline'

                # Limpiar memoria
                del all_chunks, combined_data
                free_memory(f"Después de combinar chunks para {var_name}")

            except Exception as e_combine:
                logger.error(f"Error al combinar chunks para {var_name}: {e_combine}")
                logger.error(traceback.format_exc())

        logger.info("Proceso de downscaling completado exitosamente!")
        return downscaled_ds

    except Exception as e:
        logger.error(f"Error durante el downscaling: {e}")
        logger.error(traceback.format_exc())
        raise

def merge_datasets(downscaled_ds, dem_ds):
    """
    Combinar el dataset con downscaling y el dataset DEM.

    Parameters:
        downscaled_ds (xarray.Dataset): Dataset con downscaling.
        dem_ds (xarray.Dataset): Dataset DEM.

    Returns:
        xarray.Dataset: Dataset combinado.
    """
    try:
        logger.info("Combinando dataset con downscaling y dataset DEM...")

        # Extraer solo la variable de elevación del DEM
        dem_vars = {}
        for var_name in dem_ds.data_vars:
            if any(term in var_name.lower() for term in ["dem", "elev", "altura", "height", "elevation"]):
                dem_vars[var_name] = dem_ds[var_name]

        if not dem_vars:
            logger.warning("No se encontraron variables de elevación específicas en el DEM. Usando todas.")
            dem_vars = {var: dem_ds[var] for var in dem_ds.data_vars}

        # Crear un nuevo dataset combinado
        merged_ds = downscaled_ds.copy(deep=True)

        # Añadir variables DEM
        for var_name, var_data in dem_vars.items():
            logger.info(f"Añadiendo variable DEM: {var_name}")
            merged_ds[var_name] = var_data

        # Actualizar atributos
        merged_ds.attrs.update({
            'description': 'Dataset combinado con precipitación downscaled y datos de elevación DEM',
            'source_downscaled': downscaled_ds.attrs.get('source', 'CHIRPS-2.0'),
            'source_dem': dem_ds.attrs.get('source', 'DEM 90m'),
            'created': datetime.now().strftime("%Y-%m-%d")
        })

        logger.info("Combinación completada exitosamente!")
        return merged_ds
    except Exception as e:
        logger.error(f"Error durante la combinación de datasets: {e}")
        logger.error(traceback.format_exc())
        raise

def save_dataset(ds, output_path):
    """
    Guardar el dataset en un archivo NetCDF con compresión optimizada.

    Parameters:
        ds (xarray.Dataset): Dataset a guardar.
        output_path (str): Ruta para guardar el archivo NetCDF.
    """
    try:
        logger.info(f"Guardando dataset en: {output_path}")

        # Hacer backup si el archivo ya existe
        if os.path.exists(output_path):
            backup_path = output_path + ".bak"
            try:
                os.rename(output_path, backup_path)
                logger.info(f"Se creó respaldo del archivo existente en: {backup_path}")
            except Exception as e_backup:
                logger.warning(f"No se pudo crear respaldo: {e_backup}")

        # Definir codificación para compresión óptima
        encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}

        # Guardar el archivo
        ds.to_netcdf(output_path, encoding=encoding)

        # Calcular y mostrar tamaño del archivo
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Dataset guardado exitosamente en: {output_path}")
        logger.info(f"Tamaño del archivo: {file_size_mb:.2f} MB")

        # Información adicional del dataset guardado
        logger.info(f"Variables en el dataset: {list(ds.data_vars)}")
        logger.info(f"Dimensiones: {ds.dims}")

    except Exception as e:
        logger.error(f"Error al guardar dataset en {output_path}: {e}")
        logger.error(traceback.format_exc())

        # Intentar restaurar desde backup si existe
        if 'backup_path' in locals() and os.path.exists(backup_path):
            try:
                os.rename(backup_path, output_path)
                logger.info(f"Se restauró la versión anterior desde el respaldo")
            except Exception as e_restore:
                logger.error(f"Error al restaurar backup: {e_restore}")

        raise

def main():
    """
    Función principal para realizar el downscaling y la combinación de datasets.

    Esta función controla todo el flujo del proceso:
    1. Carga los datasets
    2. Realiza el downscaling del dataset CHIRPS agregado por mes y coordenadas a la resolución del DEM (90m)
    3. Combina los datos de precipitación con downscaling con los datos de elevación
    4. Guarda el resultado
    """
    start_time = datetime.now()
    logger.info(f"Iniciando proceso de downscaling a las {start_time}")
    print(f"Iniciando proceso de downscaling CHIRPS (0.05°) → DEM (90m)...")

    try:
        # Configuración para monitoreo de memoria
        print_memory_usage()

        # Verificar existencia de archivos
        for filepath in [AGGREGATED_DATASET_PATH, DEM_90M_PATH]:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

        logger.info("Cargando datasets...")
        aggregated_ds = load_dataset(AGGREGATED_DATASET_PATH)
        dem_ds = load_dataset(DEM_90M_PATH)

        logger.info("Verificando proyecciones y sistemas de coordenadas...")
        # Asegurarse de que ambos datasets tienen CRS compatible
        if hasattr(aggregated_ds, 'rio') and hasattr(dem_ds, 'rio'):
            logger.info(f"CRS del dataset CHIRPS: {aggregated_ds.rio.crs}")
            logger.info(f"CRS del dataset DEM: {dem_ds.rio.crs}")
            if aggregated_ds.rio.crs != dem_ds.rio.crs:
                logger.warning(f"Los CRS no coinciden. Podría ser necesario reproyectar.")

        # Realizar downscaling
        logger.info("Ejecutando proceso de downscaling...")
        downscaled_ds = perform_downscaling(aggregated_ds, dem_ds)

        # Liberar memoria del dataset original
        del aggregated_ds
        free_memory("Después de liberar dataset original")

        # Combinar datasets
        logger.info("Combinando datasets...")
        merged_ds = merge_datasets(downscaled_ds, dem_ds)

        # Liberar memoria del dataset con downscaling
        del downscaled_ds
        free_memory("Después de liberar dataset intermedio")

        # Guardar el dataset combinado
        logger.info("Guardando dataset final...")
        save_dataset(merged_ds, OUTPUT_PATH)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0

        logger.info(f"Proceso completado exitosamente en {duration:.2f} minutos!")
        print(f"¡Proceso de downscaling completado exitosamente en {duration:.2f} minutos!")
        print(f"Archivo generado: {OUTPUT_PATH}")

    except Exception as e:
        logger.error(f"Error en el proceso principal: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
