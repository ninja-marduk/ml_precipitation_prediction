import xarray as xr
import numpy as np
from rasterio.warp import reproject, Resampling
import os
import logging
from datetime import datetime
import pandas as pd
import sys
import gc
import psutil
import traceback
import signal
import json
import glob
from pathlib import Path

# Agregar la ruta del directorio padre al path para poder importar los módulos personalizados
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from utils.memory_utils import print_memory_usage, limit_memory_usage, load_data_with_chunks, process_by_chunks

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../../logs/")
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists

# Generate log file name with the required format
log_filename = datetime.now().strftime("log-%Y-%m-%d.log")
LOG_FILE = os.path.join(LOG_DIR, log_filename)

# Custom log format
log_format = "%(asctime)s - %(levelname)s - [%(pathname)s-%(funcName)s-%(lineno)d] - %(message)s"

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format=log_format,  # Log message format
    handlers=[
        logging.StreamHandler(),  # Output logs to the terminal
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")  # Save logs to a daily file
    ]
)

# Configurar un manejador para capturar señales de "kill"
def handle_kill_signal(signum, frame):
    mem_info = psutil.virtual_memory()
    logging.critical(f"Recibida señal de terminación: {signum}")
    logging.critical(f"Información de memoria en el momento del error: {mem_info}")
    logging.critical(f"Uso de memoria del proceso: {psutil.Process().memory_info().rss / (1024**2):.2f} MB")
    logging.critical(f"Stack trace en el momento de la terminación:\n{traceback.format_stack()}")

    # Guardar información detallada sobre el estado de memoria
    try:
        # Obtener información de memoria detallada
        process = psutil.Process()
        memory_info = {
            "rss": process.memory_info().rss / (1024**2),
            "vms": process.memory_info().vms / (1024**2),
            "system_percent": psutil.virtual_memory().percent,
            "system_available": psutil.virtual_memory().available / (1024**2),
            "swap_used": psutil.swap_memory().used / (1024**2),
            "swap_percent": psutil.swap_memory().percent
        }

        # Guardar en un archivo específico para errores OOM
        oom_log = os.path.join(LOG_DIR, "oom_error_details.log")
        with open(oom_log, "a") as f:
            f.write(f"\n--- OOM ERROR DETECTED AT {datetime.now()} ---\n")
            f.write(f"Signal: {signum}\n")
            f.write(f"Memory info: {memory_info}\n")
            f.write(f"Stack trace:\n{traceback.format_stack()}\n")

            # Intentar obtener los procesos que más consumen memoria
            f.write("Top 5 procesos por consumo de memoria:\n")
            for proc in sorted(psutil.process_iter(['pid', 'name', 'memory_percent']),
                              key=lambda x: x.info['memory_percent'] if x.info['memory_percent'] is not None else 0,
                              reverse=True)[:5]:
                f.write(f"  PID: {proc.info['pid']}, Nombre: {proc.info['name']}, Memoria: {proc.info['memory_percent']:.2f}%\n")
    except Exception as e:
        logging.error(f"Error al guardar detalles de OOM: {e}")

    sys.exit(1)

# Registrar el manejador para señales comunes de terminación
signal.signal(signal.SIGTERM, handle_kill_signal)
signal.signal(signal.SIGINT, handle_kill_signal)
if hasattr(signal, 'SIGKILL'):  # SIGKILL no se puede capturar directamente en la mayoría de sistemas
    try:
        signal.signal(signal.SIGKILL, handle_kill_signal)
    except:
        pass

# Intentar registrar SIGABRT que puede ocurrir en caso de OOM en macOS
if hasattr(signal, 'SIGABRT'):
    signal.signal(signal.SIGABRT, handle_kill_signal)

# En macOS también es útil capturar SIGQUIT
if hasattr(signal, 'SIGQUIT'):
    signal.signal(signal.SIGQUIT, handle_kill_signal)

# Constants
CHIRPS_MONTHLY_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_monthly_coordinates_sum.nc"
DEM_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/qgis_output/dem_boyaca_90.nc"
OUTPUT_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/boyaca_region_monthly_coordinates_sum_downscaled_90m.nc"
TEMP_DIR = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/temp"
CHECKPOINT_DIR = os.path.join(TEMP_DIR, "checkpoints")

# Asegurar que existan los directorios necesarios
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Función para liberar memoria de forma agresiva
def free_memory(message="Liberando memoria"):
    """Liberar memoria de forma agresiva"""
    mem_before = psutil.Process().memory_info().rss / (1024**2)
    gc.collect()  # Primera colección

    # Forzar segunda colección más agresiva
    import ctypes
    if hasattr(ctypes, 'windll'):  # Solo en Windows
        try:
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
        except:
            pass

    # Intentar reducir caché de Python
    try:
        import sys
        if hasattr(sys, 'getsizeof'):
            sys.stderr.flush()
            sys.stdout.flush()
    except:
        pass

    gc.collect()  # Tercera colección
    mem_after = psutil.Process().memory_info().rss / (1024**2)
    logging.info(f"{message}: {mem_before:.2f}MB -> {mem_after:.2f}MB (liberados {mem_before - mem_after:.2f}MB)")
    return mem_before - mem_after

# Función para guardar y cargar checkpoints
def save_checkpoint(variable, state):
    """Guarda un punto de control para el procesamiento de una variable"""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{variable}_checkpoint.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(state, f)
    logging.info(f"Checkpoint guardado para {variable}: {state}")

def load_checkpoint(variable):
    """Carga un punto de control si existe"""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{variable}_checkpoint.json")
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                state = json.load(f)
            logging.info(f"Checkpoint cargado para {variable}: {state}")
            return state
        except Exception as e:
            logging.error(f"Error al cargar checkpoint para {variable}: {e}")
    return None

def get_memory_info():
    """
    Obtener información detallada sobre el uso de memoria del sistema y del proceso actual
    """
    process = psutil.Process()
    process_mem = process.memory_info()
    system_mem = psutil.virtual_memory()

    mem_info = {
        "proceso_rss_mb": process_mem.rss / (1024**2),  # MB
        "proceso_vms_mb": process_mem.vms / (1024**2),  # MB
        "sistema_total_mb": system_mem.total / (1024**2),  # MB
        "sistema_disponible_mb": system_mem.available / (1024**2),  # MB
        "sistema_usado_porcentaje": system_mem.percent,
        "sistema_swap_mb": psutil.swap_memory().used / (1024**2)
    }

    return mem_info

def log_memory_info(mensaje="Estado actual de memoria"):
    """
    Registrar información detallada sobre el uso de memoria
    """
    mem_info = get_memory_info()
    logging.info(f"{mensaje}:")
    logging.info(f"  - Proceso: {mem_info['proceso_rss_mb']:.2f} MB (RSS) / {mem_info['proceso_vms_mb']:.2f} MB (VMS)")
    logging.info(f"  - Sistema: {mem_info['sistema_usado_porcentaje']}% usado, {mem_info['sistema_disponible_mb']:.2f} MB disponible de {mem_info['sistema_total_mb']:.2f} MB total")
    logging.info(f"  - Swap: {mem_info['sistema_swap_mb']:.2f} MB")
    return mem_info

def load_nc_file(file_path):
    """
    Load a NetCDF file as an xarray Dataset with chunking.

    Parameters:
        file_path (str): Path to the NetCDF file.

    Returns:
        xarray.Dataset: Loaded dataset.
    """
    logging.info(f"Loading NetCDF file: {file_path}")
    try:
        # Intentar importar rioxarray para asegurarnos que está disponible
        import rioxarray
        logging.info("rioxarray importado correctamente")
    except ImportError:
        logging.error("rioxarray no está instalado. Instalándolo ahora...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "rioxarray"])
            import rioxarray
            logging.info("rioxarray instalado correctamente")
        except Exception as e:
            logging.error(f"Error al instalar rioxarray: {e}")
            raise

    try:
        # Intentar importar dask para el procesamiento en chunks
        import dask
        logging.info("dask importado correctamente")
    except ImportError:
        logging.error("dask no está instalado. Instalándolo ahora...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "dask"])
            import dask
            logging.info("dask instalado correctamente")
        except Exception as e:
            logging.error(f"Error al instalar dask: {e}")

    # Limpiar memoria antes de cargar el archivo
    free_memory("Liberando memoria antes de cargar archivo")

    # Usar load_data_with_chunks para optimizar el uso de memoria
    chunk_sizes = {'time': 1, 'latitude': 50, 'longitude': 50}  # Chunks más pequeños
    try:
        dataset = load_data_with_chunks(file_path, chunk_sizes)
        return dataset
    except Exception as e:
        logging.error(f"Error al cargar el archivo {file_path} con chunks: {e}")
        # Si falla con chunks, intentar abrir normalmente
        try:
            # Intentar con chunks más pequeños primero
            try:
                return xr.open_dataset(file_path, chunks={'time': 1, 'latitude': 20, 'longitude': 20})
            except:
                # Si falla, intentar sin chunks
                return xr.open_dataset(file_path)
        except Exception as e2:
            logging.critical(f"Error crítico al cargar el archivo: {e2}")
            raise


def downscale_variable_to_dem_chunked(variable, chirps_ds, dem_ds, dem_transform, dem_crs, dem_width, dem_height, chunk_size=1):
    """
    Downscale a specific CHIRPS variable to match the resolution of the DEM using a chunked approach.

    Parameters:
        variable (str): Name of the variable to downscale.
        chirps_ds (xarray.Dataset): CHIRPS dataset.
        dem_ds (xarray.Dataset): DEM dataset.
        dem_transform (Affine): Transform of the DEM.
        dem_crs (CRS): CRS of the DEM.
        dem_width (int): Width of the DEM.
        dem_height (int): Height of the DEM.
        chunk_size (int): Number of time steps to process at once.

    Returns:
        xr.DataArray: Downscaled data for the variable.
    """
    logging.info(f"Downscaling variable: {variable}")

    # Verificar si hay un checkpoint guardado
    checkpoint = load_checkpoint(variable)

    # Crear un archivo netCDF temporal para el resultado final
    result_file = os.path.join(TEMP_DIR, f"{variable}_result.nc")

    # Procesar por chunks de tiempo y guardar cada chunk como un archivo separado
    time_steps = chirps_ds.time.size

    # Si hay checkpoint, obtener los chunks ya procesados
    if checkpoint and 'processed_chunks' in checkpoint:
        processed_chunks = checkpoint['processed_chunks']
        logging.info(f"Retomando desde el checkpoint: {len(processed_chunks)} chunks ya procesados")
    else:
        processed_chunks = []

    # Determinar qué patrones de archivo buscar
    chunk_pattern = os.path.join(TEMP_DIR, f"{variable}_chunk_*.nc")
    existing_chunks = glob.glob(chunk_pattern)
    if existing_chunks and not processed_chunks:
        logging.info(f"Se encontraron {len(existing_chunks)} archivos de chunk existentes")
        # Verificar si estos chunks corresponden a los que necesitamos
        for chunk_file in sorted(existing_chunks):
            # Extraer el número de chunk del nombre de archivo
            try:
                chunk_num = int(os.path.basename(chunk_file).split('_')[-1].split('.')[0])
                if chunk_num not in processed_chunks:
                    processed_chunks.append(chunk_num)
            except:
                pass
        logging.info(f"Se recuperaron {len(processed_chunks)} chunks de archivos existentes")

    try:
        # Determinar desde qué chunk empezar
        remaining_chunks = [i for i in range(0, time_steps, chunk_size)
                           if i not in processed_chunks]

        if not remaining_chunks and processed_chunks:
            logging.info(f"Todos los chunks ya están procesados para {variable}, pasando a la combinación")

        # Procesar los chunks restantes
        chunk_files = []

        for chunk_start in remaining_chunks:
            # Monitorear uso de memoria
            log_memory_info(f"Memoria antes de procesar chunk de tiempo {chunk_start+1}-{min(chunk_start+chunk_size, time_steps)}")

            chunk_end = min(chunk_start + chunk_size, time_steps)
            logging.info(f"Procesando chunk de tiempo {chunk_start+1}-{chunk_end} de {time_steps}")

            # Crear array solo para este chunk
            chunk_data = np.empty((chunk_end - chunk_start, dem_height, dem_width), dtype=np.float32)

            for i in range(chunk_start, chunk_end):
                time_idx = i - chunk_start  # Índice relativo dentro del chunk
                logging.debug(f"Downscaling time step {i + 1}/{time_steps}")

                # Extraer el timestep específico y liberar memoria
                timestep_data = chirps_ds[variable].isel(time=i).values

                # Realizar el reproject
                try:
                    reproject(
                        source=timestep_data,
                        destination=chunk_data[time_idx],
                        src_transform=chirps_ds.rio.transform(),
                        src_crs=chirps_ds.rio.crs,
                        dst_transform=dem_transform,
                        dst_crs=dem_crs,
                        resampling=Resampling.bilinear
                    )
                except AttributeError:
                    # Si falla el acceso a rio, intentamos obtener la transformación manualmente
                    logging.warning("No se pudo acceder al atributo 'rio'. Intentando método alternativo...")

                    # Método alternativo para reproject sin usar rio
                    from rasterio.transform import Affine

                    # Crear transformaciones usando los límites y resoluciones del dataset
                    src_lats = chirps_ds.latitude.values
                    src_lons = chirps_ds.longitude.values
                    dst_lats = dem_ds.latitude.values
                    dst_lons = dem_ds.longitude.values

                    src_res_lat = abs(src_lats[1] - src_lats[0]) if len(src_lats) > 1 else 0.05
                    src_res_lon = abs(src_lons[1] - src_lons[0]) if len(src_lons) > 1 else 0.05

                    src_transform = Affine(src_res_lon, 0.0, min(src_lons),
                                          0.0, -src_res_lat, max(src_lats))

                    dst_res_lat = abs(dst_lats[1] - dst_lats[0]) if len(dst_lats) > 1 else 0.001
                    dst_res_lon = abs(dst_lons[1] - dst_lons[0]) if len(dst_lons) > 1 else 0.001

                    dst_transform = Affine(dst_res_lon, 0.0, min(dst_lons),
                                          0.0, -dst_res_lat, max(dst_lats))

                    # Intentar reproject con las transformaciones manuales
                    reproject(
                        source=timestep_data,
                        destination=chunk_data[time_idx],
                        src_transform=src_transform,
                        src_crs="EPSG:4326",  # Asumimos WGS84
                        dst_transform=dst_transform,
                        dst_crs="EPSG:4326",  # Asumimos WGS84
                        resampling=Resampling.bilinear
                    )

                # Liberar memoria del timestep
                del timestep_data

                # Limpiar memoria después de cada timestep
                limit_memory_usage(threshold_mb=3000)  # Más agresivo

                # Cada 5 timesteps, forzar limpieza más agresiva
                if i % 5 == 0:
                    free_memory(f"Limpieza agresiva después del timestep {i}")

            # Obtener las coordenadas de tiempo para este chunk
            chunk_times = chirps_ds.time.isel(time=slice(chunk_start, chunk_end))

            # Crear un DataArray con los datos procesados
            chunk_da = xr.DataArray(
                chunk_data,
                dims=["time", "latitude", "longitude"],
                coords={
                    "time": chunk_times,
                    "latitude": dem_ds.latitude,
                    "longitude": dem_ds.longitude
                },
                attrs={"units": "mm/month", "description": f"{variable} downscaled to DEM resolution"}
            )

            # Guardar en un archivo netCDF temporal
            chunk_file = os.path.join(TEMP_DIR, f"{variable}_chunk_{chunk_start:05d}.nc")
            chunk_da.to_netcdf(chunk_file)
            chunk_files.append(chunk_file)
            processed_chunks.append(chunk_start)

            # Guardar el checkpoint después de cada chunk
            save_checkpoint(variable, {'processed_chunks': processed_chunks})

            # Liberar memoria
            del chunk_data, chunk_da, chunk_times
            free_memory(f"Limpieza después de guardar chunk {chunk_start}")
            log_memory_info(f"Memoria después de procesar chunk {chunk_start+1}-{chunk_end}")

        # Recopilar todos los archivos de chunks existentes
        if not chunk_files:
            chunk_pattern = os.path.join(TEMP_DIR, f"{variable}_chunk_*.nc")
            chunk_files = sorted(glob.glob(chunk_pattern))
            logging.info(f"Se encontraron {len(chunk_files)} archivos de chunks para {variable}")

        # Si ya existe el archivo de resultado y todos los chunks están procesados, simplemente cargarlo
        if os.path.exists(result_file) and checkpoint and checkpoint.get('combined', False):
            logging.info(f"Usando archivo de resultado existente para {variable}: {result_file}")

            # Liberar memoria antes de cargar
            free_memory("Limpieza antes de cargar resultado final")

            # Abrir con trozos pequeños para controlar memoria
            final_result = xr.open_dataarray(result_file, chunks={'time': 10, 'latitude': 100, 'longitude': 100})
            logging.info(f"Resultado cargado exitosamente con forma {final_result.shape}")
            return final_result

        # Ahora unir todos los chunks en un solo DataArray - de forma más cuidadosa
        logging.info(f"Uniendo {len(chunk_files)} chunks de {variable}...")

        # Verificar si hay un checkpoint para la combinación y crear un resultado parcial
        if result_file and os.path.exists(result_file) and checkpoint and 'combined_chunks' in checkpoint:
            logging.info(f"Encontrado archivo de resultado parcial: {result_file}")
            combined_chunks = checkpoint['combined_chunks']
            logging.info(f"Se han combinado previamente {len(combined_chunks)} de {len(chunk_files)} chunks")

            # Cargar el resultado parcial - pero antes liberar memoria
            free_memory("Limpieza antes de cargar resultado parcial")
            partial_result = xr.open_dataarray(result_file)

            # Filtrar los chunks que faltan
            remaining_chunks = [f for f in chunk_files if int(os.path.basename(f).split('_')[-1].split('.')[0]) not in combined_chunks]
            logging.info(f"Faltan {len(remaining_chunks)} chunks por combinar")

            if not remaining_chunks:
                logging.info(f"Todos los chunks ya están combinados para {variable}")
                return partial_result

            chunk_files = remaining_chunks
        else:
            combined_chunks = []
            partial_result = None

        # Ordenar los chunk_files para asegurar que estamos procesando en orden cronológico
        chunk_files = sorted(chunk_files, key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))

        # Procesar los archivos en lotes EXTREMADAMENTE pequeños para la fase final
        batch_size = 1  # Procesar un solo archivo a la vez para minimizar el uso de memoria
        for batch_start in range(0, len(chunk_files), batch_size):
            batch_end = min(batch_start + batch_size, len(chunk_files))
            logging.info(f"Procesando lote de archivos {batch_start+1}-{batch_end} de {len(chunk_files)}")

            # Liberar memoria antes de cargar un nuevo lote
            free_memory(f"Limpieza antes de cargar lote {batch_start+1}-{batch_end}")

            # Cargar este lote de archivos
            batch_das = []
            for i in range(batch_start, batch_end):
                chunk_file = chunk_files[i]
                chunk_num = int(os.path.basename(chunk_file).split('_')[-1].split('.')[0])

                # Verificar si este chunk ya está combinado
                if chunk_num in combined_chunks:
                    logging.info(f"Saltando chunk {chunk_num} que ya está combinado")
                    continue

                # Cargar y cerrar cada archivo inmediatamente para reducir handles abiertos
                try:
                    chunk_da = xr.open_dataarray(chunk_file)
                    batch_das.append(chunk_da)

                    # Añadir a la lista de chunks combinados
                    combined_chunks.append(chunk_num)
                except Exception as e:
                    logging.error(f"Error al cargar chunk {chunk_file}: {e}")

            # Concatenar este lote - si hay un resultado parcial, agregar a él
            if batch_das:
                try:
                    if partial_result is not None:
                        # Cerrar el resultado parcial para evitar problemas de acceso
                        if hasattr(partial_result, 'close'):
                            partial_result.close()

                        # Primero concatenar los nuevos chunks entre sí
                        if len(batch_das) > 1:
                            batch_combined = xr.concat(batch_das, dim="time")
                        else:
                            batch_combined = batch_das[0]

                        # Liberar memoria de batch_das ya que tenemos batch_combined
                        for da in batch_das:
                            if hasattr(da, 'close'):
                                da.close()
                        del batch_das
                        free_memory("Limpieza después de crear batch_combined")

                        # Luego cargar el resultado parcial nuevamente y combinarlo con los nuevos
                        partial_result = xr.open_dataarray(result_file)
                        result = xr.concat([partial_result, batch_combined], dim="time")

                        # Cerrar y liberar memorias
                        partial_result.close()
                        batch_combined.close()
                        del partial_result, batch_combined
                        free_memory("Limpieza después de combinar batch con resultado parcial")
                    else:
                        # Si es el primer lote, simplemente combinar
                        if len(batch_das) > 1:
                            result = xr.concat(batch_das, dim="time")
                        else:
                            result = batch_das[0]

                        # Liberar memoria de batch_das
                        for da in batch_das:
                            if hasattr(da, 'close'):
                                da.close()
                        del batch_das
                        free_memory("Limpieza después de crear primer resultado")

                    # Guardar resultado parcial
                    result.to_netcdf(result_file)
                    partial_result = result  # Actualizar para el próximo lote

                    # Actualizar checkpoint
                    save_checkpoint(variable, {'processed_chunks': processed_chunks, 'combined_chunks': combined_chunks})

                    # Liberar completamente la memoria
                    if hasattr(result, 'close'):
                        result.close()
                    del result
                    free_memory(f"Limpieza después de guardar lote {batch_start+1}-{batch_end}")

                    # Forzar recolección de basura
                    gc.collect()
                    log_memory_info(f"Memoria después de guardar lote {batch_start+1}-{batch_end}")
                except Exception as e:
                    # En caso de error en la combinación, guardamos el progreso actual
                    logging.error(f"Error al combinar chunks del lote {batch_start+1}-{batch_end}: {e}")
                    logging.error(f"Detalles: {traceback.format_exc()}")
                    save_checkpoint(variable, {'processed_chunks': processed_chunks, 'combined_chunks': combined_chunks})

                    # Intentar liberar toda la memoria posible
                    for da in locals().get('batch_das', []):
                        if hasattr(da, 'close'):
                            da.close()
                    for item in ['batch_das', 'batch_combined', 'result', 'partial_result']:
                        if item in locals():
                            del locals()[item]
                    free_memory("Limpieza de emergencia tras error")

                    # Si hay suficientes chunks combinados, intentamos continuar
                    if len(combined_chunks) > len(chunk_files) * 0.8:
                        logging.info(f"Suficientes chunks combinados ({len(combined_chunks)}), continuando...")
                        break
                    else:
                        raise  # Reenviar el error si no tenemos suficientes chunks

        # Si se procesaron todos los chunks, marcar como combinado
        if len(combined_chunks) == len(chunk_files):
            save_checkpoint(variable, {'processed_chunks': processed_chunks, 'combined_chunks': combined_chunks, 'combined': True})

        # Cargar el resultado final (ya debería estar guardado en disco)
        if os.path.exists(result_file):
            # Liberar memoria antes de cargar
            free_memory("Limpieza final antes de cargar resultado completo")

            # Cargar con chunks para controlar memoria
            final_result = xr.open_dataarray(result_file, chunks={'time': 10, 'latitude': 100, 'longitude': 100})
            logging.info(f"Downscaling de {variable} completado exitosamente con {len(combined_chunks)} chunks")
            return final_result
        else:
            raise ValueError(f"No se pudo crear el archivo de resultado para {variable}")

    except Exception as e:
        logging.critical(f"Error durante el downscaling de {variable}: {e}")
        logging.critical(f"Traza completa:\n{traceback.format_exc()}")
        log_memory_info("Estado de memoria en el momento del error")

        # Actualizar el checkpoint incluso en caso de error
        if 'processed_chunks' in locals():
            save_checkpoint(variable, {'processed_chunks': processed_chunks})

        raise


def downscale_chirps_to_dem(chirps_ds, dem_ds):
    """
    Downscale CHIRPS data (total, max, min precipitation) to match the resolution of the DEM.
    """
    logging.info("Ensuring CRS for CHIRPS and DEM datasets...")
    try:
        # Ensure CHIRPS dataset has a CRS
        if not hasattr(chirps_ds, 'rio') or not chirps_ds.rio.crs:
            logging.info("Añadiendo información CRS a CHIRPS dataset")
            # Importar rioxarray para acceder a la funcionalidad rio
            import rioxarray
            chirps_ds = chirps_ds.rio.write_crs("EPSG:4326")

        # Ensure DEM dataset has a CRS
        if not hasattr(dem_ds, 'rio') or not dem_ds.rio.crs:
            logging.info("Añadiendo información CRS a DEM dataset")
            import rioxarray
            dem_ds = dem_ds.rio.write_crs("EPSG:4326")

    except Exception as e:
        logging.error(f"Error al configurar CRS: {e}")
        logging.info("Continuando sin configurar CRS explícitamente...")

    try:
        # Extract DEM metadata
        from rasterio.transform import Affine

        # Intentar usar rio si está disponible
        if hasattr(dem_ds, 'rio'):
            dem_transform = dem_ds.rio.transform()
            dem_crs = dem_ds.rio.crs
        else:
            # Crear manualmente si no está disponible rio
            dem_lats = dem_ds.latitude.values
            dem_lons = dem_ds.longitude.values
            dem_res_lat = abs(dem_lats[1] - dem_lats[0]) if len(dem_lats) > 1 else 0.001
            dem_res_lon = abs(dem_lons[1] - dem_lons[0]) if len(dem_lons) > 1 else 0.001

            dem_transform = Affine(dem_res_lon, 0.0, min(dem_lons),
                                  0.0, -dem_res_lat, max(dem_lats))
            dem_crs = "EPSG:4326"  # Asumimos WGS84

        dem_width = dem_ds.sizes["longitude"]
        dem_height = dem_ds.sizes["latitude"]

    except Exception as e:
        logging.critical(f"Error al extraer metadatos del DEM: {e}")
        logging.critical(f"Traza completa:\n{traceback.format_exc()}")
        raise

    logging.info("Downscaling precipitation variables...")
    # Reducir el uso de memoria procesando una variable a la vez

    try:
        # Procesar total precipitation
        log_memory_info("Memoria antes de procesar total_precipitation")
        downscaled_total = downscale_variable_to_dem_chunked("total_precipitation", chirps_ds, dem_ds,
                                                      dem_transform, dem_crs, dem_width, dem_height)
        # Forzar limpieza
        free_memory("Limpieza después de procesar total_precipitation")
        log_memory_info("Memoria después de procesar total_precipitation")

        # Procesar max precipitation
        downscaled_max = downscale_variable_to_dem_chunked("max_daily_precipitation", chirps_ds, dem_ds,
                                                    dem_transform, dem_crs, dem_width, dem_height)
        # Forzar limpieza
        free_memory("Limpieza después de procesar max_daily_precipitation")
        log_memory_info("Memoria después de procesar max_daily_precipitation")

        # Procesar min precipitation
        downscaled_min = downscale_variable_to_dem_chunked("min_daily_precipitation", chirps_ds, dem_ds,
                                                    dem_transform, dem_crs, dem_width, dem_height)
        # Forzar limpieza
        free_memory("Limpieza después de procesar min_daily_precipitation")
        log_memory_info("Memoria después de procesar min_daily_precipitation")

        return downscaled_total, downscaled_max, downscaled_min

    except Exception as e:
        logging.critical(f"Error en el proceso de downscaling: {e}")
        logging.critical(f"Traza completa:\n{traceback.format_exc()}")
        raise


def save_combined_dataset(downscaled_total, downscaled_max, downscaled_min, dem_ds, output_path):
    """
    Save the combined dataset (downscaled CHIRPS and DEM) to a NetCDF file de manera incremental
    para minimizar el uso de memoria.
    """
    logging.info("Combining datasets and saving to NetCDF...")

    # Verificar si existe un checkpoint para el guardado
    save_checkpoint_data = load_checkpoint("save_combined")
    if save_checkpoint_data and save_checkpoint_data.get('completed', False):
        logging.info("El dataset combinado ya ha sido guardado previamente")
        return

    try:
        # Hacer una copia de seguridad si el archivo ya existe
        if os.path.exists(output_path):
            backup_path = output_path + ".bak"
            try:
                os.rename(output_path, backup_path)
                logging.info(f"Archivo existente respaldado como {backup_path}")
            except Exception as e:
                logging.warning(f"No se pudo crear respaldo del archivo existente: {e}")

        # Guardar de forma incremental, trabajando con una variable a la vez
        variables_to_save = [
            ('downscaled_total_precipitation', downscaled_total),
            ('downscaled_max_daily_precipitation', downscaled_max),
            ('downscaled_min_daily_precipitation', downscaled_min)
        ]

        # Variables ya guardadas (recuperadas del checkpoint)
        saved_vars = save_checkpoint_data.get('saved_vars', []) if save_checkpoint_data else []

        # Crear el dataset inicial con la estructura y coordenadas
        # Si el archivo ya existe, lo recreamos completamente
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logging.info(f"Eliminando archivo existente {output_path} para recrearlo")
            except Exception as e:
                logging.warning(f"No se pudo eliminar archivo existente: {e}")

        logging.info("Creando estructura del dataset combinado...")

        # Liberar memoria antes de crear el dataset base
        free_memory("Limpieza antes de crear estructura base")

        # Crear el dataset base completo con todas las variables
        # En lugar de intentar agregar los datos por chunks, crearemos un dataset completo desde el principio
        try:
            # Crear dataset con coordenadas
            base_ds = xr.Dataset(
                coords={
                    "time": downscaled_total.time,
                    "latitude": dem_ds.latitude,
                    "longitude": dem_ds.longitude
                },
                attrs={
                    "description": "Combined dataset with downscaled precipitation and elevation",
                    "source": "CHIRPS and DEM",
                    "created": datetime.now().strftime("%Y-%m-%d")
                }
            )

            # Añadir elevación primero (no tiene dimensión temporal y es más pequeña)
            base_ds["elevation"] = dem_ds["elevation"]

            # Guardar la estructura base con solo la elevación
            logging.info(f"Guardando estructura base con elevación en {output_path}...")
            encoding = {"elevation": {'zlib': True, 'complevel': 4}}
            base_ds.to_netcdf(output_path, encoding=encoding)

            # Limpiar memoria
            del base_ds
            free_memory("Limpieza después de guardar estructura base")
            log_memory_info("Memoria después de guardar estructura base")

            # Ahora procesamos cada variable una por una y la guardamos en un archivo separado
            for var_idx, (var_name, var_data) in enumerate(variables_to_save):
                if var_name in saved_vars:
                    logging.info(f"Variable {var_name} ya guardada previamente, saltando...")
                    continue

                logging.info(f"Procesando variable {var_name} ({var_idx+1}/{len(variables_to_save)})...")

                # Crear archivo temporal para esta variable
                temp_file = os.path.join(TEMP_DIR, f"{var_name}_combined.nc")

                # Liberar memoria antes de procesar esta variable
                free_memory(f"Limpieza antes de procesar {var_name}")

                # Crear un dataset solo con esta variable
                var_ds = xr.Dataset({var_name: var_data})

                # Guardar este dataset en un archivo separado
                logging.info(f"Guardando {var_name} en archivo temporal {temp_file}...")
                var_encoding = {var_name: {'zlib': True, 'complevel': 4}}
                var_ds.to_netcdf(temp_file, encoding=var_encoding)

                # Liberar memoria
                del var_ds
                free_memory(f"Limpieza después de guardar {var_name}")

                # Ahora añadimos esta variable al archivo principal usando NCO (operadores de NetCDF)
                # que son más eficientes para esto
                try:
                    import subprocess
                    logging.info(f"Combinando {temp_file} con {output_path}...")

                    # Usar NCO para combinar los archivos - ncks para extraer la variable y ncks para añadirla
                    cmd = f"ncks -A {temp_file} {output_path}"
                    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                    logging.info(f"Comando NCO completado: {result.stdout}")

                    # Marcar esta variable como guardada
                    saved_vars.append(var_name)
                    save_checkpoint_data = save_checkpoint_data or {}
                    save_checkpoint_data['saved_vars'] = saved_vars
                    save_checkpoint("save_combined", save_checkpoint_data)

                    # Borrar archivo temporal
                    os.remove(temp_file)
                    logging.info(f"Archivo temporal {temp_file} eliminado")

                except Exception as e:
                    logging.error(f"Error al combinar archivo usando NCO: {e}")
                    logging.error(traceback.format_exc())

                    # Intentar método alternativo con xarray si NCO falla
                    try:
                        logging.info("Intentando método alternativo con xarray...")

                        # Cargar el archivo principal actual
                        main_ds = xr.open_dataset(output_path)

                        # Cargar el archivo temporal
                        var_ds = xr.open_dataset(temp_file)

                        # Añadir la variable al dataset principal
                        main_ds[var_name] = var_ds[var_name]

                        # Guardar el dataset actualizado
                        main_ds.to_netcdf(output_path + ".tmp")

                        # Cerrar datasets
                        main_ds.close()
                        var_ds.close()

                        # Reemplazar archivo original
                        os.replace(output_path + ".tmp", output_path)

                        # Marcar esta variable como guardada
                        saved_vars.append(var_name)
                        save_checkpoint_data = save_checkpoint_data or {}
                        save_checkpoint_data['saved_vars'] = saved_vars
                        save_checkpoint("save_combined", save_checkpoint_data)

                        # Borrar archivo temporal
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                    except Exception as e2:
                        logging.critical(f"Error también en método alternativo: {e2}")
                        logging.critical(traceback.format_exc())
                        if var_name not in saved_vars:
                            logging.warning(f"No se pudo guardar {var_name}, continuando con la siguiente variable")

                # Si la variable es downscaled_total o downscaled_max, liberamos su memoria después de guardarla
                if var_name == 'downscaled_total_precipitation':
                    if hasattr(downscaled_total, 'close'):
                        downscaled_total.close()
                    del downscaled_total
                elif var_name == 'downscaled_max_daily_precipitation':
                    if hasattr(downscaled_max, 'close'):
                        downscaled_max.close()
                    del downscaled_max

                free_memory(f"Limpieza después de procesar variable {var_name}")
                log_memory_info(f"Memoria después de procesar variable {var_name}")

            # Marcar como completado
            save_checkpoint_data = save_checkpoint_data or {}
            save_checkpoint_data['saved_vars'] = saved_vars
            save_checkpoint_data['completed'] = True
            save_checkpoint("save_combined", save_checkpoint_data)

            # Liberar última variable
            if 'downscaled_min_daily_precipitation' in saved_vars:
                if hasattr(downscaled_min, 'close'):
                    downscaled_min.close()
                del downscaled_min

            logging.info(f"Combined dataset saved to: {output_path}")

        except Exception as e:
            logging.critical(f"Error al crear dataset combinado: {e}")
            logging.critical(f"Traza completa:\n{traceback.format_exc()}")
            raise

    except Exception as e:
        logging.critical(f"Error al guardar dataset combinado: {e}")
        logging.critical(f"Traza completa:\n{traceback.format_exc()}")

        # Restaurar backup si existe y ocurrió un error
        if 'backup_path' in locals() and os.path.exists(backup_path):
            try:
                os.rename(backup_path, output_path)
                logging.info("Se restauró la versión anterior del archivo de salida desde el respaldo")
            except Exception as e2:
                logging.error(f"No se pudo restaurar la versión anterior: {e2}")

        raise


def main():
    """
    Main function to perform downscaling and save the combined dataset.
    """
    try:
        # Verificar el checkpoint de progreso general
        main_checkpoint = load_checkpoint("main")

        # Configurar límite de memoria más restrictivo para evitar out-of-memory
        try:
            import resource
            # Intentar limitar a un valor fijo más conservador, en lugar de un porcentaje
            # En macOS, los límites son más restrictivos
            mem_limit_mb = 4096  # 4 GB como valor fijo y conservador
            mem_limit = mem_limit_mb * 1024 * 1024  # Convertir a bytes

            current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
            logging.info(f"Límites actuales: soft={current_soft/(1024**2):.0f}MB, hard={current_hard/(1024**2):.0f}MB")

            # Verificar que el nuevo límite no exceda el hard limit actual
            if current_hard != resource.RLIM_INFINITY and mem_limit > current_hard:
                logging.warning(f"El límite propuesto ({mem_limit/(1024**2):.0f}MB) excede el límite máximo actual ({current_hard/(1024**2):.0f}MB)")
                mem_limit = current_hard  # Usar el hard limit como máximo

            # Configurar el nuevo límite
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                logging.info(f"Límite de memoria configurado a {mem_limit/(1024**2):.0f}MB")
            except Exception as e:
                logging.warning(f"No se pudo configurar límite de memoria: {e}")
                # En caso de error, intentar con un valor más conservador
                try:
                    mem_limit = int(current_hard * 0.9) if current_hard != resource.RLIM_INFINITY else 2 * 1024 * 1024 * 1024  # 2GB
                    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                    logging.info(f"Límite de memoria configurado a valor alternativo: {mem_limit/(1024**2):.0f}MB")
                except Exception as e2:
                    logging.warning(f"No se pudo configurar límite alternativo de memoria: {e2}")
        except ImportError:
            logging.warning("Módulo 'resource' no disponible para establecer límites de memoria")

        # Limpieza inicial
        free_memory("Limpieza inicial antes de comenzar procesamiento")

        # Cargar datasets solo si es necesario
        chirps_ds = None
        dem_ds = None

        if not main_checkpoint or not main_checkpoint.get('downscaling_completed', False):
            logging.info("Loading CHIRPS and DEM datasets...")
            print("Loading CHIRPS and DEM datasets...")

            chirps_ds = load_nc_file(CHIRPS_MONTHLY_PATH)
            dem_ds = load_nc_file(DEM_PATH)
            logging.info("Datasets loaded successfully!")
            log_memory_info("Memoria después de cargar datasets")

            logging.info("Performing downscaling...")
            downscaled_total, downscaled_max, downscaled_min = downscale_chirps_to_dem(chirps_ds, dem_ds)
            logging.info("Downscaling completed!")
            log_memory_info("Memoria después de downscaling")

            # Marcar esta fase como completada
            save_checkpoint("main", {"downscaling_completed": True})

            # Liberar memoria de los datasets originales, ya no los necesitamos
            if hasattr(chirps_ds, 'close'):
                chirps_ds.close()
            del chirps_ds
            free_memory("Limpieza después de descartar chirps_ds")
        else:
            # Si el downscaling ya está completo, cargar desde los archivos de resultados
            logging.info("Downscaling ya completado, cargando resultados...")

            # Cargar los resultados del downscaling desde archivos guardados
            total_file = os.path.join(TEMP_DIR, "total_precipitation_result.nc")
            max_file = os.path.join(TEMP_DIR, "max_daily_precipitation_result.nc")
            min_file = os.path.join(TEMP_DIR, "min_daily_precipitation_result.nc")

            # Verificar que existan los archivos
            if not all(os.path.exists(f) for f in [total_file, max_file, min_file]):
                logging.error("No se encontraron todos los archivos de resultado, reprocesando...")
                # Forzar reprocesamiento
                chirps_ds = load_nc_file(CHIRPS_MONTHLY_PATH)
                dem_ds = load_nc_file(DEM_PATH)
                downscaled_total, downscaled_max, downscaled_min = downscale_chirps_to_dem(chirps_ds, dem_ds)

                # Liberar memoria de los datasets originales
                if hasattr(chirps_ds, 'close'):
                    chirps_ds.close()
                del chirps_ds
                free_memory("Limpieza después de descartar chirps_ds (reprocesado)")
            else:
                # Limpieza antes de cargar
                free_memory("Limpieza antes de cargar resultados")

                # Cargar los resultados guardados - con chunks para controlar memoria
                downscaled_total = xr.open_dataarray(total_file, chunks={'time': 10, 'latitude': 100, 'longitude': 100})
                downscaled_max = xr.open_dataarray(max_file, chunks={'time': 10, 'latitude': 100, 'longitude': 100})
                downscaled_min = xr.open_dataarray(min_file, chunks={'time': 10, 'latitude': 100, 'longitude': 100})
                logging.info("Resultados cargados correctamente")

            # Necesitamos dem_ds para guardar
            if dem_ds is None:
                dem_ds = load_nc_file(DEM_PATH)

        # Verificar si el guardado ya está completo
        if not main_checkpoint or not main_checkpoint.get('saving_completed', False):
            logging.info("Saving combined dataset...")
            save_combined_dataset(downscaled_total, downscaled_max, downscaled_min, dem_ds, OUTPUT_PATH)
            logging.info("Saving completed!")

            # Marcar esta fase como completada
            save_checkpoint("main", {"downscaling_completed": True, "saving_completed": True})

            # Liberar memoria de las variables procesadas
            for var in ['downscaled_total', 'downscaled_max', 'downscaled_min']:
                if var in locals() and locals()[var] is not None:
                    if hasattr(locals()[var], 'close'):
                        locals()[var].close()
                    del locals()[var]
            free_memory("Limpieza después de guardar")
        else:
            logging.info("Combined dataset already saved!")

            # Liberar memoria de las variables procesadas
            for var in ['downscaled_total', 'downscaled_max', 'downscaled_min']:
                if var in locals() and locals()[var] is not None:
                    if hasattr(locals()[var], 'close'):
                        locals()[var].close()
                    del locals()[var]
            free_memory("Limpieza después de cargar datos ya procesados")

        # Limpiar archivos temporales si todo está completo
        if main_checkpoint and main_checkpoint.get('downscaling_completed', False) and main_checkpoint.get('saving_completed', False):
            if main_checkpoint.get('cleanup_completed', False):
                logging.info("Ya se realizó limpieza previamente, proceso completo!")
            else:
                # Opción para limpiar archivos temporales (con precaución)
                logging.info("Limpiando archivos temporales...")
                try:
                    # Liberar memoria antes de limpiar
                    free_memory("Limpieza antes de eliminar archivos temporales")

                    # Solo eliminamos archivos chunk, mantenemos los result por si se necesitan después
                    pattern = "*_chunk_*.nc"
                    temp_files = glob.glob(os.path.join(TEMP_DIR, pattern))
                    for f in temp_files:
                        try:
                            os.remove(f)
                            logging.debug(f"Eliminado archivo temporal: {f}")
                        except Exception as e:
                            logging.warning(f"No se pudo eliminar {f}: {e}")

                    # Marcar limpieza como completa
                    save_checkpoint("main", {"downscaling_completed": True, "saving_completed": True, "cleanup_completed": True})
                except Exception as e:
                    logging.warning(f"Error durante la limpieza de archivos temporales: {e}")

        # Limpieza final
        free_memory("Limpieza final")
        logging.info("Process completed successfully!")
        print("¡Proceso completado exitosamente!")

    except Exception as e:
        logging.critical(f"Error en el proceso principal: {e}")
        logging.critical(f"Traza completa:\n{traceback.format_exc()}")
        log_memory_info("Estado de memoria en el momento del error crítico")
        print(f"Error: {e}")
        free_memory("Último intento de limpieza después de error crítico")
        sys.exit(1)

if __name__ == "__main__":
    main()
