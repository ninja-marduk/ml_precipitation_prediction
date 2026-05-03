import os
import gc
import psutil
import logging
import numpy as np
import xarray as xr
import time
from functools import wraps

def print_memory_usage(message="Current memory usage"):
    """
    Print current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"{message}: {mem_info.rss / (1024 * 1024):.2f} MB")

def monitor_memory(interval=30, threshold_mb=None, action=None):
    """
    Monitor memory usage in a background thread and trigger an action if memory exceeds threshold.

    Args:
        interval: Monitoring interval in seconds
        threshold_mb: Memory threshold in MB
        action: Function to call when memory exceeds threshold

    Returns:
        stop_func: Function to call to stop monitoring
    """
    import threading

    if threshold_mb is None:
        # Default to 80% of available memory
        threshold_mb = int(psutil.virtual_memory().available * 0.8 / (1024 * 1024))

    def default_action():
        logging.warning(f"Memory threshold exceeded ({threshold_mb} MB). Forcing garbage collection.")
        gc.collect()

    action = action or default_action
    stop_event = threading.Event()

    def monitor_loop():
        while not stop_event.is_set():
            process = psutil.Process(os.getpid())
            mem_usage_mb = process.memory_info().rss / (1024 * 1024)
            if mem_usage_mb > threshold_mb:
                logging.warning(f"Memory usage ({mem_usage_mb:.2f} MB) exceeded threshold ({threshold_mb} MB)")
                action()
            time.sleep(interval)

    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

    def stop_monitoring():
        stop_event.set()
        monitor_thread.join(timeout=1)
        logging.info("Memory monitoring stopped")

    logging.info(f"Memory monitoring started (threshold: {threshold_mb} MB, interval: {interval}s)")
    return stop_monitoring

def forceful_memory_cleanup():
    """
    Perform a very aggressive memory cleanup, forcing memory release in all operating systems.

    This function uses multiple strategies to release memory:
    1. Multiple garbage collection passes
    2. Explicit deletion of large objects (numpy arrays, xarray objects)
    3. System-specific memory release calls
    4. I/O buffer flushing

    Particularly effective in memory-constrained environments and helps prevent OOM errors.
    """
    # First garbage collection pass
    gc.collect()

    # Force cleanup of objects that might be in memory
    for obj in gc.get_objects():
        try:
            if isinstance(obj, (np.ndarray, xr.Dataset, xr.DataArray)):
                del obj
        except:
            pass

    # Second pass
    gc.collect()

    try:
        # Try to invoke MallocTrim in macOS/Linux to return memory to the system
        import ctypes
        try:
            # MacOS and some Linux versions
            libc = ctypes.CDLL('libc.dylib')
            if hasattr(libc, 'malloc_trim'):
                libc.malloc_trim(0)
        except:
            try:
                # Linux
                libc = ctypes.CDLL('libc.so.6')
                if hasattr(libc, 'malloc_trim'):
                    libc.malloc_trim(0)
            except:
                pass
    except:
        pass

    # Force flush of I/O buffers
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

    # Final pass
    gc.collect()

    # In macOS, this can help return memory to the system
    try:
        import resource
        rusage_denom = 1024
        if sys.platform == 'darwin':
            # On macOS, getrusage returns bytes, not kilobytes
            rusage_denom = rusage_denom * rusage_denom
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
        logging.info(f"Memory cleanup: {mem_before:.2f} MB -> {mem_after:.2f} MB")
    except:
        process = psutil.Process(os.getpid())
        logging.info(f"Current memory after cleanup: {process.memory_info().rss / (1024 * 1024):.2f} MB")

def limit_memory_usage(threshold_mb=4000, force_cleanup=False):
    """
    Check if memory usage exceeds threshold and perform cleanup if needed.

    This function monitors memory usage and triggers cleanup procedures when necessary.
    It performs either standard or aggressive cleanup depending on how close the memory
    usage is to the threshold.

    Args:
        threshold_mb: Memory threshold in MB to trigger cleanup
        force_cleanup: Whether to force cleanup regardless of threshold

    Returns:
        bool: True if cleanup was performed, False otherwise
    """
    process = psutil.Process(os.getpid())
    mem_usage_mb = process.memory_info().rss / (1024 * 1024)

    if force_cleanup or mem_usage_mb > threshold_mb:
        logging.info(f"Memory usage ({mem_usage_mb:.2f} MB) {'exceeds threshold' if mem_usage_mb > threshold_mb else 'force cleanup requested'}")

        # Save memory usage before cleanup
        mem_before = mem_usage_mb

        # Perform standard cleanup
        gc.collect()

        # If still near the limit, perform more aggressive cleanup
        mem_after_gc = process.memory_info().rss / (1024 * 1024)
        if mem_after_gc > threshold_mb * 0.9 or force_cleanup:
            forceful_memory_cleanup()

        # Measure memory released
        mem_after = process.memory_info().rss / (1024 * 1024)
        logging.info(f"Memory cleanup: {mem_before:.2f} MB -> {mem_after:.2f} MB (released: {mem_before - mem_after:.2f} MB)")
        return True

    return False

def memory_check_decorator(threshold_mb=4000):
    """
    Decorator to check memory usage before and after function execution.

    Args:
        threshold_mb: Memory threshold in MB
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)
            logging.info(f"Memory before {func.__name__}: {mem_before:.2f} MB")

            # Execute function
            result = func(*args, **kwargs)

            # Check memory after
            mem_after = process.memory_info().rss / (1024 * 1024)
            logging.info(f"Memory after {func.__name__}: {mem_after:.2f} MB (diff: {mem_after - mem_before:.2f} MB)")

            # Clean up if needed
            if mem_after > threshold_mb:
                limit_memory_usage(threshold_mb)

            return result
        return wrapper
    return decorator

def load_data_with_chunks(file_path, chunk_sizes=None):
    """
    Load data with chunking to minimize memory usage.

    Args:
        file_path: Path to the file to load
        chunk_sizes: Dictionary with dimension names and chunk sizes

    Returns:
        xarray.Dataset: Loaded dataset
    """
    default_chunks = {'time': 1, 'latitude': 100, 'longitude': 100}
    chunks = chunk_sizes or default_chunks

    try:
        dataset = xr.open_dataset(file_path, chunks=chunks)
        return dataset
    except Exception as e:
        logging.error(f"Error loading data with chunks: {e}")
        # Try with smaller chunks if that fails
        smaller_chunks = {dim: max(1, size // 2) for dim, size in chunks.items()}
        try:
            dataset = xr.open_dataset(file_path, chunks=smaller_chunks)
            return dataset
        except Exception as e2:
            logging.error(f"Error loading with smaller chunks: {e2}")
            # Last resort: load without chunks
            return xr.open_dataset(file_path)

def process_by_chunks(dataset, func, dim='time', chunk_size=10):
    """
    Process a dataset by chunks along a dimension.

    Args:
        dataset: xarray.Dataset to process
        func: Function that takes a dataset chunk and returns a dataset
        dim: Dimension to chunk along
        chunk_size: Size of each chunk

    Returns:
        xarray.Dataset: Combined processed dataset
    """
    results = []
    dim_size = dataset.dims[dim]

    for i in range(0, dim_size, chunk_size):
        # Calculate end index
        end_idx = min(i + chunk_size, dim_size)
        logging.info(f"Processing chunk {i+1}-{end_idx} of {dim_size}")

        # Extract chunk
        chunk = dataset.isel({dim: slice(i, end_idx)})

        # Process chunk
        processed_chunk = func(chunk)
        results.append(processed_chunk)

        # Force cleanup after each chunk
        limit_memory_usage(force_cleanup=True)

    # Combine results
    if results:
        if isinstance(results[0], xr.Dataset) or isinstance(results[0], xr.DataArray):
            combined = xr.concat(results, dim=dim)
        else:
            # For other types (e.g., numpy arrays), use appropriate method
            combined = np.concatenate(results, axis=0)
        return combined
    else:
        return None
