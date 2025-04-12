import subprocess
import time
import psutil
import os
import sys

def run_script(script_path):
    """
    Run a Python script using subprocess and monitor its performance.

    Parameters:
        script_path (str): Path to the Python script to execute.
    """
    if not os.path.isfile(script_path):
        print(f"Error: The script {script_path} does not exist.")
        sys.exit(1)

    print(f"Running script: {script_path}")

    try:
        # Start monitoring
        start_time = time.time()
        process = psutil.Process(os.getpid())
        # Memory in MB
        memory_before = process.memory_info().rss / (1024 * 1024)
        cpu_cores = psutil.cpu_count(logical=True)

        # Run the script
        result = subprocess.run(["python", script_path], capture_output=True, text=True)

        # End monitoring
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Memory in MB
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_used = memory_after - memory_before
        cpu_percent = process.cpu_percent(interval=None)
        threads_used = process.num_threads()

        # Print performance metrics
        print(f"Script {script_path} executed in {elapsed_time:.2f} seconds.")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"CPU cores available: {cpu_cores}")
        print(f"CPU usage: {cpu_percent:.2f}%")
        print(f"Threads used: {threads_used}")

        # Handle errors
        if result.returncode != 0:
            print(f"Error executing {script_path}:\n{result.stderr}")
            sys.exit(1)
        else:
            print(f"Script {script_path} executed successfully!")

    except Exception as e:
        print(f"An unexpected error occurred while executing {script_path}: {e}")
        sys.exit(1)

def main():
    """
    Main function to execute the pipeline.
    """
    # Define the scripts to run in order
    scripts = [
        "/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/load/chirps-2.0-daily.py",
        "/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/load/dem-90m.py",
        "/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/transformation/downscaling/chirps-2.0-monthly-90m.py",
        "/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/transformation/moving-avg/chirps-2.0-monthly-90m-moving-avg.py",
        "/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/transformation/clustering/chirps-2.0.py",
        "/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/load/test/load-datasets-chirps-2.0-distinct-elevations.py",
        "/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/transformation/correlation/chirps-2.0-elevation-categories-correlation-precipitation.py"
    ]

    # Execute each script in order
    for script in scripts:
        run_script(script)

if __name__ == "__main__":
    main()
