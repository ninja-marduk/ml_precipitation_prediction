import subprocess
import time
import psutil
import os
import sys
import logging
from datetime import datetime

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
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

def run_script(script_path):
    """
    Run a Python script using subprocess and monitor its performance.

    Parameters:
        script_path (str): Path to the Python script to execute.
    """
    if not os.path.isfile(script_path):
        logging.error(f"The script {script_path} does not exist.")
        sys.exit(1)

    logging.info(f"Running script: {script_path}")

    try:
        # Start monitoring
        start_time = time.time()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        cpu_cores = psutil.cpu_count(logical=True)

        # Run the script
        result = subprocess.run(["python", script_path], capture_output=True, text=True)

        # End monitoring
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_after = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        memory_used = memory_after - memory_before
        cpu_percent = process.cpu_percent(interval=None)
        threads_used = process.num_threads()

        # Log performance metrics
        logging.info(f"Script {script_path} executed in {elapsed_time:.2f} seconds.")
        logging.info(f"Memory used: {memory_used:.2f} MB")
        logging.info(f"CPU cores available: {cpu_cores}")
        logging.info(f"CPU usage: {cpu_percent:.2f}%")
        logging.info(f"Threads used: {threads_used}")

        # Handle errors
        if result.returncode != 0:
            logging.error(f"Error executing {script_path}:\n{result.stderr}")
            sys.exit(1)
        else:
            logging.info(f"Script {script_path} executed successfully!")

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess error while executing {script_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while executing {script_path}: {e}")
        sys.exit(1)

def main():
    """
    Main function to execute the pipeline.
    """
    try:
        logging.info("Starting ETL pipeline...")

        # Define the scripts to run in order
        scripts = [

            "/Users/riperez/Conda/anaconda3/envs/precipitation_prediction/github.com/ml_precipitation_prediction/data/transformation/clustering/chirps-2.0.py"
        ]

        # Execute each script in order
        for script in scripts:
            run_script(script)

        logging.info("ETL pipeline completed successfully!")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the ETL pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
