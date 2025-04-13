import logging
import os
from datetime import datetime

def get_logger(name):
    """
    Create and configure a logger.

    Parameters:
        name (str): Name of the logger (usually __name__).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Configure logging directory
    LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists

    # Generate log file name with the required format
    log_filename = datetime.now().strftime("log-%Y-%m-%d.log")
    LOG_FILE = os.path.join(LOG_DIR, log_filename)

    # Custom log format
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # Extract only the internal project path
            base_path = os.path.dirname(os.path.dirname(__file__))  # Root of the project
            if record.pathname.startswith(base_path):
                record.track = record.pathname.replace(base_path, "")  # Internal path
            else:
                record.track = record.pathname  # Full path if external

            # Add function and line number
            record.function = record.funcName
            record.line = record.lineno

            # Format the log message
            return super().format(record)

    log_format = (
        "%(asctime)s - %(levelname)s - [track: %(track)s | function: %(function)s | line: %(line)d] - %(message)s"
    )

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Stream handler (console output)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(CustomFormatter(log_format))
        logger.addHandler(stream_handler)

        # File handler (write logs to file)
        file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        file_handler.setFormatter(CustomFormatter(log_format))
        logger.addHandler(file_handler)

    return logger
