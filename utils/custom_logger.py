import logging
import os
from datetime import datetime
import sys

def get_logger(name):
    """
    Create and configure a logger for the project.

    This function implements a custom logger that:
    - Writes logs to both console and a daily file
    - Uses a consistent format with project-relative path information
    - Includes function, line, and log level details
    - Automatically manages log directory creation

    Parameters:
        name (str): Name of the logger (usually __name__).

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Informational message")
        >>> logger.error("Error message")
    """
    # Configure logging directory
    LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists

    # Generate log file name with the required format
    log_filename = datetime.now().strftime("log-%Y-%m-%d.log")
    LOG_FILE = os.path.join(LOG_DIR, log_filename)

    # Custom log format
    class CustomFormatter(logging.Formatter):
        """
        Custom formatter that extracts the project-relative path
        and adds additional contextual information to the log message.
        """
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

    # Prevent adding handlers multiple times
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

def get_pipeline_logger(name, pipeline_id=None):
    """
    Creates a specialized logger for data processing pipelines.

    Similar to the standard logger but adds a pipeline identifier
    to facilitate tracking of long-running processes.

    Parameters:
        name (str): Name of the logger (usually __name__).
        pipeline_id (str, optional): Unique pipeline ID. If None,
                                    one will be automatically generated.

    Returns:
        logging.Logger: Configured logger instance for pipelines.
    """
    logger = get_logger(name)

    # If no pipeline ID is provided, generate one based on timestamp
    if pipeline_id is None:
        pipeline_id = f"pipeline-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Add the pipeline ID as a filter
    class PipelineFilter(logging.Filter):
        def filter(self, record):
            record.pipeline_id = pipeline_id
            return True

    # Apply the filter to all handlers
    for handler in logger.handlers:
        handler.addFilter(PipelineFilter())

    logger.info(f"Initializing pipeline with ID: {pipeline_id}")
    return logger
