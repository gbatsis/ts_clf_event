import sys
import logging

def setup_logger(log_level=logging.INFO):
    """
    Sets up a logger to log messages to the console only.

    Args:
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logger (logging.Logger): Configured logger object.
    """
    # Create a logger
    logger = logging.getLogger("TS Event Clf")
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times
    if not logger.hasHandlers():
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Define log format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(console_handler)

    return logger