"""Centralized logging configuration for kakapo.

This module provides unified logging setup for the application, ensuring
consistent log formatting and level configuration across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Default logging format with timestamp and level
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Mapping of verbosity levels to logging levels
VERBOSITY_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def setup_logging(
    verbosity: int = 0,
    log_file: Optional[Path] = None,
    module_name: str = "kakapo",
    format_string: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Initialize and configure logging for the application.

    Sets up a logger with console output (and optionally file output) with
    consistent formatting. Maps verbosity levels to Python logging levels:
    - verbosity=0: WARNING
    - verbosity=1: INFO
    - verbosity=2: DEBUG

    Args:
        verbosity: Verbosity level (0-2). Higher values produce more output.
                   Defaults to 0 (WARNING).
        log_file: Optional path to write logs to a file. If None, only
                  console output is configured.
        module_name: Name of the logger module. Defaults to "kakapo".
        format_string: Log format string. Defaults to DEFAULT_FORMAT.
        date_format: Date format string. Defaults to DEFAULT_DATE_FORMAT.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logging(verbosity=1)
        >>> logger.info("Application started")
    """
    # Get or create the logger
    logger = logging.getLogger(module_name)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set the logging level based on verbosity
    log_level = VERBOSITY_LEVELS.get(verbosity, logging.DEBUG)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)

    # Set up console (stream) handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Set up optional file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    return logger


def get_logger(module_name: str = "kakapo") -> logging.Logger:
    """Get a logger instance for a specific module.

    Returns the logger for the given module name. Use this function to get
    a logger in submodules after setup_logging() has been called.

    Args:
        module_name: Name of the logger module. Defaults to "kakapo".

    Returns:
        Logger instance for the module.

    Example:
        >>> from kakapo.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message")
    """
    return logging.getLogger(module_name)
