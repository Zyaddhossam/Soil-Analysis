"""Logging configuration for the application."""

import logging
import sys
from typing import Literal

from src.core.config import settings


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None,
) -> logging.Logger:
    """Configure application logging.

    Args:
        level: Log level. Defaults to DEBUG in development, INFO otherwise.

    Returns:
        Configured root logger.
    """
    if level is None:
        level = "DEBUG" if settings.debug else "INFO"

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger("soil_analysis")
    root_logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level))
    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.INFO)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"soil_analysis.{name}")
