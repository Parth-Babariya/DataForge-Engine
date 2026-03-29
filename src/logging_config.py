"""Structured logging configuration for production use.

This module provides JSON-formatted logging that works well with log aggregation
systems (ELK, CloudWatch, etc.). All log entries are structured JSON for easier
parsing and analysis.

Design principles:
  - Structured logs (JSON) are machine-readable and queryable
  - Context (timestamps, levels, logger name) automatically added
  - Exception tracebacks included for error debugging
  - Easy to integrate with monitoring platforms
"""
import logging
import json
import sys
from datetime import datetime
from typing import Any, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON for structured logging.

    Each log entry becomes a JSON object with:
    - timestamp: ISO 8601 timestamp
    - level: Log level (INFO, WARNING, ERROR, etc.)
    - logger: Logger name (module path)
    - message: The log message
    - exception: Stack trace (if error context exists)
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom fields if present in the record
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        return json.dumps(log_data, default=str)


def setup_logging(
    name: Optional[str] = None,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Initialize structured logging with JSON output.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to

    Returns:
        Configured logger ready for use

    Example:
        >>> logger = setup_logging(__name__, level="DEBUG")
        >>> logger.info("Application started", extra={"request_id": "123"})
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


# Root logger for the application
_root_logger = setup_logging("dataforge_engine")


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name.

    Use this in your modules:
        from src.logging_config import get_logger
        logger = get_logger(__name__)

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger
    """
    return logging.getLogger(name)
