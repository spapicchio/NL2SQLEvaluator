"""Logging utility module for standardized application-wide logging.

Why:
    Provides a consistent, color-coded output format and simplifies 
    the setup of loguru for users familiar with the standard logging pattern.

How:
    >>> logger = get_logger("my_module")
    >>> logger.info("Process started")
"""

import sys
from typing import Literal, Optional, Any

from loguru import logger as loguru_logger

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

# Track initialized sinks to avoid duplicates
_initialized_names = set()


def get_logger(
        name: str,
        level: LogLevel = "INFO",
        log_file: Optional[str] = None
) -> Any:
    """Create and return a customized logger with error message coloring.

    Args:
        name: Name of the logger (usually __name__).
        level: Logging level as a string (e.g., "DEBUG").
        log_file: Optional file path to write logs to.

    Returns:
        A loguru logger instance bound to the specific name.
    """
    level_str = level.upper()

    # We only remove the default handler once globally
    if not _initialized_names:
        loguru_logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[passed_name]}</cyan>:<cyan>{line}</cyan> | - <level>{message}</level>"
    )

    sink = sys.stdout if log_file is None else log_file

    # Register the sink specifically for this logger name
    if name not in _initialized_names:
        loguru_logger.add(
            sink,
            format=fmt,
            level=level_str,
            filter=lambda record: record["extra"].get("passed_name") == name,
        )
        _initialized_names.add(name)

    return loguru_logger.bind(passed_name=name)


if __name__ == "__main__":
    # Test cases
    logger_A = get_logger("A", level="INFO")
    logger_B = get_logger("B", level="DEBUG")

    logger_A.warning("This is a warning from A.")
    logger_B.info("This is an info message from B.")
    logger_B.debug("This is a debug message from B (visible).")
    logger_A.debug("This is hidden for A (level is INFO).")
