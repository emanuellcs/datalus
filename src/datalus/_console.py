"""Shared Rich console and logging configuration for DATALUS.

This module provides a single `Console` instance shared across all layers
(CLI, application, infrastructure) so that Rich's ``Progress`` live display
and ``RichHandler`` logging never fight over terminal output.

It also exposes ``setup_logging`` and the ``VERBOSE_CHOICES`` constant so that
every command can offer a ``--verbose`` option with consistent behaviour.
"""

from __future__ import annotations

import logging
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler

VERBOSE_CHOICES = ("WARNING", "INFO", "DEBUG")

VERBOSE_LEVEL_MAP: dict[str, int] = {
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

# Singleton console shared by RichHandler, Progress, and any other Rich component.
console = Console()

_logger = logging.getLogger("datalus")


def setup_logging(verbose: str = "WARNING") -> None:
    """Configure global logging level based on --verbose flag.

    Args:
        verbose: Log level - "WARNING", "errors only"), "INFO" (progress),
                 or "DEBUG" (detailed).
    """
    log_level = VERBOSE_LEVEL_MAP.get(verbose, logging.WARNING)

    # Clear existing handlers to avoid duplicates on repeated calls.
    _logger.handlers.clear()

    handler = RichHandler(
        console=console,
        show_time=False,
        show_level=True,
        rich_tracebacks=True,
    )
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(log_level)

    # Align root logger so third-party warnings at lower levels are silenced.
    logging.getLogger().setLevel(log_level)
