"""Shared Rich console and logging configuration for DATALUS.

This module provides a single `Console` instance shared across the package
(CLI, training, generation) so that Rich's ``Progress`` live display
and ``RichHandler`` logging never fight over terminal output.

It also exposes ``setup_logging`` and the ``VERBOSE_CHOICES`` constant so that
every command can offer a ``--verbose`` option with consistent behaviour.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

VERBOSE_CHOICES = ("WARNING", "INFO", "DEBUG")

VERBOSE_LEVEL_MAP: dict[str, int] = {
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

# Single console shared by RichHandler, Progress, and other Rich components.
console = Console()

_logger = logging.getLogger("datalus")


def setup_logging(verbose: str = "WARNING") -> None:
    """Configure the global logging level and Rich handler for DATALUS.

    Args:
        verbose: Log level - "WARNING" (errors only), "INFO" (progress),
                 or "DEBUG" (detailed).
    """
    log_level = VERBOSE_LEVEL_MAP.get(verbose, logging.WARNING)

    # Replace existing handlers so repeated setup calls do not duplicate them.
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

    # Align the root logger so lower-level third-party logs stay quiet.
    logging.getLogger().setLevel(log_level)
