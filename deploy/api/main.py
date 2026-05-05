"""Compatibility FastAPI entrypoint for DATALUS artifact serving."""

from datalus.api import app, create_app

__all__ = ["app", "create_app"]
