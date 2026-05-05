"""Compatibility exports for DATALUS zero-shot preprocessing."""

from datalus.preprocessing import ColumnProfile, ZeroShotPreprocessor

TabularAutoPrep = ZeroShotPreprocessor

__all__ = ["ColumnProfile", "TabularAutoPrep", "ZeroShotPreprocessor"]
