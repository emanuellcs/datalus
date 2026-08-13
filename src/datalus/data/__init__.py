"""Tabular data ingestion, reversible encoding, and batched loading."""

from datalus.data.encoding import TabularEncoder
from datalus.data.ingestion import ZeroShotPreprocessor
from datalus.data.loader import ChunkedParquetBatches

__all__ = ["ChunkedParquetBatches", "TabularEncoder", "ZeroShotPreprocessor"]
