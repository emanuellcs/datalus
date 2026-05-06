"""Polars batch-loading adapters used by application use cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


class ChunkedParquetBatches:
    """Expose deterministic Parquet slices without materializing full datasets."""

    def __init__(self, path: str | Path, batch_size: int, seed: int) -> None:
        self.path = str(path)
        self.batch_size = batch_size
        self.seed = seed
        self.lazy_frame = pl.scan_parquet(self.path)
        self.num_rows = int(self.lazy_frame.select(pl.len()).collect().item())

    def offsets_for_epoch(self, epoch: int, shuffle: bool = True) -> list[int]:
        """Return deterministic batch offsets for one epoch."""

        offsets = list(range(0, self.num_rows, self.batch_size))
        if shuffle:
            rng = np.random.default_rng(self.seed + epoch)
            offsets = [offsets[idx] for idx in rng.permutation(len(offsets))]
        return offsets

    def read_offset(self, offset: int) -> pl.DataFrame:
        """Collect one bounded row slice as a Polars DataFrame."""

        return self.lazy_frame.slice(offset, self.batch_size).collect()
