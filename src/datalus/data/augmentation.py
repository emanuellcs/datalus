# Adapted from TabFM (https://github.com/google-research/tabfm),
# Copyright 2026 Google LLC, Apache License 2.0.
"""TabFM-inspired training-time feature crosses and SVD structural features."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from datalus.data.encoding import EncodedBatch, TabularEncoder


def _unrank_pair(index: int) -> tuple[int, int]:
    """Map a flat combination index to an unordered pair using combinadics."""

    j = (1 + math.isqrt(1 + 8 * index)) // 2
    i = index - j * (j - 1) // 2
    return (i, j)


@dataclass(slots=True)
class FeatureAugmentation:
    """Fit and apply training-only feature crosses or SVD structural features."""

    mode: str = "none"
    random_state: int = 42
    pairs: list[tuple[int, int]] = field(default_factory=list)
    svd: object | None = None
    column_names: list[str] = field(default_factory=list)

    def _resolve_count(self, n_features: int) -> int:
        """Return the number of augmented features (sqrt rule by default)."""

        return max(1, int(math.sqrt(n_features)))

    def fit(self, encoder: TabularEncoder, frame: pl.DataFrame) -> FeatureAugmentation:
        """Fit the augmentation on a bounded sample of the base-encoded data."""

        encoded = encoder.transform(frame)
        n_numeric = len(encoder.numerical_columns)
        rng = np.random.default_rng(self.random_state)
        if self.mode == "crosses":
            total_pairs = n_numeric * (n_numeric - 1) // 2
            if total_pairs <= 0:
                raise ValueError("Crosses augmentation requires at least two numerical columns.")
            k = min(self._resolve_count(n_numeric), total_pairs)
            sample = rng.choice(total_pairs, size=k, replace=False)
            self.pairs = [tuple(_unrank_pair(int(index))) for index in sample]
            self.column_names = [f"aug_cross_{i}_{j}" for i, j in self.pairs]
        elif self.mode == "svd":
            from sklearn.decomposition import TruncatedSVD

            matrix = self._prepare_matrix(encoded)
            k = min(self._resolve_count(matrix.shape[1]), matrix.shape[0] - 1, matrix.shape[1] - 1)
            if k <= 0:
                raise ValueError("SVD augmentation requires at least two encoded features.")
            self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
            self.svd.fit(matrix)
            self.column_names = [f"aug_svd_{idx}" for idx in range(k)]
        elif self.mode != "none":
            raise ValueError(f"Unsupported augmentation mode: {self.mode}")
        return self

    def transform_encoded(self, encoded: EncodedBatch) -> dict[str, np.ndarray]:
        """Compute augmented column values from one base-encoded batch."""

        columns: dict[str, np.ndarray] = {}
        if self.mode == "crosses":
            x_num = encoded.x_num
            if x_num is None:
                raise ValueError("Crosses augmentation requires numerical columns.")
            for (i, j), name in zip(self.pairs, self.column_names, strict=False):
                columns[name] = x_num[:, i] * x_num[:, j]
        elif self.mode == "svd" and self.svd is not None:
            matrix = self._prepare_matrix(encoded)
            components = self.svd.transform(matrix)
            for idx, name in enumerate(self.column_names):
                columns[name] = components[:, idx]
        return columns

    def transform_frame(self, frame: pl.DataFrame, encoder: TabularEncoder) -> pl.DataFrame:
        """Append the augmented columns to one base-encoded frame."""

        encoded = encoder.transform(frame)
        columns = self.transform_encoded(encoded)
        if not columns:
            return frame
        return frame.with_columns(
            [pl.Series(name, values.astype(np.float32)) for name, values in columns.items()]
        )

    def _prepare_matrix(self, encoded: EncodedBatch) -> np.ndarray:
        """Build a dense numeric matrix from encoded numerical and categorical batches."""

        import scipy.sparse as sp

        parts: list[object] = []
        if encoded.x_cat is not None:
            rows = encoded.x_cat.shape[0]
            for col_idx in range(encoded.x_cat.shape[1]):
                values = encoded.x_cat[:, col_idx]
                cardinality = int(values.max()) + 1
                sparse = sp.csr_matrix(
                    (np.ones(rows, dtype=np.float64), (np.arange(rows), values)),
                    shape=(rows, cardinality),
                )
                parts.append(sparse)
        if encoded.x_num is not None:
            parts.append(encoded.x_num.astype(np.float64))
        if not parts:
            raise ValueError("SVD augmentation requires at least one feature.")
        if len(parts) == 1:
            return parts[0].toarray() if sp.issparse(parts[0]) else parts[0]
        has_sparse = any(sp.issparse(part) for part in parts)
        combined = sp.hstack(parts, format="csr") if has_sparse else np.hstack(parts)
        return combined.toarray() if sp.issparse(combined) else combined

    @staticmethod
    def schema_metadata_for(mode: str, columns: list[str]) -> dict[str, dict[str, object]]:
        """Build schema metadata entries for the augmented numeric columns."""

        metadata: dict[str, dict[str, object]] = {}
        for name in columns:
            metadata[name] = {
                "column_name": name,
                "original_dtype": "float64",
                "inferred_topology": "NUMERICAL_CONTINUOUS",
                "encoding_strategy": "QUANTILE_TRANSFORM",
                "cardinality": None,
                "null_ratio": 0.0,
                "category_frequencies": None,
                "rare_category_count": None,
                "rare_category_threshold": None,
                "rare_categories_preserved": True,
                "is_target": False,
                "retained": True,
                "reason": f"augmentation:{mode}",
                "augmented": True,
            }
        return metadata
