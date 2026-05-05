"""Reversible heterogeneous tabular encoding for DATALUS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


@dataclass(slots=True)
class EncodedBatch:
    """Numerical and categorical tensors before learned embedding projection."""

    x_num: np.ndarray | None
    x_cat: np.ndarray | None


@dataclass(slots=True)
class NumericQuantileTransform:
    """One-dimensional quantile normalization with an inverse map."""

    column: str
    quantiles: list[float]
    references: list[float]
    fill_value: float

    @classmethod
    def fit(
        cls, column: str, values: np.ndarray, n_quantiles: int = 1_000
    ) -> "NumericQuantileTransform":
        clean = values.astype(np.float64)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            clean = np.array([0.0], dtype=np.float64)
        quantile_count = int(min(max(clean.size, 2), n_quantiles))
        references = np.linspace(0.0, 1.0, quantile_count)
        quantiles = np.quantile(clean, references)
        quantiles = np.maximum.accumulate(quantiles)
        return cls(
            column=column,
            quantiles=quantiles.astype(float).tolist(),
            references=references.astype(float).tolist(),
            fill_value=float(np.median(clean)),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = values.astype(np.float64)
        arr = np.where(np.isfinite(arr), arr, self.fill_value)
        encoded = np.interp(arr, self.quantiles, self.references, left=0.0, right=1.0)
        return (encoded * 2.0 - 1.0).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip((values.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0)
        return np.interp(clipped, self.references, self.quantiles).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "quantiles": self.quantiles,
            "references": self.references,
            "fill_value": self.fill_value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NumericQuantileTransform":
        return cls(**payload)


@dataclass(slots=True)
class CategoricalVocabulary:
    """Stable category-to-index mapping with null and unknown sentinels."""

    column: str
    categories: list[str]
    null_token: str = "__NULL__"
    unknown_token: str = "__UNKNOWN__"

    @classmethod
    def fit(cls, column: str, values: np.ndarray) -> "CategoricalVocabulary":
        normalized = [_normalize_category(value) for value in values]
        categories = sorted(
            {value for value in normalized if value not in {"__NULL__", "__UNKNOWN__"}}
        )
        return cls(column=column, categories=categories)

    @property
    def vocab(self) -> dict[str, int]:
        tokens = [self.unknown_token, self.null_token, *self.categories]
        return {token: idx for idx, token in enumerate(tokens)}

    @property
    def inverse_vocab(self) -> dict[int, str]:
        return {idx: token for token, idx in self.vocab.items()}

    @property
    def size(self) -> int:
        return len(self.categories) + 2

    def transform(self, values: np.ndarray) -> np.ndarray:
        vocab = self.vocab
        return np.array(
            [
                vocab.get(_normalize_category(value), vocab[self.unknown_token])
                for value in values
            ],
            dtype=np.int64,
        )

    def inverse(self, values: np.ndarray) -> list[str | None]:
        inverse_vocab = self.inverse_vocab
        decoded: list[str | None] = []
        for value in values.astype(np.int64):
            token = inverse_vocab.get(int(value), self.unknown_token)
            decoded.append(None if token == self.null_token else token)
        return decoded

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "categories": self.categories,
            "null_token": self.null_token,
            "unknown_token": self.unknown_token,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CategoricalVocabulary":
        return cls(**payload)


def _normalize_category(value: Any) -> str:
    if value is None:
        return "__NULL__"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "__NULL__"
    except TypeError:
        pass
    return str(value)


class TabularEncoder:
    """Fit and apply DATALUS reversible tabular transforms."""

    def __init__(
        self,
        schema_metadata: dict[str, dict[str, Any]],
        numeric_transforms: dict[str, NumericQuantileTransform] | None = None,
        categorical_vocabs: dict[str, CategoricalVocabulary] | None = None,
    ) -> None:
        self.schema_metadata = schema_metadata
        self.numeric_transforms = numeric_transforms or {}
        self.categorical_vocabs = categorical_vocabs or {}

    @property
    def active_schema(self) -> dict[str, dict[str, Any]]:
        return {
            column: meta
            for column, meta in self.schema_metadata.items()
            if meta.get("retained", meta.get("encoding_strategy") != "DROP")
            and meta.get("encoding_strategy") != "DROP"
            and not meta.get("is_target", False)
        }

    @property
    def numerical_columns(self) -> list[str]:
        return [
            column
            for column, meta in self.active_schema.items()
            if "NUMERICAL" in meta.get("inferred_topology", "")
        ]

    @property
    def categorical_columns(self) -> list[str]:
        return [
            column
            for column, meta in self.active_schema.items()
            if "CATEGORICAL" in meta.get("inferred_topology", "")
            or meta.get("inferred_topology") == "BOOLEAN"
        ]

    def fit(self, frame: pl.DataFrame) -> "TabularEncoder":
        for column in self.numerical_columns:
            values = frame.get_column(column).cast(pl.Float64, strict=False).to_numpy()
            self.numeric_transforms[column] = NumericQuantileTransform.fit(
                column, values
            )
        for column in self.categorical_columns:
            values = frame.get_column(column).to_numpy()
            self.categorical_vocabs[column] = CategoricalVocabulary.fit(column, values)
            self.schema_metadata[column]["cardinality"] = self.categorical_vocabs[
                column
            ].size
        return self

    def transform(self, frame: pl.DataFrame) -> EncodedBatch:
        x_num = None
        x_cat = None
        if self.numerical_columns:
            encoded_num = [
                self.numeric_transforms[column].transform(
                    frame.get_column(column).cast(pl.Float64, strict=False).to_numpy()
                )
                for column in self.numerical_columns
            ]
            x_num = np.stack(encoded_num, axis=1).astype(np.float32)
        if self.categorical_columns:
            encoded_cat = [
                self.categorical_vocabs[column].transform(
                    frame.get_column(column).to_numpy()
                )
                for column in self.categorical_columns
            ]
            x_cat = np.stack(encoded_cat, axis=1).astype(np.int64)
        return EncodedBatch(x_num=x_num, x_cat=x_cat)

    def inverse_transform(
        self,
        x_num: np.ndarray | None,
        x_cat: np.ndarray | None,
    ) -> pl.DataFrame:
        data: dict[str, Any] = {}
        if x_num is not None:
            for idx, column in enumerate(self.numerical_columns):
                data[column] = self.numeric_transforms[column].inverse(x_num[:, idx])
        if x_cat is not None:
            for idx, column in enumerate(self.categorical_columns):
                data[column] = self.categorical_vocabs[column].inverse(x_cat[:, idx])
        return pl.DataFrame(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_metadata": self.schema_metadata,
            "numeric_transforms": {
                column: transform.to_dict()
                for column, transform in self.numeric_transforms.items()
            },
            "categorical_vocabs": {
                column: vocab.to_dict()
                for column, vocab in self.categorical_vocabs.items()
            },
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TabularEncoder":
        return cls(
            schema_metadata=payload["schema_metadata"],
            numeric_transforms={
                column: NumericQuantileTransform.from_dict(item)
                for column, item in payload.get("numeric_transforms", {}).items()
            },
            categorical_vocabs={
                column: CategoricalVocabulary.from_dict(item)
                for column, item in payload.get("categorical_vocabs", {}).items()
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "TabularEncoder":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
