# Adapted from TabFM (https://github.com/google-research/tabfm),
# Copyright 2026 Google LLC, Apache License 2.0.
"""Reversible heterogeneous tabular encoding for DATALUS.

Mirrors TabFM preprocessing (quantile noise, outlier clipping, ordinal modes)
while keeping the transforms reversible for diffusion generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl


@dataclass(slots=True)
class EncodedBatch:
    """Numerical and categorical tensors before learned embedding projection."""

    x_num: np.ndarray | None
    x_cat: np.ndarray | None


def _clip_outliers(values: np.ndarray, threshold: float) -> np.ndarray:
    """Clip extreme values with two-stage robust z-score bounds."""

    arr = values.astype(np.float64)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1 if arr.size > 1 else 0))
    std = max(std, 1e-6)
    lower = mean - threshold * std
    upper = mean + threshold * std
    cleaned = np.where((arr < lower) | (arr > upper), np.nan, arr)
    robust_mean = float(np.nanmean(cleaned))
    robust_std = float(np.nanstd(cleaned, ddof=1 if cleaned.size > 1 else 0))
    if not np.isfinite(robust_mean):
        return np.clip(arr, lower, upper)
    robust_std = max(robust_std, 1e-6)
    return np.clip(arr, robust_mean - threshold * robust_std, robust_mean + threshold * robust_std)


@dataclass(slots=True)
class NumericQuantileTransform:
    """One-dimensional quantile normalization with an inverse map."""

    column: str
    quantiles: list[float]
    references: list[float]
    fill_value: float
    mean: float | None = None
    scale: float | None = None

    @classmethod
    def fit(
        cls,
        column: str,
        values: np.ndarray,
        n_quantiles: int = 1_000,
        noise: float = 0.0,
        random_state: int = 42,
        outlier_threshold: float | None = None,
        standardize: bool = False,
    ) -> NumericQuantileTransform:
        """Fit monotone quantiles, optionally with RTDL noise, outlier clipping, and standardization."""

        clean = values.astype(np.float64)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            clean = np.array([0.0], dtype=np.float64)
        if noise > 0:
            std = float(np.std(clean))
            noise_std = noise / max(std, noise)
            rng = np.random.default_rng(random_state)
            clean = clean + noise_std * rng.standard_normal(clean.shape)
        if outlier_threshold is not None:
            clean = _clip_outliers(clean, outlier_threshold)
        quantile_count = int(min(max(clean.size, 2), n_quantiles))
        references = np.linspace(0.0, 1.0, quantile_count)
        quantiles = np.quantile(clean, references)
        quantiles = np.maximum.accumulate(quantiles)
        mean = None
        scale = None
        if standardize:
            encoded = np.interp(clean, quantiles, references) * 2.0 - 1.0
            mean = float(np.mean(encoded))
            scale = float(np.std(encoded) + 1e-6)
        return cls(
            column=column,
            quantiles=quantiles.astype(float).tolist(),
            references=references.astype(float).tolist(),
            fill_value=float(np.median(clean)),
            mean=mean,
            scale=scale,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Map raw values to the normalized interval."""

        arr = values.astype(np.float64)
        arr = np.where(np.isfinite(arr), arr, self.fill_value)
        encoded = np.interp(arr, self.quantiles, self.references, left=0.0, right=1.0)
        encoded = encoded * 2.0 - 1.0
        if self.scale is not None:
            encoded = (encoded - self.mean) / self.scale
            encoded = np.clip(encoded, -100.0, 100.0)
        return encoded.astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Map normalized values back to the original numeric scale."""

        arr = values.astype(np.float64)
        if self.scale is not None:
            arr = arr * self.scale + self.mean
        clipped = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        return np.interp(clipped, self.references, self.quantiles).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the transform for artifact persistence."""

        return {
            "column": self.column,
            "quantiles": self.quantiles,
            "references": self.references,
            "fill_value": self.fill_value,
            "mean": self.mean,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NumericQuantileTransform:
        """Rebuild a transform from a serialized payload."""

        return cls(**payload)


@dataclass(slots=True)
class CategoricalVocabulary:
    """Stable category-to-index mapping with null and unknown sentinels."""

    column: str
    categories: list[str]
    frequencies: dict[str, int] | None = None
    null_token: str = "__NULL__"
    unknown_token: str = "__UNKNOWN__"

    @classmethod
    def fit(
        cls,
        column: str,
        values: np.ndarray,
        mode: Literal["alphabetical", "appearance", "frequency"] = "alphabetical",
        min_frequency: int = 1,
    ) -> CategoricalVocabulary:
        """Collect categories by ``mode``, excluding those below ``min_frequency``."""

        normalized = [_normalize_category(value) for value in values]
        frequencies: dict[str, int] = {}
        for value in normalized:
            frequencies[value] = frequencies.get(value, 0) + 1
        sentinels = {"__NULL__", "__UNKNOWN__"}
        base = {value for value in set(normalized) if value not in sentinels}
        if mode == "frequency":
            categories = sorted(base, key=lambda value: (-frequencies[value], value))
        elif mode == "appearance":
            ordered = list(dict.fromkeys(normalized))
            categories = [value for value in ordered if value in base]
        elif mode == "alphabetical":
            categories = sorted(base)
        else:
            raise ValueError(f"Unsupported categorical mode: {mode}")
        if min_frequency > 1:
            categories = [value for value in categories if frequencies[value] >= min_frequency]
        return cls(column=column, categories=categories, frequencies=frequencies)

    @property
    def vocab(self) -> dict[str, int]:
        """Return the token-to-index mapping including the sentinels."""

        tokens = [self.unknown_token, self.null_token, *self.categories]
        return {token: idx for idx, token in enumerate(tokens)}

    @property
    def inverse_vocab(self) -> dict[int, str]:
        """Return the index-to-token reverse mapping."""

        return {idx: token for token, idx in self.vocab.items()}

    @property
    def size(self) -> int:
        """Return the vocabulary size including the two sentinels."""

        return len(self.categories) + 2

    def transform(self, values: np.ndarray) -> np.ndarray:
        vocab = self.vocab
        # Keep fitted categories as distinct tokens; only unseen values map to the sentinel.
        return np.array(
            [vocab.get(_normalize_category(value), vocab[self.unknown_token]) for value in values],
            dtype=np.int64,
        )

    def inverse(self, values: np.ndarray) -> list[str | None]:
        """Map category indices back to tokens, decoding null as None."""

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
            "frequencies": self.frequencies or {},
            "null_token": self.null_token,
            "unknown_token": self.unknown_token,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CategoricalVocabulary:
        """Rebuild a vocabulary from a serialized payload."""

        return cls(**payload)


def _normalize_category(value: Any) -> str:
    """Return a stable string token for a raw category value."""

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
        quantile_noise: float = 0.0,
        rtdl_quantile_dynamic: bool = False,
        outlier_threshold: float | None = None,
        numeric_standardize: bool = False,
        cat_encoder_mode: Literal["alphabetical", "appearance", "frequency"] = "alphabetical",
        min_cat_frequency: int = 1,
        random_state: int = 42,
    ) -> None:
        """Initialize the encoder with schema metadata and an optional encoding policy."""

        self.schema_metadata = schema_metadata
        self.numeric_transforms = numeric_transforms or {}
        self.categorical_vocabs = categorical_vocabs or {}
        self.quantile_noise = quantile_noise
        self.rtdl_quantile_dynamic = rtdl_quantile_dynamic
        self.outlier_threshold = outlier_threshold
        self.numeric_standardize = numeric_standardize
        self.cat_encoder_mode = cat_encoder_mode
        self.min_cat_frequency = min_cat_frequency
        self.random_state = random_state

    @property
    def augmented_columns(self) -> list[str]:
        """Return training-time augmented column names flagged in the schema."""

        return [
            column
            for column, meta in self.schema_metadata.items()
            if meta.get("augmented", False)
        ]

    @property
    def active_schema(self) -> dict[str, dict[str, Any]]:
        """Return schema entries that are not marked for dropping."""

        return {
            column: meta
            for column, meta in self.schema_metadata.items()
            if meta.get("retained", meta.get("encoding_strategy") != "DROP")
            and meta.get("encoding_strategy") != "DROP"
        }

    @property
    def numerical_columns(self) -> list[str]:
        """Return the retained numerical column names."""

        return [
            column
            for column, meta in self.active_schema.items()
            if "NUMERICAL" in meta.get("inferred_topology", "")
        ]

    @property
    def categorical_columns(self) -> list[str]:
        """Return the retained categorical and boolean column names."""

        return [
            column
            for column, meta in self.active_schema.items()
            if "CATEGORICAL" in meta.get("inferred_topology", "")
            or meta.get("inferred_topology") == "BOOLEAN"
        ]

    def fit(self, frame: pl.DataFrame) -> TabularEncoder:
        """Fit numeric transforms and categorical vocabularies on a frame."""

        for column in self.numerical_columns:
            values = frame.get_column(column).cast(pl.Float64, strict=False).to_numpy()
            clean_size = int(np.sum(np.isfinite(values.astype(np.float64, copy=False))))
            n_quantiles = 1_000
            if self.rtdl_quantile_dynamic:
                n_quantiles = int(min(max(clean_size // 30, 10), 1_000))
            self.numeric_transforms[column] = NumericQuantileTransform.fit(
                column,
                values,
                n_quantiles=n_quantiles,
                noise=self.quantile_noise,
                random_state=self.random_state,
                outlier_threshold=self.outlier_threshold,
                standardize=self.numeric_standardize,
            )
        for column in self.categorical_columns:
            values = frame.get_column(column).to_numpy()
            self.categorical_vocabs[column] = CategoricalVocabulary.fit(
                column,
                values,
                mode=self.cat_encoder_mode,
                min_frequency=self.min_cat_frequency,
            )
            self.schema_metadata[column]["cardinality"] = self.categorical_vocabs[column].size
            self.schema_metadata[column]["category_frequencies"] = (
                self.categorical_vocabs[column].frequencies or {}
            )
            rare_threshold = int(self.schema_metadata[column].get("rare_category_threshold") or 5)
            self.schema_metadata[column]["rare_category_count"] = sum(
                1
                for count in (self.categorical_vocabs[column].frequencies or {}).values()
                if count <= rare_threshold
            )
            self.schema_metadata[column]["rare_categories_preserved"] = True
        return self

    def transform(self, frame: pl.DataFrame) -> EncodedBatch:
        """Encode a frame into stacked numerical and categorical arrays."""

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
                self.categorical_vocabs[column].transform(frame.get_column(column).to_numpy())
                for column in self.categorical_columns
            ]
            x_cat = np.stack(encoded_cat, axis=1).astype(np.int64)
        return EncodedBatch(x_num=x_num, x_cat=x_cat)

    def inverse_transform(
        self,
        x_num: np.ndarray | None,
        x_cat: np.ndarray | None,
    ) -> pl.DataFrame:
        """Decode numerical and categorical arrays back into a DataFrame."""

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
                column: transform.to_dict() for column, transform in self.numeric_transforms.items()
            },
            "categorical_vocabs": {
                column: vocab.to_dict() for column, vocab in self.categorical_vocabs.items()
            },
            "quantile_noise": self.quantile_noise,
            "rtdl_quantile_dynamic": self.rtdl_quantile_dynamic,
            "outlier_threshold": self.outlier_threshold,
            "numeric_standardize": self.numeric_standardize,
            "cat_encoder_mode": self.cat_encoder_mode,
            "min_cat_frequency": self.min_cat_frequency,
            "random_state": self.random_state,
        }

    def save(self, path: str | Path) -> None:
        """Write the encoder to a JSON artifact file."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TabularEncoder:
        """Rebuild an encoder from a serialized payload."""

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
            quantile_noise=float(payload.get("quantile_noise", 0.0)),
            rtdl_quantile_dynamic=bool(payload.get("rtdl_quantile_dynamic", False)),
            outlier_threshold=payload.get("outlier_threshold"),
            numeric_standardize=bool(payload.get("numeric_standardize", False)),
            cat_encoder_mode=payload.get("cat_encoder_mode", "alphabetical"),
            min_cat_frequency=int(payload.get("min_cat_frequency", 1)),
            random_state=int(payload.get("random_state", 42)),
        )

    @classmethod
    def load(cls, path: str | Path) -> TabularEncoder:
        """Load an encoder from a JSON artifact file."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
