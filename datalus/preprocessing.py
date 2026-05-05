"""Lazy zero-shot preprocessing and topology inference for DATALUS."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

logger = logging.getLogger(__name__)

IDENTIFIER_NAME_RE = re.compile(
    r"(^id$|_id$|cpf|cnpj|cns|cartao|prontuario|aih|nis|sus|uuid|guid|email|telefone|phone)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ColumnProfile:
    """Serializable metadata for one input column."""

    column_name: str
    original_dtype: str
    inferred_topology: str
    encoding_strategy: str
    cardinality: int | None = None
    null_ratio: float | None = None
    is_target: bool = False
    retained: bool = True
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame using the newest available Polars streaming API."""

    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:
        return lazy_frame.collect(streaming=True)


class ZeroShotPreprocessor:
    """Infer schema topology lazily and materialize safe Parquet outputs."""

    def __init__(
        self,
        high_cardinality_threshold: int = 50,
        sample_size: int = 100_000,
        target_column: str | None = None,
        null_values: Iterable[str] | None = None,
    ) -> None:
        self.high_cardinality_threshold = high_cardinality_threshold
        self.sample_size = sample_size
        self.target_column = target_column
        self.null_values = list(
            null_values or ["", "NA", "N/A", "null", "NULL", "None"]
        )
        self.schema_registry: dict[str, ColumnProfile] = {}

    def scan(self, file_path: str | Path) -> pl.LazyFrame:
        """Return a lazy scan for CSV, Parquet, or ORC inputs."""

        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pl.scan_csv(
                path,
                infer_schema_length=10_000,
                null_values=self.null_values,
                try_parse_dates=True,
            )
        if suffix == ".parquet":
            return pl.scan_parquet(path)
        if suffix == ".orc":
            try:
                import pyarrow.dataset as ds
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise ValueError("ORC inputs require pyarrow to be installed.") from exc
            return pl.scan_pyarrow_dataset(ds.dataset(str(path), format="orc"))
        raise ValueError(f"Unsupported input format: {suffix}")

    def fit(self, file_path: str | Path) -> "ZeroShotPreprocessor":
        """Infer topology from a deterministic bounded sample."""

        logger.info("Inferring DATALUS schema from %s", file_path)
        sampled = _collect_streaming(self.scan(file_path).head(self.sample_size))
        self.schema_registry = {
            column: self._infer_column_topology(sampled[column])
            for column in sampled.columns
        }
        return self

    def fit_transform_to_parquet(
        self,
        input_path: str | Path,
        output_path: str | Path,
        schema_path: str | Path | None = None,
    ) -> dict[str, ColumnProfile]:
        """Fit the schema and stream the retained columns to Parquet."""

        self.fit(input_path)
        if schema_path is not None:
            self.export_schema(schema_path)
        self.transform_to_parquet(input_path, output_path)
        return self.schema_registry

    def transform_to_parquet(
        self, input_path: str | Path, output_path: str | Path
    ) -> None:
        """Stream the lazy graph directly to Snappy-compressed Parquet."""

        if not self.schema_registry:
            raise RuntimeError("fit() must be called before transform_to_parquet().")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        columns_to_drop = [
            name
            for name, profile in self.schema_registry.items()
            if not profile.retained
        ]
        graph = self.scan(input_path).drop(columns_to_drop)
        try:
            graph.sink_parquet(
                str(output), compression="snappy", row_group_size=100_000
            )
        except TypeError:
            graph.sink_parquet(str(output), compression="snappy")

    def export_schema(self, output_path: str | Path) -> None:
        """Write `schema_config.json` metadata."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: profile.to_dict() for name, profile in self.schema_registry.items()
        }
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def from_schema(cls, schema_path: str | Path) -> "ZeroShotPreprocessor":
        """Load an existing schema registry."""

        instance = cls()
        payload = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        instance.schema_registry = {
            name: ColumnProfile(**profile) for name, profile in payload.items()
        }
        return instance

    def _infer_column_topology(self, series: pl.Series) -> ColumnProfile:
        name = series.name
        dtype = str(series.dtype)
        total_count = len(series)
        null_count = series.null_count()
        null_ratio = null_count / total_count if total_count else 1.0
        cardinality = int(series.n_unique()) if total_count else 0
        is_target = name == self.target_column

        def profile(
            topology: str,
            strategy: str,
            retained: bool = True,
            reason: str | None = None,
        ):
            return ColumnProfile(
                column_name=name,
                original_dtype=dtype,
                inferred_topology=topology,
                encoding_strategy=strategy,
                cardinality=cardinality,
                null_ratio=float(null_ratio),
                is_target=is_target,
                retained=retained,
                reason=reason,
            )

        if null_ratio > 0.95 and not is_target:
            return profile(
                "SPARSE", "DROP", retained=False, reason="null_ratio_gt_0.95"
            )

        if not is_target and self._looks_like_identifier(
            name, cardinality, total_count
        ):
            return profile(
                "IDENTIFIER", "DROP", retained=False, reason="identifier_policy"
            )

        if series.dtype in {pl.Boolean} or cardinality == 2:
            return profile("BOOLEAN", "BINARY_ENCODING")

        if series.dtype in {pl.Date, pl.Datetime, pl.Time}:
            return profile("DATETIME", "CYCLICAL_ENCODING")

        integer_types = {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        }
        if series.dtype in integer_types:
            if cardinality <= min(15, self.high_cardinality_threshold):
                return profile("NUMERICAL_DISCRETE", "QUANTILE_TRANSFORM")
            return profile("NUMERICAL_CONTINUOUS", "QUANTILE_TRANSFORM")

        if series.dtype in {pl.Float32, pl.Float64}:
            return profile("NUMERICAL_CONTINUOUS", "QUANTILE_TRANSFORM")

        if series.dtype in {pl.String, pl.Utf8, pl.Categorical}:
            avg_len = self._average_string_length(series)
            if avg_len > 80 and cardinality > self.high_cardinality_threshold:
                return profile("FREE_TEXT", "DROP", retained=False, reason="free_text")
            if cardinality > self.high_cardinality_threshold:
                return profile("CATEGORICAL_HIGH_CARDINALITY", "HIERARCHICAL_EMBEDDING")
            return profile("CATEGORICAL_LOW_CARDINALITY", "CONTINUOUS_EMBEDDING")

        return profile("UNKNOWN", "DROP", retained=False, reason="unsupported_dtype")

    @staticmethod
    def _looks_like_identifier(name: str, cardinality: int, total_count: int) -> bool:
        if IDENTIFIER_NAME_RE.search(name):
            return True
        return total_count > 1_000 and cardinality / max(total_count, 1) > 0.98

    @staticmethod
    def _average_string_length(series: pl.Series) -> float:
        values = series.drop_nulls().head(2_000).cast(pl.String)
        if values.is_empty():
            return 0.0
        return float(values.str.len_chars().mean())


def load_schema(schema_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load schema metadata as plain dictionaries."""

    return json.loads(Path(schema_path).read_text(encoding="utf-8"))
