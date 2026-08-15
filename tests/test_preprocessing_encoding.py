"""Tests for lazy preprocessing and reversible tabular encoding."""

import json

import numpy as np
import polars as pl

from datalus.data.encoding import (
    CategoricalVocabulary,
    NumericQuantileTransform,
    TabularEncoder,
)
from datalus.data.ingestion import ZeroShotPreprocessor


def test_lazy_preprocessing_and_reversible_encoding(tmp_path):
    """Ingest, encode, and decode a frame without losing data."""

    raw = tmp_path / "raw.csv"
    processed = tmp_path / "processed.parquet"
    schema_path = tmp_path / "schema_config.json"
    frame = pl.DataFrame(
        {
            "record_id": [f"id-{idx}" for idx in range(20)],
            "age": list(range(20)),
            "sex": ["M", "F"] * 10,
            "municipality": ["small-town"] + ["capital"] * 19,
            "target": [0, 1] * 10,
        }
    )
    frame.write_csv(raw)

    prep = ZeroShotPreprocessor(sample_size=20, target_column="target")
    prep.fit_transform_to_parquet(raw, processed, schema_path)
    schema = json.loads(schema_path.read_text())

    assert schema["record_id"]["encoding_strategy"] == "DROP"
    assert schema["municipality"]["rare_category_count"] == 1
    assert schema["municipality"]["rare_categories_preserved"] is True
    assert processed.exists()

    processed_frame = pl.read_parquet(processed)
    encoder = TabularEncoder(schema).fit(processed_frame)
    encoded = encoder.transform(processed_frame)
    decoded = encoder.inverse_transform(encoded.x_num, encoded.x_cat)

    assert encoded.x_num is not None
    assert encoded.x_cat is not None
    assert "target" in encoder.categorical_columns
    assert encoder.categorical_vocabs["municipality"].vocab["small-town"] > 1
    assert encoder.categorical_vocabs["municipality"].transform(np.array(["never-seen-town"]))[0] == 0
    assert np.isfinite(encoded.x_num).all()
    assert len(decoded) == len(processed_frame)


def test_rtdl_quantile_noise_fit_is_deterministic_and_reversible():
    """RTDL-style noise smooths quantile fitting without breaking roundtrip."""

    rng = np.random.default_rng(7)
    values = rng.normal(size=500).astype(np.float64)
    first = NumericQuantileTransform.fit("col", values, noise=1e-3, random_state=42)
    second = NumericQuantileTransform.fit("col", values, noise=1e-3, random_state=42)
    assert first.quantiles == second.quantiles
    encoded = first.transform(values)
    assert np.isfinite(encoded).all()
    decoded = first.inverse(encoded)
    assert np.allclose(decoded, values, atol=0.1)


def test_rtdl_dynamic_quantile_count_caps_at_floor():
    """The dynamic quantile rule uses min(n // 30, n_quantiles) with floor 10."""

    rng = np.random.default_rng(0)
    values = rng.normal(size=50)
    transform = NumericQuantileTransform.fit("col", values)
    transform_rtdl = NumericQuantileTransform.fit(
        "col", values, n_quantiles=1_000, noise=0.0
    )
    assert len(transform_rtdl.quantiles) == 50
    frame = pl.DataFrame({"col": values})
    encoder = TabularEncoder(
        {"col": {"inferred_topology": "NUMERICAL_CONTINUOUS", "encoding_strategy": "QUANTILE_TRANSFORM"}},
        rtdl_quantile_dynamic=True,
    ).fit(frame)
    assert len(encoder.numeric_transforms["col"].quantiles) == 10
    assert len(transform.quantiles) > 0


def test_outlier_clipping_maps_extremes_to_boundary():
    """Robust z-score clipping keeps extreme values inside the table bounds."""

    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1e6], dtype=np.float64)
    transform = NumericQuantileTransform.fit(
        "col", values, outlier_threshold=4.0, standardize=False
    )
    encoded = transform.transform(np.array([1e6]))
    assert np.isfinite(encoded).all()
    clipped = transform.transform(values)
    assert float(np.max(np.abs(clipped))) <= 1.0


def test_numeric_standardize_reverses_roundtrip():
    """Standardized quantile outputs have zero mean and invert back."""

    rng = np.random.default_rng(3)
    values = rng.exponential(size=400).astype(np.float64)
    transform = NumericQuantileTransform.fit(
        "col", values, standardize=True, noise=0.0
    )
    encoded = transform.transform(values)
    assert abs(float(np.mean(encoded))) < 0.1
    decoded = transform.inverse(encoded)
    assert np.allclose(decoded, values, atol=0.1)
    assert transform.to_dict()["mean"] is not None
    rebuilt = NumericQuantileTransform.from_dict(transform.to_dict())
    assert np.allclose(rebuilt.transform(values), encoded, atol=1e-6)


def test_categorical_frequency_mode_orders_most_frequent_first():
    """Frequency mode assigns the most common category the lowest index."""

    values = np.array(["rare", "common", "common", "common", "mid", "mid"])
    vocab = CategoricalVocabulary.fit("col", values, mode="frequency")
    assert vocab.vocab["common"] == 2
    assert vocab.vocab["mid"] == 3
    assert vocab.vocab["rare"] == 4


def test_categorical_appearance_mode_preserves_first_occurrence():
    """Appearance mode orders categories by first occurrence."""

    values = np.array(["b", "a", "b", "c", "a", "c"])
    vocab = CategoricalVocabulary.fit("col", values, mode="appearance")
    assert vocab.vocab["b"] == 2
    assert vocab.vocab["a"] == 3
    assert vocab.vocab["c"] == 4


def test_min_cat_frequency_collapses_rare_categories_to_unknown():
    """Categories below min_frequency map to __UNKNOWN__ at transform time."""

    values = np.array(["common", "common", "common", "common", "lonely"])
    vocab = CategoricalVocabulary.fit("col", values, min_frequency=2)
    assert "lonely" not in vocab.categories
    assert vocab.transform(np.array(["lonely"]))[0] == 0
    assert vocab.transform(np.array(["common"]))[0] == 2


def test_encoder_policy_roundtrips_through_json(tmp_path):
    """The TabFM-style encoder policy survives save and load."""

    frame = pl.DataFrame(
        {
            "age": [1.0, 2.0, 3.0, 4.0],
            "sex": ["M", "F", "M", "F"],
        }
    )
    schema = {
        "age": {"inferred_topology": "NUMERICAL_CONTINUOUS", "encoding_strategy": "QUANTILE_TRANSFORM"},
        "sex": {"inferred_topology": "CATEGORICAL_LOW_CARDINALITY",
            "encoding_strategy": "CONTINUOUS_EMBEDDING"},
    }
    encoder = TabularEncoder(
        schema,
        quantile_noise=1e-3,
        cat_encoder_mode="frequency",
        min_cat_frequency=2,
        random_state=5,
    ).fit(frame)
    path = tmp_path / "encoder_config.json"
    encoder.save(path)
    loaded = TabularEncoder.load(path)
    assert loaded.quantile_noise == encoder.quantile_noise
    assert loaded.cat_encoder_mode == "frequency"
    assert loaded.min_cat_frequency == 2
    assert loaded.rtdl_quantile_dynamic is False


def test_encoder_reports_augmented_columns():
    """Columns flagged augmented in the schema are reported by the encoder."""

    schema = {
        "age": {"inferred_topology": "NUMERICAL_CONTINUOUS", "encoding_strategy": "QUANTILE_TRANSFORM"},
        "aug_cross_0_1": {
            "inferred_topology": "NUMERICAL_CONTINUOUS",
            "encoding_strategy": "QUANTILE_TRANSFORM",
            "augmented": True,
        },
    }
    frame = pl.DataFrame({"age": [1.0, 2.0, 3.0], "aug_cross_0_1": [1.0, 4.0, 9.0]})
    encoder = TabularEncoder(schema).fit(frame)
    assert encoder.augmented_columns == ["aug_cross_0_1"]
    assert "age" not in encoder.augmented_columns
