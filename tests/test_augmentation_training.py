"""Tests for TabFM-inspired feature augmentation and end-to-end training."""

import json

import numpy as np
import polars as pl

from datalus.config import TrainingConfig
from datalus.data.augmentation import FeatureAugmentation
from datalus.data.encoding import TabularEncoder
from datalus.generation.workflows import balance_records, sample_records
from datalus.training.trainer import DatalusTrainer


def _make_dataset(tmp_path):
    """Write a small mixed-type dataset plus schema JSON and return paths."""

    rng = np.random.default_rng(0)
    n = 48
    frame = pl.DataFrame(
        {
            "age": rng.normal(45.0, 12.0, n),
            "income": rng.lognormal(11.0, 0.8, n),
            "sex": ["M", "F"] * (n // 2),
            "municipality": ["capital"] * 40 + ["small-town"] * 8,
            "target": ["0", "1"] * (n // 2),
        }
    )
    data_path = tmp_path / "train.parquet"
    frame.write_parquet(data_path)
    schema = {
        "age": {
            "inferred_topology": "NUMERICAL_CONTINUOUS",
            "encoding_strategy": "QUANTILE_TRANSFORM",
            "retained": True,
            "is_target": False,
        },
        "income": {
            "inferred_topology": "NUMERICAL_CONTINUOUS",
            "encoding_strategy": "QUANTILE_TRANSFORM",
            "retained": True,
            "is_target": False,
        },
        "sex": {
            "inferred_topology": "CATEGORICAL_LOW_CARDINALITY",
            "encoding_strategy": "CONTINUOUS_EMBEDDING",
            "retained": True,
            "is_target": False,
        },
        "municipality": {
            "inferred_topology": "CATEGORICAL_HIGH_CARDINALITY",
            "encoding_strategy": "CONTINUOUS_EMBEDDING",
            "retained": True,
            "is_target": False,
        },
        "target": {
            "inferred_topology": "CATEGORICAL_LOW_CARDINALITY",
            "encoding_strategy": "CONTINUOUS_EMBEDDING",
            "retained": True,
            "is_target": True,
        },
    }
    schema_path = tmp_path / "schema_config.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    return data_path, schema_path


def test_feature_crosses_augmentation_is_deterministic():
    """Crosses fit and transform produce the expected product columns."""

    frame = pl.DataFrame(
        {
            "age": [1.0, 2.0, 3.0],
            "income": [4.0, 5.0, 6.0],
            "sex": ["M", "F", "M"],
        }
    )
    schema = {
        "age": {"inferred_topology": "NUMERICAL_CONTINUOUS", "encoding_strategy": "QUANTILE_TRANSFORM"},
        "income": {"inferred_topology": "NUMERICAL_CONTINUOUS", "encoding_strategy": "QUANTILE_TRANSFORM"},
        "sex": {"inferred_topology": "CATEGORICAL_LOW_CARDINALITY",
            "encoding_strategy": "CONTINUOUS_EMBEDDING"},
    }
    encoder = TabularEncoder(schema).fit(frame)
    aug = FeatureAugmentation(mode="crosses", random_state=42).fit(encoder, frame)
    assert aug.column_names
    augmented = aug.transform_frame(frame, encoder)
    assert all(name in augmented.columns for name in aug.column_names)
    name = aug.column_names[0]
    i, j = aug.pairs[0]
    expected = encoder.transform(frame).x_num[:, i] * encoder.transform(frame).x_num[:, j]
    assert np.allclose(augmented[name].to_numpy(), expected, atol=1e-6)
    metadata = FeatureAugmentation.schema_metadata_for("crosses", aug.column_names)
    assert all(meta["augmented"] is True for meta in metadata.values())


def test_svd_augmentation_fit_and_transform_shapes():
    """SVD structural features fit on the sample and apply to batches."""

    frame = pl.DataFrame(
        {
            "age": np.linspace(0.0, 10.0, 20),
            "income": np.linspace(100.0, 200.0, 20),
            "sex": ["M", "F"] * 10,
        }
    )
    schema = {
        "age": {"inferred_topology": "NUMERICAL_CONTINUOUS", "encoding_strategy": "QUANTILE_TRANSFORM"},
        "income": {"inferred_topology": "NUMERICAL_CONTINUOUS", "encoding_strategy": "QUANTILE_TRANSFORM"},
        "sex": {"inferred_topology": "CATEGORICAL_LOW_CARDINALITY",
            "encoding_strategy": "CONTINUOUS_EMBEDDING"},
    }
    encoder = TabularEncoder(schema).fit(frame)
    aug = FeatureAugmentation(mode="svd", random_state=42).fit(encoder, frame)
    assert aug.column_names
    augmented = aug.transform_frame(frame, encoder)
    assert all(name in augmented.columns for name in aug.column_names)


def test_trainer_smoke_with_all_tabfm_techniques(tmp_path):
    """End-to-end training with augmentation, transformer, CFG, and CE loss."""

    data_path, schema_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "artifacts"
    config = TrainingConfig(
        schema_path=str(schema_path),
        data_path=str(data_path),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=16,
        hidden_dims=(16, 16),
        checkpoint_every_steps=1_000_000,
        save_every=1,
        denoiser_type="transformer",
        transformer_d_model=16,
        transformer_blocks=1,
        transformer_heads=4,
        transformer_num_inds=4,
        transformer_num_cls=2,
        transformer_ffn_chunk_size=4,
        transformer_row_chunk_size=4,
        target_column="target",
        lambda_cat=0.5,
        feature_augmentation="crosses",
        quantile_noise=1e-3,
    )
    trainer = DatalusTrainer(config)
    checkpoint = trainer.train(max_steps=2)
    assert checkpoint.exists()
    assert trainer.loss_history and np.isfinite(trainer.loss_history[-1])

    encoder_path = output_dir / "encoder_config.json"
    assert (output_dir / "augmented.parquet").exists()
    encoder = TabularEncoder.load(encoder_path)
    assert encoder.augmented_columns, "augmented columns must be serialized"

    synthetic = sample_records(
        checkpoint,
        encoder_path,
        n_records=16,
        ddim_steps=5,
        seed=7,
        cfg_scale=1.5,
        conditions={"target": "1"},
    )
    assert len(synthetic) == 16
    assert "target" in synthetic.columns
    assert all(name not in synthetic.columns for name in encoder.augmented_columns)
    assert synthetic["age"].is_null().sum() == 0

    balanced = balance_records(
        checkpoint,
        encoder_path,
        data_path,
        target_column="target",
        target_distribution={"0": 10, "1": 10},
        ddim_steps=5,
        seed=11,
        cfg_scale=1.5,
        max_attempts=3,
    )
    assert "target" in balanced.columns
    assert "age" in balanced.columns
    assert len(balanced) >= 48


def test_trainer_smoke_mlp_default_remains_stable(tmp_path):
    """The default MLP path trains and samples without augmentation."""

    data_path, schema_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "mlp_artifacts"
    config = TrainingConfig(
        schema_path=str(schema_path),
        data_path=str(data_path),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=16,
        hidden_dims=(16, 16),
        checkpoint_every_steps=1_000_000,
    )
    trainer = DatalusTrainer(config)
    checkpoint = trainer.train(max_steps=2)
    encoder_path = output_dir / "encoder_config.json"
    synthetic = sample_records(checkpoint, encoder_path, n_records=8, ddim_steps=5, seed=3)
    assert len(synthetic) == 8
    assert (output_dir / "augmented.parquet").exists() is False
