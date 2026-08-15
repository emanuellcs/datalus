"""Tests for checkpoint cadence, rotation, integrity, and disk guards."""

import json
import logging
from collections import namedtuple

import numpy as np
import polars as pl
import pytest
import torch

from datalus.config import TrainingConfig
from datalus.training import checkpointing as ckpt
from datalus.training.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    validate_checkpoint_for_resume,
)
from datalus.training.trainer import DatalusTrainer


def _make_dataset(tmp_path, n_rows: int = 48):
    """Write a small mixed-type dataset plus schema JSON and return paths."""

    rng = np.random.default_rng(0)
    frame = pl.DataFrame(
        {
            "age": rng.normal(45.0, 12.0, n_rows),
            "income": rng.lognormal(11.0, 0.8, n_rows),
            "sex": ["M", "F"] * (n_rows // 2),
            "target": ["0", "1"] * (n_rows // 2),
        }
    )
    data_path = tmp_path / "train.parquet"
    frame.write_parquet(data_path)
    schema = {
        "age": {
            "inferred_topology": "NUMERICAL_CONTINUOUS",
            "encoding_strategy": "QUANTILE_TRANSFORM",
            "retained": True,
        },
        "income": {
            "inferred_topology": "NUMERICAL_CONTINUOUS",
            "encoding_strategy": "QUANTILE_TRANSFORM",
            "retained": True,
        },
        "sex": {
            "inferred_topology": "CATEGORICAL_LOW_CARDINALITY",
            "encoding_strategy": "CONTINUOUS_EMBEDDING",
            "retained": True,
        },
        "target": {
            "inferred_topology": "CATEGORICAL_LOW_CARDINALITY",
            "encoding_strategy": "CONTINUOUS_EMBEDDING",
            "retained": True,
        },
    }
    schema_path = tmp_path / "schema_config.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    return data_path, schema_path


def _trainer_config(data_path, schema_path, output_dir, **overrides):
    defaults = {
        "epochs": 1,
        "batch_size": 16,
        "hidden_dims": (16, 16),
        "checkpoint_every_steps": 0,
    }
    defaults.update(overrides)
    return TrainingConfig(
        schema_path=str(schema_path),
        data_path=str(data_path),
        output_dir=str(output_dir),
        **defaults,
    )


def _step_files(trainer):
    return sorted(trainer.checkpoint_dir.glob("checkpoint_step_*.pt"))


def test_save_every_writes_one_checkpoint_at_epoch_boundary(tmp_path):
    """Three batches in one epoch produce exactly one step checkpoint."""

    data_path, schema_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "artifacts"
    trainer = DatalusTrainer(
        _trainer_config(data_path, schema_path, output_dir, epochs=1, save_every=1)
    )
    checkpoint = trainer.train()
    files = _step_files(trainer)
    assert len(files) == 1
    assert checkpoint == files[0]
    latest = trainer.checkpoint_dir / "checkpoint_latest.pt"
    assert latest.is_symlink()
    assert latest.resolve() == files[0]
    assert load_checkpoint(latest)["global_step"] == 3


def test_save_every_two_epochs_skips_first_epoch(tmp_path):
    """save_every=2 saves at the second epoch boundary only."""

    data_path, schema_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "artifacts"
    trainer = DatalusTrainer(
        _trainer_config(data_path, schema_path, output_dir, epochs=3, save_every=2)
    )
    trainer.train()
    assert len(_step_files(trainer)) == 2


def test_step_cadence_saves_every_n_steps(tmp_path):
    """checkpoint_every_steps=2 writes step checkpoints at steps 2 and 4."""

    data_path, schema_path = _make_dataset(tmp_path, n_rows=64)
    output_dir = tmp_path / "artifacts"
    trainer = DatalusTrainer(
        _trainer_config(
            data_path,
            schema_path,
            output_dir,
            epochs=1,
            save_every=2,
            checkpoint_every_steps=2,
        )
    )
    trainer.train()
    steps = {path.stem.rsplit("_", 1)[1] for path in _step_files(trainer)}
    assert steps == {"00000002", "00000004"}


def test_step_cadence_zero_disables_intermediate_saves(tmp_path):
    """checkpoint_every_steps=0 leaves only the final step checkpoint."""

    data_path, schema_path = _make_dataset(tmp_path, n_rows=64)
    output_dir = tmp_path / "artifacts"
    trainer = DatalusTrainer(
        _trainer_config(data_path, schema_path, output_dir, epochs=1, save_every=2)
    )
    trainer.train()
    assert len(_step_files(trainer)) == 1


def test_max_steps_mid_epoch_saves_single_checkpoint(tmp_path):
    """Stopping mid-epoch writes one step file plus the latest symlink."""

    data_path, schema_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "artifacts"
    trainer = DatalusTrainer(
        _trainer_config(data_path, schema_path, output_dir, epochs=1)
    )
    checkpoint = trainer.train(max_steps=2)
    assert len(_step_files(trainer)) == 1
    assert checkpoint.name == "checkpoint_step_00000002.pt"
    latest = trainer.checkpoint_dir / "checkpoint_latest.pt"
    assert latest.is_symlink() and latest.resolve() == checkpoint


def test_keep_last_default_rotates_to_three(tmp_path):
    """The default rotation keeps only the three newest step checkpoints."""

    data_path, schema_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "artifacts"
    trainer = DatalusTrainer(
        _trainer_config(data_path, schema_path, output_dir, epochs=5)
    )
    trainer.train()
    files = _step_files(trainer)
    assert len(files) == 3
    assert files[-1].stem.endswith("00000015")


def test_keep_last_zero_keeps_all(tmp_path):
    """keep_last=0 disables rotation and keeps every step checkpoint."""

    data_path, schema_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "artifacts"
    trainer = DatalusTrainer(
        _trainer_config(data_path, schema_path, output_dir, epochs=5, keep_last=0)
    )
    trainer.train()
    assert len(_step_files(trainer)) == 5


def test_rng_state_is_portable_and_roundtrips():
    """Captured RNG state contains no numpy arrays and restores exactly."""

    ckpt.seed_everything(123)
    state = capture_rng_state()

    def has_ndarray(value):
        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, dict):
            return any(has_ndarray(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(has_ndarray(item) for item in value)
        return False

    assert not has_ndarray(state)

    ckpt.seed_everything(123)
    expected = (np.random.random(), torch.rand(1))
    restore_rng_state(state)
    actual = (np.random.random(), torch.rand(1))
    assert expected[0] == actual[0]
    assert torch.equal(expected[1], actual[1])


def test_stale_tmp_files_are_cleaned(tmp_path):
    """Interrupted-save temporary files are removed on the next save."""

    stale = [tmp_path / "checkpoint_step_00000042.pt.tmp", tmp_path / "checkpoint_latest.pt.tmp"]
    for path in stale:
        path.write_bytes(b"partial")
    save_checkpoint(tmp_path / "checkpoint_step_00000001.pt", {"loss": 0.5})
    assert all(not path.exists() for path in stale)


def test_legacy_checkpoint_loads_with_warning(tmp_path, caplog):
    """Pre-portable checkpoints fall back to the legacy loader with a warning."""

    path = tmp_path / "legacy.pt"
    payload = {
        "epoch": 1,
        "loss": 0.1,
        "rng_state": {"numpy": np.random.get_state()},
    }
    torch.save(payload, path)
    with caplog.at_level(logging.WARNING, logger="datalus.training.checkpointing"):
        loaded = load_checkpoint(path)
    assert loaded["loss"] == 0.1
    assert any("weights-only" in message or "legacy" in message for message in caplog.messages)


def test_corrupt_checkpoint_raises_clear_error(tmp_path):
    """A corrupt file produces an actionable error instead of a pickle crash."""

    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(RuntimeError, match="corrupt or not a DATALUS checkpoint"):
        load_checkpoint(path)


def test_validate_checkpoint_for_resume_rejects_architecture_change():
    """Changing architecture-defining settings blocks resume."""

    config = TrainingConfig(
        schema_path="a",
        data_path="b",
        output_dir="c",
        hidden_dims=(16, 16),
    )
    serialized = config.model_dump(mode="json")
    checkpoint = {
        "diffusion_state": {},
        "projector_state": {},
        "optimizer_state": {},
        "scheduler_state": {},
        "scaler_state": {},
        "ema_state": {},
        "rng_state": {},
        "epoch": 1,
        "batch_index": 0,
        "global_step": 3,
        "config": dict(serialized),
        "config_hash": "same",
    }
    validate_checkpoint_for_resume(checkpoint, serialized)

    changed = config.model_dump(mode="json")
    changed["hidden_dims"] = [8, 8]
    with pytest.raises(ValueError, match="hidden_dims"):
        validate_checkpoint_for_resume(checkpoint, changed)

    legacy = dict(checkpoint)
    legacy["config"] = {"hidden_dims": [16, 16]}
    validate_checkpoint_for_resume(legacy, serialized)


def test_validate_checkpoint_for_resume_missing_keys():
    """Checkpoints missing required keys are rejected with a clear message."""

    config = TrainingConfig(schema_path="a", data_path="b", output_dir="c")
    with pytest.raises(ValueError, match="missing required keys"):
        validate_checkpoint_for_resume({"config": {}}, config.model_dump(mode="json"))


def test_resume_restores_global_step(tmp_path):
    """Resuming from a checkpoint continues from the saved step."""

    data_path, schema_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "artifacts"
    config = _trainer_config(data_path, schema_path, output_dir, epochs=1)
    trainer = DatalusTrainer(config)
    checkpoint = trainer.train(max_steps=2)

    resumed = DatalusTrainer(config)
    resumed.resume(checkpoint)
    assert resumed.global_step == 2


def test_free_space_guard_warns_and_blocks(tmp_path, caplog, monkeypatch):
    """Low free space warns; critically low space refuses to write."""

    import shutil

    DiskUsage = namedtuple("DiskUsage", "total used free")
    data_path, schema_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "artifacts"
    trainer = DatalusTrainer(
        _trainer_config(data_path, schema_path, output_dir, epochs=1, min_free_space_gb=2.0)
    )

    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: DiskUsage(10 * 1024**3, 8.5 * 1024**3, 1.5 * 1024**3),
    )
    with caplog.at_level(logging.WARNING, logger="datalus.training.trainer"):
        trainer.save_checkpoint(0, 1, 0.5)
    assert any("min-free-space" in message for message in caplog.messages)

    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: DiskUsage(10 * 1024**3, 10 * 1024**3 - 100 * 1024**2, 100 * 1024**2),
    )
    with pytest.raises(RuntimeError, match="Insufficient free disk space"):
        trainer.save_checkpoint(0, 1, 0.5)
