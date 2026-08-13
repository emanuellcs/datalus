"""Tests for checkpoint save, restore, pruning, and best-checkpoint tracking."""

import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from datalus.training.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
    update_best_checkpoint,
)


def test_rng_state_roundtrip():
    """Restoring captured RNG state reproduces the exact same draws."""
    seed_everything(123)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_prune_checkpoints_keeps_most_recent():
    """Verify that prune_checkpoints keeps the highest step-numbered files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for i in (0, 2, 4, 1, 3):
            (tmpdir / f"checkpoint_step_{i:08d}.pt").touch()

        prune_checkpoints(tmpdir, keep_last=3)

        remaining = sorted(tmpdir.glob("checkpoint_step_*.pt"))
        assert len(remaining) == 3

        remaining_names = [f.name for f in remaining]
        assert "checkpoint_step_00000002.pt" in remaining_names
        assert "checkpoint_step_00000003.pt" in remaining_names
        assert "checkpoint_step_00000004.pt" in remaining_names


def test_prune_preserves_special_files():
    """Verify that preserved files like latest/best are not pruned."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for i in range(5):
            (tmpdir / f"checkpoint_step_{i:08d}.pt").touch()

        (tmpdir / "checkpoint_latest.pt").touch()
        (tmpdir / "checkpoint_best.pt").touch()

        prune_checkpoints(tmpdir, keep_last=2)

        assert (tmpdir / "checkpoint_latest.pt").exists()
        assert (tmpdir / "checkpoint_best.pt").exists()

        regular = sorted(tmpdir.glob("checkpoint_step_*.pt"))
        assert len(regular) == 2


def test_update_best_checkpoint_tracks_lowest_loss():
    """Verify that best checkpoint is updated based on lowest loss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        ckpt1 = tmpdir / "checkpoint_step_00000001.pt"
        payload1 = {"loss": 0.5, "data": "checkpoint1"}
        save_checkpoint(ckpt1, payload1)

        update_best_checkpoint(tmpdir, 0.5, ckpt1)
        best_path = tmpdir / "checkpoint_best.pt"
        assert best_path.exists()

        best_ckpt = load_checkpoint(best_path)
        assert best_ckpt["loss"] == 0.5

        ckpt2 = tmpdir / "checkpoint_step_00000002.pt"
        payload2 = {"loss": 0.3, "data": "checkpoint2"}
        save_checkpoint(ckpt2, payload2)

        update_best_checkpoint(tmpdir, 0.3, ckpt2)
        best_ckpt = load_checkpoint(best_path)
        assert best_ckpt["loss"] == 0.3
        assert best_ckpt["data"] == "checkpoint2"

        ckpt3 = tmpdir / "checkpoint_step_00000003.pt"
        payload3 = {"loss": 0.4, "data": "checkpoint3"}
        save_checkpoint(ckpt3, payload3)

        update_best_checkpoint(tmpdir, 0.4, ckpt3)
        best_ckpt = load_checkpoint(best_path)
        assert best_ckpt["loss"] == 0.3
        assert best_ckpt["data"] == "checkpoint2"


def test_checkpoint_save_and_load():
    """Verify checkpoint save/load roundtrip preserves data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        payload = {
            "epoch": 5,
            "loss": 0.123,
            "model_state": torch.randn(10, 10),
            "metadata": {"key": "value"},
        }

        ckpt_path = tmpdir / "test_checkpoint.pt"
        save_checkpoint(ckpt_path, payload)

        assert ckpt_path.exists()

        loaded = load_checkpoint(ckpt_path)
        assert loaded["epoch"] == 5
        assert loaded["loss"] == 0.123
        assert torch.equal(loaded["model_state"], payload["model_state"])
        assert loaded["metadata"]["key"] == "value"
